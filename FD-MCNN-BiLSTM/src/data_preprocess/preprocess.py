"""IQ 数据预处理模块 / IQ data preprocessing pipeline.

功能概述
--------
1) 读取数据集根目录下的 .mat 复数 IQ 信号（字段名 data）。
2) 按 2048 采样点不重叠切分，尾部不足部分丢弃。
3) 每个切片生成两路输出：
   - 频域特征：STFT（nperseg=128, noverlap=120, nfft=1024）幅值谱，缩放到 384×256。
   - 时域特征：将 I/Q 拆分为 2×2048，并对每片分别做 min-max 归一化。
4) 结果按 split/train|test|val 落盘为 .npy 或 .pt，便于后续训练直接加载。

备注：
- 仅依赖 numpy/scipy；save_format="pt" 时需要 torch（可选）。
- 默认使用幅值谱（非 dB），后续可根据模型需要再调整。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.signal import stft

try:  # Optional dependency for progress visualization
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:  # Optional dependency for MATLAB v7.3 (.mat in HDF5 format)
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:  # Optional dependency for save_format='pt'
    import torch
except ImportError:  # pragma: no cover
    torch = None


class DataPreprocessor:
    """混叠信号数据预处理 / Aliased IQ preprocessing pipeline."""

    def __init__(
        self,
        data_root: Path | str,
        save_root: Path | str,
        slice_len: int = 2048,
        stft_win: int = 128,
        stft_overlap: int = 120,
        stft_nfft: int = 1024,
        spec_size: Tuple[int, int] = (384, 256),  # (H, W)
        save_format: str = "npy",
        dtype: np.dtype = np.float32,
        split_names: Sequence[str] = ("train", "test", "val"),
    ) -> None:
        self.data_root = Path(data_root)
        self.save_root = Path(save_root)
        self.slice_len = slice_len
        self.stft_win = stft_win
        self.stft_overlap = stft_overlap
        self.stft_nfft = stft_nfft
        self.spec_size = spec_size
        self.save_format = save_format.lower()
        self.dtype = dtype
        self.split_names = tuple(split_names)
        self.label_to_id = self._build_label_map()

        if self.save_format not in {"npy", "pt"}:
            raise ValueError("save_format must be 'npy' or 'pt'")
        if self.save_format == "pt" and torch is None:
            raise ImportError("save_format='pt' requires torch to be installed.")

    # 公共接口 --------------------------------------------------------------
    def process_all(self) -> Dict[str, List[Dict[str, str]]]:
        """处理所有 split（train/test/val），返回元数据列表。"""
        summary: Dict[str, List[Dict[str, str]]] = {}
        for split in self.split_names:
            summary[split] = self.process_split(split)
        self._save_label_map()
        return summary

    def process_split(self, split: str) -> List[Dict[str, str]]:
        """处理单个数据划分，返回该划分的切片元数据列表。"""
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        mat_files = sorted(split_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files under {split_dir}")

        results: List[Dict[str, str]] = []
        if tqdm is not None:
            mat_iter = tqdm(
                mat_files,
                total=len(mat_files),
                desc=f"[{split}] files",
                unit="file",
            )
        else:
            mat_iter = mat_files

        for mat_path in mat_iter:
            iq = self._load_mat(mat_path)
            slices = self._slice_iq(iq)
            basename = mat_path.stem
            class_name = basename
            class_id = self.label_to_id[class_name]

            if tqdm is not None:
                slice_iter = tqdm(
                    enumerate(slices),
                    total=len(slices),
                    desc=f"[{split}] {basename}",
                    unit="slice",
                    leave=False,
                )
            else:
                slice_iter = enumerate(slices)

            for idx, sl in slice_iter:
                spec = self._to_spectrogram(sl)
                iq_vec = self._to_iq_vector(sl)
                stft_path = self._save(
                    spec,
                    self.save_root / split / "stft" / f"{basename}_{idx}.{self.save_suffix}",
                )
                iq_path = self._save(
                    iq_vec,
                    self.save_root / split / "iq" / f"{basename}_{idx}.{self.save_suffix}",
                )
                results.append(
                    {
                        "source": str(mat_path),
                        "split": split,
                        "class_name": class_name,
                        "class_id": class_id,
                        "slice_index": idx,
                        "stft": stft_path,
                        "iq": iq_path,
                    }
                )
        self._save_split_metadata(split, results)
        return results

    # 具体步骤 --------------------------------------------------------------
    def _load_mat(self, path: Path) -> np.ndarray:
        """读取 .mat 文件并返回复数 IQ 向量。"""
        try:
            mat = loadmat(path)
            if "data" not in mat:
                raise KeyError(f"'data' field not found in {path}")
            arr = mat["data"].squeeze()
            return self._as_complex64(arr)
        except NotImplementedError as e:
            # MATLAB v7.3 files are HDF5-backed and require h5py.
            if "Please use HDF reader for matlab v7.3 files" not in str(e):
                raise
            if h5py is None:
                raise ImportError(
                    "MATLAB v7.3 file detected but h5py is not installed. "
                    "Please install h5py to read this file."
                ) from e

            with h5py.File(path, "r") as f:
                if "data" not in f:
                    raise KeyError(f"'data' field not found in {path}")
                arr = np.array(f["data"]).squeeze()
                return self._as_complex64(arr)

    def _as_complex64(self, arr: np.ndarray) -> np.ndarray:
        """Convert common MATLAB IQ storage layouts to 1-D complex64."""
        arr = np.asarray(arr).squeeze()

        # Case 1: already complex
        if np.iscomplexobj(arr):
            return np.asarray(arr, dtype=np.complex64).reshape(-1)

        # Case 2: structured dtype, e.g. [('real','<f4'), ('imag','<f4')]
        names = arr.dtype.names
        if names:
            lower_names = {n.lower(): n for n in names}
            if "real" in lower_names and "imag" in lower_names:
                real_key = lower_names["real"]
                imag_key = lower_names["imag"]
                out = arr[real_key].astype(np.float32, copy=False) + 1j * arr[
                    imag_key
                ].astype(np.float32, copy=False)
                return np.asarray(out, dtype=np.complex64).reshape(-1)

        # Case 3: last dim is 2 -> [real, imag]
        if arr.ndim >= 1 and arr.shape[-1] == 2:
            out = arr[..., 0].astype(np.float32, copy=False) + 1j * arr[
                ..., 1
            ].astype(np.float32, copy=False)
            return np.asarray(out, dtype=np.complex64).reshape(-1)

        # Fallback: treat as real-valued IQ (imag=0)
        return np.asarray(arr, dtype=np.float32).astype(np.complex64).reshape(-1)

    def _slice_iq(self, x: np.ndarray) -> np.ndarray:
        """按 slice_len 不重叠切片，尾部不足部分丢弃。"""
        n_full = len(x) // self.slice_len
        trimmed = x[: n_full * self.slice_len]
        return trimmed.reshape(n_full, self.slice_len)

    def _to_spectrogram(self, x_slice: np.ndarray) -> np.ndarray:
        """对单个切片执行 STFT，返回 RGB 频谱图 (3, H, W)。"""
        _, _, zxx = stft(
            x_slice,
            window="hann",
            nperseg=self.stft_win,
            noverlap=self.stft_overlap,
            nfft=self.stft_nfft,
            return_onesided=True,
            padded=False,
            boundary=None,
        )
        spec = np.abs(zxx)  # 幅值谱

        # Normalize to [0, 1] before RGB mapping.
        smin, smax = spec.min(), spec.max()
        if np.isclose(smax, smin):
            spec_norm = np.zeros_like(spec, dtype=np.float32)
        else:
            spec_norm = ((spec - smin) / (smax - smin)).astype(np.float32, copy=False)

        # Convert single-channel spectrogram to RGB (C, H, W).
        spec_rgb = self._pseudo_color_rgb(spec_norm)

        target_h, target_w = self.spec_size
        zoom_h = target_h / spec_rgb.shape[1]
        zoom_w = target_w / spec_rgb.shape[2]
        spec_resized = zoom(spec_rgb, (1.0, zoom_h, zoom_w), order=1)
        return spec_resized.astype(self.dtype, copy=False)

    def _pseudo_color_rgb(self, spec_norm: np.ndarray) -> np.ndarray:
        """Map normalized 2D spectrogram to RGB channels with shape (3, H, W)."""
        x = np.asarray(spec_norm, dtype=np.float32)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=0)

    def _to_iq_vector(self, x_slice: np.ndarray) -> np.ndarray:
        """拆分 I/Q 并做逐片 min-max 归一化，返回形状 (2, slice_len)。"""
        i = np.real(x_slice)
        q = np.imag(x_slice)
        iq = np.stack([i, q], axis=0)

        def _minmax(channel: np.ndarray) -> np.ndarray:
            cmin, cmax = channel.min(), channel.max()
            if np.isclose(cmax, cmin):
                return np.zeros_like(channel, dtype=self.dtype)
            return ((channel - cmin) / (cmax - cmin)).astype(self.dtype, copy=False)

        iq_norm = np.stack([_minmax(i), _minmax(q)], axis=0)
        return iq_norm

    def _save(self, arr: np.ndarray, path: Path) -> str:
        """根据格式保存数组，返回保存路径字符串。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_format == "npy":
            np.save(path, arr)
        else:  # pt
            assert torch is not None
            torch.save(torch.as_tensor(arr), path)
        return str(path)

    def _build_label_map(self) -> Dict[str, int]:
        """Build stable class-id mapping from all .mat file names under configured splits."""
        class_names: set[str] = set()
        for split in self.split_names:
            split_dir = self.data_root / split
            if not split_dir.exists():
                continue
            for mat_path in split_dir.glob("*.mat"):
                class_names.add(mat_path.stem)
        return {name: idx for idx, name in enumerate(sorted(class_names))}

    def _save_split_metadata(self, split: str, rows: List[Dict[str, str]]) -> None:
        """Save per-split metadata for training/eval label lookup."""
        out_path = self.save_root / split / "metadata.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "split",
            "class_name",
            "class_id",
            "source",
            "slice_index",
            "stft",
            "iq",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_label_map(self) -> None:
        """Save global class-name to class-id mapping."""
        out_path = self.save_root / "label_map.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.label_to_id, f, ensure_ascii=False, indent=2)

    # 属性 ------------------------------------------------------------------
    @property
    def save_suffix(self) -> str:
        return "pt" if self.save_format == "pt" else "npy"


# CLI 入口 ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IQ 数据预处理 (STFT + Min-Max).")
    parser.add_argument(
        "--data-root",
        type=Path,
        # required=True,
        default="E:/1My_Research_Content/SCI_code/dataset/30dB",
        help="原始数据根目录",  # E:/1My_Research_Content/SCI_code/dataset/30dB
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        # required=True,
        default="E:/1My_Research_Content/SCI_code/dataset_preprocessed/30dB",
        help="预处理结果输出目录",  # E:/1My_Research_Content/SCI_code/dataset_preprocessed/30dB
    )
    parser.add_argument(
        "--format",
        choices=["npy", "pt"],
        default="npy",
        help="输出格式：npy 或 pt (torch.save)。",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="输出数据类型。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre = DataPreprocessor(
        data_root=args.data_root,
        save_root=args.save_root,
        save_format=args.format,
        dtype=np.float32 if args.dtype == "float32" else np.float64,
    )
    summary = pre.process_all()

    total_slices = sum(len(v) for v in summary.values())
    print(f"预处理完成，共生成 {total_slices} 个切片。")
    for split, items in summary.items():
        print(f"  {split}: {len(items)} slices, 例：{items[0]['stft']} / {items[0]['iq']}")


if __name__ == "__main__":
    main()
