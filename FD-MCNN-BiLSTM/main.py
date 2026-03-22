"""主训练脚本：预处理 -> FD-MCNN -> 注意力 -> BiLSTM 全流程。"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from data_preprocess.preprocess import DataPreprocessor
from first_network.fd_mcnn import FDMcnnDetector
from attention_mechanism.attention_layer import WeakSignalAttention
from second_network.bilstm_classifier import BiLstmClassifier

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None


BASE_CLASSES = ["2ASK", "4FSK", "16QAM", "AM", "FM"]
BASE_TO_ID = {c: i for i, c in enumerate(BASE_CLASSES)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FD-MCNN-BiLSTM end-to-end training")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\1My_Research_Content\SCI_code\dataset_preprocessed\30dB"),
        help="预处理后数据根目录，包含 train/test/val 与 metadata.csv",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(r"E:\1My_Research_Content\SCI_code\experimental_results\data_results"),
        help="保存指标、混淆矩阵的目录",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(r"E:\1My_Research_Content\SCI_code\experimental_results\model_results"),
        help="保存模型权重的目录",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-preprocess", action="store_true", help="如需重新运行预处理")
    return parser.parse_args()


def one_hot(idx: int, num_classes: int = 5) -> Tensor:
    t = torch.zeros(num_classes, dtype=torch.float32)
    t[idx] = 1.0
    return t


def parse_strong_weak(class_name: str) -> Tuple[int, int]:
    """从文件名解析强/弱基类，格式示例: '2×16QAM+2ASK'."""
    # 兼容 '×' 与 'x'
    parts = class_name.replace("×", "x").split("+")
    if len(parts) != 2:
        raise ValueError(f"无法解析强弱标签: {class_name}")
    strong_part = parts[0]  # 形如 '2x16QAM'
    weak_part = parts[1]
    def _strip_coeff(p: str) -> str:
        return p.split("x", 1)[-1]
    strong_cls = _strip_coeff(strong_part)
    weak_cls = _strip_coeff(weak_part)
    if strong_cls not in BASE_TO_ID or weak_cls not in BASE_TO_ID:
        raise ValueError(f"超出预期类别: {class_name}")
    return BASE_TO_ID[strong_cls], BASE_TO_ID[weak_cls]


class SignalDataset(Dataset):
    """读取预处理后的 IQ / STFT 与 metadata，生成强/弱标签。"""

    def __init__(self, split_dir: Path, metadata_path: Path) -> None:
        self.split_dir = split_dir
        with metadata_path.open("r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        header = lines[0].split(",")
        rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        iq_path = Path(row["iq"])
        stft_path = Path(row["stft"])
        class_name = row["class_name"]

        iq = torch.from_numpy(np.load(iq_path)).float()  # (2,2048)
        stft = torch.from_numpy(np.load(stft_path)).float()  # (384,256)
        stft = stft.unsqueeze(0)  # (1,384,256)

        strong_id, weak_id = parse_strong_weak(class_name)
        strong_tgt = one_hot(strong_id)
        weak_tgt = one_hot(weak_id)

        return iq, stft, strong_tgt, weak_tgt, class_name


def build_loaders(data_root: Path, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        meta = data_root / split / "metadata.csv"
        if not meta.exists():
            continue
        ds = SignalDataset(data_root / split, meta)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False
        )
    return loaders


def compute_confusion(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> np.ndarray:
    if confusion_matrix is not None:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # fallback manual
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def train_one_epoch(
    loader: DataLoader,
    fd_mcnn: nn.Module,
    attn: nn.Module,
    bilstm: nn.Module,
    proj: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float, float]:
    fd_mcnn.train(); attn.train(); bilstm.train(); proj.train()
    total_loss = 0.0
    correct_strong = 0
    correct_weak = 0
    total = 0
    for iq, stft, y_strong, y_weak, _ in loader:
        iq, stft = iq.to(device), stft.to(device)
        y_strong, y_weak = y_strong.to(device), y_weak.to(device)

        optimizer.zero_grad()

        strong_logits, feat = fd_mcnn(iq, stft)  # feat: (B,2,128)
        feat_proj = proj(feat)                   # (B,2,2048)
        attn_out = attn(feat_proj)               # 若 attention 改为 pass-through，则仍 (B,2,2048)
        weak_logits = bilstm(attn_out)           # (B,5)

        loss_strong = criterion(strong_logits, y_strong)
        loss_weak = criterion(weak_logits, y_weak)
        loss = loss_strong + loss_weak
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * iq.size(0)
        total += iq.size(0)
        correct_strong += (strong_logits.argmax(dim=1) == y_strong.argmax(dim=1)).sum().item()
        correct_weak += (weak_logits.argmax(dim=1) == y_weak.argmax(dim=1)).sum().item()

    return total_loss / total, correct_strong / total, correct_weak / total


@torch.no_grad()
def eval_epoch(
    loader: DataLoader,
    fd_mcnn: nn.Module,
    attn: nn.Module,
    bilstm: nn.Module,
    proj: nn.Module,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    fd_mcnn.eval(); attn.eval(); bilstm.eval(); proj.eval()
    total_loss = 0.0
    correct_strong = 0
    correct_weak = 0
    total = 0
    y_true_s: List[int] = []
    y_pred_s: List[int] = []
    y_true_w: List[int] = []
    y_pred_w: List[int] = []

    for iq, stft, y_strong, y_weak, _ in loader:
        iq, stft = iq.to(device), stft.to(device)
        y_strong, y_weak = y_strong.to(device), y_weak.to(device)

        strong_logits, feat = fd_mcnn(iq, stft)
        feat_proj = proj(feat)
        attn_out = attn(feat_proj)
        weak_logits = bilstm(attn_out)

        loss = criterion(strong_logits, y_strong) + criterion(weak_logits, y_weak)
        total_loss += loss.item() * iq.size(0)
        total += iq.size(0)

        s_pred = strong_logits.argmax(dim=1)
        w_pred = weak_logits.argmax(dim=1)
        s_true = y_strong.argmax(dim=1)
        w_true = y_weak.argmax(dim=1)
        correct_strong += (s_pred == s_true).sum().item()
        correct_weak += (w_pred == w_true).sum().item()

        y_true_s.extend(s_true.cpu().tolist()); y_pred_s.extend(s_pred.cpu().tolist())
        y_true_w.extend(w_true.cpu().tolist()); y_pred_w.extend(w_pred.cpu().tolist())

    cm_s = compute_confusion(y_true_s, y_pred_s)
    cm_w = compute_confusion(y_true_w, y_pred_w)
    return total_loss / total, correct_strong / total, correct_weak / total, cm_s, cm_w


def maybe_run_preprocess(data_root: Path) -> None:
    # 若 metadata 已存在则跳过；否则运行预处理
    if (data_root / "train" / "metadata.csv").exists():
        return
    raw_root = Path(r"E:\1My_Research_Content\SCI_code\dataset\30dB")
    pre = DataPreprocessor(data_root=raw_root, save_root=data_root)
    pre.process_all()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.models_root.mkdir(parents=True, exist_ok=True)

    maybe_run_preprocess(args.data_root)

    loaders = build_loaders(args.data_root, args.batch_size, args.num_workers)
    if "train" not in loaders:
        raise RuntimeError("未找到训练集 metadata，请先运行预处理。")

    device = torch.device(args.device)

    fd_mcnn = FDMcnnDetector(num_classes=5).to(device)
    # 线性投影，将 (B,2,128) 特征映射到 (B,2,2048)
    proj = nn.Linear(128, 2048).to(device)
    attn = WeakSignalAttention(embed_dim=2048, num_heads=8, mlp_hidden=4096, se_reduction=8).to(device)
    bilstm = BiLstmClassifier(num_classes=5).to(device)

    params = list(fd_mcnn.parameters()) + list(proj.parameters()) + list(attn.parameters()) + list(bilstm.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc_s, train_acc_w = train_one_epoch(
            loaders["train"], fd_mcnn, attn, bilstm, proj, criterion, optimizer, device
        )

        val_loss = val_acc_s = val_acc_w = 0.0
        cm_s = cm_w = None
        if "val" in loaders:
            val_loss, val_acc_s, val_acc_w, cm_s, cm_w = eval_epoch(
                loaders["val"], fd_mcnn, attn, bilstm, proj, criterion, device
            )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"TrainLoss {train_loss:.4f} | TrainAcc S {train_acc_s:.3f} W {train_acc_w:.3f} | "
            f"ValLoss {val_loss:.4f} | ValAcc S {val_acc_s:.3f} W {val_acc_w:.3f}"
        )

        # 保存最佳
        if val_acc_s + val_acc_w > best_val_acc and "val" in loaders:
            best_val_acc = val_acc_s + val_acc_w
            torch.save(
                {
                    "fd_mcnn": fd_mcnn.state_dict(),
                    "proj": proj.state_dict(),
                    "attn": attn.state_dict(),
                    "bilstm": bilstm.state_dict(),
                    "epoch": epoch,
                    "val_acc_s": val_acc_s,
                    "val_acc_w": val_acc_w,
                },
                args.models_root / "model_best.pt",
            )
            # 保存混淆矩阵
            if cm_s is not None:
                np.save(args.results_root / "confusion_val_strong.npy", cm_s)
                np.save(args.results_root / "confusion_val_weak.npy", cm_w)

    # 测试集评估
    if "test" in loaders:
        test_loss, test_acc_s, test_acc_w, cm_s, cm_w = eval_epoch(
            loaders["test"], fd_mcnn, attn, bilstm, proj, criterion, device
        )
        print(f"[Test] Loss {test_loss:.4f} | Acc S {test_acc_s:.3f} W {test_acc_w:.3f}")
        np.save(args.results_root / "confusion_test_strong.npy", cm_s)
        np.save(args.results_root / "confusion_test_weak.npy", cm_w)
        with (args.results_root / "metrics_test.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "loss": test_loss,
                    "acc_strong": test_acc_s,
                    "acc_weak": test_acc_w,
                    "classes": BASE_CLASSES,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
