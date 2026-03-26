"""主训练脚本：预处理 -> FD-MCNN -> 注意力 -> BiLSTM 全流程。"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.data_preprocess.preprocess import DataPreprocessor
from src.first_network.fd_mcnn import FDMcnnDetector
from src.attention_mechanism.attention_layer import WeakSignalAttention
from src.second_network.bilstm_classifier import BiLstmClassifier

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None

BASE_CLASSES = ["2ASK", "4FSK", "16QAM", "AM", "FM"]
BASE_TO_ID = {c: i for i, c in enumerate(BASE_CLASSES)}
MIXED_CLASSES = [f"{s}+{w}" for s in BASE_CLASSES for w in BASE_CLASSES if s != w]
MIXED_TO_ID = {c: i for i, c in enumerate(MIXED_CLASSES)}


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
        help="预处理后数据根目录，包含 train/test/val 及 metadata.csv",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(r"E:\1My_Research_Content\SCI_code\experimental_results\30dB\data_results"),
        help="保存指标、混淆矩阵的目录",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(r"E:\1My_Research_Content\SCI_code\experimental_results\30dB\model_results"),
        help="保存模型权重的目录",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-preprocess", action="store_true", help="若需重新运行预处理")
    parser.add_argument(
        "--test-checkpoint",
        type=str,
        choices=["last", "best"],
        default="best",
        help="测试时使用的权重：last=最后一轮，best=验证最佳",
    )
    parser.add_argument(
        "--feat-dim",
        type=int,
        choices=[128, 256, 512, 2048],
        default=128,
        help="attention 输出/投影后的序列长度，影响 BiLSTM 计算量。",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="梯度累积步数，用于显存不足时扩大等效 batch。",
    )
    parser.add_argument(
        "--amp",
        type=str,
        choices=["off", "on", "auto"],
        default="off",
        help="自动混合精度开关：off=关闭，on=开启，auto=仅CUDA开启。",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（<=0 表示关闭）。",
    )
    parser.add_argument(
        "--nan-log-max-batches",
        type=int,
        default=3,
        help="每个阶段最多打印多少个 NaN/Inf batch 诊断日志。",
    )
    parser.add_argument(
        "--nan-detect",
        dest="nan_detect",
        action="store_true",
        default=True,
        help="启用 NaN/Inf 检测并打印定位日志。",
    )
    parser.add_argument(
        "--no-nan-detect",
        dest="nan_detect",
        action="store_false",
        help="关闭 NaN/Inf 检测日志。",
    )
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
        self.samples = self._read_metadata_rows(metadata_path)

    @staticmethod
    def _read_metadata_rows(metadata_path: Path) -> List[Dict[str, str]]:
        tried_encodings = ["utf-8", "utf-8-sig", "gb18030"]
        last_error: Optional[Exception] = None
        required_fields = {"iq", "stft", "class_name"}

        for enc in tried_encodings:
            try:
                with metadata_path.open("r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise RuntimeError(f"metadata 文件为空或缺少表头: {metadata_path}")
                    missing = required_fields.difference(set(reader.fieldnames))
                    if missing:
                        missing_str = ", ".join(sorted(missing))
                        raise RuntimeError(f"metadata 缺少必需列: {missing_str} ({metadata_path})")

                    rows: List[Dict[str, str]] = []
                    for row in reader:
                        # 跳过空行，防止出现全 None 或空字符串记录。
                        if not row or all((v is None or str(v).strip() == "") for v in row.values()):
                            continue
                        rows.append({k: (v if v is not None else "") for k, v in row.items()})
                    return rows
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                continue

        enc_text = ", ".join(tried_encodings)
        raise RuntimeError(
            f"无法读取 metadata 文件: {metadata_path}. 已尝试编码: {enc_text}. "
            f"最后错误: {last_error}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        iq_path = Path(row["iq"])
        stft_path = Path(row["stft"])
        class_name = row["class_name"]

        iq = torch.from_numpy(np.load(iq_path)).float()  # (2,2048)
        stft = torch.from_numpy(np.load(stft_path)).float()  # (384,256)

        strong_id, weak_id = parse_strong_weak(class_name)
        strong_tgt = one_hot(strong_id)
        weak_tgt = one_hot(weak_id)

        return iq, stft, strong_tgt, weak_tgt, class_name


class FeatureProjector(nn.Module):
    """轻量级序列长度调整，避免大矩阵线性映射带来的计算开销。"""

    def __init__(self, target_len: int) -> None:
        super().__init__()
        self.target_len = target_len

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 2, L)
        if x.size(-1) == self.target_len:
            return x
        return F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)


def build_loaders(data_root: Path, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        meta = data_root / split / "metadata.csv"
        if not meta.exists():
            continue
        ds = SignalDataset(data_root / split, meta)
        shuffle = split == "train"
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": False,
            "pin_memory": True,
        }
        if num_workers > 0:
            kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        loaders[split] = DataLoader(ds, **kwargs)
    return loaders


def compute_confusion(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> np.ndarray:
    if confusion_matrix is not None:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # fallback manual
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def mixed_id_from_pair(strong_id: int, weak_id: int) -> int:
    return MIXED_TO_ID[f"{BASE_CLASSES[strong_id]}+{BASE_CLASSES[weak_id]}"]


def choose_weak_for_mixed(weak_logit_row: Tensor, strong_pred: int) -> int:
    # 20类混叠信号不包含强弱同类组合；若预测相同，取弱信号的次优类别。
    for idx in weak_logit_row.argsort(descending=True).tolist():
        if idx != strong_pred:
            return int(idx)
    return int((strong_pred + 1) % len(BASE_CLASSES))


def _bar(v: float, width: int = 22) -> str:
    v = max(0.0, min(1.0, float(v)))
    n = int(round(v * width))
    return "█" * n + "·" * (width - n)


def print_epoch_dashboard(
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc_s: float,
        train_acc_w: float,
        val_loss: float,
        val_acc_s: float,
        val_acc_w: float,
) -> None:
    train_acc_mean = 0.5 * (train_acc_s + train_acc_w)
    val_acc_mean = 0.5 * (val_acc_s + val_acc_w)
    print(
        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
        f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
        f"| TrainAcc={train_acc_mean:.3f} ({_bar(train_acc_mean)}) "
        f"| ValAcc={val_acc_mean:.3f} ({_bar(val_acc_mean)})"
    )


def print_test_dashboard(test_loss: float, test_acc_mixed: float) -> None:
    # 将 loss 映射到 [0,1] 区间用于文本条形图（loss 越小条越长）
    loss_score = 1.0 / (1.0 + max(0.0, test_loss))
    print(
        f"[Test Visual] MixedAcc={test_acc_mixed:.3f} ({_bar(test_acc_mixed)}) "
        f"| Loss={test_loss:.4f} ({_bar(loss_score)})"
    )


def get_next_run_dir(results_root: Path) -> Path:
    numeric_dirs: List[int] = []
    for p in results_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            numeric_dirs.append(int(p.name))
    next_id = (max(numeric_dirs) + 1) if numeric_dirs else 0
    run_dir = results_root / str(next_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: Path) -> None:
    if plt is None:
        print(f"[Plot] matplotlib 未安装，跳过图像保存: {save_path}")
        return

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    num_classes = len(classes)
    side = max(8, 0.55 * num_classes)

    if sns is not None:
        sns.set_context("paper", font_scale=1.2)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
        fig, ax = plt.subplots(figsize=(side, side * 0.8), dpi=300)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={"size": 8},
            square=True,
            linewidths=0,
            linecolor=None,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
        ax.set_ylabel("True Label", fontsize=10, labelpad=10)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
    else:
        fig, ax = plt.subplots(figsize=(side, side * 0.8), dpi=300)
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
        ax.set_ylabel("True Label", fontsize=10, labelpad=10)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 混淆矩阵已保存: {save_path}")


def plot_loss_curves(train_losses: List[float], val_losses: List[float], save_path: Path) -> None:
    if plt is None:
        print(f"[Plot] matplotlib 未安装，跳过图像保存: {save_path}")
        return
    if not train_losses:
        print("[Plot] 无训练损失记录，跳过 Loss 曲线绘制。")
        return

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, color="#1f77b4", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, color="#ff7f0e", linewidth=2, label="Val Loss")
    ax.set_title("Train/Val Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Loss 曲线已保存: {save_path}")


def _format_tensor_stats(x: Tensor) -> str:
    x32 = x.detach().float()
    return (
        f"shape={tuple(x32.shape)} "
        f"min={x32.min().item():.4e} max={x32.max().item():.4e} mean={x32.mean().item():.4e}"
    )


def _check_finite_tensors(
        split: str,
        epoch: int,
        batch_idx: int,
        log_counter: List[int],
        max_logs: int,
        named_tensors: Dict[str, Tensor],
) -> bool:
    bad_items: List[Tuple[str, Tensor]] = []
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            bad_items.append((name, tensor))
    if not bad_items:
        return True

    if log_counter[0] < max_logs:
        print(f"[NaNDetect][{split}] Epoch={epoch} Batch={batch_idx} 检测到非有限值：")
        for name, tensor in bad_items:
            print(f"  - {name}: {_format_tensor_stats(tensor)}")
        log_counter[0] += 1
    return False


def train_one_epoch(
        loader: DataLoader,
        fd_mcnn: nn.Module,
        attn: nn.Module,
        bilstm: nn.Module,
        proj: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: GradScaler,
        grad_accum: int,
        epoch: int,
        amp_enabled: bool,
        grad_clip_norm: float,
        nan_detect: bool,
        nan_log_max_batches: int,
) -> Tuple[float, float, float]:
    fd_mcnn.train()
    attn.train()
    bilstm.train()
    proj.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct_first = 0
    correct_second = 0
    total = 0
    nan_logs = [0]
    params_for_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    for step, (iq, stft, y_strong, y_weak, _) in enumerate(loader, start=1):
        iq = iq.to(device, non_blocking=True)
        stft = stft.to(device, non_blocking=True)
        y_strong = y_strong.to(device, non_blocking=True)
        y_weak = y_weak.to(device, non_blocking=True)

        if nan_detect and not _check_finite_tensors(
                split="train-input",
                epoch=epoch,
                batch_idx=step,
                log_counter=nan_logs,
                max_logs=nan_log_max_batches,
                named_tensors={"iq": iq, "stft": stft, "y_strong": y_strong, "y_weak": y_weak},
        ):
            return float("nan"), float("nan"), float("nan")

        with autocast(enabled=amp_enabled):
            first_logits, feat = fd_mcnn(iq, stft)  # feat: (B,2,128)
            feat_proj = proj(feat)  # (B,2,feat_dim)
            attn_out = attn(feat_proj)
            second_logits = bilstm(attn_out)  # (B,5)

            loss_first = criterion(first_logits, y_strong)
            loss_second = criterion(second_logits, y_weak)
            loss = loss_first + loss_second

        if nan_detect and not _check_finite_tensors(
                split="train-forward",
                epoch=epoch,
                batch_idx=step,
                log_counter=nan_logs,
                max_logs=nan_log_max_batches,
                named_tensors={"first_logits": first_logits, "second_logits": second_logits, "loss": loss.unsqueeze(0)},
        ):
            return float("nan"), float("nan"), float("nan")

        batch_loss = loss.item()
        loss = loss / grad_accum
        scaler.scale(loss).backward()

        total_loss += batch_loss * iq.size(0)
        total += iq.size(0)

        first_pred = first_logits.argmax(dim=1)
        second_pred = second_logits.argmax(dim=1)
        strong_true = y_strong.argmax(dim=1)
        weak_true = y_weak.argmax(dim=1)

        correct_first += ((first_pred == strong_true) | (first_pred == weak_true)).sum().item()
        correct_second += ((second_pred == strong_true) | (second_pred == weak_true)).sum().item()

        if step % grad_accum == 0:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, grad_clip_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    print(f"[Warn][train] Epoch={epoch} Batch={step} 梯度非有限，跳过该优化步。")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # 处理未整除的残余梯度
    if len(loader) % grad_accum != 0:
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, grad_clip_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                print(f"[Warn][train] Epoch={epoch} 尾批次梯度非有限，跳过该优化步。")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if total == 0:
                    return float("nan"), float("nan"), float("nan")
                return total_loss / total, correct_first / total, correct_second / total
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if total == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / total, correct_first / total, correct_second / total


@torch.no_grad()
def eval_epoch(
        loader: DataLoader,
        fd_mcnn: nn.Module,
        attn: nn.Module,
        bilstm: nn.Module,
        proj: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        split_name: str,
        amp_enabled: bool,
        nan_detect: bool,
        nan_log_max_batches: int,
        collect_confusion: bool = False,
) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    fd_mcnn.eval();
    attn.eval();
    bilstm.eval();
    proj.eval()
    total_loss = 0.0
    correct_first = 0
    correct_second = 0
    total = 0
    y_true_s: List[int] = []
    y_pred_s: List[int] = []
    y_true_w: List[int] = []
    y_pred_w: List[int] = []
    nan_logs = [0]

    for step, (iq, stft, y_strong, y_weak, _) in enumerate(loader, start=1):
        iq = iq.to(device, non_blocking=True)
        stft = stft.to(device, non_blocking=True)
        y_strong = y_strong.to(device, non_blocking=True)
        y_weak = y_weak.to(device, non_blocking=True)

        if nan_detect and not _check_finite_tensors(
                split=f"{split_name}-input",
                epoch=epoch,
                batch_idx=step,
                log_counter=nan_logs,
                max_logs=nan_log_max_batches,
                named_tensors={"iq": iq, "stft": stft, "y_strong": y_strong, "y_weak": y_weak},
        ):
            return float("nan"), float("nan"), float("nan"), None, None

        with autocast(enabled=amp_enabled):
            first_logits, feat = fd_mcnn(iq, stft)
            feat_proj = proj(feat)
            attn_out = attn(feat_proj)
            second_logits = bilstm(attn_out)
            loss = criterion(first_logits, y_strong) + criterion(second_logits, y_weak)

        if nan_detect and not _check_finite_tensors(
                split=f"{split_name}-forward",
                epoch=epoch,
                batch_idx=step,
                log_counter=nan_logs,
                max_logs=nan_log_max_batches,
                named_tensors={"first_logits": first_logits, "second_logits": second_logits, "loss": loss.unsqueeze(0)},
        ):
            return float("nan"), float("nan"), float("nan"), None, None
        total_loss += loss.item() * iq.size(0)
        total += iq.size(0)

        first_pred = first_logits.argmax(dim=1)
        second_pred = second_logits.argmax(dim=1)
        strong_true = y_strong.argmax(dim=1)
        weak_true = y_weak.argmax(dim=1)

        correct_first += ((first_pred == strong_true) | (first_pred == weak_true)).sum().item()
        correct_second += ((second_pred == strong_true) | (second_pred == weak_true)).sum().item()

        if collect_confusion:
            y_true_s.extend(strong_true.cpu().tolist())
            y_pred_s.extend(first_pred.cpu().tolist())
            y_true_w.extend(weak_true.cpu().tolist())
            y_pred_w.extend(second_pred.cpu().tolist())

    cm_s: Optional[np.ndarray] = None
    cm_w: Optional[np.ndarray] = None
    if collect_confusion:
        cm_s = compute_confusion(y_true_s, y_pred_s)
        cm_w = compute_confusion(y_true_w, y_pred_w)
    if total == 0:
        return float("nan"), float("nan"), float("nan"), cm_s, cm_w
    return total_loss / total, correct_first / total, correct_second / total, cm_s, cm_w


@torch.no_grad()
def eval_test_mixed(
        loader: DataLoader,
        fd_mcnn: nn.Module,
        attn: nn.Module,
        bilstm: nn.Module,
        proj: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        amp_enabled: bool,
) -> Tuple[float, float, np.ndarray]:
    fd_mcnn.eval()
    attn.eval()
    bilstm.eval()
    proj.eval()
    total_loss = 0.0
    total = 0
    mixed_score = 0.0
    y_true_mixed: List[int] = []
    y_pred_mixed: List[int] = []

    for iq, stft, y_strong, y_weak, _ in loader:
        iq = iq.to(device, non_blocking=True)
        stft = stft.to(device, non_blocking=True)
        y_strong = y_strong.to(device, non_blocking=True)
        y_weak = y_weak.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            first_logits, feat = fd_mcnn(iq, stft)
            feat_proj = proj(feat)
            attn_out = attn(feat_proj)
            second_logits = bilstm(attn_out)

            loss = criterion(first_logits, y_strong) + criterion(second_logits, y_weak)
        total_loss += loss.item() * iq.size(0)
        total += iq.size(0)

        first_pred = first_logits.argmax(dim=1)
        second_pred = second_logits.argmax(dim=1)
        strong_true = y_strong.argmax(dim=1)
        weak_true = y_weak.argmax(dim=1)

        pred_label = torch.stack([first_pred, second_pred], dim=1)
        true_label = torch.stack([strong_true, weak_true], dim=1)

        both_true = 0
        one_true = 0

        for i in range(pred_label.size(0)):
            pred_set = set(pred_label[i].tolist())
            true_set = set(true_label[i].tolist())

            match_num = len(pred_set & true_set)

            if match_num == 2:
                both_true += 1
            elif match_num == 1:
                one_true += 1

        # both_true = ((first_pred == strong_true) & (second_pred == weak_true)).sum().item()
        # one_true = ((first_pred == strong_true) ^ (second_pred == weak_true)).sum().item()
        mixed_score += both_true + 0.5 * one_true

        for i in range(iq.size(0)):
            st = int(strong_true[i].item())
            wt = int(weak_true[i].item())
            sp = int(first_pred[i].item())
            wp = int(second_pred[i].item())
            if sp == wp:
                wp = choose_weak_for_mixed(second_logits[i], sp)
            y_true_mixed.append(mixed_id_from_pair(st, wt))
            y_pred_mixed.append(mixed_id_from_pair(sp, wp))

    cm_mixed = compute_confusion(y_true_mixed, y_pred_mixed, num_classes=len(MIXED_CLASSES))
    if total == 0:
        return float("nan"), float("nan"), cm_mixed
    return total_loss / total, mixed_score / total, cm_mixed


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
    run_dir = get_next_run_dir(args.results_root)
    print(f"[Result] 当前运行输出目录: {run_dir}")
    args.models_root.mkdir(parents=True, exist_ok=True)

    maybe_run_preprocess(args.data_root)

    loaders = build_loaders(args.data_root, args.batch_size, args.num_workers)
    if "train" not in loaders:
        raise RuntimeError("未找到训练集 metadata，请先运行预处理。")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    fd_mcnn = FDMcnnDetector(num_classes=5).to(device)
    proj = FeatureProjector(target_len=args.feat_dim).to(device)
    attn = WeakSignalAttention(
        embed_dim=args.feat_dim, num_heads=8, mlp_hidden=max(512, args.feat_dim * 4), se_reduction=8
    ).to(device)
    bilstm = BiLstmClassifier(num_classes=5).to(device)

    params = list(fd_mcnn.parameters()) + list(proj.parameters()) + list(attn.parameters()) + list(bilstm.parameters())
    optim_kwargs = {"lr": args.lr}
    if device.type == "cuda":
        optim_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(params, **optim_kwargs)
    except TypeError:
        optim_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(params, **optim_kwargs)
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    print(f"[Train] AMP={amp_enabled} | grad_clip_norm={args.grad_clip_norm} | nan_detect={args.nan_detect}")

    best_val_acc = 0.0
    best_ckpt_path = args.models_root / "model_best.pt"
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc_s, train_acc_w = train_one_epoch(
            loaders["train"],
            fd_mcnn,
            attn,
            bilstm,
            proj,
            criterion,
            optimizer,
            device,
            scaler,
            args.grad_accum,
            epoch,
            amp_enabled,
            args.grad_clip_norm,
            args.nan_detect,
            args.nan_log_max_batches,
        )

        val_loss = val_acc_s = val_acc_w = 0.0
        if "val" in loaders:
            val_loss, val_acc_s, val_acc_w, _, _ = eval_epoch(
                loaders["val"],
                fd_mcnn,
                attn,
                bilstm,
                proj,
                criterion,
                device,
                epoch,
                "val",
                amp_enabled,
                args.nan_detect,
                args.nan_log_max_batches,
                collect_confusion=False,
            )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print_epoch_dashboard(
            epoch, args.epochs, train_loss, train_acc_s, train_acc_w, val_loss, val_acc_s, val_acc_w
        )
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            print(f"[Stop] Epoch={epoch} 检测到 NaN/Inf，提前停止训练以避免污染后续参数。")
            break

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
                best_ckpt_path,
            )

    plot_loss_curves(train_loss_history, val_loss_history, run_dir / "curve_train_val_loss.png")

    # 测试集评估
    if "test" in loaders:
        if args.test_checkpoint == "best":
            if best_ckpt_path.exists():
                ckpt = torch.load(best_ckpt_path, map_location=device)
                fd_mcnn.load_state_dict(ckpt["fd_mcnn"])
                proj.load_state_dict(ckpt["proj"])
                attn.load_state_dict(ckpt["attn"])
                bilstm.load_state_dict(ckpt["bilstm"])
                print(f"[Test] Loaded best checkpoint from {best_ckpt_path}")
            else:
                print(
                    f"[Test] test-checkpoint=best, but checkpoint not found: {best_ckpt_path}. "
                    f"Fallback to last epoch weights."
                )

        test_loss, test_acc_mixed, cm_mixed = eval_test_mixed(
            loaders["test"], fd_mcnn, attn, bilstm, proj, criterion, device, amp_enabled
        )
        print(f"[Test] Loss {test_loss:.4f} | MixedAcc {test_acc_mixed:.3f}")
        print_test_dashboard(test_loss, test_acc_mixed)
        np.save(run_dir / "confusion_test_mixed.npy", cm_mixed)
        plot_confusion_matrix(
            cm_mixed,
            MIXED_CLASSES,
            title="Confusion Matrix (20-Class Mixed Signal)",
            save_path=run_dir / "confusion_test_mixed.png",
        )
        with (run_dir / "test_result.txt").open("w", encoding="utf-8") as f:
            f.write(f"loss={test_loss:.6f}\n")
            f.write(f"acc_mixed={test_acc_mixed:.6f}\n")


if __name__ == "__main__":
    main()
