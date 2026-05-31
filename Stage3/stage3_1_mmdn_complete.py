# -*- coding: utf-8 -*-
"""
Stage 3.1 complete MMDN reproduction-oriented implementation.

This script implements a practical version of the multi-scale memory dynamic-learning
network (MMDN) for speckle-to-pattern recovery:

1. fixed-state supervised pretraining without using future dynamic labels;
2. frozen StaticNN baseline;
3. three dynamic expert networks: S1, S2, and L3;
4. confidence-weighted ensemble inference;
5. self-supervised online update from pseudo-labels;
6. alternating short-memory rebuild for S1/S2;
7. long-memory replay update for L3;
8. metrics, plots, pseudo-labels, spatial accuracy maps, and representative examples.

Expected input files in DATA_DIR:
- Either one or more .npz files containing arrays, or separate .npy files.
- Required arrays: speckles, pattern/patterns.
- Optional arrays: time_index, counts_by_time, kappa_by_time, kappa_per_sample.

Typical shapes:
- speckles: (N, H, W), uint8/float32
- patterns: (N, P), binary or gray-level targets, e.g. P = 256 for 16 x 16 patterns.

Run:
    python stage3_1_mmdn_complete.py
    python stage3_1_mmdn_complete.py --data-dir "./mmdn_dynamic_1500m_16x16_100x100"

If DATA_DIR has no .npz/.npy files, the script automatically searches immediate
subfolders and prefers folders whose names contain "mmdn_dynamic".

Outputs are saved under OUTPUT_DIR.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 42

    # Use Path.cwd() when this script is placed in the dataset directory.
    data_dir: str = "."
    output_dir: str = "mmdn_stage3_1_complete_outputs"

    # Data controls. None means inferred from arrays or metadata.
    speckle_dim: Optional[int] = None
    pretrain_samples: Optional[int] = None
    samples_per_dynamic_state: Optional[int] = None
    limit_dynamic_states: Optional[int] = None
    normalize_speckles: bool = True
    per_sample_standardize: bool = False
    graylevel: int = 2

    # Train/validation split inside the initial fixed-state segment only.
    pretrain_val_fraction: float = 0.10

    # Online interval. If your dynamic segment is 5000 samples and this is 1000,
    # every segment is split into five 10-s-like online intervals.
    dynamic_chunk_size: Optional[int] = 1000

    # Optimization.
    batch_size: int = 128
    num_workers: int = 0
    pretrain_epochs: int = 60
    update_epochs: int = 8
    rebuild_epochs: int = 12
    early_stop_patience: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0

    # Model.
    model_width: int = 24
    dropout: float = 0.15

    # Memory logic.
    # S1/S2 are rebuilt alternately every rebuild_interval online updates.
    rebuild_interval: int = 5
    short_memory_chunks: int = 3
    long_memory_limit: Optional[int] = None
    anchor_samples_for_l3: int = 12000
    l3_replay_pseudo_samples: int = 12000
    short_replay_samples: int = 6000

    # Pseudo-label controls.
    use_soft_pseudo_labels: bool = False
    pseudo_conf_threshold: float = 0.72
    min_pseudo_keep_fraction: float = 0.35
    ensemble_temperature: float = 8.0

    # Evaluation / plotting.
    max_eval_samples_per_chunk: Optional[int] = None
    save_example_every: int = 999999  # normally only selected examples are saved later
    example_count: int = 8


CFG = Config()


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def canonical_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def has_np_dataset_files(path: Path) -> bool:
    return any(path.glob("*.npz")) or any(path.glob("*.npy"))


def dataset_dir_score(path: Path) -> Tuple[int, str]:
    """Rank candidate dataset folders. Smaller score is better."""
    name = canonical_name(path.name)
    score = 100
    if "mmdn_dynamic" in name:
        score -= 50
    if "16x16" in name:
        score -= 15
    if "100x100" in name:
        score -= 15
    if "output" in name or "outputs" in name:
        score += 80
    return score, str(path).lower()


def find_dataset_dirs(root: Path) -> List[Path]:
    """Find folders that directly contain .npz/.npy arrays."""
    candidates: List[Path] = []

    if has_np_dataset_files(root):
        candidates.append(root)

    # First search immediate children. This avoids accidentally scanning old outputs deeply.
    for child in sorted(root.iterdir() if root.exists() else []):
        if child.is_dir() and has_np_dataset_files(child):
            candidates.append(child)

    # If still nothing is found, search recursively.
    if not candidates and root.exists():
        seen = set()
        for file_path in list(root.rglob("*.npz")) + list(root.rglob("*.npy")):
            parent = file_path.parent.resolve()
            if parent not in seen:
                seen.add(parent)
                candidates.append(parent)

    # Remove output folders unless they are the only possible folders.
    non_output = [p for p in candidates if "output" not in canonical_name(p.name)]
    if non_output:
        candidates = non_output

    candidates = sorted(set(candidates), key=dataset_dir_score)
    return candidates


def resolve_dataset_dir(requested_dir: Path) -> Path:
    """Resolve the actual dataset directory.

    If the requested directory directly contains arrays, use it.
    Otherwise, search subfolders and prefer the real MMF dataset folder.
    """
    requested_dir = requested_dir.expanduser().resolve()
    if not requested_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {requested_dir}")

    if has_np_dataset_files(requested_dir):
        return requested_dir

    candidates = find_dataset_dirs(requested_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No .npz or .npy dataset files found in {requested_dir} or its subfolders."
        )

    # Use the best-ranked candidate automatically.
    best = candidates[0].resolve()
    print("No .npz/.npy files were found directly in DATA_DIR.")
    print("Automatically selected dataset subfolder:", best)
    if len(candidates) > 1:
        print("Other candidate data folders:")
        for cand in candidates[1:8]:
            print("  -", cand)
    return best


def load_arrays_from_npz(data_dir: Path) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for npz_path in sorted(data_dir.glob("*.npz")):
        archive = np.load(npz_path, allow_pickle=False)
        for key in archive.files:
            arrays[canonical_name(key)] = archive[key]
        if len(archive.files) == 1:
            arrays[canonical_name(npz_path.stem)] = archive[archive.files[0]]
    return arrays


def load_arrays_from_npy(data_dir: Path) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for npy_path in sorted(data_dir.glob("*.npy")):
        arrays[canonical_name(npy_path.stem)] = np.load(npy_path, mmap_mode="r")
    return arrays


def pick_array(
    arrays: Dict[str, np.ndarray],
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[np.ndarray]:
    for name in candidates:
        key = canonical_name(name)
        if key in arrays:
            return arrays[key]
    if required:
        raise KeyError(
            f"Could not find any of these arrays: {list(candidates)}. "
            f"Available keys: {sorted(arrays.keys())}"
        )
    return None


def infer_segments(
    n_samples: int,
    time_index: Optional[np.ndarray],
    counts_by_time: Optional[np.ndarray],
    metadata: Dict,
    cfg: Config,
) -> List[Dict[str, int]]:
    if counts_by_time is not None:
        counts = np.asarray(counts_by_time).astype(int).tolist()
    elif time_index is not None:
        unique, counts_arr = np.unique(np.asarray(time_index), return_counts=True)
        order = np.argsort(unique)
        counts = counts_arr[order].astype(int).tolist()
    else:
        hint = metadata.get("mmdn_training_hint", {})
        pre = int(cfg.pretrain_samples or hint.get("sizeOfPretrain", n_samples // 2))
        interval = int(cfg.samples_per_dynamic_state or hint.get("sizeOfUpdateInvertal", max(1, n_samples - pre)))
        dyn = max(0, (n_samples - pre) // interval)
        counts = [pre] + [interval] * dyn

    if sum(counts) > n_samples:
        raise ValueError(f"Segment counts sum to {sum(counts)}, but only {n_samples} samples exist.")

    segments: List[Dict[str, int]] = []
    start = 0
    for state_id, count in enumerate(counts):
        stop = start + int(count)
        segments.append({"state": state_id, "start": start, "stop": stop, "count": int(count)})
        start = stop

    if cfg.limit_dynamic_states is not None:
        segments = [segments[0]] + segments[1:1 + int(cfg.limit_dynamic_states)]

    if cfg.pretrain_samples is not None:
        segments[0]["stop"] = segments[0]["start"] + int(cfg.pretrain_samples)
        segments[0]["count"] = int(cfg.pretrain_samples)

    if cfg.samples_per_dynamic_state is not None:
        base = segments[0]["stop"]
        updated = [segments[0]]
        for i, old in enumerate(segments[1:], start=1):
            start = base + (i - 1) * int(cfg.samples_per_dynamic_state)
            stop = min(start + int(cfg.samples_per_dynamic_state), n_samples)
            if start < stop:
                updated.append({"state": old["state"], "start": start, "stop": stop, "count": stop - start})
        segments = updated

    return segments


def indices_for_segment(seg: Dict[str, int]) -> np.ndarray:
    return np.arange(int(seg["start"]), int(seg["stop"]), dtype=np.int64)


def split_train_val(indices: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    n_val = max(1, int(round(len(indices) * val_fraction)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def split_dynamic_indices(indices: np.ndarray, chunk_size: Optional[int]) -> List[np.ndarray]:
    if chunk_size is None or int(chunk_size) <= 0 or len(indices) <= int(chunk_size):
        return [indices]
    chunk_size = int(chunk_size)
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def infer_side_length(outsize: int) -> Optional[int]:
    side = int(round(math.sqrt(outsize)))
    return side if side * side == outsize else None


def safe_sample(indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= n:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=n, replace=False))


# ============================================================
# 3. Dataset and target processing
# ============================================================

class MMFDataset(Dataset):
    def __init__(
        self,
        x_array: np.ndarray,
        sample_indices: Sequence[int],
        labels: Optional[np.ndarray] = None,
        speckle_dim: Optional[int] = None,
        normalize: bool = True,
        per_sample_standardize: bool = False,
    ):
        self.x_array = x_array
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.speckle_dim = speckle_dim
        self.normalize = normalize
        self.per_sample_standardize = per_sample_standardize

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, pos: int):
        sample_id = int(self.sample_indices[pos])
        x = np.asarray(self.x_array[sample_id], dtype=np.float32)
        if self.normalize:
            # Support both uint8 [0,255] and float [0,1] datasets.
            if np.nanmax(x) > 2.0:
                x = x / 255.0
        if self.per_sample_standardize:
            x = (x - float(np.mean(x))) / (float(np.std(x)) + 1e-6)
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        if self.speckle_dim is not None and (
            x_tensor.shape[-2] != self.speckle_dim or x_tensor.shape[-1] != self.speckle_dim
        ):
            x_tensor = F.interpolate(
                x_tensor.unsqueeze(0),
                size=(self.speckle_dim, self.speckle_dim),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.labels is None:
            return x_tensor
        y = torch.from_numpy(self.labels[pos])
        return x_tensor, y


def normalize_targets(patterns: np.ndarray, graylevel: int) -> np.ndarray:
    y = np.asarray(patterns)
    if graylevel == 2:
        if y.max() > 1:
            y = (y > 0).astype(np.float32)
        else:
            y = y.astype(np.float32)
        return y
    y = y.astype(np.float32)
    max_val = float(graylevel - 1)
    if y.max() > 1.0:
        y = y / max_val
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def labels_for_indices(targets: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(targets[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def make_loader(
    speckles: np.ndarray,
    indices: Sequence[int],
    labels: Optional[np.ndarray],
    speckle_dim: int,
    cfg: Config,
    shuffle: bool,
    batch_size: Optional[int] = None,
) -> DataLoader:
    dataset = MMFDataset(
        speckles,
        indices,
        labels=labels,
        speckle_dim=speckle_dim,
        normalize=cfg.normalize_speckles,
        per_sample_standardize=cfg.per_sample_standardize,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size or cfg.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# 4. Model
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMDNSubNetwork(nn.Module):
    def __init__(self, outsize: int, speckle_dim: int, width: int = 24, dropout: float = 0.15):
        super().__init__()
        w = int(width)
        self.features = nn.Sequential(
            ConvBlock(1, w, stride=2, dropout=dropout),
            ConvBlock(w, 2 * w, stride=2, dropout=dropout),
            ConvBlock(2 * w, 4 * w, stride=2, dropout=dropout),
            ConvBlock(4 * w, 4 * w, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * w * 6 * 6, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, outsize),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def new_model(outsize: int, speckle_dim: int, cfg: Config, device: torch.device) -> nn.Module:
    return MMDNSubNetwork(outsize, speckle_dim, cfg.model_width, cfg.dropout).to(device)


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ============================================================
# 5. Inference, confidence, metrics
# ============================================================

def binarize_prob(prob: np.ndarray, graylevel: int) -> np.ndarray:
    prob = np.clip(np.asarray(prob), 0.0, 1.0)
    if graylevel == 2:
        return (prob >= 0.5).astype(np.float32)
    levels = graylevel - 1
    return (np.rint(prob * levels) / levels).astype(np.float32)


def confidence_per_sample(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    return np.mean(np.abs(prob - 0.5) * 2.0, axis=1)


def expert_weight_from_confidences(conf_means: Sequence[float], temperature: float) -> np.ndarray:
    logits = np.asarray(conf_means, dtype=np.float64) * float(temperature)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    return w / np.sum(w)


def predict_prob(
    model: nn.Module,
    speckles: np.ndarray,
    indices: Sequence[int],
    speckle_dim: int,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = make_loader(speckles, indices, labels=None, speckle_dim=speckle_dim, cfg=cfg, shuffle=False)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            pred = torch.sigmoid(model(x)).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def ensemble_predict(
    models: Dict[str, nn.Module],
    speckles: np.ndarray,
    indices: Sequence[int],
    speckle_dim: int,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    probs = {
        "s1": predict_prob(models["s1"], speckles, indices, speckle_dim, cfg, device),
        "s2": predict_prob(models["s2"], speckles, indices, speckle_dim, cfg, device),
        "l3": predict_prob(models["l3"], speckles, indices, speckle_dim, cfg, device),
    }
    conf = {name: float(np.mean(confidence_per_sample(prob))) for name, prob in probs.items()}
    w_arr = expert_weight_from_confidences([conf["s1"], conf["s2"], conf["l3"]], cfg.ensemble_temperature)
    weights = {"s1": float(w_arr[0]), "s2": float(w_arr[1]), "l3": float(w_arr[2])}
    prob_ens = weights["s1"] * probs["s1"] + weights["s2"] * probs["s2"] + weights["l3"] * probs["l3"]
    pred_ens = binarize_prob(prob_ens, cfg.graylevel)
    return pred_ens, prob_ens, probs, conf, weights


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    correct = np.isclose(y_true, y_pred)
    sample_acc = np.mean(correct, axis=1) * 100.0
    errors_per_sample = np.sum(~correct, axis=1)
    return {
        "pixel_accuracy": float(np.mean(correct) * 100.0),
        "bit_error_rate": float(1.0 - np.mean(correct)),
        "sample_accuracy_mean": float(np.mean(sample_acc)),
        "sample_accuracy_std": float(np.std(sample_acc)),
        "exact_match_percent": float(np.mean(np.all(correct, axis=1)) * 100.0),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "mean_error_pixels": float(np.mean(errors_per_sample)),
        "median_error_pixels": float(np.median(errors_per_sample)),
    }


def spatial_accuracy_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.isclose(y_true, y_pred), axis=0)


# ============================================================
# 6. Training and memory buffers
# ============================================================

class MemoryBuffer:
    def __init__(self, limit: Optional[int] = None):
        self.limit = limit
        self.indices = np.empty((0,), dtype=np.int64)
        self.labels = np.empty((0, 0), dtype=np.float32)
        self.confidence = np.empty((0,), dtype=np.float32)
        self.initialized = False

    def add(self, indices: Sequence[int], labels: np.ndarray, confidence: Optional[np.ndarray] = None) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float32)
        if confidence is None:
            confidence = np.ones(len(indices), dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        if len(indices) == 0:
            return
        if not self.initialized:
            self.indices = indices.copy()
            self.labels = labels.copy()
            self.confidence = confidence.copy()
            self.initialized = True
        else:
            self.indices = np.concatenate([self.indices, indices])
            self.labels = np.concatenate([self.labels, labels])
            self.confidence = np.concatenate([self.confidence, confidence])
        if self.limit is not None and len(self.indices) > int(self.limit):
            keep = int(self.limit)
            self.indices = self.indices[-keep:]
            self.labels = self.labels[-keep:]
            self.confidence = self.confidence[-keep:]

    def latest(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        n = min(int(n), len(self.indices))
        return self.indices[-n:], self.labels[-n:]

    def sample(self, n: int, seed: int, prefer_high_conf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.indices) <= int(n):
            return self.indices, self.labels
        rng = np.random.default_rng(seed)
        if prefer_high_conf:
            # Sample from the most reliable 70% to reduce pseudo-label drift.
            threshold = np.quantile(self.confidence, 0.30)
            pool = np.flatnonzero(self.confidence >= threshold)
            if len(pool) < int(n):
                pool = np.arange(len(self.indices))
            chosen = rng.choice(pool, size=int(n), replace=False)
        else:
            chosen = rng.choice(np.arange(len(self.indices)), size=int(n), replace=False)
        chosen = np.sort(chosen)
        return self.indices[chosen], self.labels[chosen]

    def __len__(self) -> int:
        return len(self.indices)


def run_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            total_loss += float(loss.item()) * x.size(0)
            total_count += x.size(0)
    return total_loss / max(1, total_count)


def fit_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    speckles: np.ndarray,
    train_indices: Sequence[int],
    train_labels: np.ndarray,
    speckle_dim: int,
    cfg: Config,
    device: torch.device,
    epochs: int,
    name: str,
    output_dir: Path,
    val_indices: Optional[Sequence[int]] = None,
    val_labels: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.float32)
    if len(train_indices) == 0:
        raise ValueError(f"{name}: empty training set")

    train_loader = make_loader(speckles, train_indices, train_labels, speckle_dim, cfg, shuffle=True)
    val_loader = None
    if val_indices is not None and val_labels is not None and len(val_indices) > 0:
        val_loader = make_loader(speckles, val_indices, val_labels, speckle_dim, cfg, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_metric = float("inf")
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip_norm))
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            total_count += x.size(0)

        train_loss = total_loss / max(1, total_count)
        val_loss = np.nan
        monitor = train_loss
        if val_loader is not None:
            val_loss = run_validation_loss(model, val_loader, criterion, device)
            monitor = val_loss

        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})
        print(f"{name} epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if monitor < best_metric - 1e-6:
            best_metric = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg.early_stop_patience):
                print(f"{name}: early stopping at epoch {epoch}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    optimizer.load_state_dict(best_optimizer_state)
    pd.DataFrame(history).to_csv(output_dir / f"{name}.csv", index=False)
    return history


def select_pseudo_labels(
    indices: np.ndarray,
    hard_labels: np.ndarray,
    soft_probs: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    conf = confidence_per_sample(soft_probs)
    keep_mask = conf >= float(cfg.pseudo_conf_threshold)
    min_keep = max(1, int(round(len(indices) * float(cfg.min_pseudo_keep_fraction))))
    if int(np.sum(keep_mask)) < min_keep:
        order = np.argsort(conf)[::-1]
        keep_mask = np.zeros_like(conf, dtype=bool)
        keep_mask[order[:min_keep]] = True
    selected_indices = np.asarray(indices)[keep_mask]
    selected_conf = conf[keep_mask]
    selected_labels = soft_probs[keep_mask] if cfg.use_soft_pseudo_labels else hard_labels[keep_mask]
    info = {
        "pseudo_keep_count": int(len(selected_indices)),
        "pseudo_keep_fraction": float(len(selected_indices) / max(1, len(indices))),
        "pseudo_conf_mean_all": float(np.mean(conf)),
        "pseudo_conf_mean_kept": float(np.mean(selected_conf)) if len(selected_conf) else float("nan"),
        "pseudo_conf_min_kept": float(np.min(selected_conf)) if len(selected_conf) else float("nan"),
    }
    return selected_indices, np.asarray(selected_labels, dtype=np.float32), selected_conf, info


# ============================================================
# 7. Plotting
# ============================================================

def save_line_plots(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    x = metrics_df["kappa"] if "kappa" in metrics_df and metrics_df["kappa"].notna().all() else metrics_df["online_update"]
    xlabel = "kappa" if "kappa" in metrics_df and metrics_df["kappa"].notna().all() else "online update"

    plt.figure(figsize=(10, 5))
    plt.plot(x, metrics_df["ensemble_pixel_acc"], marker="o", label="MMDN ensemble")
    plt.plot(x, metrics_df["s1_pixel_acc"], marker=".", alpha=0.80, label="S1 short memory")
    plt.plot(x, metrics_df["s2_pixel_acc"], marker=".", alpha=0.80, label="S2 short memory")
    plt.plot(x, metrics_df["l3_pixel_acc"], marker=".", alpha=0.80, label="L3 long memory")
    plt.plot(x, metrics_df["static_l3_pixel_acc"], marker="x", linestyle="--", label="StaticNN")
    plt.xlabel(xlabel)
    plt.ylabel("pixel accuracy (%)")
    plt.title("Stage 3.1 dynamic tracking accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_over_drift.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(metrics_df["online_update"], metrics_df["ensemble_bit_error_rate"], marker="o", label="MMDN ensemble")
    plt.plot(metrics_df["online_update"], metrics_df["static_bit_error_rate"], marker="x", linestyle="--", label="StaticNN")
    plt.yscale("log")
    plt.xlabel("online update")
    plt.ylabel("bit error rate")
    plt.title("Error-rate comparison")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bit_error_rate_log.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.stackplot(
        metrics_df["online_update"],
        metrics_df["weight_s1"],
        metrics_df["weight_s2"],
        metrics_df["weight_l3"],
        labels=["S1", "S2", "L3"],
        alpha=0.85,
    )
    plt.xlabel("online update")
    plt.ylabel("ensemble weight")
    plt.title("Confidence-based ensemble weights")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_weights.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(metrics_df["online_update"], metrics_df["pseudo_keep_fraction"], marker="o")
    plt.xlabel("online update")
    plt.ylabel("kept pseudo-label fraction")
    plt.title("Pseudo-label selection rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pseudo_label_keep_fraction.png", dpi=200)
    plt.close()


def save_spatial_map(acc_vec: np.ndarray, side: Optional[int], path: Path, title: str) -> None:
    if side is None:
        np.save(path.with_suffix(".npy"), acc_vec)
        return
    img = np.asarray(acc_vec).reshape(side, side)
    plt.figure(figsize=(4.6, 4.2))
    plt.imshow(img, vmin=0.0, vmax=1.0)
    plt.colorbar(label="accuracy")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_examples(
    y_true: np.ndarray,
    y_ens: np.ndarray,
    y_static: np.ndarray,
    side: Optional[int],
    path: Path,
    title: str,
    max_examples: int,
) -> None:
    if side is None:
        return
    n = min(max_examples, len(y_true))
    if n <= 0:
        return
    # Prefer examples with errors, otherwise take the first examples.
    err = np.sum(~np.isclose(y_true, y_ens), axis=1)
    order = np.argsort(err)[::-1]
    chosen = order[:n]

    fig, axes = plt.subplots(n, 4, figsize=(8, 2.1 * n))
    if n == 1:
        axes = axes[None, :]
    for row_i, idx in enumerate(chosen):
        truth = y_true[idx].reshape(side, side)
        ens = y_ens[idx].reshape(side, side)
        sta = y_static[idx].reshape(side, side)
        wrong = np.logical_xor(truth > 0.5, ens > 0.5).astype(float)
        imgs = [truth, ens, sta, wrong]
        names = ["truth", "ensemble", "static", "ensemble wrong"]
        for col_i, (img, name) in enumerate(zip(imgs, names)):
            axes[row_i, col_i].imshow(img, vmin=0, vmax=1)
            axes[row_i, col_i].set_title(name)
            axes[row_i, col_i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# 8. Main workflow
# ============================================================

def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested_data_dir = Path(cfg.data_dir).expanduser().resolve()
    data_dir = resolve_dataset_dir(requested_data_dir)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "pseudo_labels").mkdir(exist_ok=True)
    (output_dir / "spatial_maps").mkdir(exist_ok=True)
    (output_dir / "examples").mkdir(exist_ok=True)

    print("requested_data_dir:", requested_data_dir)
    print("actual_data_dir:", data_dir)
    print("output_dir:", output_dir)
    print("torch:", torch.__version__)
    print("device:", device)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    arrays = load_arrays_from_npz(data_dir)
    source_kind = "npz"
    if not arrays:
        arrays = load_arrays_from_npy(data_dir)
        source_kind = "npy"
    if not arrays:
        raise FileNotFoundError(f"No readable .npz or .npy dataset arrays found in {data_dir}")

    speckles = pick_array(arrays, ["speckles", "speckle", "x", "inputs"])
    patterns_raw = pick_array(arrays, ["pattern", "patterns", "y", "labels", "targets"])
    time_index = pick_array(arrays, ["time_index", "time", "state_index", "state"], required=False)
    counts_by_time = pick_array(arrays, ["counts_by_time", "counts"], required=False)
    kappa_by_time = pick_array(arrays, ["kappa_by_time"], required=False)
    kappa_per_sample = pick_array(arrays, ["kappa_per_sample", "kappa"], required=False)

    if speckles is None or patterns_raw is None:
        raise RuntimeError("Required arrays were not loaded correctly.")

    targets = normalize_targets(patterns_raw, cfg.graylevel)
    n_samples = int(targets.shape[0])
    outsize = int(targets.shape[1])
    side = infer_side_length(outsize)
    speckle_dim = int(cfg.speckle_dim or metadata.get("mmdn_training_hint", {}).get("speckle_dim", speckles.shape[1]))

    print("source_kind:", source_kind)
    print("available arrays:", sorted(arrays.keys()))
    print("speckles:", speckles.shape, speckles.dtype)
    print("targets:", targets.shape, targets.dtype, "min/max", float(np.min(targets)), float(np.max(targets)))
    print("speckle_dim:", speckle_dim, "outsize:", outsize, "pattern_side:", side)

    segments = infer_segments(n_samples, time_index, counts_by_time, metadata, cfg)
    if len(segments) < 2:
        raise ValueError("Need one fixed-state pretraining segment plus at least one dynamic segment.")

    segment_df = pd.DataFrame(segments)
    if kappa_by_time is not None and len(kappa_by_time) >= len(segment_df):
        segment_df["kappa"] = np.asarray(kappa_by_time)[:len(segment_df)]
    segment_df.to_csv(output_dir / "segments.csv", index=False)
    print(segment_df)

    pretrain_all_indices = indices_for_segment(segments[0])
    pretrain_train_idx, pretrain_val_idx = split_train_val(pretrain_all_indices, cfg.pretrain_val_fraction, cfg.seed)
    pretrain_train_y = labels_for_indices(targets, pretrain_train_idx)
    pretrain_val_y = labels_for_indices(targets, pretrain_val_idx)

    print("pretrain train samples:", len(pretrain_train_idx))
    print("pretrain val samples:", len(pretrain_val_idx))
    print("dynamic states:", len(segments) - 1)

    # Supervised base pretraining. This avoids dynamic-label leakage.
    base = new_model(outsize, speckle_dim, cfg, device)
    base_opt = make_optimizer(base, cfg)
    t0 = time.time()
    fit_model(
        base,
        base_opt,
        speckles,
        pretrain_train_idx,
        pretrain_train_y,
        speckle_dim,
        cfg,
        device,
        cfg.pretrain_epochs,
        "pretrain_base_no_future_leakage",
        output_dir,
        val_indices=pretrain_val_idx,
        val_labels=pretrain_val_y,
    )
    print(f"pretraining finished in {(time.time() - t0) / 60:.2f} min")

    models: Dict[str, nn.Module] = {
        "s1": new_model(outsize, speckle_dim, cfg, device),
        "s2": new_model(outsize, speckle_dim, cfg, device),
        "l3": new_model(outsize, speckle_dim, cfg, device),
        "static": new_model(outsize, speckle_dim, cfg, device),
    }
    for m in models.values():
        m.load_state_dict(copy.deepcopy(base.state_dict()))

    optimizers = {
        "s1": make_optimizer(models["s1"], cfg),
        "s2": make_optimizer(models["s2"], cfg),
        "l3": make_optimizer(models["l3"], cfg),
    }

    torch.save(base.state_dict(), output_dir / "models" / "pretrain_base.pt")
    torch.save(models["static"].state_dict(), output_dir / "models" / "static_baseline.pt")

    # Memory buffers.
    dynamic_total = sum(int(seg["count"]) for seg in segments[1:])
    long_limit = cfg.long_memory_limit or (len(pretrain_train_idx) + dynamic_total)
    long_memory = MemoryBuffer(limit=int(long_limit))
    short_memory = MemoryBuffer(limit=int(cfg.short_memory_chunks * (cfg.dynamic_chunk_size or segments[1]["count"])))

    # L3 anchor keeps clean fixed-state labels and prevents pure pseudo-label collapse.
    anchor_idx = safe_sample(pretrain_train_idx, int(cfg.anchor_samples_for_l3), cfg.seed + 100)
    anchor_y = labels_for_indices(targets, anchor_idx)
    long_memory.add(anchor_idx, anchor_y, confidence=np.ones(len(anchor_idx), dtype=np.float32))

    # Fixed-state validation evaluation.
    fixed_eval_idx = pretrain_val_idx
    fixed_true = labels_for_indices(targets, fixed_eval_idx)
    fixed_ens, fixed_prob, fixed_probs, fixed_conf, fixed_weights = ensemble_predict(
        models, speckles, fixed_eval_idx, speckle_dim, cfg, device
    )
    fixed_static = binarize_prob(predict_prob(models["static"], speckles, fixed_eval_idx, speckle_dim, cfg, device), cfg.graylevel)
    fixed_metrics = evaluate_prediction(fixed_true, fixed_ens)
    fixed_row = {
        "state": int(segments[0]["state"]),
        "samples_evaluated": int(len(fixed_eval_idx)),
        "ensemble_pixel_acc": fixed_metrics["pixel_accuracy"],
        "ensemble_exact_match": fixed_metrics["exact_match_percent"],
        "ensemble_mae": fixed_metrics["mae"],
        "static_l3_pixel_acc": evaluate_prediction(fixed_true, fixed_static)["pixel_accuracy"],
        "conf_s1": fixed_conf["s1"],
        "conf_s2": fixed_conf["s2"],
        "conf_l3": fixed_conf["l3"],
        "weight_s1": fixed_weights["s1"],
        "weight_s2": fixed_weights["s2"],
        "weight_l3": fixed_weights["l3"],
    }
    pd.DataFrame([fixed_row]).to_csv(output_dir / "fixed_state_validation_metrics.csv", index=False)
    print("fixed-state validation metrics:")
    print(pd.DataFrame([fixed_row]))

    rows: List[Dict[str, float]] = []
    saved_examples: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    global_update = 0
    rebuild_toggle = 0

    # Dynamic loop: predict current unlabeled batch, evaluate only for analysis,
    # then update models using pseudo-labels from this current batch for future batches.
    for state_step, seg in enumerate(segments[1:], start=1):
        state = int(seg["state"])
        state_indices = indices_for_segment(seg)
        chunks = split_dynamic_indices(state_indices, cfg.dynamic_chunk_size)
        print("=" * 90)
        print(f"dynamic state {state_step}/{len(segments)-1}, state={state}, samples={len(state_indices)}, chunks={len(chunks)}")

        for chunk_id, current_indices in enumerate(chunks, start=1):
            global_update += 1
            if cfg.max_eval_samples_per_chunk is not None and len(current_indices) > int(cfg.max_eval_samples_per_chunk):
                eval_indices = safe_sample(current_indices, int(cfg.max_eval_samples_per_chunk), cfg.seed + global_update)
            else:
                eval_indices = current_indices
            true_labels = labels_for_indices(targets, eval_indices)

            print("-" * 90)
            print(f"online_update={global_update}, state={state}, chunk={chunk_id}/{len(chunks)}, eval_samples={len(eval_indices)}")

            ens_pred, ens_prob, probs, conf, weights = ensemble_predict(
                models, speckles, eval_indices, speckle_dim, cfg, device
            )
            pred_s1 = binarize_prob(probs["s1"], cfg.graylevel)
            pred_s2 = binarize_prob(probs["s2"], cfg.graylevel)
            pred_l3 = binarize_prob(probs["l3"], cfg.graylevel)
            pred_static = binarize_prob(
                predict_prob(models["static"], speckles, eval_indices, speckle_dim, cfg, device),
                cfg.graylevel,
            )

            ens_m = evaluate_prediction(true_labels, ens_pred)
            s1_m = evaluate_prediction(true_labels, pred_s1)
            s2_m = evaluate_prediction(true_labels, pred_s2)
            l3_m = evaluate_prediction(true_labels, pred_l3)
            static_m = evaluate_prediction(true_labels, pred_static)

            kappa_value = np.nan
            if kappa_by_time is not None and state < len(kappa_by_time):
                kappa_value = float(np.asarray(kappa_by_time)[state])
            elif kappa_per_sample is not None:
                kappa_value = float(np.mean(np.asarray(kappa_per_sample)[eval_indices]))

            row = {
                "online_update": int(global_update),
                "state_step": int(state_step),
                "state": int(state),
                "chunk": int(chunk_id),
                "chunks_in_state": int(len(chunks)),
                "start": int(eval_indices[0]),
                "stop": int(eval_indices[-1]) + 1,
                "samples": int(len(eval_indices)),
                "kappa": kappa_value,
                "ensemble_pixel_acc": ens_m["pixel_accuracy"],
                "ensemble_bit_error_rate": ens_m["bit_error_rate"],
                "ensemble_exact_match": ens_m["exact_match_percent"],
                "ensemble_mae": ens_m["mae"],
                "ensemble_mean_error_pixels": ens_m["mean_error_pixels"],
                "s1_pixel_acc": s1_m["pixel_accuracy"],
                "s2_pixel_acc": s2_m["pixel_accuracy"],
                "l3_pixel_acc": l3_m["pixel_accuracy"],
                "static_l3_pixel_acc": static_m["pixel_accuracy"],
                "static_bit_error_rate": static_m["bit_error_rate"],
                "static_exact_match": static_m["exact_match_percent"],
                "static_mean_error_pixels": static_m["mean_error_pixels"],
                "conf_s1": conf["s1"],
                "conf_s2": conf["s2"],
                "conf_l3": conf["l3"],
                "weight_s1": weights["s1"],
                "weight_s2": weights["s2"],
                "weight_l3": weights["l3"],
            }

            # Save spatial map per update.
            np.save(output_dir / "spatial_maps" / f"spatial_acc_update_{global_update:04d}.npy", spatial_accuracy_map(true_labels, ens_pred))
            if side is not None:
                save_spatial_map(
                    spatial_accuracy_map(true_labels, ens_pred),
                    side,
                    output_dir / "spatial_maps" / f"spatial_acc_update_{global_update:04d}.png",
                    f"Spatial accuracy, update {global_update}",
                )

            # Generate pseudo-labels using the same eval/current interval.
            # If max_eval_samples_per_chunk is None, this is the full current chunk.
            pseudo_indices, pseudo_labels, pseudo_conf, pseudo_info = select_pseudo_labels(
                eval_indices,
                ens_pred,
                ens_prob,
                cfg,
            )
            row.update(pseudo_info)
            rows.append(row)

            np.save(output_dir / "pseudo_labels" / f"pseudo_update_{global_update:04d}_indices.npy", pseudo_indices)
            np.save(output_dir / "pseudo_labels" / f"pseudo_update_{global_update:04d}_labels.npy", pseudo_labels.astype(np.float32))
            np.save(output_dir / "pseudo_labels" / f"pseudo_update_{global_update:04d}_confidence.npy", pseudo_conf.astype(np.float32))

            # Keep examples for final visualization.
            if global_update == 1:
                saved_examples["first"] = (true_labels.copy(), ens_pred.copy(), pred_static.copy())
            saved_examples["last"] = (true_labels.copy(), ens_pred.copy(), pred_static.copy())
            if "worst_ensemble" not in saved_examples or row["ensemble_pixel_acc"] < saved_examples.get("worst_ensemble_acc", (1e9,))[0]:
                saved_examples["worst_ensemble"] = (true_labels.copy(), ens_pred.copy(), pred_static.copy())
                saved_examples["worst_ensemble_acc"] = (row["ensemble_pixel_acc"],)  # type: ignore[assignment]

            # Update memory buffers.
            long_memory.add(pseudo_indices, pseudo_labels, confidence=pseudo_conf)
            short_memory.add(pseudo_indices, pseudo_labels, confidence=pseudo_conf)

            metrics_df = pd.DataFrame(rows)
            metrics_df.to_csv(output_dir / "dynamic_tracking_metrics.csv", index=False)
            print(
                f"ensemble={row['ensemble_pixel_acc']:.4f}%, "
                f"static={row['static_l3_pixel_acc']:.4f}%, "
                f"kept_pseudo={row['pseudo_keep_fraction']:.3f}"
            )

            # Self-supervised model update for subsequent intervals.
            # L3: long memory with clean anchors + replayed pseudo labels.
            l3_pseudo_idx, l3_pseudo_y = long_memory.sample(
                int(cfg.l3_replay_pseudo_samples),
                seed=cfg.seed + 1000 + global_update,
                prefer_high_conf=True,
            )
            l3_train_idx = np.concatenate([anchor_idx, l3_pseudo_idx])
            l3_train_y = np.concatenate([anchor_y, l3_pseudo_y])
            fit_model(
                models["l3"],
                optimizers["l3"],
                speckles,
                l3_train_idx,
                l3_train_y,
                speckle_dim,
                cfg,
                device,
                cfg.update_epochs,
                f"update_{global_update:04d}_l3_long_memory",
                output_dir,
            )

            # S1/S2: short memory, alternating rebuild to implement forgetting.
            short_idx, short_y = short_memory.sample(
                int(cfg.short_replay_samples),
                seed=cfg.seed + 2000 + global_update,
                prefer_high_conf=True,
            )
            if len(short_idx) > 0:
                do_rebuild = (global_update % int(cfg.rebuild_interval) == 0)
                if do_rebuild:
                    rebuild_name = "s1" if rebuild_toggle % 2 == 0 else "s2"
                    rebuild_toggle += 1
                    print(f"Rebuilding {rebuild_name.upper()} from short-memory pseudo-labels")
                    models[rebuild_name] = new_model(outsize, speckle_dim, cfg, device)
                    models[rebuild_name].load_state_dict(copy.deepcopy(base.state_dict()))
                    optimizers[rebuild_name] = make_optimizer(models[rebuild_name], cfg)
                    fit_model(
                        models[rebuild_name],
                        optimizers[rebuild_name],
                        speckles,
                        short_idx,
                        short_y,
                        speckle_dim,
                        cfg,
                        device,
                        cfg.rebuild_epochs,
                        f"rebuild_{global_update:04d}_{rebuild_name}",
                        output_dir,
                    )

                # Frequent short-memory update for both short experts.
                for expert in ["s1", "s2"]:
                    fit_model(
                        models[expert],
                        optimizers[expert],
                        speckles,
                        short_idx,
                        short_y,
                        speckle_dim,
                        cfg,
                        device,
                        cfg.update_epochs,
                        f"update_{global_update:04d}_{expert}_short_memory",
                        output_dir,
                    )

            # Save rolling checkpoints.
            torch.save(models["s1"].state_dict(), output_dir / "models" / "latest_s1.pt")
            torch.save(models["s2"].state_dict(), output_dir / "models" / "latest_s2.pt")
            torch.save(models["l3"].state_dict(), output_dir / "models" / "latest_l3.pt")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "dynamic_tracking_metrics.csv", index=False)
    metrics_df.to_excel(output_dir / "dynamic_tracking_metrics.xlsx", index=False)

    torch.save(models["s1"].state_dict(), output_dir / "models" / "final_s1.pt")
    torch.save(models["s2"].state_dict(), output_dir / "models" / "final_s2.pt")
    torch.save(models["l3"].state_dict(), output_dir / "models" / "final_l3.pt")

    # Plots and summary.
    save_line_plots(metrics_df, output_dir)

    for name, value in saved_examples.items():
        if not isinstance(value, tuple):
            continue
        y_true, y_ens, y_static = value
        save_examples(
            y_true,
            y_ens,
            y_static,
            side,
            output_dir / "examples" / f"examples_{name}.png",
            f"Representative examples: {name}",
            int(cfg.example_count),
        )

    summary = {
        "fixed_state_validation_ensemble_pixel_acc": float(fixed_row["ensemble_pixel_acc"]),
        "dynamic_updates": int(len(metrics_df)),
        "dynamic_ensemble_mean_pixel_acc": float(metrics_df["ensemble_pixel_acc"].mean()) if len(metrics_df) else float("nan"),
        "dynamic_ensemble_min_pixel_acc": float(metrics_df["ensemble_pixel_acc"].min()) if len(metrics_df) else float("nan"),
        "dynamic_ensemble_final_pixel_acc": float(metrics_df["ensemble_pixel_acc"].iloc[-1]) if len(metrics_df) else float("nan"),
        "dynamic_static_mean_pixel_acc": float(metrics_df["static_l3_pixel_acc"].mean()) if len(metrics_df) else float("nan"),
        "mean_gain_over_static_percent_points": float((metrics_df["ensemble_pixel_acc"] - metrics_df["static_l3_pixel_acc"]).mean()) if len(metrics_df) else float("nan"),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("Stage 3.1 complete workflow finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved:", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3.1 MMDN dynamic self-supervised input recovery."
    )
    parser.add_argument("--data-dir", type=str, default=CFG.data_dir, help="Dataset folder. If this folder has no .npy/.npz files, subfolders are searched automatically.")
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir, help="Output folder.")
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--pretrain-epochs", type=int, default=CFG.pretrain_epochs)
    parser.add_argument("--update-epochs", type=int, default=CFG.update_epochs)
    parser.add_argument("--rebuild-epochs", type=int, default=CFG.rebuild_epochs)
    parser.add_argument("--dynamic-chunk-size", type=int, default=-1, help="Online update chunk size. Use -1 to infer/default.")
    parser.add_argument("--limit-dynamic-states", type=int, default=-1, help="Limit number of dynamic states. Use -1 for all.")
    parser.add_argument("--pretrain-samples", type=int, default=-1, help="Limit fixed-state pretraining samples. Use -1 for all available.")
    parser.add_argument("--samples-per-dynamic-state", type=int, default=-1, help="Limit samples per dynamic state. Use -1 for inferred/all available.")
    parser.add_argument("--pseudo-conf-threshold", type=float, default=CFG.pseudo_conf_threshold)
    parser.add_argument("--model-width", type=int, default=CFG.model_width)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    parser.add_argument("--cpu", action="store_true", help="Accepted for command compatibility. Device is still selected inside main; set CUDA_VISIBLE_DEVICES=-1 if needed.")
    return parser


def config_from_args() -> Config:
    args = build_arg_parser().parse_args()
    cfg = copy.deepcopy(CFG)
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.pretrain_epochs = args.pretrain_epochs
    cfg.update_epochs = args.update_epochs
    cfg.rebuild_epochs = args.rebuild_epochs
    cfg.pseudo_conf_threshold = args.pseudo_conf_threshold
    cfg.model_width = args.model_width
    cfg.dropout = args.dropout
    if args.dynamic_chunk_size >= 0:
        cfg.dynamic_chunk_size = args.dynamic_chunk_size
    if args.limit_dynamic_states >= 0:
        cfg.limit_dynamic_states = args.limit_dynamic_states
    if args.pretrain_samples >= 0:
        cfg.pretrain_samples = args.pretrain_samples
    if args.samples_per_dynamic_state >= 0:
        cfg.samples_per_dynamic_state = args.samples_per_dynamic_state
    return cfg


if __name__ == "__main__":
    main(config_from_args())
