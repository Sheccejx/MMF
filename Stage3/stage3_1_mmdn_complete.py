# -*- coding: utf-8 -*-
"""
Stage 3.1 MMDN dynamic self-supervised tracking, tuned standalone version.

Purpose
-------
This script runs the Stage 3.1 MMDN-style dynamic input recovery experiment:

    speckle image -> 16 x 16 input pattern

It is designed for the dataset folder containing files such as:

    speckles.npy
    pattern.npy
    counts_by_time.npy
    kappa_by_time.npy
    kappa_per_sample.npy
    time_index.npy

Main features
-------------
1. Automatically finds the real dataset folder, including nested folders.
2. Uses the lightweight CNN and Adadelta setup from Stage3_v1.ipynb by default.
3. Exposes all important hyperparameters through argparse.
4. Avoids future-label leakage by default through a fixed-state train/validation split.
5. Can reproduce the legacy Stage3_v1 validation behavior if explicitly requested.
6. Prints complete logs to terminal and also saves them to run_log.txt.
7. Saves the same main figures as Stage3_v1.ipynb into a new folder named Stage3_1图片.
8. Saves metrics, model weights, pseudo-labels, and training histories.
9. Supports NVIDIA CUDA GPU acceleration, mixed precision AMP, channels-last tensors, and faster DataLoader settings.

Recommended run from the MMF root directory
-------------------------------------------
    python stage3_1_mmdn_tuned_full.py

Or explicitly specify the dataset folder
----------------------------------------
    python stage3_1_mmdn_tuned_full.py --data-dir "./mmdn_dynamic_1500m_16x16_100x100"

Fast test run
-------------
    python stage3_1_mmdn_tuned_full.py --pretrain-epochs 3 --update-epochs 1 --chunk-size 1000

Legacy Stage3_v1-like run, not recommended for formal reporting because it uses
first dynamic-state true labels for validation during pretraining
--------------------------------------------------------------------------
    python stage3_1_mmdn_tuned_full.py --pretrain-val-mode first_dynamic --fixed-eval-mode train_tail
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. Hyperparameter configuration
# ============================================================


@dataclass
class Config:
    # Paths
    data_dir: str = "."
    output_dir: Optional[str] = None
    output_folder_name: str = "Stage3_1图片"
    preferred_dataset_name: str = "mmdn_dynamic_1500m_16x16_100x100"

    # Reproducibility
    seed: int = 42

    # Data controls
    speckle_dim: Optional[int] = None
    normalize_speckles: bool = True
    pretrain_samples: Optional[int] = None
    samples_per_dynamic_state: Optional[int] = None
    limit_dynamic_states: Optional[int] = None
    pretrain_val_ratio: float = 0.10
    pretrain_val_mode: str = "holdout"  # holdout, first_dynamic, none
    fixed_eval_mode: str = "holdout"    # holdout, train_tail
    fixed_eval_count: int = 10000

    # Model hyperparameters, Stage3_v1-like by default
    base_channels: int = 8
    dropout: float = 0.40
    graylevel: int = 2

    # Optimization
    batch_size: int = 128
    pretrain_epochs: int = 40
    update_epochs: int = 20
    early_stop_patience: int = 6
    learning_rate: float = 0.10
    optimizer: str = "adadelta"  # adadelta, adamw, adam
    weight_decay: float = 0.0
    num_workers: int = 0

    # GPU / acceleration controls
    device: str = "auto"          # auto, cuda, cpu
    use_amp: bool = True           # mixed precision on CUDA
    amp_dtype: str = "float16"     # float16 or bfloat16
    channels_last: bool = True     # use NHWC memory format on CUDA for Conv2d
    pin_memory: bool = True        # faster CPU -> GPU transfer
    persistent_workers: bool = True
    prefetch_factor: int = 2
    cudnn_benchmark: bool = True
    deterministic: bool = False
    compile_model: bool = False    # optional torch.compile; default off for stability

    # Dynamic online tracking
    chunk_size: int = 1000
    update_train_window: Optional[int] = None
    rebuild_interval: int = 5
    memory_limit: Optional[int] = None

    # Pseudo-label filtering. Default 0.0 reproduces Stage3_v1 behavior: keep all pseudo-labels.
    pseudo_sample_threshold: float = 0.0
    min_pseudo_keep_ratio: float = 0.0

    # Confidence ensemble
    confidence_scale: float = 1.8
    confidence_bias: float = 0.1
    confidence_weight_gain: float = 10.0

    # Evaluation and saving
    save_pseudo_labels: bool = True
    save_models: bool = True
    save_example_images: bool = True
    example_count: int = 8

    # If true, dynamic-stage true labels can be used as validation labels during online update.
    # Keep false for a clean self-supervised protocol.
    use_true_labels_for_dynamic_early_stopping: bool = False


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Stage 3.1 MMDN dynamic self-supervised tracking, tuned full version."
    )

    # Paths
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--output-folder-name", type=str, default=Config.output_folder_name)
    parser.add_argument("--preferred-dataset-name", type=str, default=Config.preferred_dataset_name)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=Config.seed)

    # Data controls
    parser.add_argument("--speckle-dim", type=int, default=Config.speckle_dim)
    parser.add_argument("--normalize-speckles", type=str2bool, default=Config.normalize_speckles)
    parser.add_argument("--pretrain-samples", type=int, default=Config.pretrain_samples)
    parser.add_argument("--samples-per-dynamic-state", type=int, default=Config.samples_per_dynamic_state)
    parser.add_argument("--limit-dynamic-states", type=int, default=Config.limit_dynamic_states)
    parser.add_argument("--pretrain-val-ratio", type=float, default=Config.pretrain_val_ratio)
    parser.add_argument(
        "--pretrain-val-mode",
        type=str,
        choices=["holdout", "first_dynamic", "none"],
        default=Config.pretrain_val_mode,
    )
    parser.add_argument(
        "--fixed-eval-mode",
        type=str,
        choices=["holdout", "train_tail"],
        default=Config.fixed_eval_mode,
    )
    parser.add_argument("--fixed-eval-count", type=int, default=Config.fixed_eval_count)

    # Model
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--graylevel", type=int, default=Config.graylevel)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--pretrain-epochs", type=int, default=Config.pretrain_epochs)
    parser.add_argument("--update-epochs", type=int, default=Config.update_epochs)
    parser.add_argument("--early-stop-patience", type=int, default=Config.early_stop_patience)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--optimizer", type=str, choices=["adadelta", "adamw", "adam"], default=Config.optimizer)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)

    # GPU / acceleration controls
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=Config.device)
    parser.add_argument("--use-amp", type=str2bool, default=Config.use_amp)
    parser.add_argument("--amp-dtype", type=str, choices=["float16", "bfloat16"], default=Config.amp_dtype)
    parser.add_argument("--channels-last", type=str2bool, default=Config.channels_last)
    parser.add_argument("--pin-memory", type=str2bool, default=Config.pin_memory)
    parser.add_argument("--persistent-workers", type=str2bool, default=Config.persistent_workers)
    parser.add_argument("--prefetch-factor", type=int, default=Config.prefetch_factor)
    parser.add_argument("--cudnn-benchmark", type=str2bool, default=Config.cudnn_benchmark)
    parser.add_argument("--deterministic", type=str2bool, default=Config.deterministic)
    parser.add_argument("--compile-model", type=str2bool, default=Config.compile_model)

    # Dynamic tracking
    parser.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    parser.add_argument("--update-train-window", type=int, default=Config.update_train_window)
    parser.add_argument("--rebuild-interval", type=int, default=Config.rebuild_interval)
    parser.add_argument("--memory-limit", type=int, default=Config.memory_limit)
    parser.add_argument("--pseudo-sample-threshold", type=float, default=Config.pseudo_sample_threshold)
    parser.add_argument("--min-pseudo-keep-ratio", type=float, default=Config.min_pseudo_keep_ratio)

    # Confidence ensemble
    parser.add_argument("--confidence-scale", type=float, default=Config.confidence_scale)
    parser.add_argument("--confidence-bias", type=float, default=Config.confidence_bias)
    parser.add_argument("--confidence-weight-gain", type=float, default=Config.confidence_weight_gain)

    # Saving
    parser.add_argument("--save-pseudo-labels", type=str2bool, default=Config.save_pseudo_labels)
    parser.add_argument("--save-models", type=str2bool, default=Config.save_models)
    parser.add_argument("--save-example-images", type=str2bool, default=Config.save_example_images)
    parser.add_argument("--example-count", type=int, default=Config.example_count)
    parser.add_argument(
        "--use-true-labels-for-dynamic-early-stopping",
        type=str2bool,
        default=Config.use_true_labels_for_dynamic_early_stopping,
    )

    args = parser.parse_args()
    return Config(**vars(args))


# ============================================================
# 2. Logging utilities
# ============================================================


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_output_and_logging(cfg: Config, requested_data_dir: Path) -> Path:
    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir).expanduser().resolve()
    else:
        output_dir = requested_data_dir / cfg.output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "pseudo_labels").mkdir(exist_ok=True)
    (output_dir / "training_logs").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    log_path = output_dir / "run_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return output_dir


# ============================================================
# 3. Dataset loading
# ============================================================


def canonical_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def load_arrays_from_npz(data_dir: Path) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for npz_path in sorted(data_dir.glob("*.npz")):
        try:
            archive = np.load(npz_path, allow_pickle=False)
        except Exception as exc:
            print(f"Warning: failed to load npz {npz_path}: {exc}")
            continue
        for key in archive.files:
            arrays[canonical_name(key)] = archive[key]
        if len(archive.files) == 1:
            arrays[canonical_name(npz_path.stem)] = archive[archive.files[0]]
    return arrays


def load_arrays_from_npy(data_dir: Path, mmap: bool = True) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for npy_path in sorted(data_dir.glob("*.npy")):
        try:
            arrays[canonical_name(npy_path.stem)] = np.load(npy_path, mmap_mode="r" if mmap else None)
        except Exception as exc:
            print(f"Warning: failed to load npy {npy_path}: {exc}")
    return arrays


def load_arrays_direct(data_dir: Path) -> Tuple[Dict[str, np.ndarray], str]:
    arrays = load_arrays_from_npz(data_dir)
    if arrays:
        return arrays, "npz"
    arrays = load_arrays_from_npy(data_dir)
    if arrays:
        return arrays, "npy"
    return {}, "none"


def contains_required_arrays(arrays: Dict[str, np.ndarray]) -> bool:
    keys = set(arrays.keys())
    speckle_names = {"speckles", "speckle", "x", "inputs"}
    pattern_names = {"pattern", "patterns", "y", "labels", "targets"}
    return bool(keys & speckle_names) and bool(keys & pattern_names)


def find_dataset_dir(requested_dir: Path, preferred_name: str) -> Tuple[Path, Dict[str, np.ndarray], str]:
    arrays, source_kind = load_arrays_direct(requested_dir)
    if contains_required_arrays(arrays):
        print("Dataset files found directly in DATA_DIR.")
        return requested_dir, arrays, source_kind

    print("No complete .npz/.npy dataset was found directly in DATA_DIR.")
    print("Searching subfolders recursively...")

    candidate_dirs: List[Path] = []
    for root, dirs, files in os.walk(requested_dir):
        root_path = Path(root)
        if root_path.name in {"models", "pseudo_labels", "training_logs", "figures", "__pycache__"}:
            continue
        if any(str(file).lower().endswith((".npz", ".npy")) for file in files):
            candidate_dirs.append(root_path)

    def score_dir(path: Path) -> Tuple[int, int, str]:
        name = path.name.lower()
        preferred = preferred_name.lower()
        score = 0
        if preferred and preferred in name:
            score -= 1000
        if "mmdn" in name:
            score -= 100
        if "dynamic" in name:
            score -= 50
        if "output" in name or "outputs" in name:
            score += 200
        depth = len(path.relative_to(requested_dir).parts) if path != requested_dir else 0
        return score, depth, str(path)

    candidate_dirs = sorted(set(candidate_dirs), key=score_dir)
    for candidate in candidate_dirs:
        arrays, source_kind = load_arrays_direct(candidate)
        if contains_required_arrays(arrays):
            print(f"Automatically selected dataset subfolder: {candidate}")
            return candidate, arrays, source_kind

    raise FileNotFoundError(
        f"Could not find a dataset folder containing speckles and pattern arrays under {requested_dir}."
    )


def pick_array(arrays: Dict[str, np.ndarray], candidates: Sequence[str], required: bool = True):
    for name in candidates:
        key = canonical_name(name)
        if key in arrays:
            return arrays[key]
    if required:
        raise KeyError(f"Could not find any of these arrays: {candidates}. Available keys: {sorted(arrays)}")
    return None


def load_metadata(actual_data_dir: Path) -> dict:
    metadata_path = actual_data_dir / "metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: failed to read metadata.json: {exc}")
    return {}


# ============================================================
# 4. Segment inference
# ============================================================


def infer_segments(
    patterns: np.ndarray,
    time_index: Optional[np.ndarray],
    counts_by_time: Optional[np.ndarray],
    metadata: dict,
    cfg: Config,
) -> List[dict]:
    n = int(patterns.shape[0])
    if counts_by_time is not None:
        counts = np.asarray(counts_by_time).astype(int).reshape(-1).tolist()
    elif time_index is not None:
        unique, counts_arr = np.unique(np.asarray(time_index), return_counts=True)
        order = np.argsort(unique)
        counts = counts_arr[order].astype(int).tolist()
    else:
        hint = metadata.get("mmdn_training_hint", {}) if isinstance(metadata, dict) else {}
        pre = int(cfg.pretrain_samples or hint.get("sizeOfPretrain", n // 2))
        interval = int(cfg.samples_per_dynamic_state or hint.get("sizeOfUpdateInvertal", max(1, n - pre)))
        dyn = max(0, (n - pre) // interval)
        counts = [pre] + [interval] * dyn

    if sum(counts) > n:
        raise ValueError(f"Segment counts sum to {sum(counts)}, but only {n} samples exist.")

    segments: List[dict] = []
    start = 0
    for state_id, count in enumerate(counts):
        stop = start + int(count)
        segments.append({"state": state_id, "start": start, "stop": stop, "count": int(count)})
        start = stop

    if cfg.limit_dynamic_states is not None:
        segments = [segments[0]] + segments[1 : 1 + int(cfg.limit_dynamic_states)]

    if cfg.pretrain_samples is not None:
        segments[0]["stop"] = segments[0]["start"] + int(cfg.pretrain_samples)
        segments[0]["count"] = int(cfg.pretrain_samples)

    if cfg.samples_per_dynamic_state is not None:
        base = int(segments[0]["stop"])
        updated = [segments[0]]
        for i, old in enumerate(segments[1:], start=1):
            start = base + (i - 1) * int(cfg.samples_per_dynamic_state)
            stop = start + int(cfg.samples_per_dynamic_state)
            if stop > n:
                break
            updated.append({"state": old["state"], "start": start, "stop": stop, "count": stop - start})
        segments = updated

    if len(segments) < 2:
        raise ValueError("Need one pretraining segment plus at least one dynamic segment.")

    return segments


def make_segment_dataframe(segments: List[dict], kappa_by_time: Optional[np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(segments)
    if kappa_by_time is not None and len(kappa_by_time) >= len(df):
        df["kappa"] = np.asarray(kappa_by_time).reshape(-1)[: len(df)].astype(float)
    return df


# ============================================================
# 5. Dataset and model
# ============================================================


class MMFDataset(Dataset):
    def __init__(
        self,
        x_array: np.ndarray,
        sample_indices: Sequence[int],
        labels: Optional[np.ndarray] = None,
        speckle_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        self.x_array = x_array
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.speckle_dim = speckle_dim
        self.normalize = normalize

        if self.labels is not None and len(self.labels) != len(self.sample_indices):
            raise ValueError(
                f"labels length {len(self.labels)} does not match sample_indices length {len(self.sample_indices)}"
            )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, pos: int):
        sample_id = int(self.sample_indices[pos])
        x = np.asarray(self.x_array[sample_id], dtype=np.float32)
        if self.normalize:
            if x.max() > 2.0:
                x = x / 255.0
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        if self.speckle_dim is not None and (
            x_tensor.shape[-2] != self.speckle_dim or x_tensor.shape[-1] != self.speckle_dim
        ):
            x_tensor = F.interpolate(
                x_tensor.unsqueeze(0),
                size=(self.speckle_dim, self.speckle_dim),
                mode="nearest",
            ).squeeze(0)

        if self.labels is None:
            return x_tensor
        y = torch.from_numpy(self.labels[pos])
        return x_tensor, y


class MMDNSubNetwork(nn.Module):
    def __init__(self, outsize: int, speckle_dim: int, base_channels: int = 8, dropout: float = 0.4):
        super().__init__()
        c = int(base_channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv2d(c, c, kernel_size=3, stride=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, speckle_dim, speckle_dim)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Linear(flat_dim, outsize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# ============================================================
# 6. Training, prediction, metrics
# ============================================================


def choose_device(cfg: Config) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and check your NVIDIA driver."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype(cfg: Config):
    if cfg.amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def move_x_to_device(x: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    if channels_last and device.type == "cuda" and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


class Experiment:
    def __init__(
        self,
        cfg: Config,
        output_dir: Path,
        speckles: np.ndarray,
        patterns: np.ndarray,
        segments: List[dict],
        kappa_by_time: Optional[np.ndarray],
        kappa_per_sample: Optional[np.ndarray],
        speckle_dim: int,
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.speckles = speckles
        self.patterns = patterns
        self.segments = segments
        self.kappa_by_time = kappa_by_time
        self.kappa_per_sample = kappa_per_sample
        self.speckle_dim = int(speckle_dim)
        self.outsize = int(patterns.shape[1])
        self.pattern_side = int(round(math.sqrt(self.outsize)))
        self.device = choose_device(cfg)
        self.use_amp = bool(cfg.use_amp and self.device.type == "cuda")
        self.amp_dtype = get_amp_dtype(cfg)
        self.channels_last = bool(cfg.channels_last and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.clf1 = self.new_model()
        self.clf2 = self.new_model()
        self.clf3 = self.new_model()
        self.optimizers = {
            "clf1": self.make_optimizer(self.clf1),
            "clf2": self.make_optimizer(self.clf2),
            "clf3": self.make_optimizer(self.clf3),
        }
        self.static_clf3: Optional[nn.Module] = None

    def indices_for_segment(self, seg: dict) -> np.ndarray:
        return np.arange(int(seg["start"]), int(seg["stop"]), dtype=np.int64)

    def labels_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self.patterns[np.asarray(indices, dtype=np.int64)], dtype=np.uint8)

    def make_loader(
        self,
        indices: Sequence[int],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        dataset = MMFDataset(
            self.speckles,
            indices,
            labels=labels,
            speckle_dim=self.speckle_dim,
            normalize=self.cfg.normalize_speckles,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=int(batch_size or self.cfg.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.num_workers),
            pin_memory=bool(self.cfg.pin_memory and self.device.type == "cuda"),
        )
        if int(self.cfg.num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(self.cfg.persistent_workers)
            loader_kwargs["prefetch_factor"] = int(self.cfg.prefetch_factor)
        return DataLoader(**loader_kwargs)

    def new_model(self) -> nn.Module:
        model = MMDNSubNetwork(
            outsize=self.outsize,
            speckle_dim=self.speckle_dim,
            base_channels=self.cfg.base_channels,
            dropout=self.cfg.dropout,
        )
        model = model.to(self.device)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if bool(self.cfg.compile_model) and hasattr(torch, "compile"):
            print("torch.compile enabled for a newly created subnetwork")
            model = torch.compile(model)
        return model

    def make_optimizer(self, model: nn.Module):
        name = self.cfg.optimizer.lower()
        if name == "adadelta":
            return torch.optim.Adadelta(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

    def print_model_summary(self):
        print("=" * 100)
        print("MODEL SUMMARY")
        print(self.clf3)
        print("parameters clf3:", sum(p.numel() for p in self.clf3.parameters()))
        print("device:", self.device)
        print("use_amp:", self.use_amp, "amp_dtype:", str(self.amp_dtype))
        print("channels_last:", self.channels_last)
        if self.device.type == "cuda":
            print("gpu:", torch.cuda.get_device_name(0))
            print("cuda version in torch:", torch.version.cuda)
            print("gpu total memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        print("=" * 100)

    def binarize(self, pred: np.ndarray) -> np.ndarray:
        pred = np.clip(pred, 0.0, 1.0)
        if int(self.cfg.graylevel) == 2:
            return (pred >= 0.5).astype(np.uint8)
        return ((0.5 + pred * (self.cfg.graylevel - 1)).astype(np.uint8) / (self.cfg.graylevel - 1)).astype(np.float16)

    def confidence_level(self, pred: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - 0.5) * self.cfg.confidence_scale + self.cfg.confidence_bias))

    def confidence_weights(self, c1: float, c2: float, c3: float) -> np.ndarray:
        denom = max(1e-6, 3.0 - c1 - c2 - c3)
        gain = float(self.cfg.confidence_weight_gain)
        logits = np.array(
            [
                gain * (2.0 - c2 - c3) / denom,
                gain * (2.0 - c1 - c3) / denom,
                gain * (2.0 - c1 - c2) / denom,
            ],
            dtype=np.float64,
        )
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        return weights / np.sum(weights)

    def predict_prob(self, model: nn.Module, indices: Sequence[int], batch_size: Optional[int] = None) -> np.ndarray:
        model.eval()
        preds = []
        loader = self.make_loader(indices, labels=None, shuffle=False, batch_size=batch_size or self.cfg.batch_size)
        with torch.no_grad():
            for x in loader:
                x = move_x_to_device(x, self.device, self.channels_last)
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = model(x)
                preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
        return np.concatenate(preds, axis=0)

    @staticmethod
    def evaluate_prediction(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict:
        y_true = np.asarray(y_true)
        y_pred_binary = np.asarray(y_pred_binary)
        sample_accuracy = np.mean(y_true == y_pred_binary, axis=1) * 100.0
        return {
            "pixel_accuracy": float(np.mean(y_true == y_pred_binary) * 100.0),
            "sample_accuracy_mean": float(np.mean(sample_accuracy)),
            "sample_accuracy_std": float(np.std(sample_accuracy)),
            "exact_match_percent": float(np.mean(np.all(y_true == y_pred_binary, axis=1)) * 100.0),
            "mae": float(np.mean(np.abs(y_true.astype(np.float32) - y_pred_binary.astype(np.float32)))),
        }

    def ensemble_predict(self, indices: Sequence[int]):
        p1 = self.predict_prob(self.clf1, indices)
        p2 = self.predict_prob(self.clf2, indices)
        p3 = self.predict_prob(self.clf3, indices)
        c1, c2, c3 = self.confidence_level(p1), self.confidence_level(p2), self.confidence_level(p3)
        weights = self.confidence_weights(c1, c2, c3)
        prob = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
        pred = self.binarize(prob)
        return pred, prob, (p1, p2, p3), (c1, c2, c3), weights

    def run_validation_loss(self, model: nn.Module, val_loader: DataLoader, criterion) -> float:
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = move_x_to_device(x, self.device, self.channels_last)
                y = y.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                total_count += x.size(0)
        return total_loss / max(1, total_count)

    def fit_on_memory(
        self,
        model: nn.Module,
        optimizer,
        train_indices: Sequence[int],
        train_labels: np.ndarray,
        val_indices: Optional[Sequence[int]] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 1,
        name: str = "train",
    ) -> List[dict]:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        train_labels = np.asarray(train_labels, dtype=np.float32)
        if len(train_indices) == 0:
            print(f"{name}: skipped because train_indices is empty")
            return []

        train_loader = self.make_loader(train_indices, labels=train_labels, shuffle=True)
        val_loader = None
        if val_indices is not None and val_labels is not None and len(val_indices) > 0:
            val_loader = self.make_loader(val_indices, labels=np.asarray(val_labels, dtype=np.float32), shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_metric = float("inf")
        wait = 0
        history: List[dict] = []

        t0 = time.time()
        for epoch in range(1, int(epochs) + 1):
            model.train()
            total_loss = 0.0
            total_count = 0
            for x, y in train_loader:
                x = move_x_to_device(x, self.device, self.channels_last)
                y = y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss.detach().item()) * x.size(0)
                total_count += x.size(0)

            train_loss = total_loss / max(1, total_count)
            if val_loader is not None:
                val_loss = self.run_validation_loss(model, val_loader, criterion)
                monitor = val_loss
            else:
                val_loss = np.nan
                monitor = train_loss

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_samples": int(len(train_indices)),
                "val_samples": 0 if val_indices is None else int(len(val_indices)),
                "elapsed_sec": float(time.time() - t0),
            }
            history.append(row)
            print(
                f"{name} epoch {epoch:03d}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"train_samples={len(train_indices)}, "
                f"val_samples={0 if val_indices is None else len(val_indices)}"
            )

            if monitor < best_metric - 5e-6:
                best_metric = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(self.cfg.early_stop_patience):
                    print(f"{name}: early stopping at epoch {epoch}")
                    break

        model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        optimizer.load_state_dict(best_optimizer_state)
        pd.DataFrame(history).to_csv(self.output_dir / "training_logs" / f"{name}.csv", index=False)
        return history

    def split_pretrain_indices(self, pretrain_indices: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.cfg.pretrain_val_mode != "holdout" or self.cfg.pretrain_val_ratio <= 0:
            return pretrain_indices, None
        n_total = len(pretrain_indices)
        n_val = int(round(n_total * float(self.cfg.pretrain_val_ratio)))
        n_val = max(1, min(n_total - 1, n_val))
        train_indices = pretrain_indices[:-n_val]
        val_indices = pretrain_indices[-n_val:]
        return train_indices, val_indices

    def get_pretrain_validation(
        self,
        fixed_val_indices: Optional[np.ndarray],
        first_dynamic_indices: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        mode = self.cfg.pretrain_val_mode
        if mode == "holdout":
            if fixed_val_indices is None:
                return None, None, "none"
            return fixed_val_indices, self.labels_for_indices(fixed_val_indices), "fixed_holdout"
        if mode == "first_dynamic":
            return first_dynamic_indices, self.labels_for_indices(first_dynamic_indices), "first_dynamic_legacy"
        if mode == "none":
            return None, None, "none"
        raise ValueError(f"Unsupported pretrain_val_mode: {mode}")

    def split_dynamic_indices(self, indices: np.ndarray) -> List[np.ndarray]:
        chunk_size = int(self.cfg.chunk_size)
        if chunk_size <= 0 or len(indices) <= chunk_size:
            return [indices]
        return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    def select_pseudo_labels(
        self,
        indices: np.ndarray,
        pred_binary: np.ndarray,
        prob: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        indices = np.asarray(indices, dtype=np.int64)
        pred_binary = np.asarray(pred_binary)
        prob = np.asarray(prob)

        confidence_score = np.mean(np.abs(prob - 0.5) * 2.0, axis=1)
        threshold = float(self.cfg.pseudo_sample_threshold)
        if threshold <= 0:
            mask = np.ones(len(indices), dtype=bool)
        else:
            mask = confidence_score >= threshold

        min_keep = int(math.ceil(float(self.cfg.min_pseudo_keep_ratio) * len(indices)))
        if min_keep > 0 and np.sum(mask) < min_keep:
            order = np.argsort(-confidence_score)
            mask = np.zeros(len(indices), dtype=bool)
            mask[order[:min_keep]] = True

        kept_indices = indices[mask]
        kept_labels = pred_binary[mask].astype(np.uint8)
        stats = {
            "pseudo_keep_count": int(np.sum(mask)),
            "pseudo_keep_fraction": float(np.mean(mask)) if len(mask) else 0.0,
            "pseudo_conf_mean": float(np.mean(confidence_score)) if len(confidence_score) else np.nan,
            "pseudo_conf_min": float(np.min(confidence_score)) if len(confidence_score) else np.nan,
            "pseudo_conf_max": float(np.max(confidence_score)) if len(confidence_score) else np.nan,
        }
        return kept_indices, kept_labels, stats

    def get_kappa_value(self, state: int, current_indices: np.ndarray) -> Optional[float]:
        if self.kappa_by_time is not None and state < len(self.kappa_by_time):
            return float(np.asarray(self.kappa_by_time).reshape(-1)[state])
        if self.kappa_per_sample is not None:
            return float(np.mean(np.asarray(self.kappa_per_sample)[current_indices]))
        return None

    def pretrain(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        pretrain_indices_all = self.indices_for_segment(self.segments[0])
        first_dynamic_indices = self.indices_for_segment(self.segments[1])
        pretrain_train_indices, fixed_val_indices = self.split_pretrain_indices(pretrain_indices_all)
        pretrain_train_labels = self.labels_for_indices(pretrain_train_indices)
        val_indices, val_labels, val_name = self.get_pretrain_validation(fixed_val_indices, first_dynamic_indices)

        print("=" * 100)
        print("PRETRAINING SETUP")
        print("pretrain_val_mode:", self.cfg.pretrain_val_mode)
        print("validation source:", val_name)
        print("pretrain total fixed-state samples:", len(pretrain_indices_all))
        print("pretrain train samples:", len(pretrain_train_indices))
        print("pretrain validation samples:", 0 if val_indices is None else len(val_indices))
        print("dynamic states:", len(self.segments) - 1)
        print("samples per first dynamic state:", int(self.segments[1]["count"]))
        print("=" * 100)

        t0 = time.time()
        print("Pretraining L3/clf3 on fixed-state memory")
        self.fit_on_memory(
            self.clf3,
            self.optimizers["clf3"],
            pretrain_train_indices,
            pretrain_train_labels,
            val_indices,
            val_labels,
            self.cfg.pretrain_epochs,
            "pretrain_clf3_L3_long_memory",
        )

        print("Pretraining S2/clf2 on last 3/4 of fixed-state memory")
        n2 = max(1, int(len(pretrain_train_indices) * 3 / 4))
        self.fit_on_memory(
            self.clf2,
            self.optimizers["clf2"],
            pretrain_train_indices[-n2:],
            pretrain_train_labels[-n2:],
            val_indices,
            val_labels,
            self.cfg.pretrain_epochs,
            "pretrain_clf2_S2_short_memory",
        )

        print("Pretraining S1/clf1 on last 2/3 of fixed-state memory")
        n1 = max(1, int(len(pretrain_train_indices) * 2 / 3))
        self.fit_on_memory(
            self.clf1,
            self.optimizers["clf1"],
            pretrain_train_indices[-n1:],
            pretrain_train_labels[-n1:],
            val_indices,
            val_labels,
            self.cfg.pretrain_epochs,
            "pretrain_clf1_S1_short_memory",
        )

        self.static_clf3 = self.new_model()
        self.static_clf3.load_state_dict({k: v.detach().clone() for k, v in self.clf3.state_dict().items()})

        if self.cfg.save_models:
            torch.save(self.clf1.state_dict(), self.output_dir / "models" / "pretrain_clf1_S1.pt")
            torch.save(self.clf2.state_dict(), self.output_dir / "models" / "pretrain_clf2_S2.pt")
            torch.save(self.clf3.state_dict(), self.output_dir / "models" / "pretrain_clf3_L3.pt")
            torch.save(self.static_clf3.state_dict(), self.output_dir / "models" / "static_clf3_L3.pt")

        print(f"pretraining finished in {(time.time() - t0) / 60:.2f} min")

        if self.cfg.fixed_eval_mode == "holdout" and fixed_val_indices is not None:
            fixed_eval_indices = fixed_val_indices[-min(int(self.cfg.fixed_eval_count), len(fixed_val_indices)) :]
            fixed_eval_source = "fixed_holdout"
        else:
            fixed_eval_indices = pretrain_indices_all[-min(int(self.cfg.fixed_eval_count), len(pretrain_indices_all)) :]
            fixed_eval_source = "train_tail_or_all_fixed"

        fixed_metrics_df = self.evaluate_fixed_state(fixed_eval_indices, fixed_eval_source)
        return fixed_metrics_df, pretrain_train_indices, pretrain_train_labels

    def evaluate_fixed_state(self, fixed_eval_indices: np.ndarray, source_name: str) -> pd.DataFrame:
        if self.static_clf3 is None:
            raise RuntimeError("static_clf3 is not initialized")

        fixed_true_labels = self.labels_for_indices(fixed_eval_indices)
        fixed_pred_ensemble, _, fixed_indiv_probs, fixed_confs, fixed_weights = self.ensemble_predict(fixed_eval_indices)
        fixed_pred1 = self.binarize(fixed_indiv_probs[0])
        fixed_pred2 = self.binarize(fixed_indiv_probs[1])
        fixed_pred3 = self.binarize(fixed_indiv_probs[2])
        fixed_static = self.binarize(self.predict_prob(self.static_clf3, fixed_eval_indices))

        ens = self.evaluate_prediction(fixed_true_labels, fixed_pred_ensemble)
        m1 = self.evaluate_prediction(fixed_true_labels, fixed_pred1)
        m2 = self.evaluate_prediction(fixed_true_labels, fixed_pred2)
        m3 = self.evaluate_prediction(fixed_true_labels, fixed_pred3)
        st = self.evaluate_prediction(fixed_true_labels, fixed_static)

        fixed_row = {
            "state": int(self.segments[0]["state"]),
            "eval_source": source_name,
            "samples_evaluated": int(len(fixed_eval_indices)),
            "ensemble_pixel_acc": ens["pixel_accuracy"],
            "s1_pixel_acc": m1["pixel_accuracy"],
            "s2_pixel_acc": m2["pixel_accuracy"],
            "l3_pixel_acc": m3["pixel_accuracy"],
            "static_l3_pixel_acc": st["pixel_accuracy"],
            "ensemble_exact_match": ens["exact_match_percent"],
            "ensemble_mae": ens["mae"],
            "conf_s1": fixed_confs[0],
            "conf_s2": fixed_confs[1],
            "conf_l3": fixed_confs[2],
            "weight_s1": float(fixed_weights[0]),
            "weight_s2": float(fixed_weights[1]),
            "weight_l3": float(fixed_weights[2]),
        }
        df = pd.DataFrame([fixed_row])
        df.to_csv(self.output_dir / "fixed_state_pretrain_metrics.csv", index=False)
        print("=" * 100)
        print("FIXED-STATE VALIDATION METRICS")
        print(df.to_string(index=False))
        print(f"fixed-state ensemble accuracy: {fixed_row['ensemble_pixel_acc']:.4f}%")
        print(f"fixed-state exact match: {fixed_row['ensemble_exact_match']:.4f}%")
        print(f"fixed-state MAE: {fixed_row['ensemble_mae']:.6f}")
        print("=" * 100)
        return df

    def run_dynamic_tracking(
        self,
        memory_indices: np.ndarray,
        memory_labels: np.ndarray,
    ) -> pd.DataFrame:
        if self.static_clf3 is None:
            raise RuntimeError("static_clf3 is not initialized")

        rows: List[dict] = []
        global_update = 0
        memory_indices = np.asarray(memory_indices, dtype=np.int64)
        memory_labels = np.asarray(memory_labels, dtype=np.uint8)
        memory_limit = int(self.cfg.memory_limit or len(memory_indices))

        last_example_payload = None

        print("=" * 100)
        print("DYNAMIC SELF-SUPERVISED TRACKING")
        print("memory_limit:", memory_limit)
        print("chunk_size:", self.cfg.chunk_size)
        print("update_train_window:", self.cfg.update_train_window)
        print("pseudo_sample_threshold:", self.cfg.pseudo_sample_threshold)
        print("min_pseudo_keep_ratio:", self.cfg.min_pseudo_keep_ratio)
        print("=" * 100)

        for state_step, seg in enumerate(self.segments[1:], start=1):
            state = int(seg["state"])
            state_indices = self.indices_for_segment(seg)
            state_chunks = self.split_dynamic_indices(state_indices)
            print("=" * 100)
            print(
                f"dynamic state {state_step}/{len(self.segments) - 1}, "
                f"state={state}, samples={len(state_indices)}, chunks={len(state_chunks)}, "
                f"kappa={self.get_kappa_value(state, state_indices)}"
            )

            for chunk_id, current_indices in enumerate(state_chunks, start=1):
                global_update += 1
                current_indices = np.asarray(current_indices, dtype=np.int64)
                true_labels = self.labels_for_indices(current_indices)
                print("-" * 100)
                print(
                    f"online_update={global_update}, state={state}, "
                    f"chunk={chunk_id}/{len(state_chunks)}, eval_samples={len(current_indices)}"
                )

                if global_update > 1:
                    train_window = int(self.cfg.update_train_window or self.cfg.chunk_size or int(seg["count"]))
                    train_n = min(train_window, len(memory_indices))
                    train_indices = memory_indices[-train_n:]
                    train_labels = memory_labels[-train_n:]
                    dyn_val_indices = current_indices if self.cfg.use_true_labels_for_dynamic_early_stopping else None
                    dyn_val_labels = true_labels if self.cfg.use_true_labels_for_dynamic_early_stopping else None

                    # Stage3_v1-style alternating rebuild at state boundaries.
                    if chunk_id == 1 and self.cfg.rebuild_interval > 0:
                        if state_step % (2 * self.cfg.rebuild_interval) == self.cfg.rebuild_interval + 1:
                            print("Rebuilding S1/clf1 from recent pseudo-labeled memory")
                            self.clf1 = self.new_model()
                            self.optimizers["clf1"] = self.make_optimizer(self.clf1)
                            rebuild_n = min(len(memory_indices), max(train_n, int(memory_limit * 2 / 3)))
                            self.fit_on_memory(
                                self.clf1,
                                self.optimizers["clf1"],
                                memory_indices[-rebuild_n:],
                                memory_labels[-rebuild_n:],
                                dyn_val_indices,
                                dyn_val_labels,
                                self.cfg.update_epochs,
                                f"rebuild_state{state:02d}_chunk{chunk_id:02d}_clf1_S1",
                            )

                        if state_step % (2 * self.cfg.rebuild_interval) == 1 and state_step > 1:
                            print("Rebuilding S2/clf2 from recent pseudo-labeled memory")
                            self.clf2 = self.new_model()
                            self.optimizers["clf2"] = self.make_optimizer(self.clf2)
                            rebuild_n = min(len(memory_indices), max(train_n, int(memory_limit * 3 / 4)))
                            self.fit_on_memory(
                                self.clf2,
                                self.optimizers["clf2"],
                                memory_indices[-rebuild_n:],
                                memory_labels[-rebuild_n:],
                                dyn_val_indices,
                                dyn_val_labels,
                                self.cfg.update_epochs,
                                f"rebuild_state{state:02d}_chunk{chunk_id:02d}_clf2_S2",
                            )

                    print("Fine-tuning L3/clf3 on latest pseudo-labeled interval")
                    self.fit_on_memory(
                        self.clf3,
                        self.optimizers["clf3"],
                        train_indices,
                        train_labels,
                        dyn_val_indices,
                        dyn_val_labels,
                        self.cfg.update_epochs,
                        f"update_{global_update:04d}_state{state:02d}_chunk{chunk_id:02d}_clf3_L3",
                    )
                    print("Fine-tuning S2/clf2 on latest pseudo-labeled interval")
                    self.fit_on_memory(
                        self.clf2,
                        self.optimizers["clf2"],
                        train_indices,
                        train_labels,
                        dyn_val_indices,
                        dyn_val_labels,
                        self.cfg.update_epochs,
                        f"update_{global_update:04d}_state{state:02d}_chunk{chunk_id:02d}_clf2_S2",
                    )
                    print("Fine-tuning S1/clf1 on latest pseudo-labeled interval")
                    self.fit_on_memory(
                        self.clf1,
                        self.optimizers["clf1"],
                        train_indices,
                        train_labels,
                        dyn_val_indices,
                        dyn_val_labels,
                        self.cfg.update_epochs,
                        f"update_{global_update:04d}_state{state:02d}_chunk{chunk_id:02d}_clf1_S1",
                    )

                pred_ensemble, prob_ensemble, indiv_probs, confs, weights = self.ensemble_predict(current_indices)
                pred1 = self.binarize(indiv_probs[0])
                pred2 = self.binarize(indiv_probs[1])
                pred3 = self.binarize(indiv_probs[2])
                pred_static = self.binarize(self.predict_prob(self.static_clf3, current_indices))

                ens_metrics = self.evaluate_prediction(true_labels, pred_ensemble)
                m1_metrics = self.evaluate_prediction(true_labels, pred1)
                m2_metrics = self.evaluate_prediction(true_labels, pred2)
                m3_metrics = self.evaluate_prediction(true_labels, pred3)
                static_metrics = self.evaluate_prediction(true_labels, pred_static)

                kept_indices, kept_labels, pseudo_stats = self.select_pseudo_labels(current_indices, pred_ensemble, prob_ensemble)
                if len(kept_indices) == 0:
                    print("Warning: pseudo-label filter kept zero samples. Falling back to keeping current chunk.")
                    kept_indices = current_indices
                    kept_labels = pred_ensemble.astype(np.uint8)
                    pseudo_stats["pseudo_keep_count"] = int(len(current_indices))
                    pseudo_stats["pseudo_keep_fraction"] = 1.0

                kappa_value = self.get_kappa_value(state, current_indices)
                row = {
                    "online_update": global_update,
                    "state_step": state_step,
                    "state": state,
                    "chunk": chunk_id,
                    "chunks_in_state": len(state_chunks),
                    "start": int(current_indices[0]),
                    "stop": int(current_indices[-1]) + 1,
                    "samples": int(len(current_indices)),
                    "kappa": kappa_value,
                    "ensemble_pixel_acc": ens_metrics["pixel_accuracy"],
                    "s1_pixel_acc": m1_metrics["pixel_accuracy"],
                    "s2_pixel_acc": m2_metrics["pixel_accuracy"],
                    "l3_pixel_acc": m3_metrics["pixel_accuracy"],
                    "static_l3_pixel_acc": static_metrics["pixel_accuracy"],
                    "ensemble_exact_match": ens_metrics["exact_match_percent"],
                    "s1_exact_match": m1_metrics["exact_match_percent"],
                    "s2_exact_match": m2_metrics["exact_match_percent"],
                    "l3_exact_match": m3_metrics["exact_match_percent"],
                    "static_exact_match": static_metrics["exact_match_percent"],
                    "ensemble_mae": ens_metrics["mae"],
                    "static_mae": static_metrics["mae"],
                    "conf_s1": confs[0],
                    "conf_s2": confs[1],
                    "conf_l3": confs[2],
                    "weight_s1": float(weights[0]),
                    "weight_s2": float(weights[1]),
                    "weight_l3": float(weights[2]),
                }
                row.update(pseudo_stats)
                rows.append(row)

                if self.cfg.save_pseudo_labels:
                    np.save(
                        self.output_dir / "pseudo_labels" / f"pseudo_state_{state:02d}_chunk_{chunk_id:02d}.npy",
                        pred_ensemble.astype(np.uint8),
                    )

                memory_indices = np.concatenate([memory_indices, kept_indices])[-memory_limit:]
                memory_labels = np.concatenate([memory_labels, kept_labels])[-memory_limit:]

                metrics_df = pd.DataFrame(rows)
                metrics_df.to_csv(self.output_dir / "dynamic_tracking_metrics.csv", index=False)
                print("current metrics:")
                print(pd.DataFrame([row]).to_string(index=False))
                print(
                    f"ensemble={row['ensemble_pixel_acc']:.4f}%, "
                    f"static={row['static_l3_pixel_acc']:.4f}%, "
                    f"gain={row['ensemble_pixel_acc'] - row['static_l3_pixel_acc']:.4f} pp, "
                    f"kept_pseudo={row['pseudo_keep_fraction']:.3f}"
                )

                last_example_payload = {
                    "indices": current_indices.copy(),
                    "true": true_labels.copy(),
                    "ensemble": pred_ensemble.copy(),
                    "static": pred_static.copy(),
                    "state": state,
                    "kappa": kappa_value,
                }

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(self.output_dir / "dynamic_tracking_metrics.csv", index=False)
        try:
            metrics_df.to_excel(self.output_dir / "dynamic_tracking_metrics.xlsx", index=False)
        except Exception as exc:
            print(f"Warning: failed to save Excel file. Install openpyxl if needed. Error: {exc}")

        if self.cfg.save_models:
            torch.save(self.clf1.state_dict(), self.output_dir / "models" / "final_clf1_S1.pt")
            torch.save(self.clf2.state_dict(), self.output_dir / "models" / "final_clf2_S2.pt")
            torch.save(self.clf3.state_dict(), self.output_dir / "models" / "final_clf3_L3.pt")

        print("=" * 100)
        print("DYNAMIC TRACKING FINISHED")
        print(metrics_df.to_string(index=False))
        print("=" * 100)

        if self.cfg.save_example_images and last_example_payload is not None:
            self.save_recovery_examples(last_example_payload)

        return metrics_df

    def save_recovery_examples(self, payload: dict):
        if self.pattern_side * self.pattern_side != self.outsize:
            print("Skipping example image grid because output size is not square.")
            return
        n = min(int(self.cfg.example_count), len(payload["indices"]))
        if n <= 0:
            return
        true = payload["true"][:n].reshape(n, self.pattern_side, self.pattern_side)
        ensemble = payload["ensemble"][:n].reshape(n, self.pattern_side, self.pattern_side)
        static = payload["static"][:n].reshape(n, self.pattern_side, self.pattern_side)

        fig, axes = plt.subplots(3, n, figsize=(1.45 * n, 4.8))
        if n == 1:
            axes = np.asarray(axes).reshape(3, 1)
        for j in range(n):
            axes[0, j].imshow(true[j], cmap="gray", vmin=0, vmax=1)
            axes[0, j].set_title(f"GT {j+1}", fontsize=8)
            axes[1, j].imshow(static[j], cmap="gray", vmin=0, vmax=1)
            axes[1, j].set_title("Static", fontsize=8)
            axes[2, j].imshow(ensemble[j], cmap="gray", vmin=0, vmax=1)
            axes[2, j].set_title("MMDN", fontsize=8)
            for i in range(3):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        fig.suptitle(f"Recovery examples at state={payload['state']}, kappa={payload['kappa']}")
        fig.tight_layout()
        path = self.output_dir / "recovery_examples_last_state.png"
        fig.savefig(path, dpi=180)
        fig.savefig(self.output_dir / "figures" / "recovery_examples_last_state.png", dpi=180)
        plt.close(fig)
        print("saved:", path)

    def make_plots(self, fixed_metrics_df: pd.DataFrame, metrics_df: pd.DataFrame):
        if metrics_df.empty:
            print("No dynamic metrics to plot.")
            return

        metrics_path = self.output_dir / "dynamic_tracking_metrics.csv"
        fixed_path = self.output_dir / "fixed_state_pretrain_metrics.csv"
        print("=" * 100)
        print("SAVING FIGURES")
        print("metrics source:", metrics_path)
        print("fixed metrics source:", fixed_path)

        use_kappa = "kappa" in metrics_df.columns and metrics_df["kappa"].notna().all()
        x_axis = metrics_df["kappa"].to_numpy() if use_kappa else metrics_df["online_update"].to_numpy()
        x_label = "kappa" if use_kappa else "dynamic step"

        # Same main accuracy figure as Stage3_v1.ipynb.
        plt.figure(figsize=(9, 5))
        plt.plot(x_axis, metrics_df["ensemble_pixel_acc"], marker="o", label="MMDN ensemble")
        plt.plot(x_axis, metrics_df["s1_pixel_acc"], marker=".", label="S1 short memory", alpha=0.8)
        plt.plot(x_axis, metrics_df["s2_pixel_acc"], marker=".", label="S2 short memory", alpha=0.8)
        plt.plot(x_axis, metrics_df["l3_pixel_acc"], marker=".", label="L3 long memory", alpha=0.8)
        plt.plot(x_axis, metrics_df["static_l3_pixel_acc"], marker="x", label="static pretrained L3", linestyle="--")
        plt.xlabel(x_label)
        plt.ylabel("pixel accuracy (%)")
        plt.title("Dynamic tracking accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = self.output_dir / "accuracy_over_drift.png"
        plt.savefig(p, dpi=180)
        plt.savefig(self.output_dir / "figures" / "accuracy_over_drift.png", dpi=180)
        plt.close()
        print("saved:", p)

        # Same ensemble-weight figure as Stage3_v1.ipynb.
        plt.figure(figsize=(9, 4))
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
        p = self.output_dir / "ensemble_weights.png"
        plt.savefig(p, dpi=180)
        plt.savefig(self.output_dir / "figures" / "ensemble_weights.png", dpi=180)
        plt.close()
        print("saved:", p)

        # State-averaged accuracy plot, useful when each kappa has multiple chunks.
        group_cols = ["state", "kappa"] if use_kappa else ["state"]
        state_mean = metrics_df.groupby(group_cols, as_index=False).agg(
            ensemble_pixel_acc=("ensemble_pixel_acc", "mean"),
            static_l3_pixel_acc=("static_l3_pixel_acc", "mean"),
            s1_pixel_acc=("s1_pixel_acc", "mean"),
            s2_pixel_acc=("s2_pixel_acc", "mean"),
            l3_pixel_acc=("l3_pixel_acc", "mean"),
            ensemble_exact_match=("ensemble_exact_match", "mean"),
            ensemble_mae=("ensemble_mae", "mean"),
        )
        state_mean.to_csv(self.output_dir / "dynamic_tracking_state_mean.csv", index=False)
        try:
            state_mean.to_excel(self.output_dir / "dynamic_tracking_state_mean.xlsx", index=False)
        except Exception as exc:
            print(f"Warning: failed to save state-mean Excel file: {exc}")

        x_state = state_mean["kappa"].to_numpy() if use_kappa else state_mean["state"].to_numpy()
        plt.figure(figsize=(9, 5))
        plt.plot(x_state, state_mean["static_l3_pixel_acc"], marker="x", linestyle="--", label="Static")
        plt.plot(x_state, state_mean["ensemble_pixel_acc"], marker="o", label="MMDN")
        plt.xlabel(x_label if use_kappa else "state")
        plt.ylabel("pixel accuracy (%)")
        plt.title("Dynamic tracking and input recovery accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = self.output_dir / "accuracy_state_average_static_vs_mmdn.png"
        plt.savefig(p, dpi=180)
        plt.savefig(self.output_dir / "figures" / "accuracy_state_average_static_vs_mmdn.png", dpi=180)
        plt.close()
        print("saved:", p)

        # Exact-match plot.
        plt.figure(figsize=(9, 4))
        plt.plot(x_axis, metrics_df["ensemble_exact_match"], marker="o", label="MMDN ensemble")
        if "static_exact_match" in metrics_df.columns:
            plt.plot(x_axis, metrics_df["static_exact_match"], marker="x", linestyle="--", label="static pretrained L3")
        plt.xlabel(x_label)
        plt.ylabel("exact match (%)")
        plt.title("Exact-match recovery accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = self.output_dir / "exact_match_over_drift.png"
        plt.savefig(p, dpi=180)
        plt.savefig(self.output_dir / "figures" / "exact_match_over_drift.png", dpi=180)
        plt.close()
        print("saved:", p)

        # MAE plot.
        plt.figure(figsize=(9, 4))
        plt.plot(x_axis, metrics_df["ensemble_mae"], marker="o", label="MMDN ensemble")
        if "static_mae" in metrics_df.columns:
            plt.plot(x_axis, metrics_df["static_mae"], marker="x", linestyle="--", label="static pretrained L3")
        plt.xlabel(x_label)
        plt.ylabel("MAE")
        plt.title("Recovery MAE over drift")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p = self.output_dir / "mae_over_drift.png"
        plt.savefig(p, dpi=180)
        plt.savefig(self.output_dir / "figures" / "mae_over_drift.png", dpi=180)
        plt.close()
        print("saved:", p)

        # Pseudo-label keep ratio plot.
        if "pseudo_keep_fraction" in metrics_df.columns:
            plt.figure(figsize=(9, 4))
            plt.plot(metrics_df["online_update"], metrics_df["pseudo_keep_fraction"], marker="o")
            plt.xlabel("online update")
            plt.ylabel("kept pseudo-label fraction")
            plt.title("Pseudo-label filtering ratio")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            p = self.output_dir / "pseudo_label_keep_fraction.png"
            plt.savefig(p, dpi=180)
            plt.savefig(self.output_dir / "figures" / "pseudo_label_keep_fraction.png", dpi=180)
            plt.close()
            print("saved:", p)

        summary = {
            "fixed": fixed_metrics_df.to_dict(orient="records"),
            "dynamic_mean_ensemble_pixel_acc": float(metrics_df["ensemble_pixel_acc"].mean()),
            "dynamic_mean_static_l3_pixel_acc": float(metrics_df["static_l3_pixel_acc"].mean()),
            "dynamic_mean_gain_pp": float((metrics_df["ensemble_pixel_acc"] - metrics_df["static_l3_pixel_acc"]).mean()),
            "dynamic_last_ensemble_pixel_acc": float(metrics_df["ensemble_pixel_acc"].iloc[-1]),
            "dynamic_last_static_l3_pixel_acc": float(metrics_df["static_l3_pixel_acc"].iloc[-1]),
            "dynamic_last_gain_pp": float(metrics_df["ensemble_pixel_acc"].iloc[-1] - metrics_df["static_l3_pixel_acc"].iloc[-1]),
            "num_dynamic_rows": int(len(metrics_df)),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("saved:", self.output_dir / "summary.json")
        print("summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("=" * 100)


# ============================================================
# 7. Main entry
# ============================================================


def set_seed_and_backend(seed: int, cfg: Config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark and not cfg.deterministic)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main():
    cfg = parse_args()
    requested_data_dir = Path(cfg.data_dir).expanduser().resolve()
    output_dir = setup_output_and_logging(cfg, requested_data_dir)
    set_seed_and_backend(cfg.seed, cfg)

    print("=" * 100)
    print("STAGE 3.1 MMDN TUNED FULL VERSION")
    print("requested_data_dir:", requested_data_dir)
    print("output_dir:", output_dir)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("cuda device 0:", torch.cuda.get_device_name(0))
        print("cuda total memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print("configuration:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print("=" * 100)

    actual_data_dir, arrays, source_kind = find_dataset_dir(requested_data_dir, cfg.preferred_dataset_name)
    metadata = load_metadata(actual_data_dir)

    speckles = pick_array(arrays, ["speckles", "speckle", "x", "inputs"])
    patterns = pick_array(arrays, ["pattern", "patterns", "y", "labels", "targets"])
    time_index = pick_array(arrays, ["time_index", "time", "state_index", "state"], required=False)
    counts_by_time = pick_array(arrays, ["counts_by_time", "counts"], required=False)
    kappa_by_time = pick_array(arrays, ["kappa_by_time"], required=False)
    kappa_per_sample = pick_array(arrays, ["kappa_per_sample", "kappa"], required=False)

    patterns = np.asarray(patterns, dtype=np.float32)
    if patterns.ndim != 2:
        raise ValueError(f"patterns must be a 2D array, got shape {patterns.shape}")
    if speckles.ndim != 3:
        raise ValueError(f"speckles must be a 3D array, got shape {speckles.shape}")
    if len(speckles) != len(patterns):
        raise ValueError(f"speckles length {len(speckles)} does not match patterns length {len(patterns)}")

    print("=" * 100)
    print("DATASET SUMMARY")
    print("source_kind:", source_kind)
    print("actual_data_dir:", actual_data_dir)
    print("available arrays:", sorted(arrays))
    print("speckles:", speckles.shape, speckles.dtype)
    print("targets:", patterns.shape, patterns.dtype, "min/max", float(np.min(patterns)), float(np.max(patterns)))
    if time_index is not None:
        ti = np.asarray(time_index)
        print("time_index:", ti.shape, ti.dtype, int(np.min(ti)), int(np.max(ti)))
    if counts_by_time is not None:
        print("counts_by_time:", np.asarray(counts_by_time).astype(int).reshape(-1).tolist())
    if kappa_by_time is not None:
        print("kappa_by_time:", np.asarray(kappa_by_time).astype(float).reshape(-1).tolist())
    if kappa_per_sample is not None:
        print("kappa_per_sample:", np.asarray(kappa_per_sample).shape, np.asarray(kappa_per_sample).dtype)
    print("=" * 100)

    segments = infer_segments(patterns, time_index, counts_by_time, metadata, cfg)
    segment_df = make_segment_dataframe(segments, kappa_by_time)
    segment_df.to_csv(output_dir / "segments.csv", index=False)

    speckle_dim = int(
        cfg.speckle_dim
        or metadata.get("mmdn_training_hint", {}).get("speckle_dim", speckles.shape[1])
    )
    outsize = int(patterns.shape[1])
    pattern_side = int(round(math.sqrt(outsize)))

    print("=" * 100)
    print("SEGMENTS")
    print(segment_df.to_string(index=False))
    print("speckle_dim:", speckle_dim, "outsize:", outsize, "pattern_side:", pattern_side)
    print("=" * 100)

    exp = Experiment(
        cfg=cfg,
        output_dir=output_dir,
        speckles=speckles,
        patterns=patterns,
        segments=segments,
        kappa_by_time=kappa_by_time,
        kappa_per_sample=kappa_per_sample,
        speckle_dim=speckle_dim,
    )
    exp.print_model_summary()

    t_start = time.time()
    fixed_metrics_df, memory_indices, memory_labels = exp.pretrain()
    metrics_df = exp.run_dynamic_tracking(memory_indices, memory_labels)
    exp.make_plots(fixed_metrics_df, metrics_df)
    elapsed = (time.time() - t_start) / 60.0

    print("=" * 100)
    print(f"ALL DONE. Total elapsed time: {elapsed:.2f} min")
    print("Main output folder:", output_dir)
    print("Important files:")
    for name in [
        "fixed_state_pretrain_metrics.csv",
        "dynamic_tracking_metrics.csv",
        "dynamic_tracking_metrics.xlsx",
        "dynamic_tracking_state_mean.csv",
        "accuracy_over_drift.png",
        "ensemble_weights.png",
        "accuracy_state_average_static_vs_mmdn.png",
        "exact_match_over_drift.png",
        "mae_over_drift.png",
        "pseudo_label_keep_fraction.png",
        "recovery_examples_last_state.png",
        "summary.json",
        "run_log.txt",
    ]:
        path = output_dir / name
        if path.exists():
            print(" -", path)
    print("=" * 100)


if __name__ == "__main__":
    main()
