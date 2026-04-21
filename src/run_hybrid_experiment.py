from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


SEED = 42
WINDOW_SIZE = 11
MAX_EPOCHS = 30
PATIENCE = 4
BATCH_SIZE = 128
MC_PASSES = 10
SMOOTHING_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
SELECTIVE_COVERAGES = [0.6, 0.7, 0.8, 0.9, 1.0]

FACIES_COLORS = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#7F5539",
    "#6D597A",
    "#577590",
]

VALIDATION_PALETTE = {
    "RandomSplit": "#B8D8D8",
    "BlindWell": "#1D3557",
}


@dataclass
class ExperimentConfig:
    data_path: Path
    results_dir: Path
    figures_dir: Path
    window_size: int = WINDOW_SIZE
    seed: int = SEED
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    mc_passes: int = MC_PASSES


@dataclass
class ModelingBundle:
    df: pd.DataFrame
    core_df: pd.DataFrame
    feature_cols: list[str]
    label_encoder: LabelEncoder
    x_tab_raw: np.ndarray
    x_seq_raw: np.ndarray
    y_encoded: np.ndarray
    y_original: np.ndarray
    groups: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_numeric_text(value: object) -> object:
    if isinstance(value, str):
        token = value.strip().split()[0]
        return token
    return value


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data format: {path}")


def set_publication_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.05,
        rc={
            "axes.facecolor": "#fcfcfc",
            "figure.facecolor": "white",
            "axes.edgecolor": "#b0b0b0",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        },
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def get_facies_style(
    facies_labels: list[int],
) -> tuple[list[int], list[str], ListedColormap, BoundaryNorm, dict[int, int]]:
    ordered = [int(label) for label in facies_labels]
    colors = FACIES_COLORS[: len(ordered)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(ordered) + 1) - 0.5, cmap.N)
    index_map = {label: idx for idx, label in enumerate(ordered)}
    return ordered, colors, cmap, norm, index_map


def encode_facies(values: pd.Series | np.ndarray, index_map: dict[int, int]) -> np.ndarray:
    return np.array([index_map[int(value)] for value in values], dtype=int)


def draw_facies_strip(
    ax: plt.Axes,
    depth: np.ndarray,
    facies_values: pd.Series | np.ndarray,
    index_map: dict[int, int],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    title: str,
    show_ylabel: bool = False,
) -> None:
    encoded = encode_facies(facies_values, index_map)[:, None]
    ax.imshow(
        encoded,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[0, 1, float(depth.max()), float(depth.min())],
    )
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.grid(False)
    if show_ylabel:
        ax.set_ylabel("Depth")
    else:
        ax.tick_params(axis="y", labelleft=False)


def add_facies_legend(fig: plt.Figure, facies_labels: list[int], facies_colors: list[str], anchor_y: float = 0.0) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="none", label=f"Facies {label}")
        for label, color in zip(facies_labels, facies_colors)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, anchor_y),
    )


def load_dataset(config: ExperimentConfig) -> ModelingBundle:
    df = load_table(config.data_path)
    df = df.rename(columns={"Nombre": "Well_ID", "Formacion": "Formation"})
    df["PHIE"] = df["PHIE"].map(clean_numeric_text)
    numeric_cols = [col for col in df.columns if col not in {"Well_ID", "Formation"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Well_ID", "Depth"]).reset_index(drop=True)
    df["RelDepth"] = df.groupby("Well_ID")["Depth"].transform(
        lambda s: (s - s.min()) / max(s.max() - s.min(), 1e-6)
    )
    df["DepthStep"] = (
        df.groupby("Well_ID")["Depth"].diff().groupby(df["Well_ID"]).bfill().fillna(0.5)
    )

    # PEF is entirely missing in well A129, so it is excluded from the main workflow.
    feature_cols = [
        "RelDepth",
        "DepthStep",
        "NetRes",
        "BVW",
        "GR",
        "PHID",
        "PHIE",
        "PHIN",
        "ResD",
        "ResM",
        "ResS",
        "RHOB",
        "Ro",
        "RT",
        "SwA",
        "Vshl",
    ]

    label_encoder = LabelEncoder()
    df["FaciesEncoded"] = label_encoder.fit_transform(df["Facies"].astype(int))

    core_df, x_seq_raw = build_centered_windows(df, feature_cols, config.window_size)
    x_tab_raw = core_df[feature_cols].to_numpy(dtype=float)
    y_encoded = core_df["FaciesEncoded"].to_numpy(dtype=int)
    y_original = core_df["Facies"].to_numpy(dtype=int)
    groups = core_df["Well_ID"].to_numpy()

    return ModelingBundle(
        df=df,
        core_df=core_df,
        feature_cols=feature_cols,
        label_encoder=label_encoder,
        x_tab_raw=x_tab_raw,
        x_seq_raw=x_seq_raw,
        y_encoded=y_encoded,
        y_original=y_original,
        groups=groups,
    )


def build_centered_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[pd.DataFrame, np.ndarray]:
    half = window_size // 2
    windows: list[np.ndarray] = []
    core_rows: list[pd.Series] = []

    for _, group in df.groupby("Well_ID", sort=False):
        group = group.sort_values("Depth").reset_index(drop=True)
        features = group[feature_cols].to_numpy(dtype=float)
        if len(group) < window_size:
            continue
        for idx in range(half, len(group) - half):
            windows.append(features[idx - half : idx + half + 1])
            core_rows.append(group.iloc[idx])

    core_df = pd.DataFrame(core_rows).reset_index(drop=True)
    return core_df, np.asarray(windows, dtype=np.float32)


def choose_group_validation_indices(groups: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = list(dict.fromkeys(groups.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    all_classes = set(y.tolist())
    for group in unique_groups:
        mask_val = groups == group
        mask_train = ~mask_val
        if set(y[mask_train].tolist()) == all_classes:
            return np.where(mask_train)[0], np.where(mask_val)[0]
    fallback = unique_groups[0]
    return np.where(groups != fallback)[0], np.where(groups == fallback)[0]


def choose_random_validation_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, val_idx


def fit_preprocessors(
    x_train_raw: np.ndarray,
) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_train_raw)
    scaler = StandardScaler()
    scaler.fit(x_imputed)
    return imputer, scaler


def transform_tabular(x_raw: np.ndarray, imputer: SimpleImputer) -> np.ndarray:
    return imputer.transform(x_raw).astype(np.float32)


def transform_sequence(
    x_seq_raw: np.ndarray, imputer: SimpleImputer, scaler: StandardScaler
) -> np.ndarray:
    n_samples, n_steps, n_features = x_seq_raw.shape
    flat = x_seq_raw.reshape(-1, n_features)
    flat = imputer.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(n_samples, n_steps, n_features).astype(np.float32)


def compute_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    present = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=y)
    result = {cls: 1.0 for cls in range(n_classes)}
    for cls, weight in zip(present, weights):
        result[int(cls)] = float(min(weight, 12.0))
    return result


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs.astype(np.float64), 1e-8, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def probability_entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.clip(probs, 1e-8, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)


def probability_margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
    return top2[:, 1] - top2[:, 0]


def summarize_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = normalize_probs(probs)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(probs.shape[1])))),
        "brier_score": multiclass_brier_score(y_true, probs),
        "mean_entropy": float(probability_entropy(probs).mean()),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "mean_margin": float(probability_margin(probs).mean()),
    }


def logsumexp_np(arr: np.ndarray, axis: int) -> np.ndarray:
    max_vals = np.max(arr, axis=axis, keepdims=True)
    stabilized = np.exp(arr - max_vals)
    return np.squeeze(max_vals + np.log(np.sum(stabilized, axis=axis, keepdims=True)), axis=axis)


def split_contiguous_segments(meta: pd.DataFrame) -> list[np.ndarray]:
    sorted_meta = meta.reset_index(drop=True).copy()
    sorted_meta["RowID"] = np.arange(len(sorted_meta))
    sorted_meta = sorted_meta.sort_values(["Well_ID", "Depth", "RowID"]).reset_index(drop=True)
    segments: list[np.ndarray] = []

    for _, group in sorted_meta.groupby("Well_ID", sort=False):
        row_ids = group["RowID"].to_numpy(dtype=int)
        depths = group["Depth"].to_numpy(dtype=float)
        if len(row_ids) == 0:
            continue
        diffs = np.diff(depths)
        positive_diffs = diffs[diffs > 0]
        median_step = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
        if median_step > 0:
            break_points = np.where(diffs > 1.5 * median_step)[0] + 1
        else:
            break_points = np.array([], dtype=int)
        starts = np.concatenate(([0], break_points))
        ends = np.concatenate((break_points, [len(row_ids)]))
        for start, end in zip(starts, ends):
            segments.append(row_ids[start:end])
    return segments


def estimate_transition_prior(
    meta: pd.DataFrame,
    labels: np.ndarray,
    n_classes: int,
    pseudocount: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    start_counts = np.full(n_classes, pseudocount, dtype=float)
    transition_counts = np.full((n_classes, n_classes), pseudocount, dtype=float)
    sorted_meta = meta.reset_index(drop=True).copy()
    sorted_meta["Label"] = labels.astype(int)

    for segment in split_contiguous_segments(sorted_meta):
        segment_labels = sorted_meta.iloc[segment]["Label"].to_numpy(dtype=int)
        if len(segment_labels) == 0:
            continue
        start_counts[segment_labels[0]] += 1.0
        for left, right in zip(segment_labels[:-1], segment_labels[1:]):
            transition_counts[int(left), int(right)] += 1.0

    start_probs = start_counts / start_counts.sum()
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return start_probs.astype(np.float32), transition_probs.astype(np.float32)


def hmm_smooth_segment(
    probs: np.ndarray,
    start_probs: np.ndarray,
    transition_probs: np.ndarray,
    smoothing_strength: float,
) -> np.ndarray:
    probs = normalize_probs(probs)
    if len(probs) == 0 or smoothing_strength <= 0:
        return probs

    log_emission = np.log(np.clip(probs, 1e-8, 1.0))
    log_start = np.log(np.clip(start_probs, 1e-8, 1.0))
    log_transition = smoothing_strength * np.log(np.clip(transition_probs, 1e-8, 1.0))
    n_steps, n_classes = probs.shape

    forward = np.zeros((n_steps, n_classes), dtype=np.float64)
    backward = np.zeros((n_steps, n_classes), dtype=np.float64)
    forward[0] = log_start + log_emission[0]

    for idx in range(1, n_steps):
        transition_scores = forward[idx - 1][:, None] + log_transition
        forward[idx] = log_emission[idx] + logsumexp_np(transition_scores, axis=0)

    for idx in range(n_steps - 2, -1, -1):
        transition_scores = log_transition + log_emission[idx + 1][None, :] + backward[idx + 1][None, :]
        backward[idx] = logsumexp_np(transition_scores, axis=1)

    posterior = forward + backward
    posterior -= logsumexp_np(posterior, axis=1)[:, None]
    return normalize_probs(np.exp(posterior))


def smooth_probabilities_by_well(
    meta: pd.DataFrame,
    probs: np.ndarray,
    start_probs: np.ndarray,
    transition_probs: np.ndarray,
    smoothing_strength: float,
) -> np.ndarray:
    smoothed = normalize_probs(probs.copy())
    for segment in split_contiguous_segments(meta):
        smoothed[segment] = hmm_smooth_segment(
            probs=smoothed[segment],
            start_probs=start_probs,
            transition_probs=transition_probs,
            smoothing_strength=smoothing_strength,
        )
    return normalize_probs(smoothed)


def build_boundary_depths(depths: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if len(labels) < 2:
        return np.array([], dtype=float)
    change_mask = labels[1:] != labels[:-1]
    if not np.any(change_mask):
        return np.array([], dtype=float)
    return (depths[1:][change_mask] + depths[:-1][change_mask]) / 2.0


def mean_nearest_distance(source: np.ndarray, target: np.ndarray, fallback: float) -> float:
    if len(source) == 0 and len(target) == 0:
        return 0.0
    if len(source) == 0 or len(target) == 0:
        return float(fallback)
    distances = np.abs(source[:, None] - target[None, :]).min(axis=1)
    return float(distances.mean())


def summarize_sequence_metrics(model_pred_df: pd.DataFrame, n_classes: int) -> dict[str, float]:
    thickness_mae = []
    thickness_rmse = []
    contact_mae = []
    transition_count_error = []

    ordered = model_pred_df.sort_values(["Well_ID", "Depth"]).copy()
    for _, group in ordered.groupby("Well_ID", sort=False):
        depth = group["Depth"].to_numpy(dtype=float)
        sample_thickness = group["DepthStep"].to_numpy(dtype=float)
        y_true = group["TrueEncoded"].to_numpy(dtype=int)
        y_pred = group["PredEncoded"].to_numpy(dtype=int)

        true_thickness = np.bincount(y_true, weights=sample_thickness, minlength=n_classes)
        pred_thickness = np.bincount(y_pred, weights=sample_thickness, minlength=n_classes)
        thickness_diff = pred_thickness - true_thickness
        thickness_mae.append(float(np.mean(np.abs(thickness_diff))))
        thickness_rmse.append(float(np.sqrt(np.mean(thickness_diff**2))))

        true_boundaries = build_boundary_depths(depth, y_true)
        pred_boundaries = build_boundary_depths(depth, y_pred)
        depth_range = max(float(depth.max() - depth.min()), float(np.median(sample_thickness)))
        true_to_pred = mean_nearest_distance(true_boundaries, pred_boundaries, fallback=depth_range)
        pred_to_true = mean_nearest_distance(pred_boundaries, true_boundaries, fallback=depth_range)
        contact_mae.append((true_to_pred + pred_to_true) / 2.0)
        transition_count_error.append(float(abs(len(pred_boundaries) - len(true_boundaries))))

    return {
        "thickness_mae": float(np.mean(thickness_mae)),
        "thickness_rmse": float(np.mean(thickness_rmse)),
        "contact_mae": float(np.mean(contact_mae)),
        "transition_count_error": float(np.mean(transition_count_error)),
    }


def summarize_selective_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    coverages: list[float] = SELECTIVE_COVERAGES,
) -> dict[str, float]:
    probs = normalize_probs(probs)
    confidence = probs.max(axis=1)
    order = np.argsort(-confidence)
    y_true_sorted = y_true[order]
    probs_sorted = probs[order]
    all_labels = list(range(probs.shape[1]))
    result: dict[str, float] = {}

    for coverage in coverages:
        n_keep = max(1, int(math.ceil(len(y_true_sorted) * coverage)))
        kept_true = y_true_sorted[:n_keep]
        kept_probs = probs_sorted[:n_keep]
        kept_pred = kept_probs.argmax(axis=1)
        label = int(round(coverage * 100))
        result[f"accuracy_at_{label}"] = float(accuracy_score(kept_true, kept_pred))
        result[f"macro_f1_at_{label}"] = float(
            f1_score(kept_true, kept_pred, labels=all_labels, average="macro", zero_division=0)
        )
    return result


def generate_weight_triplets(step: float = 0.1) -> list[tuple[float, float, float]]:
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    weights: list[tuple[float, float, float]] = []
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if abs(round(w3 / step) * step - w3) > 1e-6:
                continue
            weights.append((float(w1), float(w2), float(max(0.0, w3))))
    return weights


def choose_tabular_ensemble_weights(
    val_probs: dict[str, np.ndarray], y_val: np.ndarray
) -> dict[str, float]:
    names = list(val_probs.keys())
    best_weights: dict[str, float] | None = None
    best_key = (-1.0, -math.inf)
    for triplet in generate_weight_triplets():
        combo = normalize_probs(sum(weight * val_probs[name] for weight, name in zip(triplet, names)))
        macro_f1 = f1_score(y_val, combo.argmax(axis=1), average="macro", zero_division=0)
        neg_log_loss = -log_loss(y_val, combo, labels=list(range(combo.shape[1])))
        key = (macro_f1, neg_log_loss)
        if key > best_key:
            best_key = key
            best_weights = {name: float(weight) for weight, name in zip(triplet, names)}
    assert best_weights is not None
    return best_weights


def apply_weighted_fusion(probs_by_name: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    fused = None
    for name, probs in probs_by_name.items():
        weighted = weights[name] * probs
        fused = weighted if fused is None else fused + weighted
    assert fused is not None
    return normalize_probs(fused)


def choose_hybrid_alpha(tab_probs: np.ndarray, dl_probs: np.ndarray, y_val: np.ndarray) -> float:
    best_alpha = 0.5
    best_key = (-1.0, -math.inf)
    for alpha in np.linspace(0.0, 1.0, 21):
        combo = normalize_probs(alpha * tab_probs + (1.0 - alpha) * dl_probs)
        macro_f1 = f1_score(y_val, combo.argmax(axis=1), average="macro", zero_division=0)
        neg_log_loss = -log_loss(y_val, combo, labels=list(range(combo.shape[1])))
        key = (macro_f1, neg_log_loss)
        if key > best_key:
            best_key = key
            best_alpha = float(alpha)
    return best_alpha


def choose_smoothing_strength(
    meta: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    start_probs: np.ndarray,
    transition_probs: np.ndarray,
    n_classes: int,
) -> float:
    best_strength = 0.0
    best_key = (-1.0, -math.inf, math.inf)

    for strength in SMOOTHING_GRID:
        smoothed = smooth_probabilities_by_well(
            meta=meta,
            probs=probs,
            start_probs=start_probs,
            transition_probs=transition_probs,
            smoothing_strength=float(strength),
        )
        preds = smoothed.argmax(axis=1)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        neg_log_loss = -log_loss(y_true, smoothed, labels=list(range(smoothed.shape[1])))
        temp_df = meta[["Well_ID", "Depth", "DepthStep"]].copy()
        temp_df["TrueEncoded"] = y_true
        temp_df["PredEncoded"] = preds
        contact_mae = summarize_sequence_metrics(temp_df, n_classes)["contact_mae"]
        key = (macro_f1, neg_log_loss, -contact_mae)
        if key > best_key:
            best_key = key
            best_strength = float(strength)

    return best_strength


class CNNClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.mean(dim=2)
        logits = self.head(x)
        return logits, None


class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        logits = self.head(x[:, -1, :])
        return logits, None


class CNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(128, 1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        scores = torch.tanh(self.attention(x))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        logits = self.head(context)
        return logits, weights.squeeze(-1)


def build_cnn_model(window_size: int, n_features: int, n_classes: int) -> nn.Module:
    del window_size
    return CNNClassifier(n_features=n_features, n_classes=n_classes)


def build_cnn_bilstm_model(window_size: int, n_features: int, n_classes: int) -> nn.Module:
    del window_size
    return CNNBiLSTMClassifier(n_features=n_features, n_classes=n_classes)


def build_cnn_bilstm_attention_model(window_size: int, n_features: int, n_classes: int) -> nn.Module:
    del window_size
    return CNNBiLSTMAttentionClassifier(n_features=n_features, n_classes=n_classes)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_loss(class_weights: dict[int, float], n_classes: int) -> nn.Module:
    weights = torch.ones(n_classes, dtype=torch.float32)
    for cls, weight in class_weights.items():
        weights[int(cls)] = float(weight)
    return nn.CrossEntropyLoss(weight=weights)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def predict_torch(
    model: nn.Module,
    x: np.ndarray,
    mc_passes: int = 1,
    dropout: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x_tensor = torch.from_numpy(x).float()
    draws = []
    attn_draws: list[np.ndarray] = []
    if dropout:
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        for _ in range(mc_passes):
            logits, attention = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            draws.append(probs)
            if attention is not None:
                attn_draws.append(attention.cpu().numpy())
    stacked = np.stack(draws, axis=0)
    mean_probs = normalize_probs(stacked.mean(axis=0))
    variance = stacked.var(axis=0)
    mean_attention = None
    if attn_draws:
        mean_attention = np.mean(np.stack(attn_draws, axis=0), axis=0)
    model.eval()
    return mean_probs, variance, mean_attention


def fit_torch_with_inner_validation(
    builder: Callable[[int, int, int], nn.Module],
    x_subtrain: np.ndarray,
    y_subtrain: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_full_train: np.ndarray,
    y_full_train: np.ndarray,
    class_weights: dict[int, float],
    config: ExperimentConfig,
    seed: int,
    n_classes: int,
) -> tuple[nn.Module, np.ndarray, int]:
    set_seed(seed)
    model = builder(x_subtrain.shape[1], x_subtrain.shape[2], n_classes)
    criterion = make_loss(class_weights, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = make_loader(x_subtrain, y_subtrain, config.batch_size, shuffle=True)
    x_val_tensor = torch.from_numpy(x_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    best_state = clone_state_dict(model)
    best_loss = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    model.load_state_dict(best_state)
    val_probs, _, _ = predict_torch(model, x_val)

    set_seed(seed)
    final_model = builder(x_full_train.shape[1], x_full_train.shape[2], n_classes)
    final_criterion = make_loss(class_weights, n_classes)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
    full_loader = make_loader(x_full_train, y_full_train, config.batch_size, shuffle=True)
    for _ in range(best_epoch):
        final_model.train()
        for xb, yb in full_loader:
            final_optimizer.zero_grad()
            logits, _ = final_model(xb)
            loss = final_criterion(logits, yb)
            loss.backward()
            final_optimizer.step()
    final_model.eval()
    return final_model, val_probs, best_epoch


def train_tabular_models(
    x_subtrain: np.ndarray, y_subtrain: np.ndarray, seed: int
) -> dict[str, object]:
    classes = np.unique(y_subtrain)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_subtrain)
    class_weight_map = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    sample_weight = np.asarray([class_weight_map[int(label)] for label in y_subtrain], dtype=float)

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.85,
        random_state=seed,
    )
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=4,
    )

    rf.fit(x_subtrain, y_subtrain)
    gb.fit(x_subtrain, y_subtrain, sample_weight=sample_weight)
    xgb.fit(x_subtrain, y_subtrain, sample_weight=sample_weight, verbose=False)
    return {"RandomForest": rf, "GradientBoosting": gb, "XGBoost": xgb}


def build_prediction_frame(
    meta: pd.DataFrame,
    probs: np.ndarray,
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
    validation_name: str,
    split_label: str,
    extra: dict[str, np.ndarray | float] | None = None,
) -> pd.DataFrame:
    probs = normalize_probs(probs)
    out = meta[["Well_ID", "Formation", "Depth", "RelDepth", "DepthStep", "Facies"]].copy()
    out["Validation"] = validation_name
    out["SplitLabel"] = split_label
    out["Model"] = model_name
    pred_encoded = probs.argmax(axis=1)
    out["PredEncoded"] = pred_encoded
    out["PredFacies"] = label_encoder.inverse_transform(pred_encoded)
    out["TrueEncoded"] = y_encoded
    out["Confidence"] = probs.max(axis=1)
    out["Entropy"] = probability_entropy(probs)
    out["Margin"] = probability_margin(probs)
    for idx, original_label in enumerate(label_encoder.classes_):
        out[f"Prob_{original_label}"] = probs[:, idx]
    if extra:
        for key, value in extra.items():
            out[key] = value
    return out


def evaluate_split(
    bundle: ModelingBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    validation_name: str,
    split_label: str,
    config: ExperimentConfig,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    n_classes = len(bundle.label_encoder.classes_)
    print(f"[{validation_name}] {split_label}: preparing inner split", flush=True)
    train_groups = bundle.groups[train_idx]
    y_train_all = bundle.y_encoded[train_idx]

    if validation_name == "BlindWell":
        subtrain_local, val_local = choose_group_validation_indices(train_groups, y_train_all, seed)
        subtrain_idx = train_idx[subtrain_local]
        val_idx = train_idx[val_local]
    else:
        subtrain_local, val_local = choose_random_validation_indices(y_train_all, seed)
        subtrain_idx = train_idx[subtrain_local]
        val_idx = train_idx[val_local]

    imputer, scaler = fit_preprocessors(bundle.x_tab_raw[subtrain_idx])
    x_tab_sub = transform_tabular(bundle.x_tab_raw[subtrain_idx], imputer)
    x_tab_val = transform_tabular(bundle.x_tab_raw[val_idx], imputer)
    x_tab_train = transform_tabular(bundle.x_tab_raw[train_idx], imputer)
    x_tab_test = transform_tabular(bundle.x_tab_raw[test_idx], imputer)

    x_seq_sub = transform_sequence(bundle.x_seq_raw[subtrain_idx], imputer, scaler)
    x_seq_val = transform_sequence(bundle.x_seq_raw[val_idx], imputer, scaler)
    x_seq_train = transform_sequence(bundle.x_seq_raw[train_idx], imputer, scaler)
    x_seq_test = transform_sequence(bundle.x_seq_raw[test_idx], imputer, scaler)

    meta_subtrain = bundle.core_df.iloc[subtrain_idx].reset_index(drop=True)
    meta_val = bundle.core_df.iloc[val_idx].reset_index(drop=True)
    meta_train = bundle.core_df.iloc[train_idx].reset_index(drop=True)
    meta_test = bundle.core_df.iloc[test_idx].reset_index(drop=True)

    y_sub = bundle.y_encoded[subtrain_idx]
    y_val = bundle.y_encoded[val_idx]
    y_train = bundle.y_encoded[train_idx]
    y_test = bundle.y_encoded[test_idx]

    sub_start_probs, sub_transition_probs = estimate_transition_prior(
        meta=meta_subtrain,
        labels=y_sub,
        n_classes=n_classes,
    )
    train_start_probs, train_transition_probs = estimate_transition_prior(
        meta=meta_train,
        labels=y_train,
        n_classes=n_classes,
    )

    print(f"[{validation_name}] {split_label}: training tabular models", flush=True)
    tabular_models = train_tabular_models(x_tab_sub, y_sub, seed)
    val_tab_probs = {name: model.predict_proba(x_tab_val) for name, model in tabular_models.items()}
    tab_weights = choose_tabular_ensemble_weights(val_tab_probs, y_val)
    ensemble_val_probs = apply_weighted_fusion(val_tab_probs, tab_weights)

    tabular_models = train_tabular_models(x_tab_train, y_train, seed)
    test_tab_probs = {name: model.predict_proba(x_tab_test) for name, model in tabular_models.items()}
    ensemble_test_probs = apply_weighted_fusion(test_tab_probs, tab_weights)

    dl_builders = {
        "CNN": build_cnn_model,
        "CNN_BiLSTM": build_cnn_bilstm_model,
        "CNN_BiLSTM_Attention": build_cnn_bilstm_attention_model,
    }
    dl_test_probs: dict[str, np.ndarray] = {}
    dl_extra_outputs: dict[str, dict[str, object]] = {}
    class_weights = compute_class_weights(y_sub, n_classes)

    for offset, (name, builder) in enumerate(dl_builders.items(), start=1):
        print(f"[{validation_name}] {split_label}: training {name}", flush=True)
        model, val_probs, best_epoch = fit_torch_with_inner_validation(
            builder=builder,
            x_subtrain=x_seq_sub,
            y_subtrain=y_sub,
            x_val=x_seq_val,
            y_val=y_val,
            x_full_train=x_seq_train,
            y_full_train=y_train,
            class_weights=class_weights,
            config=config,
            seed=seed + offset,
            n_classes=n_classes,
        )
        if name == "CNN_BiLSTM_Attention":
            test_probs, test_var, attention_weights = predict_torch(
                model,
                x_seq_test,
                mc_passes=config.mc_passes,
                dropout=True,
            )
            dl_extra_outputs[name] = {
                "EpistemicVariance": test_var.mean(axis=1),
                "AttentionWeights": attention_weights,
                "BestEpoch": best_epoch,
            }
            val_attention_probs, _, _ = predict_torch(
                model,
                x_seq_val,
                mc_passes=config.mc_passes,
                dropout=True,
            )
            dl_val_probs = val_attention_probs
        else:
            test_probs, _, _ = predict_torch(model, x_seq_test)
            dl_extra_outputs[name] = {"BestEpoch": best_epoch}
            dl_val_probs = val_probs
        dl_test_probs[name] = test_probs
        dl_extra_outputs[name]["ValProbs"] = dl_val_probs

    print(f"[{validation_name}] {split_label}: building hybrid fusion", flush=True)
    hybrid_alpha = choose_hybrid_alpha(
        ensemble_val_probs,
        dl_extra_outputs["CNN_BiLSTM_Attention"]["ValProbs"],
        y_val,
    )
    hybrid_val_probs = normalize_probs(
        hybrid_alpha * ensemble_val_probs
        + (1.0 - hybrid_alpha) * dl_extra_outputs["CNN_BiLSTM_Attention"]["ValProbs"]
    )
    hybrid_test_probs = normalize_probs(
        hybrid_alpha * ensemble_test_probs + (1.0 - hybrid_alpha) * dl_test_probs["CNN_BiLSTM_Attention"]
    )

    print(f"[{validation_name}] {split_label}: applying geological smoothing", flush=True)
    geo_ensemble_strength = choose_smoothing_strength(
        meta=meta_val,
        probs=ensemble_val_probs,
        y_true=y_val,
        start_probs=sub_start_probs,
        transition_probs=sub_transition_probs,
        n_classes=n_classes,
    )
    geo_hybrid_strength = choose_smoothing_strength(
        meta=meta_val,
        probs=hybrid_val_probs,
        y_true=y_val,
        start_probs=sub_start_probs,
        transition_probs=sub_transition_probs,
        n_classes=n_classes,
    )
    geo_ensemble_test_probs = smooth_probabilities_by_well(
        meta=meta_test,
        probs=ensemble_test_probs,
        start_probs=train_start_probs,
        transition_probs=train_transition_probs,
        smoothing_strength=geo_ensemble_strength,
    )
    geo_hybrid_test_probs = smooth_probabilities_by_well(
        meta=meta_test,
        probs=hybrid_test_probs,
        start_probs=train_start_probs,
        transition_probs=train_transition_probs,
        smoothing_strength=geo_hybrid_strength,
    )

    predictions = [
        build_prediction_frame(
            meta=meta_test,
            probs=test_tab_probs["RandomForest"],
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="RandomForest",
            validation_name=validation_name,
            split_label=split_label,
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=test_tab_probs["GradientBoosting"],
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="GradientBoosting",
            validation_name=validation_name,
            split_label=split_label,
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=test_tab_probs["XGBoost"],
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="XGBoost",
            validation_name=validation_name,
            split_label=split_label,
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=ensemble_test_probs,
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="ProbabilisticEnsemble",
            validation_name=validation_name,
            split_label=split_label,
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=geo_ensemble_test_probs,
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="GeoConstrainedEnsemble",
            validation_name=validation_name,
            split_label=split_label,
            extra={"GeoStrength": geo_ensemble_strength},
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=dl_test_probs["CNN"],
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="CNN",
            validation_name=validation_name,
            split_label=split_label,
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=dl_test_probs["CNN_BiLSTM"],
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="CNN_BiLSTM",
            validation_name=validation_name,
            split_label=split_label,
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=dl_test_probs["CNN_BiLSTM_Attention"],
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="CNN_BiLSTM_Attention",
            validation_name=validation_name,
            split_label=split_label,
            extra={
                "DLVariance": dl_extra_outputs["CNN_BiLSTM_Attention"]["EpistemicVariance"],
            },
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=hybrid_test_probs,
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="HybridFusion",
            validation_name=validation_name,
            split_label=split_label,
            extra={
                "DLVariance": dl_extra_outputs["CNN_BiLSTM_Attention"]["EpistemicVariance"],
            },
        ),
        build_prediction_frame(
            meta=meta_test,
            probs=geo_hybrid_test_probs,
            y_encoded=y_test,
            label_encoder=bundle.label_encoder,
            model_name="GeoConstrainedHybrid",
            validation_name=validation_name,
            split_label=split_label,
            extra={
                "DLVariance": dl_extra_outputs["CNN_BiLSTM_Attention"]["EpistemicVariance"],
                "GeoStrength": geo_hybrid_strength,
            },
        ),
    ]
    prediction_df = pd.concat(predictions, ignore_index=True)

    metrics_rows = []
    for model_name, model_pred_df in prediction_df.groupby("Model", sort=False):
        probs_cols = [col for col in model_pred_df.columns if col.startswith("Prob_")]
        probs = model_pred_df[probs_cols].to_numpy()
        y_true = model_pred_df["TrueEncoded"].to_numpy()
        row = summarize_metrics(y_true, probs)
        row.update(summarize_sequence_metrics(model_pred_df, n_classes=n_classes))
        row.update(summarize_selective_metrics(y_true, probs))
        row.update(
            {
                "Validation": validation_name,
                "SplitLabel": split_label,
                "Model": model_name,
                "n_samples": len(model_pred_df),
            }
        )
        if "GeoStrength" in model_pred_df.columns and model_pred_df["GeoStrength"].notna().any():
            row["geo_strength"] = float(model_pred_df["GeoStrength"].dropna().iloc[0])
        metrics_rows.append(row)

    artifacts = {
        "tab_weights": tab_weights,
        "hybrid_alpha": hybrid_alpha,
        "geo_ensemble_strength": geo_ensemble_strength,
        "geo_hybrid_strength": geo_hybrid_strength,
        "attention_weights": dl_extra_outputs["CNN_BiLSTM_Attention"]["AttentionWeights"],
        "representative_meta": meta_test,
        "xgb_model": tabular_models["XGBoost"],
        "x_tab_train": x_tab_train,
        "feature_names": bundle.feature_cols,
        "start_probs": train_start_probs,
        "transition_matrix": train_transition_probs,
    }
    return prediction_df, metrics_rows, artifacts


def run_experiments(bundle: ModelingBundle, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    blind_predictions = []
    blind_metrics = []
    blind_artifacts = {}

    unique_wells = bundle.core_df["Well_ID"].unique().tolist()
    for fold_id, well in enumerate(unique_wells, start=1):
        print(f"[BlindWell] starting outer fold for {well}", flush=True)
        test_idx = np.where(bundle.groups == well)[0]
        train_idx = np.where(bundle.groups != well)[0]
        pred_df, metrics_rows, artifacts = evaluate_split(
            bundle=bundle,
            train_idx=train_idx,
            test_idx=test_idx,
            validation_name="BlindWell",
            split_label=f"Blind_{well}",
            config=config,
            seed=config.seed + fold_id,
        )
        blind_predictions.append(pred_df)
        blind_metrics.extend(metrics_rows)
        blind_artifacts[well] = artifacts

    all_indices = np.arange(len(bundle.core_df))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=config.seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(all_indices)), bundle.y_encoded))
    print("[RandomSplit] starting 70/30 experiment", flush=True)
    random_predictions, random_metrics_rows, random_artifacts = evaluate_split(
        bundle=bundle,
        train_idx=train_idx,
        test_idx=test_idx,
        validation_name="RandomSplit",
        split_label="Random_70_30",
        config=config,
        seed=config.seed,
    )

    predictions_df = pd.concat([pd.concat(blind_predictions, ignore_index=True), random_predictions], ignore_index=True)
    metrics_df = pd.DataFrame(blind_metrics + random_metrics_rows)
    return predictions_df, metrics_df, {"blind": blind_artifacts, "random": random_artifacts}


def aggregate_metrics(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "mean_entropy",
        "mean_confidence",
        "mean_margin",
        "thickness_mae",
        "thickness_rmse",
        "contact_mae",
        "transition_count_error",
        "accuracy_at_60",
        "macro_f1_at_60",
        "accuracy_at_70",
        "macro_f1_at_70",
        "accuracy_at_80",
        "macro_f1_at_80",
        "accuracy_at_90",
        "macro_f1_at_90",
        "accuracy_at_100",
        "macro_f1_at_100",
    ]
    optional_cols = [col for col in metric_cols if col in metrics_df.columns]
    summary = (
        metrics_df.groupby(["Validation", "Model"], as_index=False)[optional_cols]
        .mean()
        .sort_values(["Validation", "macro_f1"], ascending=[True, False])
    )
    per_split = metrics_df.sort_values(["Validation", "SplitLabel", "macro_f1"], ascending=[True, True, False])
    return summary, per_split


def choose_representative_well(per_split_df: pd.DataFrame) -> str:
    hybrid = per_split_df[
        (per_split_df["Validation"] == "BlindWell") & (per_split_df["Model"] == "HybridFusion")
    ].copy()
    hybrid = hybrid.sort_values(["macro_f1", "accuracy", "log_loss"], ascending=[True, True, False])
    worst_split = hybrid.iloc[0]["SplitLabel"]
    return str(worst_split).replace("Blind_", "", 1)


def save_artifacts(
    bundle: ModelingBundle,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_split_df: pd.DataFrame,
    artifacts: dict[str, object],
    config: ExperimentConfig,
) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(config.results_dir / "predictions_all.csv", index=False)
    metrics_df.to_csv(config.results_dir / "metrics_by_split.csv", index=False)
    summary_df.to_csv(config.results_dir / "metrics_summary.csv", index=False)
    per_split_df.to_csv(config.results_dir / "metrics_per_split.csv", index=False)
    selective_curves_df = build_selective_curves(
        predictions_df=predictions_df,
        validation_name="BlindWell",
        model_names=[
            "ProbabilisticEnsemble",
            "GeoConstrainedEnsemble",
            "HybridFusion",
            "GeoConstrainedHybrid",
        ],
    )
    selective_curves_df.to_csv(config.results_dir / "selective_curves_blindwell.csv", index=False)
    with open(config.results_dir / "label_mapping.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "encoded_to_original": {
                    int(idx): int(label) for idx, label in enumerate(bundle.label_encoder.classes_)
                },
                "feature_columns": bundle.feature_cols,
            },
            fp,
            indent=2,
        )

    representative_well = choose_representative_well(per_split_df)
    attention_weights = artifacts["blind"][representative_well]["attention_weights"]
    np.save(config.results_dir / f"attention_weights_{representative_well}.npy", attention_weights)
    with open(config.results_dir / "run_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "data_path": str(config.data_path),
                "window_size": config.window_size,
                "seed": config.seed,
                "max_epochs": config.max_epochs,
                "patience": config.patience,
                "batch_size": config.batch_size,
                "mc_passes": config.mc_passes,
                "representative_well": representative_well,
                "n_raw_samples": int(len(bundle.df)),
                "n_windowed_samples": int(len(bundle.core_df)),
                "n_wells": int(bundle.df["Well_ID"].nunique()),
            },
            fp,
            indent=2,
        )

    facies_labels = [int(label) for label in bundle.label_encoder.classes_]
    print("[Artifacts] rendering workflow figures", flush=True)
    plot_workflow(config.figures_dir / "figure_1_workflow.png")
    plot_random_vs_blind(summary_df, config.figures_dir / "figure_3_random_vs_blind.png")
    plot_model_comparison(summary_df, config.figures_dir / "figure_4_model_comparison.png")
    print(f"[Artifacts] rendering depth and uncertainty figures for {representative_well}", flush=True)
    plot_depth_predictions(
        bundle,
        predictions_df,
        facies_labels,
        representative_well,
        config.figures_dir / "figure_5_depth_predictions.png",
    )
    plot_uncertainty(
        predictions_df,
        facies_labels,
        representative_well,
        config.figures_dir / "figure_6_uncertainty.png",
    )
    print("[Artifacts] rendering SHAP and attention figure", flush=True)
    plot_shap_and_attention(
        bundle=bundle,
        predictions_df=predictions_df,
        xgb_model=artifacts["blind"][representative_well]["xgb_model"],
        x_tab_train=artifacts["blind"][representative_well]["x_tab_train"],
        feature_names=artifacts["blind"][representative_well]["feature_names"],
        attention_weights=attention_weights,
        representative_well=representative_well,
        output_path=config.figures_dir / "figure_7_shap_attention.png",
    )
    plot_sequence_value(summary_df, config.figures_dir / "figure_8_sequence_value.png")
    plot_selective_prediction(selective_curves_df, config.figures_dir / "figure_9_selective_prediction.png")


def plot_workflow(output_path: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(13.4, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    training_zone = Rectangle(
        (0.03, 0.12),
        0.69,
        0.76,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        edgecolor="#94A3B8",
        facecolor="#F8FAFC",
    )
    ax.add_patch(training_zone)
    ax.text(0.05, 0.84, "Training wells only", ha="left", va="center", fontsize=12, fontweight="bold", color="#0F172A")
    ax.text(
        0.05,
        0.80,
        "Scaling, tuning, calibration, probability fusion weights and transition priors are learned inside this zone.",
        ha="left",
        va="center",
        fontsize=9.6,
        color="#475569",
    )

    def add_card(
        x: float,
        y: float,
        w: float,
        h: float,
        accent: str,
        title: str,
        lines: list[str],
        facecolor: str = "#FFFFFF",
    ) -> None:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.1, edgecolor="#334155", facecolor=facecolor))
        ax.add_patch(Rectangle((x, y + h - 0.038), w, 0.038, linewidth=0, facecolor=accent))
        ax.text(x + 0.016, y + h - 0.06, title, ha="left", va="top", fontsize=11.2, fontweight="bold", color="#0F172A")
        line_y = y + h - 0.12
        for line in lines:
            ax.text(x + 0.018, line_y, line, ha="left", va="top", fontsize=9.6, color="#334155")
            line_y -= 0.067

    add_card(
        0.05,
        0.28,
        0.16,
        0.42,
        "#54728C",
        "Input logs",
        [
            "GR, RHOB, PHID, RT and derived curves",
            "Quality control and harmonization",
            "Depth ordering within each well",
        ],
    )
    add_card(
        0.25,
        0.28,
        0.16,
        0.42,
        "#5B7F67",
        "Blind-well protocol",
        [
            "Outer leave-one-well-out split",
            "Inner split for tuning and calibration",
            "No blind-well labels during training",
        ],
    )

    ax.add_patch(Rectangle((0.45, 0.22), 0.23, 0.50, linewidth=1.0, edgecolor="#CBD5E1", facecolor="#F8FAFC"))
    ax.text(0.466, 0.69, "Complementary modeling", ha="left", va="top", fontsize=11.2, fontweight="bold", color="#0F172A")
    ax.text(
        0.466,
        0.652,
        "Tabular interactions and vertical continuity are learned in parallel.",
        ha="left",
        va="top",
        fontsize=9.2,
        color="#475569",
    )
    add_card(
        0.47,
        0.47,
        0.19,
        0.14,
        "#6687A8",
        "Tabular branch",
        [
            "Random Forest, Gradient Boosting, XGBoost",
            "Calibrated probabilistic ensemble",
        ],
    )
    add_card(
        0.47,
        0.29,
        0.19,
        0.14,
        "#B67B52",
        "Sequence branch",
        [
            "1D-CNN + BiLSTM + attention",
            "Depth windows and Monte Carlo dropout",
        ],
    )
    add_card(
        0.50,
        0.15,
        0.13,
        0.10,
        "#6B7280",
        "Decision layer",
        [
            "Probability fusion",
            "HMM-style geological smoothing",
        ],
        facecolor="#FFFFFF",
    )

    add_card(
        0.77,
        0.28,
        0.19,
        0.42,
        "#3F4D63",
        "Blind-well outputs",
        [
            "Facies track along depth",
            "Confidence, entropy and margin",
            "SHAP, attention and sequence diagnostics",
            "Coverage-risk and contact/thickness error",
        ],
        facecolor="#FFFFFF",
    )

    arrow_style = {"arrowstyle": "-|>", "lw": 1.7, "color": "#64748B", "shrinkA": 6, "shrinkB": 6}
    ax.annotate("", xy=(0.25, 0.49), xytext=(0.21, 0.49), arrowprops=arrow_style)
    ax.annotate("", xy=(0.47, 0.54), xytext=(0.41, 0.54), arrowprops=arrow_style)
    ax.annotate("", xy=(0.47, 0.36), xytext=(0.41, 0.42), arrowprops=arrow_style)
    ax.annotate("", xy=(0.565, 0.25), xytext=(0.565, 0.47), arrowprops=arrow_style)
    ax.annotate("", xy=(0.565, 0.25), xytext=(0.565, 0.29), arrowprops=arrow_style)
    ax.annotate("", xy=(0.77, 0.49), xytext=(0.63, 0.20), arrowprops=arrow_style)

    ax.text(0.77, 0.84, "Blind well kept outside training and tuning", ha="left", va="center", fontsize=10.2, color="#334155")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_path)


def plot_random_vs_blind(summary_df: pd.DataFrame, output_path: Path) -> None:
    set_publication_style()
    model_order = ["RandomForest", "ProbabilisticEnsemble", "GeoConstrainedEnsemble", "GeoConstrainedHybrid"]
    model_labels = {
        "RandomForest": "RF",
        "ProbabilisticEnsemble": "Ensemble",
        "GeoConstrainedEnsemble": "Geo-Ensemble",
        "GeoConstrainedHybrid": "Geo-Hybrid",
    }
    plot_df = summary_df[summary_df["Model"].isin(model_order)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)

    for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        pivot = plot_df.pivot(index="Model", columns="Validation", values=metric).reindex(model_order)
        y = np.arange(len(model_order))
        ax.hlines(y, pivot["BlindWell"], pivot["RandomSplit"], color="#c7c7c7", lw=2.2, zorder=1)
        ax.scatter(
            pivot["RandomSplit"],
            y,
            s=80,
            color=VALIDATION_PALETTE["RandomSplit"],
            edgecolor="white",
            linewidth=0.7,
            label="Random split",
            zorder=3,
        )
        ax.scatter(
            pivot["BlindWell"],
            y,
            s=80,
            color=VALIDATION_PALETTE["BlindWell"],
            edgecolor="white",
            linewidth=0.7,
            label="Blind well",
            zorder=3,
        )
        for yi, (_, row) in enumerate(pivot.iterrows()):
            delta = row["BlindWell"] - row["RandomSplit"]
            ax.text(max(row["RandomSplit"], row["BlindWell"]) + 0.01, yi, f"{delta:+.02f}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([model_labels[name] for name in model_order])
        ax.set_xlabel(title)
        ax.set_title(f"{title}: optimism gap from random to blind-well testing")
        ax.set_xlim(max(0.0, pivot.min().min() - 0.08), min(1.02, pivot.max().max() + 0.10))

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], loc="lower left", frameon=True)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_model_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    set_publication_style()
    blind_df = summary_df[summary_df["Validation"] == "BlindWell"].copy()
    blind_df = blind_df.set_index("Model").reindex(
        [
            "RandomForest",
            "ProbabilisticEnsemble",
            "GeoConstrainedEnsemble",
            "XGBoost",
            "GradientBoosting",
            "HybridFusion",
            "GeoConstrainedHybrid",
            "CNN_BiLSTM_Attention",
            "CNN_BiLSTM",
            "CNN",
        ]
    ).dropna(how="all")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={"width_ratios": [2.2, 1.2]})
    higher_better = blind_df[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
    lower_better = blind_df[["log_loss", "brier_score"]]
    sns.heatmap(higher_better, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar_kws={"shrink": 0.7})
    sns.heatmap(lower_better, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=axes[1], cbar_kws={"shrink": 0.7})
    axes[0].set_title("Blind-well discrimination metrics (higher is better)")
    axes[1].set_title("Blind-well probability errors (lower is better)")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_depth_predictions(
    bundle: ModelingBundle,
    predictions_df: pd.DataFrame,
    facies_labels: list[int],
    well_id: str,
    output_path: Path,
) -> None:
    set_publication_style()
    subset = predictions_df[
        (predictions_df["Validation"] == "BlindWell")
        & (predictions_df["SplitLabel"] == f"Blind_{well_id}")
        & (
            predictions_df["Model"].isin(
                [
                    "ProbabilisticEnsemble",
                    "GeoConstrainedEnsemble",
                    "HybridFusion",
                    "GeoConstrainedHybrid",
                ]
            )
        )
    ].copy()
    logs = bundle.core_df[bundle.core_df["Well_ID"] == well_id].sort_values("Depth").copy()
    truth = subset[subset["Model"] == "GeoConstrainedHybrid"].sort_values("Depth")
    facies_labels, facies_colors, cmap, norm, index_map = get_facies_style(facies_labels)

    fig, axes = plt.subplots(
        1,
        8,
        figsize=(18.2, 9.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.15, 1.35, 0.40, 0.40, 0.40, 0.40, 0.40], "wspace": 0.08},
    )

    depth = logs["Depth"].to_numpy()
    axes[0].plot(logs["GR"], depth, color="#2A6F97", lw=1.8)
    axes[0].fill_betweenx(depth, logs["GR"].min(), logs["GR"], color="#8ECAE6", alpha=0.35)
    axes[0].set_title("GR")
    axes[0].set_xlabel("gAPI")
    axes[0].set_ylabel("Depth")

    axes[1].plot(logs["PHID"], depth, color="#F4A261", lw=1.8)
    axes[1].fill_betweenx(depth, logs["PHID"].min(), logs["PHID"], color="#F6BD60", alpha=0.35)
    axes[1].set_title("PHID")
    axes[1].set_xlabel("v/v")

    rt_track = np.clip(logs["RT"].to_numpy(), 1e-2, None)
    axes[2].plot(rt_track, depth, color="#9C6644", lw=1.8)
    axes[2].set_xscale("log")
    axes[2].set_title("RT")
    axes[2].set_xlabel("ohm.m")

    draw_facies_strip(axes[3], truth["Depth"].to_numpy(), truth["Facies"], index_map, cmap, norm, "True")
    draw_facies_strip(
        axes[4],
        truth["Depth"].to_numpy(),
        subset[subset["Model"] == "ProbabilisticEnsemble"].sort_values("Depth")["PredFacies"],
        index_map,
        cmap,
        norm,
        "Ens.",
    )
    draw_facies_strip(
        axes[5],
        truth["Depth"].to_numpy(),
        subset[subset["Model"] == "GeoConstrainedEnsemble"].sort_values("Depth")["PredFacies"],
        index_map,
        cmap,
        norm,
        "Geo-Ens.",
    )
    draw_facies_strip(
        axes[6],
        truth["Depth"].to_numpy(),
        subset[subset["Model"] == "HybridFusion"].sort_values("Depth")["PredFacies"],
        index_map,
        cmap,
        norm,
        "Hybrid",
    )
    draw_facies_strip(
        axes[7],
        truth["Depth"].to_numpy(),
        subset[subset["Model"] == "GeoConstrainedHybrid"].sort_values("Depth")["PredFacies"],
        index_map,
        cmap,
        norm,
        "Geo-Hyb.",
    )

    for ax in axes[:3]:
        ax.set_ylim(depth.max(), depth.min())
    fig.suptitle(f"Blind-well multi-track panel: {well_id}", y=0.985)
    add_facies_legend(fig, facies_labels, facies_colors, anchor_y=0.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    save_figure(fig, output_path)


def plot_uncertainty(
    predictions_df: pd.DataFrame,
    facies_labels: list[int],
    well_id: str,
    output_path: Path,
) -> None:
    set_publication_style()
    subset = predictions_df[
        (predictions_df["Validation"] == "BlindWell")
        & (predictions_df["SplitLabel"] == f"Blind_{well_id}")
        & (predictions_df["Model"] == "GeoConstrainedHybrid")
    ].copy().sort_values("Depth")
    facies_labels, facies_colors, cmap, norm, index_map = get_facies_style(facies_labels)
    prob_cols = [f"Prob_{label}" for label in facies_labels]
    prob_matrix = subset[prob_cols].to_numpy().T

    fig, axes = plt.subplots(
        1,
        7,
        figsize=(17.2, 9.2),
        sharey=True,
        gridspec_kw={"width_ratios": [0.36, 0.36, 1.8, 0.9, 0.9, 0.9, 0.9], "wspace": 0.08},
    )
    depth = subset["Depth"].to_numpy()
    draw_facies_strip(axes[0], depth, subset["Facies"], index_map, cmap, norm, "True", show_ylabel=True)
    draw_facies_strip(axes[1], depth, subset["PredFacies"], index_map, cmap, norm, "Geo-Hybrid")

    heatmap = axes[2].imshow(
        prob_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=[-0.5, len(facies_labels) - 0.5, float(depth.max()), float(depth.min())],
    )
    axes[2].set_title("Geo-Hybrid class probabilities")
    axes[2].set_xticks(np.arange(len(facies_labels)))
    axes[2].set_xticklabels([str(label) for label in facies_labels], rotation=45, ha="right")
    axes[2].set_xlabel("Facies")
    axes[2].grid(False)

    metrics = [
        ("Confidence", "#1D3557", "Pmax", (0.0, 1.02)),
        ("Entropy", "#C1121F", "Entropy", (0.0, max(1.0, float(subset["Entropy"].max()) * 1.05))),
        ("Margin", "#2A9D8F", "Top1-Top2", (0.0, 1.02)),
        ("DLVariance", "#6D597A", "MC var", (0.0, max(0.01, float(subset["DLVariance"].max()) * 1.10))),
    ]
    for ax, (column, color, xlabel, xlim) in zip(axes[3:], metrics):
        ax.plot(subset[column], depth, color=color, lw=1.7)
        ax.fill_betweenx(depth, 0, subset[column], color=color, alpha=0.15)
        ax.set_title(column)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)

    cbar = fig.colorbar(heatmap, ax=axes[2], fraction=0.045, pad=0.02)
    cbar.set_label("Probability")
    fig.suptitle(f"Geo-hybrid probability and uncertainty diagnostics: {well_id}", y=0.99)
    add_facies_legend(fig, facies_labels, facies_colors, anchor_y=0.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    save_figure(fig, output_path)


def plot_shap_and_attention(
    bundle: ModelingBundle,
    predictions_df: pd.DataFrame,
    xgb_model: XGBClassifier,
    x_tab_train: np.ndarray,
    feature_names: list[str],
    attention_weights: np.ndarray,
    representative_well: str,
    output_path: Path,
) -> None:
    set_publication_style()
    explainer = shap.TreeExplainer(xgb_model)
    sample = x_tab_train[: min(300, len(x_tab_train))]
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        values = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
    else:
        values = np.mean(np.abs(shap_values), axis=(0, 1))

    attn_df = predictions_df[
        (predictions_df["Validation"] == "BlindWell")
        & (predictions_df["SplitLabel"] == f"Blind_{representative_well}")
        & (predictions_df["Model"] == "CNN_BiLSTM_Attention")
    ].copy()
    attn_df = attn_df.sort_values("Depth").reset_index(drop=True)
    n_rows = min(len(attn_df), attention_weights.shape[0])
    averaged_attention = attention_weights[:n_rows].mean(axis=0)
    spread_attention = attention_weights[:n_rows].std(axis=0)
    top_idx = np.argsort(values)[-10:]
    top_values = values[top_idx]
    top_features = np.array(feature_names)[top_idx]
    order = np.argsort(top_values)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    axes[0].barh(top_features[order], top_values[order], color=sns.color_palette("crest", len(order)))
    axes[0].set_title("Top XGBoost drivers from SHAP")
    axes[0].set_xlabel("Importance")

    positions = np.arange(len(averaged_attention)) - len(averaged_attention) // 2
    axes[1].plot(positions, averaged_attention, color="#C1121F", lw=2)
    axes[1].fill_between(
        positions,
        np.maximum(averaged_attention - spread_attention, 0),
        averaged_attention + spread_attention,
        color="#F4ACB7",
        alpha=0.35,
    )
    axes[1].axvline(0, color="#6c757d", ls="--", lw=1.2)
    axes[1].set_title(f"Attention profile around target depth ({representative_well})")
    axes[1].set_xlabel("Window position relative to target")
    axes[1].set_ylabel("Weight")
    fig.tight_layout()
    save_figure(fig, output_path)


def build_selective_curves(
    predictions_df: pd.DataFrame,
    validation_name: str,
    model_names: list[str],
    coverages: list[float] | None = None,
) -> pd.DataFrame:
    if coverages is None:
        coverages = list(np.linspace(0.5, 1.0, 11))

    rows: list[dict[str, float | str]] = []
    subset = predictions_df[
        (predictions_df["Validation"] == validation_name) & (predictions_df["Model"].isin(model_names))
    ].copy()

    for (model_name, split_label), group in subset.groupby(["Model", "SplitLabel"], sort=False):
        probs_cols = [col for col in group.columns if col.startswith("Prob_")]
        probs = normalize_probs(group[probs_cols].to_numpy())
        y_true = group["TrueEncoded"].to_numpy(dtype=int)
        confidence = probs.max(axis=1)
        order = np.argsort(-confidence)
        probs = probs[order]
        y_true = y_true[order]
        all_labels = list(range(probs.shape[1]))

        for coverage in coverages:
            n_keep = max(1, int(math.ceil(len(group) * coverage)))
            kept_probs = probs[:n_keep]
            kept_true = y_true[:n_keep]
            kept_pred = kept_probs.argmax(axis=1)
            rows.append(
                {
                    "Validation": validation_name,
                    "Model": model_name,
                    "SplitLabel": split_label,
                    "Coverage": float(coverage),
                    "Accuracy": float(accuracy_score(kept_true, kept_pred)),
                    "MacroF1": float(
                        f1_score(
                            kept_true,
                            kept_pred,
                            labels=all_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_sequence_value(summary_df: pd.DataFrame, output_path: Path) -> None:
    set_publication_style()
    model_order = [
        "RandomForest",
        "ProbabilisticEnsemble",
        "GeoConstrainedEnsemble",
        "HybridFusion",
        "GeoConstrainedHybrid",
    ]
    label_map = {
        "RandomForest": "RF",
        "ProbabilisticEnsemble": "Ensemble",
        "GeoConstrainedEnsemble": "Geo-Ensemble",
        "HybridFusion": "Hybrid",
        "GeoConstrainedHybrid": "Geo-Hybrid",
    }
    blind_df = (
        summary_df[summary_df["Validation"] == "BlindWell"]
        .set_index("Model")
        .reindex(model_order)
        .dropna(how="all")
        .reset_index()
    )
    blind_df["Label"] = blind_df["Model"].map(label_map)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    metrics = [
        ("contact_mae", "Contact displacement\n(lower is better)", "Mean absolute depth error"),
        ("thickness_mae", "Facies thickness error\n(lower is better)", "Mean absolute thickness error"),
        ("transition_count_error", "Over-segmentation\n(lower is better)", "Absolute transition-count error"),
    ]
    palette = sns.color_palette("crest", len(blind_df))

    for ax, (metric, title, xlabel) in zip(axes, metrics):
        sns.barplot(
            data=blind_df,
            y="Label",
            x=metric,
            hue="Label",
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax,
            orient="h",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")

    fig.suptitle("Blind-well geological utility beyond aggregate accuracy", y=0.98)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_selective_prediction(selective_df: pd.DataFrame, output_path: Path) -> None:
    set_publication_style()
    model_order = [
        "ProbabilisticEnsemble",
        "GeoConstrainedEnsemble",
        "HybridFusion",
        "GeoConstrainedHybrid",
    ]
    label_map = {
        "ProbabilisticEnsemble": "Ensemble",
        "GeoConstrainedEnsemble": "Geo-Ensemble",
        "HybridFusion": "Hybrid",
        "GeoConstrainedHybrid": "Geo-Hybrid",
    }
    plot_df = (
        selective_df[selective_df["Model"].isin(model_order)]
        .groupby(["Model", "Coverage"], as_index=False)[["Accuracy", "MacroF1"]]
        .mean()
    )
    plot_df["Label"] = plot_df["Model"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    sns.lineplot(
        data=plot_df,
        x="Coverage",
        y="Accuracy",
        hue="Label",
        marker="o",
        linewidth=2,
        ax=axes[0],
    )
    sns.lineplot(
        data=plot_df,
        x="Coverage",
        y="MacroF1",
        hue="Label",
        marker="o",
        linewidth=2,
        ax=axes[1],
    )
    axes[0].set_title("Coverage-risk curve: accuracy")
    axes[1].set_title("Coverage-risk curve: macro-F1")
    for ax in axes:
        ax.set_xlabel("Coverage retained")
        ax.set_xlim(0.48, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", frameon=True)
    axes[1].get_legend().remove()
    fig.suptitle("Selective prediction from blind-well confidence", y=0.98)
    fig.tight_layout()
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the geologically constrained blind-well facies experiment."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the anonymized CSV/XLSX dataset. Defaults to data/facies_logs_anonymized.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory where CSV outputs will be written. Defaults to outputs/results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where figures will be written. Defaults to outputs/figures",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    config = ExperimentConfig(
        data_path=args.data_path or (project_root / "data" / "facies_logs_anonymized.csv"),
        results_dir=args.results_dir or (project_root / "outputs" / "results"),
        figures_dir=args.figures_dir or (project_root / "outputs" / "figures"),
    )
    torch.set_num_threads(1)
    set_seed(config.seed)
    bundle = load_dataset(config)
    predictions_df, metrics_df, artifacts = run_experiments(bundle, config)
    summary_df, per_split_df = aggregate_metrics(metrics_df)
    save_artifacts(bundle, predictions_df, metrics_df, summary_df, per_split_df, artifacts, config)


if __name__ == "__main__":
    main()
