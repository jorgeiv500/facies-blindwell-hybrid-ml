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
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
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

    # PEF is entirely missing in one well, so it is excluded from the main workflow.
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
    out = meta[["Well_ID", "Formation", "Depth", "RelDepth", "Facies"]].copy()
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

    y_sub = bundle.y_encoded[subtrain_idx]
    y_val = bundle.y_encoded[val_idx]
    y_train = bundle.y_encoded[train_idx]
    y_test = bundle.y_encoded[test_idx]

    print(f"[{validation_name}] {split_label}: training tabular models", flush=True)
    tabular_models = train_tabular_models(x_tab_sub, y_sub, seed)
    val_tab_probs = {name: model.predict_proba(x_tab_val) for name, model in tabular_models.items()}
    tab_weights = choose_tabular_ensemble_weights(val_tab_probs, y_val)

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
        apply_weighted_fusion(val_tab_probs, tab_weights),
        dl_extra_outputs["CNN_BiLSTM_Attention"]["ValProbs"],
        y_val,
    )
    hybrid_test_probs = normalize_probs(
        hybrid_alpha * ensemble_test_probs + (1.0 - hybrid_alpha) * dl_test_probs["CNN_BiLSTM_Attention"]
    )

    meta_test = bundle.core_df.iloc[test_idx].reset_index(drop=True)
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
    ]
    prediction_df = pd.concat(predictions, ignore_index=True)

    metrics_rows = []
    for model_name, model_pred_df in prediction_df.groupby("Model", sort=False):
        probs_cols = [col for col in model_pred_df.columns if col.startswith("Prob_")]
        probs = model_pred_df[probs_cols].to_numpy()
        y_true = model_pred_df["TrueEncoded"].to_numpy()
        row = summarize_metrics(y_true, probs)
        row.update(
            {
                "Validation": validation_name,
                "SplitLabel": split_label,
                "Model": model_name,
                "n_samples": len(model_pred_df),
            }
        )
        metrics_rows.append(row)

    artifacts = {
        "tab_weights": tab_weights,
        "hybrid_alpha": hybrid_alpha,
        "attention_weights": dl_extra_outputs["CNN_BiLSTM_Attention"]["AttentionWeights"],
        "representative_meta": meta_test,
        "xgb_model": tabular_models["XGBoost"],
        "x_tab_train": x_tab_train,
        "feature_names": bundle.feature_cols,
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
    ]
    summary = (
        metrics_df.groupby(["Validation", "Model"], as_index=False)[metric_cols]
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
    plot_window_construction(config.figures_dir / "figure_2_window.png", bundle, config.window_size)
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
    plot_petrophysical_crossplots(bundle.df, facies_labels, config.figures_dir / "figure_8_crossplots.png")
    plot_confusion_matrices(predictions_df, facies_labels, config.figures_dir / "figure_9_confusion_matrices.png")
    plot_facies_context(bundle.df, facies_labels, config.figures_dir / "figure_10_dataset_context.png")


def plot_workflow(output_path: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")

    boxes = [
        (0.10, 0.50, 0.18, 0.18, "Well logs\nQC + harmonization", "#DCEAF4"),
        (0.30, 0.50, 0.18, 0.18, "Leakage-aware\nblind-well split", "#E9F5DB"),
        (0.52, 0.68, 0.19, 0.18, "Tabular branch\nRF / GB / XGB", "#D9EAF7"),
        (0.52, 0.32, 0.23, 0.18, "Sequence branch\n1D-CNN + BiLSTM + Attention", "#FDE2C5"),
        (0.77, 0.50, 0.17, 0.18, "Probability\nfusion", "#EADFF8"),
        (0.92, 0.50, 0.17, 0.18, "Blind-well facies\nuncertainty + interpretation", "#F7D9D9"),
    ]

    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#37515f",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color="#22313f")

    arrows = [
        ((0.19, 0.50), (0.22, 0.50)),
        ((0.39, 0.56), (0.43, 0.65)),
        ((0.39, 0.44), (0.41, 0.35)),
        ((0.63, 0.62), (0.69, 0.54)),
        ((0.64, 0.38), (0.69, 0.46)),
        ((0.85, 0.50), (0.87, 0.50)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#445d6e"})

    ax.set_title("Hybrid reliable facies-classification workflow", fontsize=15, pad=10)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_window_construction(output_path: Path, bundle: ModelingBundle, window_size: int) -> None:
    set_publication_style()
    well = bundle.df["Well_ID"].iloc[0]
    group = bundle.df[bundle.df["Well_ID"] == well].head(30).copy()
    half = window_size // 2
    center = half + 4
    left_depth = group.iloc[center - half]["Depth"]
    right_depth = group.iloc[center + half]["Depth"]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True, gridspec_kw={"hspace": 0.08})
    tracks = [
        ("GR", "#2A6F97", "Gamma ray"),
        ("RT", "#9C6644", "True resistivity"),
    ]
    for ax, (column, color, title) in zip(axes, tracks):
        ax.plot(group["Depth"], group[column], color=color, lw=2)
        ax.axvspan(left_depth, right_depth, color="#F4D58D", alpha=0.35)
        ax.axvline(group.iloc[center]["Depth"], color="#C1121F", ls="--", lw=1.8)
        ax.set_ylabel(title)
        ax.text(group.iloc[center]["Depth"], ax.get_ylim()[1], " target ", color="#C1121F", ha="left", va="top")
    axes[1].set_xlabel("Depth")
    fig.suptitle(f"Depth-centered window used by the sequence model ({well})", y=0.99)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_random_vs_blind(summary_df: pd.DataFrame, output_path: Path) -> None:
    set_publication_style()
    model_order = ["RandomForest", "ProbabilisticEnsemble", "CNN_BiLSTM_Attention", "HybridFusion"]
    model_labels = {
        "RandomForest": "RF",
        "ProbabilisticEnsemble": "Ensemble",
        "CNN_BiLSTM_Attention": "CNN-BiLSTM-Attn",
        "HybridFusion": "Hybrid",
    }
    plot_df = summary_df[summary_df["Model"].isin(model_order)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)

    for ax, (metric, title) in zip(axes, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
        pivot = plot_df.pivot(index="Model", columns="Validation", values=metric).loc[model_order]
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
    blind_df = blind_df.set_index("Model").loc[
        [
            "RandomForest",
            "ProbabilisticEnsemble",
            "XGBoost",
            "GradientBoosting",
            "HybridFusion",
            "CNN_BiLSTM_Attention",
            "CNN_BiLSTM",
            "CNN",
        ]
    ]
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
        & (predictions_df["Model"].isin(["ProbabilisticEnsemble", "CNN_BiLSTM_Attention", "HybridFusion"]))
    ].copy()
    logs = bundle.core_df[bundle.core_df["Well_ID"] == well_id].sort_values("Depth").copy()
    truth = subset[subset["Model"] == "HybridFusion"].sort_values("Depth")
    facies_labels, facies_colors, cmap, norm, index_map = get_facies_style(facies_labels)

    fig, axes = plt.subplots(
        1,
        7,
        figsize=(15.5, 9.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.15, 1.35, 0.36, 0.36, 0.36, 0.36], "wspace": 0.08},
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
        "Ensemble",
    )
    draw_facies_strip(
        axes[5],
        truth["Depth"].to_numpy(),
        subset[subset["Model"] == "CNN_BiLSTM_Attention"].sort_values("Depth")["PredFacies"],
        index_map,
        cmap,
        norm,
        "DL",
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

    for ax in axes[:3]:
        ax.set_ylim(depth.max(), depth.min())
    fig.suptitle(f"Blind-well multi-track panel: {well_id}", y=0.99)
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
        & (predictions_df["Model"] == "HybridFusion")
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
    draw_facies_strip(axes[1], depth, subset["PredFacies"], index_map, cmap, norm, "Hybrid")

    heatmap = axes[2].imshow(
        prob_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=[-0.5, len(facies_labels) - 0.5, float(depth.max()), float(depth.min())],
    )
    axes[2].set_title("Hybrid class probabilities")
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
    fig.suptitle(f"Hybrid probability and uncertainty diagnostics: {well_id}", y=0.99)
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


def plot_petrophysical_crossplots(raw_df: pd.DataFrame, facies_labels: list[int], output_path: Path) -> None:
    set_publication_style()
    sample_df = raw_df.sample(n=min(3500, len(raw_df)), random_state=SEED).copy()
    facies_labels, facies_colors, _, _, _ = get_facies_style(facies_labels)
    color_map = dict(zip(facies_labels, facies_colors))

    panels = [
        ("GR", "PHID", "GR vs PHID", "linear"),
        ("RHOB", "PHIE", "RHOB vs PHIE", "linear"),
        ("RT", "PHIE", "RT vs PHIE", "log"),
        ("Vshl", "SwA", "Vshl vs SwA", "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
    for ax, (x_col, y_col, title, scale) in zip(axes.ravel(), panels):
        panel_df = sample_df[[x_col, y_col, "Facies"]].dropna().copy()
        if scale == "log":
            panel_df = panel_df[panel_df[x_col] > 0]
        for facies in facies_labels:
            subset = panel_df[panel_df["Facies"] == facies]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                s=16,
                alpha=0.40,
                color=color_map[facies],
                edgecolor="none",
            )
        centroids = panel_df.groupby("Facies")[[x_col, y_col]].median().reindex(facies_labels)
        ax.scatter(
            centroids[x_col],
            centroids[y_col],
            s=90,
            c=[color_map[label] for label in facies_labels],
            edgecolor="black",
            linewidth=0.7,
            marker="X",
            zorder=4,
        )
        if scale == "log":
            ax.set_xscale("log")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)

    legend_handles = [
        Patch(facecolor=color_map[label], edgecolor="none", label=f"Facies {label}")
        for label in facies_labels
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle("Petrophysical crossplots from the anonymized reservoir dataset", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    save_figure(fig, output_path)


def plot_confusion_matrices(predictions_df: pd.DataFrame, facies_labels: list[int], output_path: Path) -> None:
    set_publication_style()
    blind_df = predictions_df[predictions_df["Validation"] == "BlindWell"].copy()
    model_order = ["RandomForest", "ProbabilisticEnsemble", "CNN_BiLSTM_Attention", "HybridFusion"]
    model_labels = {
        "RandomForest": "RF",
        "ProbabilisticEnsemble": "Ensemble",
        "CNN_BiLSTM_Attention": "CNN-BiLSTM-Attn",
        "HybridFusion": "Hybrid",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    for idx, (ax, model_name) in enumerate(zip(axes.ravel(), model_order)):
        subset = blind_df[blind_df["Model"] == model_name]
        cm = confusion_matrix(
            subset["Facies"],
            subset["PredFacies"],
            labels=facies_labels,
            normalize="true",
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            xticklabels=facies_labels,
            yticklabels=facies_labels,
            cbar=idx == 0,
            cbar_ax=cbar_ax if idx == 0 else None,
        )
        ax.set_title(model_labels[model_name])
        ax.set_xlabel("Predicted facies")
        ax.set_ylabel("True facies")
    fig.suptitle("Blind-well confusion matrices (row-normalized)", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    save_figure(fig, output_path)


def plot_facies_context(raw_df: pd.DataFrame, facies_labels: list[int], output_path: Path) -> None:
    set_publication_style()
    facies_labels, facies_colors, cmap, norm, index_map = get_facies_style(facies_labels)
    wells = sorted(raw_df["Well_ID"].unique())
    formation_dist = (
        pd.crosstab(raw_df["Formation"], raw_df["Facies"], normalize="index")
        .reindex(columns=facies_labels, fill_value=0.0)
        .sort_index()
    )

    fig = plt.figure(figsize=(14.5, 6.0))
    grid = fig.add_gridspec(1, len(wells) + 1, width_ratios=[0.32] * len(wells) + [1.8], wspace=0.10)
    strip_axes = [fig.add_subplot(grid[0, idx]) for idx in range(len(wells))]
    heatmap_ax = fig.add_subplot(grid[0, -1])

    for idx, (ax, well_id) in enumerate(zip(strip_axes, wells)):
        well_df = raw_df[raw_df["Well_ID"] == well_id].sort_values("RelDepth")
        draw_facies_strip(
            ax,
            well_df["RelDepth"].to_numpy(),
            well_df["Facies"],
            index_map,
            cmap,
            norm,
            well_id.replace("_", "\n"),
            show_ylabel=idx == 0,
        )
        if idx == 0:
            ax.set_ylabel("Relative depth")

    sns.heatmap(
        formation_dist,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(0.5, float(formation_dist.max().max())),
        ax=heatmap_ax,
        cbar_kws={"shrink": 0.75, "label": "Within-formation fraction"},
    )
    heatmap_ax.set_title("Facies composition by formation")
    heatmap_ax.set_xlabel("Facies")
    heatmap_ax.set_ylabel("Formation")
    add_facies_legend(fig, facies_labels, facies_colors, anchor_y=-0.01)
    fig.suptitle("Dataset context: relative-depth facies architecture and stratigraphic composition", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the blind-well hybrid facies experiment.")
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
