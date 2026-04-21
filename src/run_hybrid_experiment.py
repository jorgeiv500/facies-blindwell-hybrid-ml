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
from matplotlib.colors import ListedColormap
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

    print("[Artifacts] rendering workflow figures", flush=True)
    plot_workflow(config.figures_dir / "figure_1_workflow.png")
    plot_window_construction(config.figures_dir / "figure_2_window.png", bundle, config.window_size)
    plot_random_vs_blind(summary_df, config.figures_dir / "figure_3_random_vs_blind.png")
    plot_model_comparison(summary_df, config.figures_dir / "figure_4_model_comparison.png")
    print(f"[Artifacts] rendering depth and uncertainty figures for {representative_well}", flush=True)
    plot_depth_predictions(
        predictions_df,
        representative_well,
        config.figures_dir / "figure_5_depth_predictions.png",
    )
    plot_uncertainty(
        predictions_df,
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


def plot_workflow(output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        "Well logs\n+ QC",
        "Leakage-aware\nsplit by well",
        "Tabular branch\nRF / GB / XGB",
        "Sequence branch\n1D-CNN + BiLSTM + Attention",
        "Probability\nfusion",
        "Uncertainty\n+ interpretability",
        "Facies along\ndepth",
    ]
    x_positions = np.linspace(0.05, 0.95, len(boxes))
    for xpos, label in zip(x_positions, boxes):
        ax.text(
            xpos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#e8f1f8", "edgecolor": "#24557a"},
        )
    for start, end in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(end - 0.05, 0.5), xytext=(start + 0.05, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_window_construction(output_path: Path, bundle: ModelingBundle, window_size: int) -> None:
    well = bundle.df["Well_ID"].iloc[0]
    group = bundle.df[bundle.df["Well_ID"] == well].head(25).copy()
    half = window_size // 2
    center = half + 4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(group["Depth"], group["GR"], color="#24557a", lw=2, label="GR")
    ax.axvspan(group.iloc[center - half]["Depth"], group.iloc[center + half]["Depth"], color="#9fd3c7", alpha=0.35)
    ax.axvline(group.iloc[center]["Depth"], color="#b23a48", ls="--", lw=2, label="Target facies")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Gamma Ray")
    ax.set_title("Depth-centered window for the sequence model")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_random_vs_blind(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[summary_df["Model"].isin(["ProbabilisticEnsemble", "CNN_BiLSTM_Attention", "HybridFusion"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=plot_df, x="Model", y="accuracy", hue="Validation", ax=axes[0], palette="Set2")
    sns.barplot(data=plot_df, x="Model", y="macro_f1", hue="Validation", ax=axes[1], palette="Set2")
    axes[0].set_title("Accuracy by validation protocol")
    axes[1].set_title("Macro-F1 by validation protocol")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    blind_df = summary_df[summary_df["Validation"] == "BlindWell"].copy()
    blind_df = blind_df.set_index("Model")[
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score"]
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(blind_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Blind-well model comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_predictions(predictions_df: pd.DataFrame, well_id: str, output_path: Path) -> None:
    subset = predictions_df[
        (predictions_df["Validation"] == "BlindWell")
        & (predictions_df["SplitLabel"] == f"Blind_{well_id}")
        & (predictions_df["Model"].isin(["ProbabilisticEnsemble", "CNN_BiLSTM_Attention", "HybridFusion"]))
    ].copy()
    facies_palette = ListedColormap(sns.color_palette("tab10", n_colors=8))
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    truth = subset[subset["Model"] == "HybridFusion"].sort_values("Depth")
    axes[0].scatter(truth["Depth"], np.ones(len(truth)), c=truth["Facies"], cmap=facies_palette, s=22)
    axes[0].set_ylabel("True")
    axes[0].set_yticks([])
    for ax_idx, model_name in enumerate(["ProbabilisticEnsemble", "CNN_BiLSTM_Attention", "HybridFusion"], start=1):
        df_model = subset[subset["Model"] == model_name].sort_values("Depth")
        axes[ax_idx].scatter(df_model["Depth"], np.ones(len(df_model)), c=df_model["PredFacies"], cmap=facies_palette, s=22)
        axes[ax_idx].set_ylabel(model_name)
        axes[ax_idx].set_yticks([])
    axes[-1].set_xlabel("Depth")
    fig.suptitle(f"Blind-well facies prediction along depth: {well_id}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(predictions_df: pd.DataFrame, well_id: str, output_path: Path) -> None:
    subset = predictions_df[
        (predictions_df["Validation"] == "BlindWell")
        & (predictions_df["SplitLabel"] == f"Blind_{well_id}")
        & (predictions_df["Model"] == "HybridFusion")
    ].copy().sort_values("Depth")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(subset["Depth"], subset["Confidence"], color="#24557a", lw=1.8)
    axes[0].set_ylabel("Confidence")
    axes[1].plot(subset["Depth"], subset["Entropy"], color="#b23a48", lw=1.8)
    axes[1].set_ylabel("Entropy")
    axes[2].plot(subset["Depth"], subset["Margin"], color="#2d6a4f", lw=1.8)
    axes[2].set_ylabel("Top1-Top2")
    axes[2].set_xlabel("Depth")
    fig.suptitle(f"Hybrid uncertainty along depth: {well_id}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = np.argsort(values)
    axes[0].barh(np.array(feature_names)[order], values[order], color="#24557a")
    axes[0].set_title("Mean |SHAP| for XGBoost")
    axes[0].set_xlabel("Importance")

    axes[1].plot(np.arange(len(averaged_attention)), averaged_attention, color="#b23a48", lw=2)
    axes[1].fill_between(np.arange(len(averaged_attention)), 0, averaged_attention, color="#f4acb7", alpha=0.35)
    axes[1].set_title(f"Average attention window weights: {representative_well}")
    axes[1].set_xlabel("Window position")
    axes[1].set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
