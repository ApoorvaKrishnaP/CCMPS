"""
train_gru.py — GRU training orchestrator.

Run this script after generating (or recording) your feature CSV:

    python train_gru.py --csv data/crowd_features.csv --epochs 80

What this script does:
  1. Load & validate the feature CSV.
  2. Normalise features with MinMaxScaler (fit on train, transform both splits).
  3. Build sliding-window sequences of length SEQUENCE_LENGTH.
  4. Train/validate split (80/20 chronological — no shuffle to respect temporal order).
  5. Balance classes with sample weights.
  6. Train GRU with early stopping & model checkpointing.
  7. Export model, scaler, encoder + diagnostic plots.
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from models.gru_model import build_gru_model
from utils.config import (
    FEATURE_COLUMNS, LABEL_COLUMN, RISK_CLASSES,
    SEQUENCE_LENGTH, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    TRAINED_MODEL, SCALER_PATH, ENCODER_PATH, OUTPUT_DIR,
    DATASET_CSV,
)


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequence generation.

    For each timestep t (starting at seq_len), the input is the window
    [t-seq_len : t] and the label is y[t] (next-timestep prediction).

    This is the ONLY correct way to train a temporal sequence model.
    Using individual rows would discard all temporal context.
    """
    Xs, ys = [], []
    for t in range(seq_len, len(X)):
        Xs.append(X[t - seq_len : t])
        ys.append(y[t])
    return np.array(Xs), np.array(ys)


# ── Training function ─────────────────────────────────────────────────────────

def train(csv_path: Path, epochs: int = MAX_EPOCHS) -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"[Train] Loaded {len(df)} rows from {csv_path}")
    print(f"[Train] Class distribution:\n{df[LABEL_COLUMN].value_counts()}")

    # Validate required columns
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    le.classes_ = np.array(RISK_CLASSES)   # fix ordering: SAFE=0, WARNING=1, HIGH=2
    y_raw = le.transform(df[LABEL_COLUMN].values)
    

    # ── 3. Train / val split (chronological — DO NOT shuffle) ─────────────────
    split = int(len(df) * 0.80)
    X_tr_raw = df[FEATURE_COLUMNS].values[:split]
    X_va_raw = df[FEATURE_COLUMNS].values[split:]
    y_tr_raw = y_raw[:split]
    y_va_raw = y_raw[split:]

    # ── 4. Normalise (fit ONLY on train split) ────────────────────────────────
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)

    # ── 5. Build sequences ────────────────────────────────────────────────────
    X_tr_seq, y_tr_seq = build_sequences(X_tr, y_tr_raw, SEQUENCE_LENGTH)
    X_va_seq, y_va_seq = build_sequences(X_va, y_va_raw, SEQUENCE_LENGTH)
    print(f"[Train] Train sequences: {X_tr_seq.shape}  Val: {X_va_seq.shape}")

    # ── 6. Class weights for imbalanced training set ──────────────────────────
    classes = np.unique(y_tr_seq)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr_seq)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"[Train] Class weights: {class_weight_dict}")

    # ── 7. Build and train model ──────────────────────────────────────────────
    model = build_gru_model(
        n_features=len(FEATURE_COLUMNS),
        sequence_length=SEQUENCE_LENGTH,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(TRAINED_MODEL),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
    ]

    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_va_seq, y_va_seq),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    # ── 8. Save artefacts ─────────────────────────────────────────────────────
    model.save(str(TRAINED_MODEL))
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"[Train] Model  → {TRAINED_MODEL}")
    print(f"[Train] Scaler → {SCALER_PATH}")
    print(f"[Train] Encoder→ {ENCODER_PATH}")

    # ── 9. Evaluation plots ───────────────────────────────────────────────────
    _plot_learning_curves(history, OUTPUT_DIR / "learning_curves.png")
    y_pred = np.argmax(model.predict(X_va_seq), axis=1)
    _plot_confusion_matrix(y_va_seq, y_pred, le.classes_, OUTPUT_DIR / "confusion_matrix.png")
    _plot_prediction_probabilities(model, X_va_seq, y_va_seq, le, OUTPUT_DIR / "prob_distribution.png")

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_learning_curves(history, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Categorical Cross-Entropy")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Saved learning curves → {path}")


def _plot_confusion_matrix(y_true, y_pred, classes, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Saved confusion matrix → {path}")


def _plot_prediction_probabilities(model, X_val, y_val, le, path: Path) -> None:
    probs = model.predict(X_val)
    fig, axes = plt.subplots(1, len(le.classes_), figsize=(14, 4))
    for ci, cls in enumerate(le.classes_):
        mask_true = y_val == ci
        axes[ci].hist(probs[:, ci][mask_true],  bins=20, alpha=0.7, label="Correct")
        axes[ci].hist(probs[:, ci][~mask_true], bins=20, alpha=0.7, label="Incorrect")
        axes[ci].set(title=f"P({cls})", xlabel="Probability", ylabel="Count")
        axes[ci].legend(fontsize=8)
    plt.suptitle("Prediction Probability Distributions by Class")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Saved prob distributions → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU Crowd Risk Model")
    parser.add_argument("--csv",    type=str, default=str(DATASET_CSV))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    train(Path(args.csv), epochs=args.epochs)
