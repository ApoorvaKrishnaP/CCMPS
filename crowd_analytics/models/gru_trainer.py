"""
models/gru_trainer.py
=====================
GRU-based temporal crowd risk prediction model.

Why GRU (not LSTM or Transformer)?
───────────────────────────────────
GRU (Gated Recurrent Unit) is ideal for this task because:

1. TEMPORAL PATTERN LEARNING:
   Congestion doesn't appear instantly — it builds over 10-60 seconds.
   GRU's recurrent state captures this temporal evolution that a
   feedforward network cannot.

2. EFFICIENCY:
   GRU has fewer parameters than LSTM (no separate cell state) while
   achieving similar accuracy on short sequences (5-8 timesteps).
   This enables real-time inference at surveillance framerate.

3. SEQUENCE SENSITIVITY:
   A GRU distinguishes between:
   - [SAFE → SAFE → SAFE → SAFE → WARNING]  →  escalating (predict HIGH)
   - [HIGH → WARNING → SAFE → SAFE → SAFE]  →  recovering (predict SAFE)
   A single-frame classifier would give identical outputs for both.

4. CONGESTION BUILDUP LEARNING:
   The GRU learns that the COMBINATION of rising density +
   falling speed + rising stagnation = impending HIGH risk.
   This causal temporal pattern is exactly what recurrence captures.

Sequence Formation:
───────────────────
Input:  [t-4, t-3, t-2, t-1, t]  →  5 consecutive 1-second feature vectors
Output: Risk class at t+1          →  SAFE / WARNING / HIGH

This design gives the GRU 5 seconds of historical context to predict
what will happen in the next second — actionable for security staff.
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import Counter

logger = logging.getLogger(__name__)

# GRU model feature names (must match analytics output)
FEATURE_NAMES = [
    "people_count", "density", "avg_speed_mps", "stagnation_ratio",
    "avg_dwell_time_sec", "flow_conflict_ratio", "directional_entropy",
    "acceleration_variance", "zone_transitions", "inflow", "outflow",
]

RISK_CLASSES = ["SAFE", "WARNING", "HIGH"]
N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = len(RISK_CLASSES)


# ─── Data Processing ─────────────────────────────────────────────────────────

def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 6,
    prediction_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding window sequences for GRU training.

    Example with sequence_length=6, prediction_horizon=1:
        Input:  features[t:t+6]     → shape (6, n_features)
        Target: labels[t+6]         → risk class at next timestep

    This is a LOOK-AHEAD design: given 6 seconds of context,
    predict the risk in the NEXT second.

    Args:
        features: Shape (T, n_features) — time series feature matrix.
        labels:   Shape (T,)            — integer risk class labels.
        sequence_length: Context window (number of past timesteps).
        prediction_horizon: How far ahead to predict (1 = next second).

    Returns:
        X: Shape (N, sequence_length, n_features)
        y: Shape (N,) — integer labels
    """
    X, y = [], []
    total_len = len(features)

    #Did not understand this logic!
    for i in range(total_len - sequence_length - prediction_horizon + 1):
        seq = features[i : i + sequence_length]
        label = labels[i + sequence_length + prediction_horizon - 1]
        X.append(seq)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)#Will throw value error if data cant be converted to numbers!


def prepare_data(
    csv_path: str,
    sequence_length: int = 6,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Dict:
    """
    Full data preparation pipeline:
    1. Load CSV
    2. Scale features
    3. Encode labels
    4. Create sequences
    5. Split into train/val/test

    Args:
        csv_path: Path to crowd analytics CSV.
        sequence_length: GRU context window size.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        random_seed: For reproducibility.

    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test,
        scaler, label_encoder, class_names.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Risk label distribution:\n{df['risk_label'].value_counts()}")

    # ── Extract features and labels ──────────────────────────────────────────
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    features_raw = df[FEATURE_NAMES].values.astype(np.float32)
    labels_raw = df["risk_label"].values

    # ── Feature normalization (StandardScaler) ───────────────────────────────
    # StandardScaler is preferred over MinMaxScaler for GRU because:
    # - Gaussian-like distribution after scaling
    # - Less sensitive to outliers (rare HIGH events)
    # - Stable gradients during training
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)

    # ── Label encoding ───────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(RISK_CLASSES)  # Fixed order: HIGH=0, SAFE=1, WARNING=2
    labels_encoded = le.transform(labels_raw)

    # ── Sequence generation ──────────────────────────────────────────────────
    X, y = create_sequences(features_scaled, labels_encoded, sequence_length)
    logger.info(f"Sequences: X.shape={X.shape}, y.shape={y.shape}")

    # ── Class distribution ───────────────────────────────────────────────────
    class_dist = Counter(y)
    logger.info(f"Sequence class distribution: {dict(sorted(class_dist.items()))}")

    # ── Train/Val/Test split ─────────────────────────────────────────────────
    # Use chronological split (not random) to preserve temporal order
    total = len(X)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - val_size - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # ── Class weights for imbalanced training ────────────────────────────────
    # HIGH events are rare — upweight them so GRU doesn't ignore them
    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    class_weights = {
        i: total / (N_CLASSES * max(count, 1))
        for i, count in enumerate(class_counts)
    }
    logger.info(f"Class weights: {class_weights}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "label_encoder": le,
        "class_names": le.classes_.tolist(),
        "class_weights": class_weights,
        "n_features": N_FEATURES,
        "sequence_length": sequence_length,
        "n_classes": N_CLASSES,
    }


# ─── GRU Model Architecture ──────────────────────────────────────────────────

def build_gru_model(
    sequence_length: int = 6,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    gru_units_1: int = 64,
    gru_units_2: int = 32,
    dropout_rate: float = 0.30,
    l2_reg: float = 1e-4,
) -> "tf.keras.Model":
    """
    Build GRU model for temporal crowd risk classification.

    Architecture:
        Input(sequence_length, n_features)
        → GRU(64, return_sequences=True)  [capture temporal patterns]
        → Dropout(0.3)                    [prevent overfitting]
        → GRU(32, return_sequences=False) [summarize sequence]
        → Dropout(0.3)
        → Dense(32, relu)                 [non-linear feature combination]
        → Dense(3, softmax)               [probability over SAFE/WARNING/HIGH]

    Two-layer GRU rationale:
    - First GRU: learns fine-grained temporal transitions (second-by-second)
    - Second GRU: integrates these into a longer-range contextual summary
    - Together: captures both immediate and accumulative congestion signals

    Args:
        sequence_length: Input sequence length (timesteps).
        n_features: Number of features per timestep.
        n_classes: Output classes (3: SAFE, WARNING, HIGH).
        gru_units_1: Units in first GRU layer.
        gru_units_2: Units in second GRU layer.
        dropout_rate: Dropout probability (0.3 standard for time series).
        l2_reg: L2 regularization strength.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    model = tf.keras.Sequential([
        layers.Input(shape=(sequence_length, n_features), name="temporal_input"),

        # First GRU: Temporal pattern extraction
        layers.GRU(
            gru_units_1,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg / 10),
            name="gru_temporal_extraction",
        ),
        layers.Dropout(dropout_rate, name="dropout_1"),
        layers.BatchNormalization(name="bn_1"),

        # Second GRU: Sequence summarization
        layers.GRU(
            gru_units_2,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            name="gru_sequence_summary",
        ),
        layers.Dropout(dropout_rate, name="dropout_2"),
        layers.BatchNormalization(name="bn_2"),

        # Dense classification head
        layers.Dense(32, activation="relu", name="dense_classification"),
        layers.Dropout(dropout_rate * 0.5, name="dropout_3"),

        # Output: softmax over SAFE / WARNING / HIGH
        layers.Dense(n_classes, activation="softmax", name="risk_output"),
    ], name="GRU_CrowdRisk_FOOD_COURT_A")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─── Training Pipeline ───────────────────────────────────────────────────────

def train_gru_model(
    data: Dict,
    output_dir: str = "outputs/models",
    epochs: int = 80,
    batch_size: int = 64,
    patience: int = 15,
) -> Tuple["tf.keras.Model", Dict]:
    """
    Full GRU training pipeline with monitoring and checkpointing.

    Args:
        data: Output of prepare_data().
        output_dir: Directory to save model artifacts.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        patience: EarlyStopping patience (epochs without val improvement).

    Returns:
        Tuple of (trained model, training history dict).
    """
    import tensorflow as tf

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_gru_model(
        sequence_length=data["sequence_length"],
        n_features=data["n_features"],
        n_classes=data["n_classes"],
    )
    model.summary(print_fn=logger.info)

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        # Stop training when validation accuracy plateaus
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(output_dir) / "gru_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce LR on plateau for fine-grained convergence
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logging
        tf.keras.callbacks.CSVLogger(
            str(Path(output_dir) / "training_log.csv"),
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        f"Training GRU — "
        f"X_train={data['X_train'].shape}, "
        f"epochs={epochs}, batch_size={batch_size}"
    )

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=data["class_weights"],
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on test set ─────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(
        data["X_test"], data["y_test"], verbose=0
    )
    logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    # Save scaler
    with open(Path(output_dir) / "scaler.pkl", "wb") as f:
        pickle.dump(data["scaler"], f)

    # Save label encoder
    with open(Path(output_dir) / "label_encoder.pkl", "wb") as f:
        pickle.dump(data["label_encoder"], f)

    # Save model config
    config = {
        "sequence_length": data["sequence_length"],
        "n_features": data["n_features"],
        "n_classes": data["n_classes"],
        "class_names": data["class_names"],
        "feature_names": FEATURE_NAMES,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
    }
    with open(Path(output_dir) / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model artifacts saved to: {output_dir}")

    return model, history.history


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(
    model: "tf.keras.Model",
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    output_dir: str = "outputs/models",
) -> Dict:
    """
    Generate evaluation metrics and visualizations.

    Produces:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix heatmap
    - Prediction probability distribution
    - Training history plots (loss + accuracy curves)

    Args:
        model: Trained Keras model.
        X_test: Test sequences.
        y_test: True labels.
        class_names: Class label names.
        output_dir: Directory to save plots.

    Returns:
        Dictionary of evaluation metrics.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Predictions
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    # Classification report
    report = classification_report(
        y_test, preds,
        target_names=class_names,
        output_dict=True,
    )
    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_title("GRU Crowd Risk — Confusion Matrix\nFOOD_COURT_A", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "confusion_matrix.png"), dpi=150)
    plt.close()

    # Probability distribution per class
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#27ae60", "#e67e22", "#e74c3c"]
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        mask = y_test == i
        if mask.sum() > 0:
            axes[i].hist(
                probs[mask, i], bins=20, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.5
            )
            axes[i].set_title(f"P({cls}) | True={cls}", fontsize=11)
            axes[i].set_xlabel("Predicted Probability")
            axes[i].set_ylabel("Count")
            axes[i].set_xlim(0, 1)
    plt.suptitle("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "probability_distribution.png"), dpi=150)
    plt.close()

    logger.info(f"Evaluation plots saved to: {output_dir}")

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": report["accuracy"],
    }


def plot_training_history(history: Dict, output_dir: str = "outputs/models"):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary from Keras history.history.
        output_dir: Directory to save plots.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    axes[0].plot(history.get("loss", []), label="Train Loss", color="#3498db", linewidth=2)
    axes[0].plot(history.get("val_loss", []), label="Val Loss", color="#e74c3c", linewidth=2, linestyle="--")
    axes[0].set_title("Training Loss — GRU CrowdRisk", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history.get("accuracy", []), label="Train Acc", color="#27ae60", linewidth=2)
    axes[1].plot(history.get("val_accuracy", []), label="Val Acc", color="#e67e22", linewidth=2, linestyle="--")
    axes[1].set_title("Training Accuracy — GRU CrowdRisk", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(str(Path(output_dir) / "training_history.png"), dpi=150)
    plt.close()
    logger.info(f"Training history plot saved to: {output_dir}")
