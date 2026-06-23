"""
gru_model.py — GRU-based temporal crowd risk classifier.

WHY GRU?
  Crowd congestion doesn't happen instantaneously. It builds over time:
    t0: density rises slightly, speeds drop
    t1: stagnation ratio increases
    t2: flow conflicts emerge
    t3: full congestion — HIGH risk

  A GRU captures these temporal dependencies across a sliding window of
  feature vectors. It learns that rising stagnation BEFORE rising density
  is a precursor to HIGH risk — something a frame-by-frame classifier cannot detect.

  GRU vs LSTM:
    GRUs have fewer parameters (no cell state), train faster, and perform
    comparably on short sequences (5–8 steps). Preferred for real-time use.

ARCHITECTURE:
  Input  → (batch, sequence_length, n_features)
  GRU-1  → 64 units, return_sequences=True
  Dropout → 0.3
  GRU-2  → 64 units, return_sequences=False
  Dropout → 0.3
  Dense  → 32 units, ReLU
  Output → 3 units, Softmax  [SAFE, WARNING, HIGH]
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from utils.config import (
    SEQUENCE_LENGTH, FEATURE_COLUMNS, RISK_CLASSES,
    GRU_UNITS, GRU_LAYERS, DROPOUT_RATE, LEARNING_RATE,
)


def build_gru_model(
    n_features: int = len(FEATURE_COLUMNS),
    sequence_length: int = SEQUENCE_LENGTH,
    gru_units: int = GRU_UNITS,
    n_layers: int = GRU_LAYERS,
    dropout_rate: float = DROPOUT_RATE,
    n_classes: int = len(RISK_CLASSES),
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """
    Build and compile the GRU risk classification model.

    Args:
        n_features:       Number of input features per timestep.
        sequence_length:  Temporal window size.
        gru_units:        Hidden units per GRU layer.
        n_layers:         Number of stacked GRU layers (1 or 2 recommended).
        dropout_rate:     Dropout applied after each GRU layer.
        n_classes:        Number of output risk classes.
        learning_rate:    Adam learning rate.

    Returns:
        Compiled keras.Model ready for training.
    """
    inp = layers.Input(shape=(sequence_length, n_features), name="temporal_input")
    x   = inp

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)   # only last GRU returns single vector
        x = layers.GRU(
            gru_units,
            return_sequences=return_seq,
            dropout=dropout_rate,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"gru_{i+1}",
        )(x)
        if return_seq:
            x = layers.Dropout(dropout_rate, name=f"drop_seq_{i+1}")(x)

    x = layers.Dropout(dropout_rate, name="drop_final")(x)
    x = layers.Dense(32, activation="relu", name="fc_1")(x)
    out = layers.Dense(n_classes, activation="softmax", name="risk_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GRU_CrowdRisk")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model
