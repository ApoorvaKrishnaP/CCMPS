import numpy as np
from crowd_analytics.models import gru_trainer


def test_create_sequences_basic():
    # Build a small feature matrix: 20 timesteps, 3 features
    total = 20
    n_features = 3
    features = np.arange(total * n_features).reshape(total, n_features)
    labels = np.arange(total)

    seq_len = 4
    pred_horizon = 1
    X, y = gru_trainer.create_sequences(features, labels, sequence_length=seq_len, prediction_horizon=pred_horizon)

    expected_n = total - seq_len - pred_horizon + 1
    assert X.shape == (expected_n, seq_len, n_features)
    assert y.shape == (expected_n,)

    # First sequence should match features[0:seq_len]
    np.testing.assert_array_equal(X[0], features[0:seq_len].astype(np.float32))

    # First label should be labels[seq_len] because prediction_horizon=1
    assert int(y[0]) == int(labels[seq_len])


def test_create_sequences_types_and_values():
    # Minimal example where values are floats
    features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    labels = np.array([0, 1, 2, 0, 1])

    X, y = gru_trainer.create_sequences(features, labels, sequence_length=3, prediction_horizon=1)

    # Expect 5 - 3 -1 +1 = 2 sequences
    assert X.shape == (2, 3, 2)
    assert y.dtype == np.int32
    assert X.dtype == np.float32
