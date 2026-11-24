"""Hand-written metrics must match sklearn to 1e-6."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.metrics import (average_precision, ece, recall_at_k, roc_auc,
                         sens_spec, youden_threshold)


def _data(seed=0, n=500):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(float)
    s = y * rng.normal(0.6, 0.3, n) + (1 - y) * rng.normal(0.2, 0.3, n)
    return y, s


def test_auc_matches_sklearn():
    y, s = _data()
    assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-6
    for seed in range(3):
        y, s = _data(seed, 300)
        assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-6


def test_auc_handles_ties():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.7, 0.9, 0.1])
    assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-6


def test_ap_matches_sklearn():
    y, s = _data(1)
    assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-6


def test_ap_is_not_auc():
    y, s = _data(2)
    assert abs(average_precision(y, s) - roc_auc(y, s)) > 0.01


def test_recall_at_k():
    rel = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    assert recall_at_k(rel, 1) == 1 / 3
    assert recall_at_k(rel, 3) == 2 / 3


def test_threshold_and_ece():
    y, s = _data(3)
    p = 1 / (1 + np.exp(-(s - 0.4) * 4))
    thr = youden_threshold(y, p)
    se, sp = sens_spec(y, p, thr)
    assert 0.0 <= se <= 1.0 and 0.0 <= sp <= 1.0
    assert 0.0 <= ece(y, p) <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
