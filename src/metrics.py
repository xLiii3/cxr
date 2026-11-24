"""NumPy implementations of the metrics used across the project.

No sklearn import: each function is checked against sklearn in tests/.
"""
import numpy as np


def _check(y, s):
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError("y and score must have the same shape")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary")
    return y, s


def roc_auc(y_true, y_score):
    """Area under ROC via the Mann-Whitney U statistic (ties averaged)."""
    y, s = _check(y_true, y_score)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y_true, y_score):
    """Average precision without interpolation, matching sklearn."""
    y, s = _check(y_true, y_score)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def recall_at_k(relevant, k):
    """Mean fraction of queries whose top-k contains a relevant item.

    relevant: bool [n_query, n_gallery]; column 0 is assumed to be the query
    itself and is skipped when self_match=True.
    """
    relevant = np.asarray(relevant, dtype=bool)
    hit = relevant[:, :k].any(axis=1)
    return float(hit.mean())


def precision_at_k(relevant, k):
    relevant = np.asarray(relevant, dtype=bool)
    return float(relevant[:, :k].sum() / (relevant.shape[0] * k))


def youden_threshold(y_true, y_score):
    """Threshold maximising sensitivity + specificity - 1."""
    y, s = _check(y_true, y_score)
    order = np.argsort(-s)
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    sens = tp / max(y.sum(), 1)
    spec = 1 - fp / max((1 - y).sum(), 1)
    return float(s[int(np.argmax(sens + spec))])


def sens_spec(y_true, y_score, thr):
    y, s = _check(y_true, y_score)
    pred = s >= thr
    tp = float((pred & (y == 1)).sum())
    tn = float((~pred & (y == 0)).sum())
    return tp / max(y.sum(), 1), tn / max((1 - y).sum(), 1)


def ece(y_true, y_prob, bins=15):
    """Expected calibration error over equal-width probability bins."""
    y, p = _check(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(total)


def summarize(y_true, y_score):
    """Per-class AUC / AP / sensitivity / specificity at the Youden point."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    aucs, aps, sens, spec = [], [], [], []
    for c in range(y_true.shape[1]):
        thr = youden_threshold(y_true[:, c], y_score[:, c])
        se, sp = sens_spec(y_true[:, c], y_score[:, c], thr)
        aucs.append(roc_auc(y_true[:, c], y_score[:, c]))
        aps.append(average_precision(y_true[:, c], y_score[:, c]))
        sens.append(se)
        spec.append(sp)
    return {
        "auc": aucs, "ap": aps, "sens": sens, "spec": spec,
        "mean_auc": float(np.nanmean(aucs)), "mean_ap": float(np.nanmean(aps)),
    }
