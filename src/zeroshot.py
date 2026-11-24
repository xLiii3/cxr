"""Stage 2: zero-shot scoring with prompt ensembling.

    python -m src.zeroshot
Scores = cos(image, class prompt mean) - cos(image, "a normal chest x-ray").
"""
import argparse
import json

import numpy as np

from . import config as C
from .metrics import summarize

SCALE = 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    d = np.load(C.EMB_NPZ)
    X, y, split = d["X"], d["y"], d["split"]
    text = d["text"].mean(axis=1)  # 14 x 512
    text /= np.linalg.norm(text, axis=1, keepdims=True)

    logits = SCALE * (X @ text.T - (X @ d["normal"])[:, None])
    m = split == args.split
    res = summarize(y[m], logits[m])

    print(f"{'class':22s} {'AUC':>6s} {'AP':>6s}")
    for i, c in enumerate(C.CLASSES):
        print(f"{c:22s} {res['auc'][i]:6.3f} {res['ap'][i]:6.3f}")
    print(f"{'mean':22s} {res['mean_auc']:6.3f} {res['mean_ap']:6.3f}")

    # why subtract the normal anchor: same features, three scoring rules
    variants = {"no_anchor": X @ text.T,
                "anchor": X @ text.T - (X @ d["normal"])[:, None],
                "ratio": (X @ text.T) / ((X @ d["normal"])[:, None] + 1e-6)}
    comp = {}
    for name, s in variants.items():
        r = summarize(y[m], s[m])
        comp[name] = {"mean_auc": r["mean_auc"], "mean_ap": r["mean_ap"]}
        print(f"{name:10s} AUC {r['mean_auc']:.3f}  AP {r['mean_ap']:.3f}")

    C.OUT.mkdir(exist_ok=True)
    (C.OUT / "zeroshot_anchor.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    np.savez(C.OUT / "scores_zeroshot.npz", y=y[m], p=1 / (1 + np.exp(-logits[m])))
    (C.OUT / "zeroshot.json").write_text(json.dumps(
        {"split": args.split, **res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
