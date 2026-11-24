"""Stage 5: comparison table, figures and the leakage ablation.

    python -m src.evaluate
"""
import json

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from . import config as C
from .data import read_split
from .metrics import summarize
from .model import embed_texts, inject_lora, load_clip, load_lora
from .train import fit_probe

plt.switch_backend("Agg")

STAGES = [("zeroshot", "Zero-shot"), ("linear", "Linear probe"), ("lora", "LoRA")]
ASSETS = C.ROOT / "assets"


def load_json(name, default=None):
    p = C.OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def table():
    rows = []
    for key, label in STAGES:
        d = load_json(f"{key}.json")
        if not d:
            continue
        ck = C.OUT / f"{key}.pt"
        n = 0 if key == "zeroshot" else sum(
            v.numel() for v in torch.load(ck, weights_only=False)["state"].values())
        rows.append((label, d["mean_auc"], d["mean_ap"], n))
    lines = ["| 方案 | 可训练参数 | mean AUC | mean AP |", "| --- | --- | --- | --- |"]
    for label, auc, ap, n in rows:
        lines.append(f"| {label} | {n:,} | {auc:.4f} | {ap:.4f} |")
    md = "\n".join(lines)
    (C.OUT / "comparison.md").write_text(md, encoding="utf-8")
    (C.OUT / "comparison.csv").write_text(
        "stage,params,mean_auc,mean_ap\n" +
        "\n".join(f"{k},{n},{a:.4f},{p:.4f}" for k, a, p, n in rows), encoding="utf-8")
    print(md)
    return rows


def fig_auc():
    data = {k: load_json(f"{k}.json") for k, _ in STAGES}
    data = {k: v for k, v in data.items() if v}
    labels = [c.replace("_", " ") for c in C.CLASSES] + ["MEAN"]
    x = np.arange(len(labels))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=160)
    for i, (key, pretty) in enumerate([s for s in STAGES if s[0] in data]):
        auc = list(data[key]["auc"]) + [data[key]["mean_auc"]]
        ax.bar(x + (i - (len(data) - 1) / 2) * w, auc, w, label=pretty)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Per-class AUC on the patient-level test split")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "auc_bars.png")
    plt.close(fig)


def fig_curves():
    hist = {k: load_json(f"history_{k}.json") for k in ("linear", "lora")}
    hist = {k: v for k, v in hist.items() if v}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    for key, label in (("linear", "Linear probe"), ("lora", "LoRA")):
        if key in hist:
            ep = [h["epoch"] for h in hist[key]]
            axes[0].plot(ep, [h["loss"] for h in hist[key]], marker="o", ms=3, label=label)
            axes[1].plot(ep, [h["val_ap"] for h in hist[key]], marker="o", ms=3, label=label)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation mAP")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend(frameon=False, fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png")
    plt.close(fig)


def fig_retrieval(device="cuda"):
    gallery = read_split()
    gallery = gallery[gallery.split == "test"].reset_index(drop=True)
    emb = np.load(C.OUT / "gallery_emb.npy")
    idx = faiss.read_index(str(C.OUT / "index_flat.faiss"))
    model, _, tokenizer = load_clip(device)
    inject_lora(model)
    load_lora(model, C.OUT / "lora.pt")
    model.eval()

    def panel(r, items, label, offset=0):
        for j, im in enumerate(items):
            ax = axes[r][j + offset]
            ax.imshow(im, cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_title(label, fontsize=7)

    fig, axes = plt.subplots(4, 6, figsize=(11, 7.6), dpi=160)
    for r in range(2):
        qi = int(np.random.default_rng(r).integers(len(emb)))
        _, I = idx.search(emb[qi:qi + 1], 6)
        imgs = [Image.open(C.IMAGES / gallery.image[qi]).convert("L")]
        imgs += [Image.open(C.IMAGES / gallery.image[j]).convert("L") for j in I[0][1:]]
        panel(r, imgs, "image query")

    for r, text in enumerate(["cardiomegaly", "pleural effusion"], start=2):
        t = embed_texts(model, tokenizer, [text], device)[0].numpy()
        t /= np.linalg.norm(t)
        _, I = idx.search(t[None], 5)
        axes[r][0].text(0.5, 0.5, f"text:\n{text}", ha="center", va="center", fontsize=8)
        axes[r][0].axis("off")
        imgs = [Image.open(C.IMAGES / gallery.image[j]).convert("L") for j in I[0]]
        panel(r, imgs, f"text: {text}", offset=1)
    fig.suptitle("Top-5 retrieval: image queries (rows 1-2) and text queries (rows 3-4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "retrieval.png")
    plt.close(fig)


def ablation(epochs=60, seeds=(0, 1, 2), device=None):
    """Same probe, random image-level split instead of patient-level."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(C.EMB_NPZ)
    X, Y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    aucs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = torch.from_numpy(rng.permutation(len(X)))
        n_tr, n_va = int(0.7 * len(X)), int(0.1 * len(X))
        tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
        head, _ = fit_probe(X[tr], Y[tr], X[va], Y[va], epochs, device=device, verbose=False)
        with torch.no_grad():
            p = torch.sigmoid(head(X[te].to(device))).cpu().numpy()
        aucs.append(summarize(Y[te].numpy(), p)["mean_auc"])
    base = load_json("linear.json")
    out = {"patient_split_auc": base["mean_auc"],
           "random_split_auc": float(np.mean(aucs)),
           "random_split_std": float(np.std(aucs)),
           "inflation": float(np.mean(aucs)) - base["mean_auc"],
           "patient_split_ap": base["mean_ap"],
           "random_split_aucs": aucs,
           "seeds": list(seeds)}
    (C.OUT / "ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"patient-level AUC {out['patient_split_auc']:.4f} vs random-split "
          f"{out['random_split_auc']:.4f}±{out['random_split_std']:.4f} "
          f"(inflation {out['inflation']:+.4f})")
    return out


def main():
    ASSETS.mkdir(exist_ok=True)
    table()
    fig_auc()
    fig_curves()
    if (C.OUT / "gallery_emb.npy").exists() and (C.OUT / "lora.pt").exists():
        fig_retrieval()
    ablation()


if __name__ == "__main__":
    main()
