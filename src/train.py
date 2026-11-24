"""Stage 3: train the linear probe or LoRA adapters.

    python -m src.train --stage linear
    python -m src.train --stage lora --epochs 3
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import loader, read_split
from .metrics import summarize
from .model import (count_params, freeze_all, inject_lora, load_clip, load_lora,
                    trainable_parameters)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_lora(model, df, args):
    """Sigmoid scores for every row of df, forward pass through the backbone."""
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader(df, args.batch, shuffle=False, workers=args.workers):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                f = model.encode_image(x.to(args.device, non_blocking=True))
            out.append(torch.sigmoid(model.head(f.float())).cpu().numpy())
    return np.concatenate(out)


def fit_probe(Xtr, Ytr, Xva, Yva, epochs=60, batch=256, lr=1e-3, wd=1e-2,
              device="cuda", verbose=True):
    """Train a linear classifier on frozen embeddings; return best head.

    Class weights counter the 1-16% prevalence, early stopping uses val AUC.
    """
    pos = Ytr.sum(0).to(device)
    pw = ((len(Ytr) - pos) / pos.clamp(min=1)).to(device)
    head = nn.Linear(Xtr.shape[1], len(C.CLASSES)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xva, Yva = Xva.to(device), Yva
    best, best_state, hist = -1.0, None, []

    for ep in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            xb, yb = Xtr[b].to(device), Ytr[b].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head(Xva)).cpu().numpy()
        res = summarize(Yva.numpy(), pv)
        hist.append({"epoch": ep, "loss": total / len(Xtr),
                     "val_auc": res["mean_auc"], "val_ap": res["mean_ap"]})
        if verbose and (ep % 5 == 0 or ep == epochs):
            print(f"epoch {ep:02d} loss {hist[-1]['loss']:.4f} val_auc {res['mean_auc']:.4f} "
                  f"val_ap {res['mean_ap']:.4f}", flush=True)
        if res["mean_auc"] > best:
            best = res["mean_auc"]
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head, hist


def run_linear(args):
    d = np.load(C.EMB_NPZ)
    X, Y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    mask = {s: d["split"] == s for s in ("train", "val", "test")}
    Xtr, Ytr = X[mask["train"]], Y[mask["train"]]
    Xva, Yva = X[mask["val"]], Y[mask["val"]]
    Xte, Yte = X[mask["test"]].to(args.device), Y[mask["test"]]

    head, hist = fit_probe(Xtr, Ytr, Xva, Yva, args.epochs, device=args.device)
    with torch.no_grad():
        p = torch.sigmoid(head(Xte)).cpu().numpy()
    res = summarize(Yte.numpy(), p)
    print(f"test mean_auc {res['mean_auc']:.4f} mean_ap {res['mean_ap']:.4f}")

    torch.save({"state": head.state_dict(), "r": 0}, C.OUT / "linear.pt")
    np.savez(C.OUT / "scores_linear.npz", y=Yte.numpy(), p=p)
    (C.OUT / "history_linear.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (C.OUT / "linear.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def run_lora(args):
    df = read_split()
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    model, _, _ = load_clip(args.device)
    freeze_all(model)
    inject_lora(model, args.r, args.alpha)
    model.head = nn.Linear(model.embed_dim, len(C.CLASSES)).to(args.device)

    train_names = {n for n, p in model.named_parameters() if p.requires_grad}
    total_params = sum(p.numel() for p in model.parameters())
    n_train = count_params(model)
    keep = [n for n in model.state_dict() if n in train_names]
    print(f"trainable {n_train:,} / {total_params:,} ({100 * n_train / total_params:.3f}%)")

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr_head], total_steps=steps, pct_start=0.15)
    loss_fn = nn.BCEWithLogitsLoss()

    best_ap, hist, t0 = -1.0, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        model.visual.eval()  # frozen backbone stays in eval mode
        total, seen = 0.0, 0
        for x, y in loader(tr, args.batch, shuffle=True, workers=args.workers):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = loss_fn(model.head(model.encode_image(x)).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            seen += len(y)
            if seen % (args.batch * 50) == 0:
                print(f"  ep{ep} {seen}/{len(tr)} loss {total / seen:.4f} "
                      f"{seen / (time.time() - t0):.0f} img/s", flush=True)

        pv = predict_lora(model, va, args)
        res = summarize(va[C.CLASSES].to_numpy("float32"), pv)
        rec = {"epoch": ep, "loss": total / seen, "val_auc": res["mean_auc"],
               "val_ap": res["mean_ap"], "vram_gb": torch.cuda.max_memory_allocated() / 2**30,
               "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        print(f"epoch {ep:02d} loss {rec['loss']:.4f} val_auc {res['mean_auc']:.4f} "
              f"val_ap {res['mean_ap']:.4f} vram {rec['vram_gb']:.2f}GB", flush=True)
        if res["mean_ap"] > best_ap:
            best_ap = res["mean_ap"]
            state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in keep}
            torch.save({"state": state, "r": args.r, "epoch": ep}, C.OUT / "lora.pt")

    ck = load_lora(model, C.OUT / "lora.pt", args.r)
    print("best epoch", ck["epoch"])
    p = predict_lora(model, te, args)
    res = summarize(te[C.CLASSES].to_numpy("float32"), p)
    print(f"test mean_auc {res['mean_auc']:.4f} mean_ap {res['mean_ap']:.4f}")
    np.savez(C.OUT / "scores_lora.npz", y=te[C.CLASSES].to_numpy("float32"), p=p)
    (C.OUT / "history_lora.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (C.OUT / "lora.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["linear", "lora"], required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.epochs = args.epochs or (60 if args.stage == "linear" else 3)
    seed_all(args.seed)
    C.OUT.mkdir(exist_ok=True)
    run_linear(args) if args.stage == "linear" else run_lora(args)


if __name__ == "__main__":
    main()
