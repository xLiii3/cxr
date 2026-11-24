"""Stage 4: embed the test gallery with the adapted encoder, build FAISS indices.

    python -m src.index                    # gallery = the patient-level test split
    python -m src.index --images <DIR>     # gallery = any folder of your own chest X-rays
Writes outputs/index_{flat,ivf,hnsw}.faiss, gallery.csv, gallery_emb.npy and index_bench.json.
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import config as C
from .data import loader, read_split, to_tensor
from .metrics import precision_at_k, recall_at_k
from .model import inject_lora, load_clip, load_lora

K = 10
N_QUERY = 100
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def embed_frame(model, df, device, batch=128):
    out = []
    with torch.no_grad():
        for x, _ in loader(df, batch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(model.encode_image(x.to(device, non_blocking=True)).float().cpu())
    e = torch.cat(out).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def embed_files(model, paths, device, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xs = torch.cat([to_tensor([Image.open(p).convert("RGB")]) for p in paths[i:i + batch]])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(model.encode_image(xs.to(device)).float().cpu())
    e = torch.cat(out).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def folder_gallery(model, folder, device):
    """Embed every image in a folder; labels stay unknown (all zeros)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"no images in {folder}")
    X = embed_files(model, paths, device)
    names = [str(p.resolve()) for p in paths]
    np.save(C.OUT / "gallery_emb.npy", X)
    (C.OUT / "gallery_y.npy").unlink(missing_ok=True)  # no labels in this mode
    (C.OUT / "gallery.csv").write_text("image\n" + "\n".join(names) + "\n", encoding="utf-8")
    kinds = ("flat", "ivf", "hnsw") if len(X) >= 64 else ("flat",)
    for kind in kinds:
        faiss.write_index(build(X, kind), str(C.OUT / f"index_{kind}.faiss"))
    print(f"gallery {X.shape} from {folder}; labels unknown, retrieval still works")


def build(X, kind):
    d = X.shape[1]
    if kind == "flat":
        idx = faiss.IndexFlatIP(d)
    elif kind == "ivf":
        idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, int(np.sqrt(len(X))),
                                 faiss.METRIC_INNER_PRODUCT)
        idx.train(X)
        idx.nprobe = 8
    elif kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 80
        idx.hnsw.efSearch = 64
    else:
        raise ValueError(kind)
    idx.add(X)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="build the gallery from this folder instead of the test split")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = load_clip(device)
    inject_lora(model)
    model.head = nn.Linear(model.embed_dim, len(C.CLASSES)).to(device)
    if (C.OUT / "lora.pt").exists():
        load_lora(model, C.OUT / "lora.pt")
        print("gallery: LoRA-adapted encoder")
    model.eval()
    if args.images:
        folder_gallery(model, args.images, device)
        return

    df = read_split()
    test = df[df.split == "test"].reset_index(drop=True)
    y = test[C.CLASSES].to_numpy("float32")
    X = embed_frame(model, test, device)
    np.save(C.OUT / "gallery_emb.npy", X)
    np.save(C.OUT / "gallery_y.npy", y)
    test[["idx", "image", "patient"]].to_csv(C.OUT / "gallery.csv", index=False)
    print(f"gallery {X.shape}")

    rng = np.random.default_rng(C.SEED)
    q_ids = rng.choice(len(X), min(N_QUERY, len(X)), replace=False)
    queries = X[q_ids]

    flat = build(X, "flat")
    _, truth = flat.search(queries, K + 1)
    stats = {}

    for kind in ("flat", "ivf", "hnsw"):
        idx = flat if kind == "flat" else build(X, kind)
        idx.search(queries[:1], K)
        lat, inter = [], 0.0
        for i, q in enumerate(queries):
            t = time.perf_counter()
            _, I = idx.search(q[None], K + 1)
            lat.append((time.perf_counter() - t) * 1000)
            inter += len(set(I[0][1:]) & set(truth[i][1:])) / K
        lat = np.array(lat)
        stats[kind] = {"p50_ms": float(np.percentile(lat, 50)),
                       "p95_ms": float(np.percentile(lat, 95)),
                       "recall_vs_flat": float(inter / len(queries))}
        faiss.write_index(idx, str(C.OUT / f"index_{kind}.faiss"))
        print(f"{kind:5s} p50 {stats[kind]['p50_ms']:.2f}ms  p95 {stats[kind]['p95_ms']:.2f}ms  "
              f"recall@10 vs flat {stats[kind]['recall_vs_flat']:.3f}")

    # label-overlap retrieval over the whole test split (self excluded)
    sims = X @ X.T
    np.fill_diagonal(sims, -1.0)
    top = np.argsort(-sims, axis=1)[:, :K // 2]
    rel_all = (y.astype(bool) @ y.astype(bool).T)
    has_finding = y.sum(1) > 0
    ranked_rel = rel_all[np.arange(len(X))[:, None], top]
    stats["label_queries"] = int(has_finding.sum())
    stats["label_recall@5"] = recall_at_k(ranked_rel[has_finding], K // 2)
    stats["label_precision@5"] = precision_at_k(ranked_rel[has_finding], K // 2)
    print(f"label recall@5 {stats['label_recall@5']:.3f}  "
          f"precision@5 {stats['label_precision@5']:.3f}  "
          f"({stats['label_queries']} queries with a finding)")

    (C.OUT / "index_bench.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
