"""Stage 0: parse labels, split by patient, cache 224x224 uint8 images.

    python -m src.prepare            # labels + split + memmap
    python -m src.prepare --dry      # only report what would be done
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C


def parse_labels(csv=C.LABELS_CSV):
    df = pd.read_csv(csv)
    df = df.rename(columns={"Image Index": "image", "Patient ID": "patient",
                            "Patient Age": "age", "Patient Gender": "gender",
                            "View Position": "view"})
    df["image"] = df["image"].astype(str)
    for c in C.CLASSES:
        df[c] = df["Finding Labels"].str.contains(c, regex=False).astype("int8")
    # keep column order stable
    return df[["image", "patient", "age", "gender", "view", "Finding Labels"] + C.CLASSES]


def patient_split(df, frac=C.SPLITS, seed=C.SEED):
    patients = np.array(sorted(df["patient"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_tr, n_va = int(n * frac[0]), int(n * frac[1])
    parts = {
        "train": set(patients[:n_tr]),
        "val": set(patients[n_tr:n_tr + n_va]),
        "test": set(patients[n_tr + n_va:]),
    }
    df = df.copy()
    df["split"] = pd.Series(
        np.select([df["patient"].isin(p) for p in (parts["train"], parts["val"])],
                  ["train", "val"], "test"), index=df.index)
    return df, {k: len(v) for k, v in parts.items()}


def _resize(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BICUBIC), dtype=np.uint8)


def _fill(args):
    npy, start, paths = args
    arr = np.lib.format.open_memmap(npy, mode="r+")
    bad = 0
    for i, p in enumerate(paths):
        try:
            arr[start + i] = _resize(p)
        except Exception:
            bad += 1
    arr.flush()
    return len(paths), bad


def build_memmap(paths, out=C.IMAGE_NPY, workers=8, chunk=128):
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8,
                                    shape=(len(paths), C.IMG_SIZE, C.IMG_SIZE, 3))
    del arr
    jobs = [(str(out), i, paths[i:i + chunk]) for i in range(0, len(paths), chunk)]
    t0, done, bad = time.time(), 0, 0
    with Pool(workers) as pool:
        for n, b in pool.imap_unordered(_fill, jobs):
            done += n
            bad += b
            if done % (chunk * 10) == 0:
                print(f"  {done}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
    print(f"  resized {done} images in {time.time() - t0:.0f}s ({bad} unreadable)")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true", help="recreate the memmap")
    args = ap.parse_args()

    df = parse_labels()
    have = {p.name for p in C.IMAGES.glob("*.png")}
    df = df[df["image"].isin(have)].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no images found in {C.IMAGES}")
    df, n_pat = patient_split(df)
    df["idx"] = np.arange(len(df))

    print(f"images={len(df)} patients={df['patient'].nunique()} {n_pat}")
    print(f"train={(df.split == 'train').sum()} val={(df.split == 'val').sum()} test={(df.split == 'test').sum()}")
    print("prevalence:", {c: round(float(df[c].mean()), 4) for c in C.CLASSES})
    if args.dry:
        return

    C.PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.SPLIT_CSV, index=False)
    if args.rebuild and C.IMAGE_NPY.exists():
        C.IMAGE_NPY.unlink()
    if not C.IMAGE_NPY.exists():
        build_memmap([str(C.IMAGES / im) for im in df["image"]], workers=args.workers)

    stats = {
        "images": int(len(df)),
        "patients": int(df["patient"].nunique()),
        "split_images": {k: int((df.split == k).sum()) for k in ("train", "val", "test")},
        "split_patients": {k: int(df[df.split == k]["patient"].nunique()) for k in ("train", "val", "test")},
        "prevalence": {c: float(df[c].mean()) for c in C.CLASSES},
    }
    (C.OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote", C.SPLIT_CSV, C.IMAGE_NPY)


if __name__ == "__main__":
    main()
