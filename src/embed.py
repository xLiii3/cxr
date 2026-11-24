"""Stage 1: cache BiomedCLIP image and text embeddings.

    python -m src.embed
Writes data/processed/embeddings.npz with keys: X, y, split, text, normal.
"""
import argparse
import time

import numpy as np
import torch

from . import config as C
from .data import loader, read_split
from .model import embed_images, embed_texts, load_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = read_split()
    model, _, tokenizer = load_clip(device)
    prompts = C.prompt_list()

    flat = [p for row in prompts for p in row] + [C.NORMAL_PROMPT]
    text = embed_texts(model, tokenizer, flat, device)
    text = text / text.norm(dim=-1, keepdim=True)
    text, normal = text[:-1].view(len(C.CLASSES), len(C.PROMPT_TEMPLATES), -1), text[-1]

    dl = loader(df, args.batch, shuffle=False, workers=args.workers)
    feats = np.zeros((len(df), text.shape[-1]), dtype="float32")
    t0 = time.time()
    pos = 0
    with torch.no_grad():
        for x, _ in dl:
            n = len(x)
            feats[pos:pos + n] = embed_images(model, x, device).numpy()
            pos += n
            if pos % (args.batch * 20) == 0:
                print(f"  {pos}/{len(df)}  {pos / (time.time() - t0):.0f} img/s", flush=True)

    y = df[C.CLASSES].to_numpy("float32")
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    np.savez_compressed(
        C.EMB_NPZ, X=feats, y=y, split=df["split"].to_numpy().astype("U5"),
        text=text.numpy(), normal=normal.numpy(),
    )
    # sanity: embeddings are unit length, so dot product == cosine
    norms = np.linalg.norm(feats[:64], axis=1)
    print(f"saved {C.EMB_NPZ} shape={feats.shape} norm={norms.mean():.4f} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
