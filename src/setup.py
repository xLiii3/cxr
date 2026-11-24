"""One-command bootstrap: model weights, demo artifacts, optional dataset.

    python -m src.setup --check                 # report what is missing
    python -m src.setup --model                 # 747 MB, from the HF mirror
    python -m src.setup --artifacts <zip|URL>   # LoRA + index + demo thumbnails
    python -m src.setup --data --shards 3       # 11 GB training subset
    python -m src.setup --all --artifacts URL   # model + artifacts

Only `--data` is needed to retrain; inference needs the model plus the artifact zip.
"""
import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import config as C

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TEXT_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

MODEL_FILES = [
    ("open_clip_pytorch_model.bin", CLIP_REPO, "main", 600_000_000),
    ("open_clip_config.json", CLIP_REPO, "main", 200),
    ("tokenizer.json", CLIP_REPO, "main", 1000),
    ("vocab.txt", CLIP_REPO, "main", 100_000),
    ("tokenizer_config.json", CLIP_REPO, "main", 100),
    ("special_tokens_map.json", CLIP_REPO, "main", 100),
    ("config.json", TEXT_REPO, "main", 200),
]

ARTIFACTS = ["outputs/lora.pt", "outputs/gallery_emb.npy", "outputs/gallery_y.npy",
             "outputs/index_flat.faiss", "outputs/gallery.csv"]


def _get(url, dst, need):
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done % (64 << 20) < (1 << 20):
                print(f"  {dst.name} {done / total * 100:5.1f}%  {done / 2**20:.0f} MB", flush=True)
    if tmp.stat().st_size < need:
        tmp.unlink()
        raise RuntimeError(f"{dst.name}: got {tmp} bytes, expected > {need}")
    tmp.replace(dst)


def fetch_model():
    C.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    for name, repo, rev, need in MODEL_FILES:
        dst = C.CLIP_DIR / name
        if dst.exists() and dst.stat().st_size >= need:
            print(f"  ok      {name}")
            continue
        url = f"{MIRROR}/{repo}/resolve/{rev}/{name}"
        print(f"  get     {name}  ({need / 2**20:.0f} MB+)" if need > 1e6 else f"  get     {name}")
        _get(url, dst, need)
    print(f"model ready: {C.CLIP_DIR}")


def fetch_artifacts(src):
    dst = Path(src)
    if str(src).startswith("http"):
        dst = C.ROOT / "cxr-artifacts.zip"
        print(f"  get     {src}")
        _get(src, dst, 1)
    if not dst.exists():
        raise FileNotFoundError(dst)
    with zipfile.ZipFile(dst) as z:
        names = z.namelist()
        bad = [n for n in names if n.startswith(("/", "\\")) or ".." in Path(n).parts]
        if bad:
            raise RuntimeError(f"refusing unsafe zip entries: {bad[:3]}")
        z.extractall(C.ROOT)
    print(f"artifacts ready: {len(names)} files -> {C.ROOT}")


def check():
    rows = [("model weights", C.CLIP_DIR / "open_clip_pytorch_model.bin", 600_000_000)]
    rows += [("text tower cfg", C.CLIP_DIR / "config.json", 200)]
    rows += [("LoRA + head", C.ROOT / "outputs" / "lora.pt", 1_000_000)]
    rows += [("retrieval index", C.ROOT / "outputs" / "index_flat.faiss", 1000)]
    rows += [("gallery vectors", C.ROOT / "outputs" / "gallery_emb.npy", 1000)]
    rows += [("gallery labels", C.ROOT / "outputs" / "gallery.csv", 1000)]
    rows += [("demo thumbnails", C.GALLERY, 1)]
    rows += [("training images", C.IMAGES, 1)]
    for label, path, size in rows:
        ok = path.exists() and (path.stat().st_size >= size if path.is_file()
                                else any(path.iterdir()))
        print(f"  {'ok  ' if ok else 'miss'}  {label:16s} {path}")
    print("\n推理只需要前 7 项；只有重新训练才需要 training images。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--artifacts", metavar="ZIP_OR_URL")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not any([args.check, args.model, args.artifacts, args.data, args.all]):
        ap.print_help()
        return 0
    if args.check:
        check()
    if args.model or args.all:
        fetch_model()
    if args.artifacts:
        fetch_artifacts(args.artifacts)
    if args.data:
        from .fetch_data import main as fetch_main
        sys.argv = ["fetch_data", "--shards", str(args.shards)]
        fetch_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
