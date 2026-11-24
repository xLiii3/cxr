"""Download the NIH ChestX-ray14 subset used in this project.

    python -m src.fetch_data --shards 3

Pulls the label CSV and image shards (5k-10k PNGs each) from the HuggingFace
mirror, unpacks only the PNGs, and deletes the archives afterwards.
"""
import argparse
import os
import shutil
import subprocess
import zipfile

from . import config as C

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO = "datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
LABELS = f"{REPO}/Data_Entry_2017_v2020.csv"


def download(url, dst):
    """Resume-capable download; uses curl when available, urllib otherwise."""
    if dst.exists() and dst.stat().st_size > 1_000:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "--retry", "3", "-C", "-", "-o", str(dst), url], check=True)
        return
    import urllib.request
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3, help="number of image shards (5k-10k images each)")
    args = ap.parse_args()

    C.RAW.mkdir(parents=True, exist_ok=True)
    C.IMAGES.mkdir(parents=True, exist_ok=True)
    download(f"{MIRROR}/{LABELS}", C.LABELS_CSV)

    for i in range(1, args.shards + 1):
        zp = C.RAW / f"images_{i:03d}.zip"
        download(f"{MIRROR}/{REPO}/images/images_{i:03d}.zip", zp)
        n = 0
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                name = os.path.basename(info.filename)
                if "__MACOSX" in info.filename or not name.lower().endswith(".png"):
                    continue
                out = C.IMAGES / name
                if not out.exists():
                    with z.open(info) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    n += 1
        print(f"{zp.name}: +{n} images, total {len(list(C.IMAGES.glob('*.png')))}", flush=True)
        zp.unlink()


if __name__ == "__main__":
    main()
