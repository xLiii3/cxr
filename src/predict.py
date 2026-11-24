"""Run a folder of chest X-rays through the trained model and write a CSV.

    python -m src.predict --images D:\my_xrays                # -> outputs/predictions/predictions.csv
    python -m src.predict --images D:\my_xrays --cam --top 3  # + Grad-CAM PNGs for the top class
    python -m src.predict --images D:\my_xrays\a.png          # one file

Needs only the model weights and outputs/lora.pt; the retrieval gallery is optional.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

from . import config as C
from .service import Engine

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect(path, recursive=False):
    p = Path(path)
    if p.is_file():
        return [p]
    it = p.rglob("*") if recursive else p.iterdir()
    files = sorted(f for f in it if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images under {p}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image file or folder")
    ap.add_argument("--out", default=str(C.OUT / "predictions"))
    ap.add_argument("--top", type=int, default=3, help="how many classes to report per image")
    ap.add_argument("--cam", action="store_true", help="also save a Grad-CAM PNG per image")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    files = collect(args.images, args.recursive)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cam:
        (out / "cam").mkdir(exist_ok=True)

    engine = Engine(gallery=False)
    top = max(1, min(args.top, len(C.CLASSES)))
    head = ["file"] + [f"top{i}_{k}" for i in range(1, top + 1) for k in ("zh", "prob")]
    head += [f"p_{c}" for c in C.CLASSES]

    rows = []
    for f in files:
        img = Image.open(f).convert("L")
        preds = engine.predict(img)
        row = {"file": f.name}
        for i in range(top):
            row[f"top{i + 1}_zh"] = preds[i]["zh"]
            row[f"top{i + 1}_prob"] = round(preds[i]["prob"], 4)
        for p in preds:
            row[f"p_{p['class']}"] = round(p["prob"], 4)
        rows.append(row)
        if args.cam:
            heat, cls, prob = engine.explain(img, preds[0]["class"])
            heat.save(out / "cam" / f"{f.stem}_cam.png")
        print(f"{f.name:28s} {preds[0]['zh']:8s} {preds[0]['prob']:.3f}  "
              f"(2nd {preds[1]['zh']} {preds[1]['prob']:.3f})")

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(rows)

    top1 = [r["top1_zh"] for r in rows]
    print(f"\n{len(rows)} images -> {out / 'predictions.csv'}")
    print("top-1 分布: " + ", ".join(f"{k} {top1.count(k)}" for k in sorted(set(top1), key=top1.count, reverse=True)))
    if args.cam:
        print(f"Grad-CAM: {out / 'cam'}")


if __name__ == "__main__":
    main()
