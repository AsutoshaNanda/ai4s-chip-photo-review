"""Rank organ-on-a-chip photos from most to least likely to be bad (the review aid).

  python inference.py --photos path/to/photos_or_folder --out ranking.csv [--sheet sheet.png]

For every photo: (1) input gate by camera format (mode/width x height): formats seen in development are scored automatically; the thin-support
format (640x480) and any unseen format are sent to MANUAL REVIEW with no score; (2) grey conversion, long side 1036; (3) frozen DINOv2 ViT-L@518
features of the whole photo and of a 3x3 grid of native-resolution tiles (mean+max); (4) the saved logistic-regression scorer
(model_x01_3x3.joblib) -> p_bad. Output CSV: photo, format, route, p_bad, rank (1 = review first). GPU if available, else CPU (slower).
The score is a ranking score; it is NOT advertised as a calibrated probability. Needs: pip install -r requirements.txt; the model file."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

import joblib
import numpy as np
import pandas as pd
from PIL import Image

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class Stub:                                # a_views.Extractor expects a run object with .log
    def log(self, *_):
        pass


def load_gray(p):
    im = Image.open(p)
    key = f"{im.mode}/{im.size[0]}x{im.size[1]}"
    w, h = im.size
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    s = 1036 / max(w, h)
    g = np.asarray(Image.fromarray(g).resize((round(w * s), round(h * s)), Image.BILINEAR), dtype=np.uint8)
    return key, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="+", required=True, help="image files and/or folders")
    ap.add_argument("--out", default="ranking.csv")
    ap.add_argument("--sheet", default="", help="optional PNG contact sheet of the lowest- and highest-ranked photos")
    ap.add_argument("--model", default=str(HERE / "model_x01_3x3.joblib"))
    a = ap.parse_args()
    meta = json.load(open(Path(a.model).with_suffix(".json")))
    model = joblib.load(a.model)
    files = []
    for p in a.photos:
        p = Path(p)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT) if p.is_dir() else [p]
    rows, imgs, idx = [], [], []
    for f in files:
        key, g = load_gray(f)
        if key in meta["automatic_formats"]:
            route = "automatic"
        elif key in meta["thin_support_formats"]:
            route = meta["small_rule"]
        else:
            route = "manual_review"
        rows.append({"photo": str(f), "format": key, "route": route, "p_bad": np.nan})
        if route == "automatic":
            imgs.append(g)
            idx.append(len(rows) - 1)
    if imgs:
        import torch
        from a_views import Extractor, tiles
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ex = Extractor("vit_large_patch14_dinov2.lvd142m", 518, dev, Stub())
        ex.quiet = True
        whole = ex("whole", imgs)
        T = ex("tiles", [t for m in imgs for t in tiles(m, 3)], native=True).reshape(len(imgs), 9, -1)
        X = np.concatenate([whole, T.mean(1), T.max(1)], 1)
        for i, p in zip(idx, model.predict_proba(X)[:, 1]):
            rows[i]["p_bad"] = float(p)
    D = pd.DataFrame(rows)
    D["rank"] = D["p_bad"].rank(ascending=False, method="first")
    D = D.sort_values(["rank", "photo"], na_position="last")
    D.to_csv(a.out, index=False)
    print(D.to_string(index=False))
    print(f"\n{(D['route'] == 'automatic').sum()} photos scored automatically; {(D['route'] != 'automatic').sum()} sent to manual review. Wrote {a.out}")
    if a.sheet and len(D.dropna(subset=['p_bad'])):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        S = D.dropna(subset=["p_bad"])
        pick = pd.concat([S.head(4), S.tail(4)]).drop_duplicates("photo")
        fig, ax = plt.subplots(2, 4, figsize=(16, 7))
        for a_, (_, r) in zip(ax.ravel(), pick.iterrows()):
            _, g = load_gray(r["photo"])
            a_.imshow(g, cmap="gray")
            a_.set_title(f"rank {int(r['rank'])}  p_bad {r['p_bad']:.2f}", fontsize=11)
            a_.axis("off")
        for a_ in ax.ravel()[len(pick):]:
            a_.axis("off")
        fig.suptitle("Review first (top row) vs review last (bottom row)")
        fig.tight_layout()
        fig.savefig(a.sheet, dpi=100)


if __name__ == "__main__":
    main()
