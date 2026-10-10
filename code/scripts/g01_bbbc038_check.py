"""G01 for B07: BBBC038 (Kaggle DSB 2018 stage-1 training, CC0) file checks and overlap with BBBC039.

BBBC039 is known to share sources with BBBC038 (D03), so overlapping photos must be found and removed before
B07 can count as an outside test. No similarity cut-off is assumed: every BBBC038 image gets its best match
score against all 200 BBBC039 images, the full distribution is saved, and the split between "same photo" and
"different photo" is read from that distribution (written down with the result).

Two checks per BBBC038 image (grayscale = mean of the colour channels; each image scaled 1st-99.8th percentile):
  whole   images with BBBC039's shape ratio (520x696, also half and double size): resized to 520x696 and
          compared pixel by pixel (Pearson correlation) on 4x reduced copies.
  crop    every image that fits inside a BBBC039 image at the same pixel scale: best normalised cross-
          correlation of its 4x reduced copy anywhere inside each 4x reduced BBBC039 image.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import match_template
from skimage.transform import resize

from ai4s import bbbc039 as B
from ai4s.paths import DATA_ROOT, SPLITS_ROOT
from ai4s.runlog import Run

ROOT = DATA_ROOT / "bbbc038" / "stage1_train"
RED = 4


def scaled(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99.8])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def reduce(a, f=RED):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def main():
    run = Run("G01", "bbbc038_check", config={"source": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
                                              "licence": "CC0 (provider page, read 6 Oct 2026)", "reduction": RED})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    ref = {s: reduce(scaled(B.read_image(s))) for s in sp["stem"]}
    role = dict(zip(sp["stem"], sp["role"]))
    H, W = 520, 696
    rows = []
    ids = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    for n, iid in enumerate(ids):
        im = np.asarray(Image.open(ROOT / iid / "images" / f"{iid}.png"))
        g = im[..., :3].mean(-1) if im.ndim == 3 else im.astype(np.float32)
        nmask = len(list((ROOT / iid / "masks").glob("*.png")))
        colour_spread = float(np.abs(im[..., 0].astype(float) - im[..., 2].astype(float)).mean()) if im.ndim == 3 else 0.0
        r = {"image_id": iid, "height": g.shape[0], "width": g.shape[1], "masks": nmask, "colour_spread": colour_spread,
             "whole_best": np.nan, "whole_stem": None, "crop_best": np.nan, "crop_stem": None}
        gs = scaled(g)
        if abs(g.shape[1] / g.shape[0] - W / H) / (W / H) < 0.01:
            t = reduce(resize(gs, (H, W), preserve_range=True, anti_aliasing=True)).ravel()
            t = (t - t.mean()) / (t.std() + 1e-9)
            best = max(((float(np.mean(t * ((v.ravel() - v.mean()) / (v.std() + 1e-9)))), s) for s, v in ref.items()))
            r["whole_best"], r["whole_stem"] = best
        if g.shape[0] <= H and g.shape[1] <= W:
            t = reduce(gs)
            if t.std() > 0:
                best = max(((float(match_template(v, t).max()), s) for s, v in ref.items()))
                r["crop_best"], r["crop_stem"] = best
        r["best"] = np.nanmax([r["whole_best"], r["crop_best"]]) if not (np.isnan(r["whole_best"]) and np.isnan(r["crop_best"])) else np.nan
        stem = r["whole_stem"] if (not np.isnan(r["whole_best"]) and r["whole_best"] == r["best"]) else r["crop_stem"]
        r["best_stem"], r["best_stem_role"] = stem, role.get(stem)
        rows.append(r)
        if n % 100 == 0:
            run.log(f"{n}/{len(ids)} checked")
    D = pd.DataFrame(rows).sort_values("best", ascending=False)
    run.save_df("per_image_overlap.csv", D)
    hist = np.histogram(D["best"].dropna(), bins=np.linspace(-0.2, 1.0, 25))
    run.log(f"BBBC038 stage-1 training: {len(D)} images, {int(D['masks'].sum())} nucleus masks; shapes {D.groupby(['height', 'width']).size().to_dict()}")
    run.log("best-match score distribution (bin start: count): " + ", ".join(f"{b:.2f}: {c}" for b, c in zip(hist[1], hist[0]) if c))
    run.log("top 40 matches:\n" + D.head(40)[["image_id", "height", "width", "whole_best", "crop_best", "best_stem", "best_stem_role"]].to_string())
    run.finish("measured", notes="BBBC038 file check + overlap scores vs all 200 BBBC039 images (no cut-off assumed)")


if __name__ == "__main__":
    main()
