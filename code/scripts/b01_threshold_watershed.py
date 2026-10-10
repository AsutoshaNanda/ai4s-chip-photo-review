"""B01: simple nucleus counting: smoothing + brightness cut-off (Otsu x factor) + watershed splitting + size filter.

Settings are chosen on the provider's TRAINING images only, then checked on the VALIDATION images.
The provider's TEST images stay locked (F01, Thursday). No single score is assumed as "the" objective:
two settings are kept, one with the lowest count error and one with the best outline score averaged over
all overlap levels. Both are reported with full curves. The search grid is wide and written to config.json.
"""
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from ai4s import bbbc039 as B
from ai4s import metrics_seg as S
from ai4s.runlog import Run

GRID = {
    "sigma": [0, 1, 2, 3],
    "otsu_factor": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
    "min_distance": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    "min_size": [0, 5, 10, 20, 40, 80],
}


def segment(img, sigma, otsu_factor, min_distance, min_size):
    x = img.astype(np.float32)
    if sigma > 0:
        x = gaussian(x, sigma=sigma, preserve_range=True)
    fg = ndi.binary_fill_holes(x > threshold_otsu(x) * otsu_factor)
    if min_size > 0:
        fg = remove_small_objects(fg, max_size=min_size)
    dist = ndi.distance_transform_edt(fg)
    lab_fg, _ = ndi.label(fg)
    peaks = peak_local_max(dist, min_distance=min_distance, labels=lab_fg, exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = watershed(-dist, markers, mask=fg)
    if min_size > 0:
        lab = remove_small_objects(lab, max_size=min_size)
    return lab


def mean_f1_over_levels(row):
    return float(np.mean([row[f"f1@{lv:.2f}"] for lv in S.DEFAULT_IOU_LEVELS]))


def grid_one_image(stem):
    """Score every grid setting on one image (training only). Returns compact rows."""
    img = B.read_image(stem)
    ref = B.read_labels(stem)
    out = []
    for sigma, fac, md, ms in itertools.product(*GRID.values()):
        lab = segment(img, sigma, fac, md, ms)
        r = S.score_image(ref, lab)
        out.append((stem, sigma, fac, md, ms, r["abs_count_error"], r["count_error"], mean_f1_over_levels(r),
                    r["tp@0.50"], r["fp@0.50"], r["fn@0.50"]))
    return out


def eval_one(args):
    stem, params = args
    img = B.read_image(stem)
    ref = B.read_labels(stem)
    t0 = time.perf_counter()
    lab = segment(img, **params)
    secs = time.perf_counter() - t0
    r = S.score_image(ref, lab)
    r.update({"stem": stem, "seconds": secs, "mean_f1_over_levels": mean_f1_over_levels(r)})
    return r, lab.astype(np.uint16)


def main():
    run = Run("B01", "threshold_watershed", config={"grid": GRID, "tune_on": "provider training (100)",
                                                     "report_on": "provider validation (50)", "test": "locked"})
    splits = B.split_lists()
    workers = max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for chunk in ex.map(grid_one_image, splits["training"]) for r in chunk]
    cols = ["stem", "sigma", "otsu_factor", "min_distance", "min_size", "abs_err", "err", "mean_f1_levels", "tp50", "fp50", "fn50"]
    g = pd.DataFrame(rows, columns=cols)
    run.log(f"grid: {len(g)} scored (image, setting) pairs in {time.time() - t0:.0f} s with {workers} processes")
    keys = ["sigma", "otsu_factor", "min_distance", "min_size"]
    agg = g.groupby(keys).agg(count_mae=("abs_err", "mean"), count_bias=("err", "mean"),
                              mean_f1_levels=("mean_f1_levels", "mean")).reset_index()
    run.save_df("grid_training_summary.csv", agg)
    best_count = agg.sort_values(["count_mae", "mean_f1_levels"], ascending=[True, False]).iloc[0]
    best_outline = agg.sort_values(["mean_f1_levels", "count_mae"], ascending=[False, True]).iloc[0]
    chosen = {"best_count": {k: (float(best_count[k]) if k in ("otsu_factor",) else int(best_count[k])) for k in keys},
              "best_outline": {k: (float(best_outline[k]) if k in ("otsu_factor",) else int(best_outline[k])) for k in keys}}
    chosen["training_scores"] = {"best_count": best_count.to_dict(), "best_outline": best_outline.to_dict()}
    run.save_json("chosen_settings.json", chosen)
    run.log(f"chosen on training: {chosen}")

    results = {}
    for name in ("best_count", "best_outline"):
        params = chosen[name]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(eval_one, [(s, params) for s in splits["validation"]]))
        rows = [r for r, _ in out]
        np.savez_compressed(run.path(f"pred_validation_{name}.npz"), **{r["stem"]: lab for r, lab in out})
        df = pd.DataFrame(rows)
        run.save_df(f"per_image_validation_{name}.csv", df)
        summ = S.summarize(rows)
        summ["seconds_per_image_median"] = float(df["seconds"].median())
        summ["mean_f1_over_levels"] = float(df["mean_f1_over_levels"].mean())
        results[name] = summ
        run.log(f"validation {name}: count MAE {summ['count_mae']:.2f}, bias {summ['count_bias']:.2f}, "
                f"pooled F1@0.50 {summ['pooled_f1@0.50']:.3f}, mean F1 over levels {summ['mean_f1_over_levels']:.3f}, "
                f"joined {summ['joined_ref']}, split {summ['split_pred']}, s/img {summ['seconds_per_image_median']:.3f}")
    run.save_json("validation_results.json", results)
    run.finish("measured", notes="B01 tuned on training, reported on validation; test locked")


if __name__ == "__main__":
    main()
