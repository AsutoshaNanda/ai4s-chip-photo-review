"""Tiny-nucleus diagnosis (improvements.md section 5, item 4): where do small nuclei disappear?

For the retained B02/B03 v2 settings (best count and best outline, saved fold models, held-out practice images):
small nuclei = reference nuclei in the smallest tenth of nucleus area on that fold's TRAINING images (same rule as B05).
Each small nucleus is followed through the pipeline, using only the pipeline's own settings (no new cut-offs):
  1 preparation   : image normalisation (percentile scaling, b_unet.norm). Recorded: the nucleus's mean prepared
                    intensity minus its surrounding ring, and the share of its pixels clipped to 0.
  2 prediction    : does the model's probability reach the setting's threshold anywhere inside the nucleus?
                    (B02: nucleus probability; B03: "inside" probability, which makes the seeds)
  3 clean-up      : present in the thresholded mask, but lost afterwards: removed by the minimum-size rule, merged
                    with a neighbour (its best predicted object also covers another reference nucleus), or kept
                    but outlined too differently.
A nucleus counts as found when its best overlap (IoU) is at least the level reported (0.50 and 0.75, as in B05).
The first stage it fails is recorded. Stage-1 signals are compared between found and missed small nuclei.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import b_unet as U
from ai4s import bbbc039 as B
from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run
from b_posts import fold_models
from g03_speed_test import UNet


def stages(prob, kind, thr, min_distance, min_size):
    """b_unet.to_objects split into its stages; returns (mask after threshold, seeds/markers mask, labels before size rule, final)."""
    if kind == "fg":
        fg = ndi.binary_fill_holes(prob[1] > thr)
        dist = ndi.distance_transform_edt(fg)
        lab_fg, _ = ndi.label(fg)
        pk = peak_local_max(dist, min_distance=min_distance, labels=lab_fg, exclude_border=False)
        mk = np.zeros(fg.shape, np.int32)
        mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        lab = watershed(-dist, mk, mask=fg)
        seedmask = prob[1] > thr
    else:
        fg = (prob[1] + prob[2]) > thr
        seedmask = prob[1] > thr
        seeds, _ = ndi.label(seedmask)
        lab = watershed(-prob[1], seeds, mask=fg)
    before = lab.copy()
    if min_size > 0:
        sizes = np.bincount(lab.ravel())
        small = np.isin(lab, np.flatnonzero(sizes <= min_size)) & (lab > 0)
        lab[small] = 0
    return fg, seedmask, before, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Run("B05", f"tiny_nuclei_diagnosis_seed{a.seed}", config={"seed": a.seed, "settings": "v2 best_count / best_outline",
                                                                     "small": "smallest tenth of training-fold nucleus areas", "levels": [0.5, 0.75]})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    refs = {s: S._relabel(B.read_labels(s)) for s in prac["stem"]}
    rows = []
    for exp, target, n_cls in (("B02", "fg", 2), ("B03", "boundary", 3)):
        fm = fold_models(exp, target, a.seed, "v2")
        for k, (d, post) in sorted(fm.items()):
            m = UNet(out_ch=n_cls)
            m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
            m = m.eval().to(U.DEV)
            train_areas = np.concatenate([np.bincount(refs[s].ravel())[1:] for s in prac.loc[prac["cv_fold"] != k, "stem"]])
            d1 = float(np.quantile(train_areas[train_areas > 0], 0.1))
            for stem in prac.loc[prac["cv_fold"] == k, "stem"]:
                raw = B.read_image(stem)
                x = U.norm(raw)
                prob = U.predict_prob(m, x[None])[0]
                ref = refs[stem]
                areas = np.bincount(ref.ravel())
                small_ids = [i for i in range(1, len(areas)) if 0 < areas[i] <= d1]
                if not small_ids:
                    continue
                objs = ndi.find_objects(ref)
                for name in ("best_count", "best_outline"):
                    s = post[name]
                    thr, md, ms = s["thr"], int(s["min_distance"]), int(s["min_size"])
                    fg, seedmask, before, final = stages(prob, target, thr, md, ms)
                    assert np.array_equal(final, U.to_objects(prob, target, thr, md, ms))
                    iou = S.iou_from_counts(S.overlap_counts(ref, final))          # rows: reference objects 1..n
                    for i in small_ids:
                        o = ref == i
                        best = float(iou[i - 1].max()) if iou.shape[1] else 0.0
                        bestj = int(iou[i - 1].argmax()) + 1 if iou.shape[1] and best > 0 else 0
                        ring = ndi.binary_dilation(o, iterations=3) & ~o & (ref == 0)
                        signal = (prob[1][o] > thr).any()
                        in_mask = (fg[o]).any()
                        kept_px = (final[o] > 0).any()
                        before_px = (before[o] > 0).any()
                        merged = bool(bestj and len(np.unique(ref[final == bestj])) > 2)    # background + 2 or more references
                        rows.append({"method": f"{exp}_{name}", "cv_fold": int(k), "stem": stem, "ref_id": i, "area": int(areas[i]),
                                     "best_iou": best, "prepared_contrast": float(x[o].mean() - x[ring].mean()) if ring.any() else np.nan,
                                     "share_clipped_to_0": float((x[o] <= 0).mean()), "raw_contrast": float(raw[o].mean() - raw[ring].mean()) if ring.any() else np.nan,
                                     "max_prob_inside": float(prob[1][o].max()), "threshold": thr,
                                     "prob_reaches_threshold": bool(signal), "in_threshold_mask": bool(in_mask),
                                     "present_before_size_rule": bool(before_px), "present_after_size_rule": bool(kept_px), "merged": merged})
            run.log(f"{exp} fold {k} done")
    D = pd.DataFrame(rows)
    run.save_df("small_nuclei.csv", D)
    out = {}
    for lvl in (0.5, 0.75):
        D[f"found@{lvl}"] = D["best_iou"] >= lvl
        miss = ~D[f"found@{lvl}"]
        stage = np.select([~D["prob_reaches_threshold"], ~D["present_before_size_rule"], ~D["present_after_size_rule"], D["merged"]],
                          ["prediction: probability never reaches threshold", "clean-up: no object formed (no seed/marker)",
                           "clean-up: removed by minimum-size rule", "clean-up: merged with a neighbour"],
                          "clean-up/outline: kept but overlap below level")
        D[f"stage@{lvl}"] = np.where(miss, stage, "found")
        tab = D.groupby(["method", f"stage@{lvl}"]).size().unstack(fill_value=0)
        run.log(f"small nuclei by first failing stage at overlap {lvl}:\n{tab.to_string()}")
        out[f"stages@{lvl}"] = tab.to_dict(orient="index")
        sig = D.groupby(["method", f"found@{lvl}"])[["prepared_contrast", "raw_contrast", "share_clipped_to_0", "max_prob_inside", "area"]].median()
        run.log(f"preparation and prediction signals, medians, found vs missed at {lvl}:\n{sig.round(4).to_string()}")
        out[f"signals@{lvl}"] = {f"{m}|found={f}": v for (m, f), v in sig.round(5).to_dict(orient="index").items()}
    out["small_nuclei_total"] = D.groupby("method").size().to_dict()
    run.save_json("results.json", out)
    run.finish("measured", notes="tiny-nucleus stage diagnosis on v2 settings; practice out-of-fold")


if __name__ == "__main__":
    main()
