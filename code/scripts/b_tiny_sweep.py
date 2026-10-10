"""Tiny-nucleus follow-up (improvements.md section 5, item 4): change the responsible stage, then test the explanation.

b_tiny.py found WHERE small nuclei are lost (retained v2 settings, saved fold models, held-out practice images):
  B02: mostly "merged with a neighbour" (marker spacing / min_distance 10-12 px, while a small nucleus is about 5 px wide),
       and, for the outline setting, "removed by the minimum-size rule".
  B03: for the count setting, the "inside" probability never reaches the threshold (0.85-0.95); small nuclei have lower contrast.
Test of each explanation: change ONLY that stage's setting over its full recorded search range, keep everything else at the
fold's retained setting, and report for every value
  - share of the smallest tenth of nuclei (training-fold deciles, as B05) found at overlap 0.50 and 0.75,
  - the cost on everything else: pooled F1 at 0.50 / 0.75, count MAE, split / joined / missed / false objects.
No value is selected and no cut-off is set here; the full curves are the result. If a stage change raises small-nucleus recall
only by giving up quality elsewhere, that trade-off is visible in the same table. Saved models only; practice images only.
"""
import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import b_unet as U
from ai4s import bbbc039 as B
from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run, n_workers
from b_posts import fold_models
from b_repost import GRID
from g03_speed_test import UNet

_G = {}


def _init(g):
    _G.update(g)


def _task(args):
    exp, setting_name, param, value = args
    target = "fg" if exp == "B02" else "boundary"
    rows, small_found = [], {0.5: [0, 0], 0.75: [0, 0]}
    for i, stem in enumerate(_G["stems"]):
        k = _G["fold_of"][i]
        s = dict(_G["posts"][exp][k][setting_name])
        post = {"thr": s["thr"], "min_distance": int(s["min_distance"]), "min_size": int(s["min_size"])}
        post[param] = value if param == "thr" else int(value)
        lab = U.to_objects(_G["probs"][exp][i].astype(np.float32), target, post["thr"], post["min_distance"], post["min_size"])
        ref = _G["refs"][i]
        rows.append(S.score_image(_G["refs_raw"][i], lab))
        iou = S.iou_from_counts(S.overlap_counts(ref, lab))
        best = iou.max(1) if iou.shape[1] else np.zeros(int(ref.max()))
        areas = np.bincount(ref.ravel())[1:]
        small = (areas > 0) & (areas <= _G["d1"][k])
        for lvl in small_found:
            small_found[lvl][0] += int((best[small] >= lvl).sum())
            small_found[lvl][1] += int(small.sum())
    sm = S.summarize(rows)
    return {"exp": exp, "setting": setting_name, "parameter": param, "value": value,
            "small_found@0.50": small_found[0.5][0] / max(small_found[0.5][1], 1),
            "small_found@0.75": small_found[0.75][0] / max(small_found[0.75][1], 1), "small_nuclei": small_found[0.5][1],
            "count_mae": sm["count_mae"], "count_bias": sm["count_bias"], "pooled_f1@0.50": sm["pooled_f1@0.50"],
            "pooled_f1@0.75": sm["pooled_f1@0.75"], "joined": sm["joined_ref"], "split": sm["split_pred"],
            "missed": sm["missed"], "false": sm["false"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Run("B05", f"tiny_nuclei_stage_sweep_seed{a.seed}", config={
        "seed": a.seed, "ranges": {k: v for k, v in GRID.items()}, "rule": "one stage setting changed at a time over its recorded search range; no selection"})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    stems = prac["stem"].tolist()
    refs_raw = [B.read_labels(s) for s in stems]
    refs = [S._relabel(r) for r in refs_raw]
    fold_of = prac["cv_fold"].tolist()
    d1 = {}
    for k in sorted(set(fold_of)):
        ta = np.concatenate([np.bincount(refs[i].ravel())[1:] for i in range(len(stems)) if fold_of[i] != k])
        d1[k] = float(np.quantile(ta[ta > 0], 0.1))
    probs, posts = {}, {}
    for exp, target, n_cls in (("B02", "fg", 2), ("B03", "boundary", 3)):
        fm = fold_models(exp, target, a.seed, "v2")
        posts[exp] = {k: v[1] for k, v in fm.items()}
        P = [None] * len(stems)
        for k, (d, _) in sorted(fm.items()):
            m = UNet(out_ch=n_cls)
            m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
            m = m.eval().to(U.DEV)
            idx = [i for i in range(len(stems)) if fold_of[i] == k]
            for i, p in zip(idx, U.predict_prob(m, np.stack([U.norm(B.read_image(stems[i])) for i in idx]))):
                P[i] = p.astype(np.float16)
        probs[exp] = P
        run.log(f"{exp}: probabilities for {len(stems)} images")
    g = {"stems": stems, "fold_of": fold_of, "refs": refs, "refs_raw": refs_raw, "d1": d1, "probs": probs, "posts": posts}
    tasks = []
    for exp in ("B02", "B03"):
        grid = GRID["fg" if exp == "B02" else "boundary"]
        params = ["thr", "min_distance", "min_size"] if exp == "B02" else ["thr", "min_size"]
        for setting in ("best_count", "best_outline"):
            for param in params:
                tasks += [(exp, setting, param, v) for v in grid[param]]
    with ProcessPoolExecutor(n_workers(), initializer=_init, initargs=(g,)) as ex:
        res = list(ex.map(_task, tasks, chunksize=1))
    D = pd.DataFrame(res)
    run.save_df("stage_sweep.csv", D)
    for (exp, setting, param), d in D.groupby(["exp", "setting", "parameter"]):
        run.log(f"{exp} {setting}, only {param} changed (others at each fold's retained value):\n"
                + d[["value", "small_found@0.50", "small_found@0.75", "pooled_f1@0.50", "pooled_f1@0.75", "count_mae", "joined", "split", "missed", "false"]]
                .round(3).to_string(index=False))
    run.save_json("results.json", D.to_dict(orient="records"))
    run.finish("measured", notes="tiny-nucleus stage sweep on saved v2 settings; no selection")


if __name__ == "__main__":
    main()
