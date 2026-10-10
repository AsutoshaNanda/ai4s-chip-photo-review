"""B02/B03 v2 clean-up search: the first search hit the edge of its range in every fold (B02: marker spacing 10 =
largest tried; B03: threshold 0.7 = largest tried), so the best setting may lie outside it. Same saved fold models,
same rule (settings chosen on the inner check fold only, two objectives kept), wider ranges; then the held-out
fold is scored again. No retraining. If a choice still sits on a range edge, that is logged."""
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from ai4s import metrics_seg as S
from ai4s.paths import RUNS_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, n_workers

RUNS = Path(__file__).resolve().parents[1] / "runs"
GRID = {"fg": {"thr": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "min_distance": [2, 4, 6, 8, 10, 12, 15, 20, 25, 30],
               "min_size": [0, 10, 25, 50, 100, 200]},
        "boundary": {"thr": [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], "min_distance": [0],
                     "min_size": [0, 10, 25, 50, 100, 200]}}
_P = {}


def _init(probs, labels, kind):
    _P.update(probs=probs, labels=labels, kind=kind)


def _score(setting):
    import b_unet as U
    thr, md, ms = setting
    rs = [S.score_image(l, U.to_objects(p, _P["kind"], thr, md, ms)) for p, l in zip(_P["probs"], _P["labels"])]
    sm = S.summarize(rs)
    return {"thr": thr, "min_distance": md, "min_size": ms, "count_mae": sm["count_mae"],
            "mean_f1_levels": float(np.mean([sm[f"mean_image_f1@{lv:.2f}"] for lv in S.DEFAULT_IOU_LEVELS]))}


def main():
    import b_unet as U
    from g03_speed_test import UNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["B02", "B03"])
    ap.add_argument("--seed", type=int, default=0, help="training seed of the saved fold models to use")
    ap.add_argument("--variant", default="", help="model variant suffix, e.g. _noisetrain")
    ap.add_argument("--thr_extra", default="", help="extra thresholds added to the grid (improvements.md: refine B03 toward the 1.0 limit)")
    a = ap.parse_args()
    target, n_cls = ("fg", 2) if a.exp == "B02" else ("boundary", 3)
    g = {k: list(v) for k, v in GRID[target].items()}
    if a.thr_extra:
        g["thr"] = sorted(set(g["thr"]) | {float(t) for t in a.thr_extra.split(",")})
    assert all(0 < t < 1 for t in g["thr"]), "thresholds must stay inside the valid probability range"
    name = f"unet_{target}{a.variant}_repost_v2" + (f"_seed{a.seed}" if a.seed else "") + ("_thr_extended" if a.thr_extra else "")
    run = Run(a.exp, name, config={"grid": g, "seed": a.seed, "variant": a.variant, "rule": "chosen on inner check fold; two objectives"})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    folds = sorted(prac["cv_fold"].unique())
    X, _, L = U.load(prac["stem"].tolist(), target)
    oof = {"best_count": [], "best_outline": []}
    edges = {}
    from b_posts import fold_models
    fm = fold_models(a.exp, target, a.seed, "orig", a.variant)
    assert sorted(fm) == folds, f"expected fold models {folds}, found {sorted(fm)}"
    run.log("fold models: " + str({k: v[0].name for k, v in fm.items()}))
    for k in folds:
        d = fm[k][0]
        inner = folds[(folds.index(k) + 1) % len(folds)]
        m = UNet(out_ch=n_cls)
        m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
        m = m.eval().to(U.DEV)
        va = np.flatnonzero(prac["cv_fold"] == inner)
        te = np.flatnonzero(prac["cv_fold"] == k)
        pva, pte = U.predict_prob(m, X[va]), U.predict_prob(m, X[te])
        settings = list(itertools.product(g["thr"], g["min_distance"], g["min_size"]))
        with ProcessPoolExecutor(n_workers(), initializer=_init, initargs=(pva, [L[i] for i in va], target)) as ex:
            G = pd.DataFrame(list(ex.map(_score, settings)))
        G.to_csv(run.path(f"post_grid_fold{k}.csv"), index=False)
        pick = {"best_count": G.sort_values(["count_mae", "mean_f1_levels"], ascending=[True, False]).iloc[0].to_dict(),
                "best_outline": G.sort_values(["mean_f1_levels", "count_mae"], ascending=[False, True]).iloc[0].to_dict()}
        for name, s in pick.items():
            on_edge = [key for key in ("thr", "min_distance", "min_size") if len(g[key]) > 1 and s[key] in (min(g[key]), max(g[key]))]
            edges[f"fold{k}|{name}"] = on_edge
            for i, p in zip(te, pte):
                r = S.score_image(L[i], U.to_objects(p, target, s["thr"], int(s["min_distance"]), int(s["min_size"])))
                r.update({"stem": prac.loc[i, "stem"], "plate": prac.loc[i, "plate"], "cv_fold": int(k)})
                oof[name].append(r)
        run.log(f"fold {k}: picks {{{', '.join(f'{n}: thr {s['thr']}, md {s['min_distance']:.0f}, ms {s['min_size']:.0f}' for n, s in pick.items())}}}; on range edge: "
                f"{ {n: edges[f'fold{k}|{n}'] for n in pick} }")
    results = {"edges": edges}
    for name, rows in oof.items():
        df = pd.DataFrame(rows)
        run.save_df(f"oof_per_image_{name}.csv", df.drop(columns=[c for c in df.columns if c.startswith(("tp@", "fp@", "fn@"))]))
        aa = S.summarize(rows)
        results[name] = aa
        run.log(f"{a.exp} v2 {name}: count MAE {aa['count_mae']:.2f} (bias {aa['count_bias']:.2f}); pooled F1 0.25/0.50/0.75/0.90 "
                f"{aa['pooled_f1@0.25']:.3f}/{aa['pooled_f1@0.50']:.3f}/{aa['pooled_f1@0.75']:.3f}/{aa['pooled_f1@0.90']:.3f}; "
                f"joined {aa['joined_ref']} split {aa['split_pred']} missed {aa['missed']} false {aa['false']}")
    run.save_json("results.json", results)
    run.finish("measured", notes=f"{a.exp} wider clean-up search on saved fold models")


if __name__ == "__main__":
    main()
