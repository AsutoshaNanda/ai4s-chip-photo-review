"""X01 follow-up (improvements.md section 4): small-photo handling and camera-format dependence.

Uses the cached features only (no new image-model runs): A02 whole-photo features (vit_large dinov2 @518) and
X01 patch features (grids 2..6, tiles at native resolution, mean+max pooled per photo). Same 5 practice folds,
same logistic regression as X01 (a_views.fit_eval), so every candidate is paired on identical photos.

Candidates
  fixed_{n}x{n}       whole + patches on the same n x n grid for every photo (the current X01 rule)
  sizeaware_{n}x{n}   large photos n x n; each smaller photo gets the grid whose tiles hold about the same number
                      of pixels: round(n * photo_width / 2048), at least 1. A 1x1 "grid" uses the whole-photo
                      features as its patch (approximation: whole features are resized to long side 518,
                      not native 640 pixels; recorded).
  smallwhole_{n}x{n}  large photos scored by whole + n x n patches; 640x480 photos scored by the whole-photo model
                      (both models trained on all training photos of the fold)
  formatbal_{name}    same features as {name}, training photos weighted so every photo format has equal total weight
                      (weighting = format-balanced sampling in expectation); for whole, fixed 3x3 and fixed 5x5
  whole               whole-photo reference
No candidate is selected here: every one is reported overall, per photo format, per cell type, with full review
(catch) curves, paired group-resampled differences and measured feature-extraction time per photo.
Practice photos only; the final test stays locked (and contains no 640x480 photos).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
import pandas as pd

from a01_simple_baselines import CS, inner_cv
from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, RUNS_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, latest_run

SEED = 20261005
A02_RUN = "20261006-012654_A02_frozen_features"
TILE_RUNS = {2: "20261006-103251_X01_whole_plus_patches", 3: "20261006-103251_X01_whole_plus_patches",
             4: "20261006-103757_X01_whole_plus_patches_grids4-5-6", 5: "20261006-103757_X01_whole_plus_patches_grids4-5-6",
             6: "20261006-103757_X01_whole_plus_patches_grids4-5-6"}
WHOLE = "vit_large_patch14_dinov2@518"
LARGE_W = 2048


def run_dir(name):
    for root in (RUNS_ROOT, CODE_ROOT / "runs"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(name)


def fit_predict(X, prac, weights=None):
    """X01's classifier (a_views.fit_eval) with optional per-photo training weights. Returns p per photo (out of fold)."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p, cut, info = np.zeros(len(prac)), np.zeros(len(prac)), []
    for k in sorted(prac["dev_fold"].unique()):
        tr, te = (prac["dev_fold"] != k).to_numpy(), (prac["dev_fold"] == k).to_numpy()
        ytr, gtr = prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy()
        splits = inner_cv(gtr, ytr)
        fit_kw = {} if weights is None else {"logisticregressioncv__sample_weight": weights[tr]}
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000, n_jobs=-1))
        model.fit(X[tr], ytr, **fit_kw)
        c = float(model[-1].C_[0])
        oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)), X[tr], ytr, cv=splits,
                                method="predict_proba", params=({} if weights is None else {"logisticregression__sample_weight": weights[tr]}))[:, 1]
        cu = C.choose_cutoff(ytr, oof)["cutoff"]
        p[te] = model.predict_proba(X[te])[:, 1]
        cut[te] = cu
        info.append({"fold": int(k), "C": c, "cutoff": cu})
    return p, cut, info


def main():
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = sp[sp.role == "practice"].merge(fmt[["member", "photo_format", "width"]], on="member").reset_index(drop=True)
    run = Run("X01", "small_photos_and_formats", config={"a02_run": A02_RUN, "tile_runs": TILE_RUNS, "whole_features": WHOLE,
                                                         "size_rule": "grid = max(1, round(n * width / 2048))", "photos": "practice only"})
    a02 = run_dir(A02_RUN)
    ids = pd.read_csv(a02 / "features_practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    whole = pd.DataFrame(np.load(a02 / f"features_practice_{WHOLE}.npy").astype(np.float32), index=ids).loc[prac["image_id"]].to_numpy()
    tiles = {n: np.load(run_dir(r) / f"features_tiles_{n}x{n}_meanmax.npy").astype(np.float32) for n, r in TILE_RUNS.items()}
    tiles[1] = np.concatenate([whole, whole], axis=1)          # 1x1 grid: mean = max = the whole photo (approximation, see top)
    width = prac["width"].to_numpy()
    small = (prac["photo_format"] == "RGB/640x480").to_numpy()

    # measured extraction time per image (whole photo and per tile) from the X01 runs
    t_whole, t_tile = [], []
    for r in set(TILE_RUNS.values()):
        ft = json.load(open(run_dir(r) / "feature_timing.json"))
        t_whole.append(ft["whole"]["seconds"] / ft["whole"]["images"])
        t_tile += [v["seconds"] / v["images"] for k, v in ft.items() if k.startswith("tiles_")]
    sec_whole, sec_tile = float(np.mean(t_whole)), float(np.mean(t_tile))

    feats, grid_of = {"whole": whole}, {"whole": np.zeros(len(prac), int)}
    for n in range(2, 7):
        feats[f"fixed_{n}x{n}"] = np.concatenate([whole, tiles[n]], axis=1)
        grid_of[f"fixed_{n}x{n}"] = np.full(len(prac), n)
        g = np.maximum(1, np.round(n * width / LARGE_W).astype(int))
        T = np.stack([tiles[gi][i] for i, gi in enumerate(g)])
        feats[f"sizeaware_{n}x{n}"] = np.concatenate([whole, T], axis=1)
        grid_of[f"sizeaware_{n}x{n}"] = g
        run.log(f"sizeaware_{n}x{n}: grids used by format " + str(pd.Series(g).groupby(prac['photo_format']).agg(lambda s: sorted(set(s))).to_dict()))
    w = 1.0 / prac["photo_format"].map(prac["photo_format"].value_counts()).to_numpy()
    w = w / w.mean()

    preds, info = {}, {}
    for name, X in feats.items():
        preds[name], _, info[name] = fit_predict(X, prac)
    for n in range(2, 7):
        p = preds[f"fixed_{n}x{n}"].copy()
        p[small] = preds["whole"][small]
        preds[f"smallwhole_{n}x{n}"] = p
        grid_of[f"smallwhole_{n}x{n}"] = np.where(small, 0, n)
    for name in ("whole", "fixed_3x3", "fixed_5x5"):
        preds[f"formatbal_{name}"], _, info[f"formatbal_{name}"] = fit_predict(feats[name], prac, weights=w)
        grid_of[f"formatbal_{name}"] = grid_of[name]

    y, grp = prac["label_bad"].to_numpy(), prac["group"].to_numpy()
    P = pd.concat([prac[["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].assign(method=k, p_bad=v)
                   for k, v in preds.items()], ignore_index=True)
    run.save_df("oof_predictions.csv", P)
    res, curves = {"support_by_format": prac.groupby("photo_format").agg(photos=("label_bad", "size"), bad=("label_bad", "sum"),
                                                                       groups=("group", "nunique")).reset_index().to_dict("records")}, []
    for name, p in preds.items():
        g = grid_of[name]
        secs = sec_whole + sec_tile * np.where(g > 0, g ** 2, 0)
        r = {"pooled": C.summary(y, p), "per_photo_format": C.by_group(y, p, prac["photo_format"].to_numpy()),
             "per_cell_type": C.by_group(y, p, prac["cell_type"].to_numpy()),
             "feature_seconds_per_photo_by_format": pd.Series(secs).groupby(prac["photo_format"]).mean().round(4).to_dict()}
        aucs = [v["roc_auc"] for v in r["per_photo_format"].values() if v["roc_auc"] is not None]
        r["worst_format_auc"] = float(min(aucs))
        for ref in ("whole", "fixed_5x5", "fixed_3x3"):
            if name == ref:
                continue
            pr = preds[ref]
            out = {}
            for scope, mask in [("pooled", np.ones(len(y), bool))] + [(f, (prac["photo_format"] == f).to_numpy()) for f in sorted(prac["photo_format"].unique())]:
                yy, gg, a, b = y[mask], grp[mask], pr[mask], p[mask]
                vals = C.grouped_bootstrap(lambda idx: (None if C.roc_auc(yy[idx], b[idx]) is None else C.roc_auc(yy[idx], b[idx]) - C.roc_auc(yy[idx], a[idx])),
                                           gg, n_boot=1000, seed=SEED)
                out[scope] = {"difference": float(C.roc_auc(yy, b) - C.roc_auc(yy, a)), "n_groups": int(len(np.unique(gg))),
                              "n_bad": int(yy.sum()),
                              "range_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None}
            r[f"paired_minus_{ref}"] = out
        res[name] = r
        for f, d in [("ALL", prac.index.to_numpy())] + [(f, np.flatnonzero((prac["photo_format"] == f).to_numpy())) for f in sorted(prac["photo_format"].unique())]:
            c = C.review_curve(y[d], p[d])["catch_rate"]
            curves.append(pd.DataFrame({"method": name, "photo_format": f, "fraction_reviewed": np.arange(len(c)) / (len(c) - 1), "catch_rate": c}))
        fm = {k: (round(v["roc_auc"], 3) if v["roc_auc"] is not None else None) for k, v in r["per_photo_format"].items()}
        run.log(f"{name:26s} AUC {r['pooled']['roc_auc']:.3f} logloss {r['pooled']['log_loss']:.3f} | per format {fm} | "
                f"worst {r['worst_format_auc']:.3f} | s/photo {r['feature_seconds_per_photo_by_format']}")
    run.save_df("review_curves.csv", pd.concat(curves, ignore_index=True))
    run.save_json("fold_settings.json", info)
    run.save_json("results.json", res)
    run.finish("measured", notes="X01 small-photo handling and format dependence on cached features; no selection; practice only")


if __name__ == "__main__":
    main()
