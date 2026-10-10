"""A01: simple methods and possible shortcuts, on the locked practice folds (final test untouched).

Methods (identical folds for all):
  majority      -> the training fold's bad-photo rate as everyone's score
  details_only  -> logistic regression on sample details (cell type, seeding density, hours, day, flow rate,
                   with "missing" flags). Never ids, prefixes, paths or the decision column.
  photo_stats   -> logistic regression on simple photo measurements (brightness, contrast, sharpness, texture)
Every data-learned step (filling gaps, scaling, regularisation strength) is learned inside each training fold;
the regularisation strength is chosen by grouped inner folds with log loss (no cut-off needed).
Reported: pooled out-of-fold threshold-free scores, per fold, per cell type, review curves, and grouped
bootstrap intervals (whole groups resampled). The final-test photos are not read at all.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai4s import imgfeat, ooc
from ai4s import metrics_cls as C
from ai4s.paths import OOC_DIR, SPLITS_ROOT
from ai4s.runlog import Run

CS = np.logspace(-4, 4, 17)
SEED = 20261005
FEAT_CACHE = OOC_DIR / "features" / "photo_stats_v1.parquet"


def _measure(member):
    t0 = time.perf_counter()
    arr = np.asarray(ooc.read_image(member))
    f = imgfeat.measure(arr)
    f["member"] = member
    f["_seconds"] = time.perf_counter() - t0
    return f


def photo_stats(members, run):
    if FEAT_CACHE.exists():
        cached = pd.read_parquet(FEAT_CACHE)
        if set(members) <= set(cached["member"]):
            run.log(f"photo stats from cache {FEAT_CACHE}")
            return cached[cached["member"].isin(members)]
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_measure, members, chunksize=8))
    feats = pd.DataFrame(rows)
    run.log(f"photo stats for {len(feats)} photos in {time.time() - t0:.0f} s ({workers} processes); "
            f"median {feats['_seconds'].median():.2f} s/photo per process")
    feats.to_parquet(FEAT_CACHE, index=False)
    return feats


def inner_cv(groups, y, n=4):
    sgk = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    return list(sgk.split(np.zeros(len(y)), y, groups))


def _pre(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="drop")


def fit_predict(Xtr, ytr, gtr, Xte, num_cols, cat_cols):
    """Fit on the training fold; regularisation chosen by grouped inner folds (log loss).
    The cut-off for yes/no answers is chosen on the training fold's inner out-of-fold scores
    (rule: highest balanced accuracy), never on the fold being tested."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    t0 = time.perf_counter()
    splits = inner_cv(gtr, ytr)
    model = make_pipeline(_pre(num_cols, cat_cols),
                          LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000, n_jobs=1))
    model.fit(Xtr, ytr)
    c = float(model[-1].C_[0])
    fit_s = time.perf_counter() - t0
    inner = make_pipeline(_pre(num_cols, cat_cols), LogisticRegression(C=c, max_iter=5000))
    oof_tr = cross_val_predict(inner, Xtr, ytr, cv=splits, method="predict_proba")[:, 1]
    cut = C.choose_cutoff(ytr, oof_tr)
    t1 = time.perf_counter()
    p = model.predict_proba(Xte)[:, 1]
    pred_s = time.perf_counter() - t1
    return p, c, cut["cutoff"], fit_s, pred_s / max(len(Xte), 1)


def main():
    run = Run("A01", "simple_baselines", config={"Cs": CS.tolist(), "seed": SEED, "splits": str(SPLITS_ROOT / "ooc_splits.csv")})
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"prefix": str, "image_id": str, "group": str})
    prac = sp[sp["role"] == "practice"].copy()
    meta = ooc.load_metadata()
    prac = prac.merge(meta[["image_id", "seeding_density", "hours_after_seeding", "day", "flow_rate_ul_per_min"]],
                      on="image_id", how="left")
    prac["log10_seeding_density"] = np.log10(prac["seeding_density"])
    feats = photo_stats(prac["member"].tolist(), run)
    stat_cols = [c for c in feats.columns if c not in ("member", "_seconds")]
    prac = prac.merge(feats, on="member", how="left")

    # photo format (mode + size) measured in G01: a possible shortcut (G01 found format tracks the label)
    g01 = sorted(p for p in (Path(__file__).resolve().parents[1] / "runs").iterdir() if "G01_ooc_check" in p.name)[-1]
    fmt = pd.read_csv(g01 / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = prac.merge(fmt[["member", "photo_format"]], on="member", how="left")
    allfmt = sp.merge(fmt[["member", "photo_format"]], on="member", how="left")
    run.save_json("format_mix.json", {r: allfmt[allfmt.role == r]["photo_format"].value_counts().to_dict()
                                      for r in ("practice", "final_test")})

    detail_num = ["log10_seeding_density", "hours_after_seeding", "day", "flow_rate_ul_per_min"]
    fixed_cols = [c for c in stat_cols if not c.startswith("native_")]   # size-normalised measures only
    methods = {"details_only": (detail_num, ["cell_type"]), "photo_stats": (stat_cols, []),
               "photo_stats_fixed_size": (fixed_cols, []), "format_only": ([], ["photo_format"])}
    preds = []
    chosen_C = []
    for k in sorted(prac["dev_fold"].unique()):
        tr, te = prac[prac.dev_fold != k], prac[prac.dev_fold == k]
        base = pd.DataFrame({"image_id": te["image_id"], "group": te["group"], "cell_type": te["cell_type"],
                             "photo_format": te["photo_format"], "label_bad": te["label_bad"], "dev_fold": k})
        rate = float(tr["label_bad"].mean())
        preds.append(base.assign(method="majority", p_bad=rate, pred_bad=int(rate >= 0.5), cutoff=np.nan))
        for name, (num, cat) in methods.items():
            p, c, cut, fit_s, pred_s = fit_predict(tr, tr["label_bad"].to_numpy(), tr["group"].to_numpy(), te, num, cat)
            preds.append(base.assign(method=name, p_bad=p, pred_bad=(p >= cut).astype(int), cutoff=cut))
            chosen_C.append({"fold": int(k), "method": name, "C": c, "cutoff_from_training_fold": cut,
                             "fit_seconds": fit_s, "predict_seconds_per_photo": pred_s})
    P = pd.concat(preds, ignore_index=True)
    run.save_df("oof_predictions.csv", P)
    run.save_json("chosen_regularisation_cutoffs_timing.json", chosen_C)

    # tables of right/wrong answers (cut-offs chosen on training folds only), overall and per cell type
    tables = []
    for (name, ct), d in list(P.groupby(["method", "cell_type"])) + [((n, "ALL"), x) for n, x in P.groupby("method")]:
        y, yh = d["label_bad"].to_numpy(), d["pred_bad"].to_numpy()
        tp, fn = int(((y == 1) & (yh == 1)).sum()), int(((y == 1) & (yh == 0)).sum())
        tn, fp = int(((y == 0) & (yh == 0)).sum()), int(((y == 0) & (yh == 1)).sum())
        sens = tp / (tp + fn) if tp + fn else None
        spec = tn / (tn + fp) if tn + fp else None
        tables.append({"method": name, "cell_type": ct, "bad_caught": tp, "bad_missed_false_pass": fn,
                       "good_passed": tn, "good_flagged": fp, "share_bad_found": sens, "share_good_found": spec,
                       "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None})
    run.save_df("right_wrong_tables.csv", pd.DataFrame(tables))

    results = {}
    for name, d in P.groupby("method"):
        y, p, g = d["label_bad"].to_numpy(), d["p_bad"].to_numpy(), d["group"].to_numpy()
        res = {"pooled": C.summary(y, p),
               "per_fold": {int(k): C.summary(x["label_bad"], x["p_bad"]) for k, x in d.groupby("dev_fold")},
               "per_cell_type": C.by_group(y, p, d["cell_type"].to_numpy()),
               "per_photo_format": C.by_group(y, p, d["photo_format"].to_numpy())}
        for metric, fn in [("roc_auc", C.roc_auc), ("avg_precision_bad", C.average_precision), ("log_loss", C.log_loss),
                           ("review_area", C.review_area)]:
            vals = C.grouped_bootstrap(lambda idx: fn(y[idx], p[idx]), g, n_boot=2000, seed=SEED)
            res[f"{metric}_bootstrap_95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None
        rc = C.review_curve(y, p)
        pd.DataFrame({k: rc[k] for k in rc}).to_csv(run.path(f"review_curve_{name}.csv"), index=False)
        results[name] = res
        s = res["pooled"]
        run.log(f"{name}: AUC {s['roc_auc']}, AP(bad) {s['avg_precision_bad']}, log loss {s['log_loss']:.4f}, "
                f"Brier {s['brier']:.4f}, review area {s['review_area']}")
    run.save_json("results.json", results)
    run.finish("measured", notes="A01 on practice folds; final test untouched")


if __name__ == "__main__":
    main()
