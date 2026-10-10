"""Chip-photo experiments built on A02's frozen features (practice photos only; final test untouched).

  A03  unseen cell types: train on 5 cell types, test on the 6th (every type), vs seen-type grouped folds
  A06  class weighting: balanced vs none, same folds
  A07  photos + sample details: photo only / details only / both / both with details missing at test time
  A08  confidence correction: none vs Platt vs isotonic, group-aware cross-fitting over folds
  A09  review order: random / blur rule / bad-probability / proposed (probability + model disagreement),
       full catch curves from 0 to 100% reviewed, per cell type and per photo format
  A10  two models: joint failures, equal-weight average, disagreement guard
  A11  how many labels: nested whole-group subsets, random vs active (uncertainty) selection, equal budgets
Every model step is learned inside training folds. Results reported overall, per cell type, per photo format.
Feature set used where one is needed: the A02 set with the lowest out-of-fold log loss (rule fixed in advance);
A03/A06/A07 are also run for every A02 feature set.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from a01_simple_baselines import CS, inner_cv
from ai4s import metrics_cls as C
from ai4s import ooc
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run, latest_run

SEED = 20261005


def load():
    a02 = latest_run("A02_frozen_features")
    ids = pd.read_csv(a02 / "features_practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    feats = {p.name[len("features_practice_"):-4]: np.load(p).astype(np.float32) for p in a02.glob("features_practice_*.npy")}
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = sp[sp.role == "practice"].merge(fmt[["member", "photo_format"]], on="member")
    meta = ooc.load_metadata()[["image_id", "seeding_density", "hours_after_seeding", "day", "flow_rate_ul_per_min"]]
    prac = prac.merge(meta, on="image_id", how="left")
    prac["log10_seeding_density"] = np.log10(prac["seeding_density"])
    prac = prac.set_index("image_id").loc[ids].reset_index()
    oof = pd.read_csv(a02 / "oof_predictions.csv", dtype={"image_id": str})
    return a02, prac, feats, oof


def best_feature_set(oof):
    ll = {m: C.log_loss(d["label_bad"], d["p_bad"]) for m, d in oof.groupby("method")}
    return min(ll, key=ll.get), ll


def fit_lr(Xtr, ytr, gtr, class_weight=None):
    splits = inner_cv(gtr, ytr)
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000,
                                                             class_weight=class_weight, n_jobs=-1))
    m.fit(Xtr, ytr)
    c = float(m[-1].C_[0])
    inner = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight=class_weight)),
                              Xtr, ytr, cv=splits, method="predict_proba")[:, 1]
    return m, C.choose_cutoff(ytr, inner)["cutoff"]


def report(df, run, name):
    y, p, g = df["label_bad"].to_numpy(), df["p_bad"].to_numpy(), df["group"].to_numpy()
    res = {"pooled": C.summary(y, p), "per_cell_type": C.by_group(y, p, df["cell_type"].to_numpy()),
           "per_photo_format": C.by_group(y, p, df["photo_format"].to_numpy())}
    vals = C.grouped_bootstrap(lambda idx: C.roc_auc(y[idx], p[idx]), g, n_boot=1000, seed=SEED)
    res["roc_auc_bootstrap_95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None
    if "pred_bad" in df:
        t = []
        for ct, d in list(df.groupby("cell_type")) + [("ALL", df)]:
            yy, yh = d["label_bad"].to_numpy(), d["pred_bad"].to_numpy()
            nb, ng = (yy == 1).sum(), (yy == 0).sum()
            t.append({"cell_type": ct, "share_bad_found": float(((yy == 1) & (yh == 1)).sum() / nb) if nb else None,
                      "share_good_passed": float(((yy == 0) & (yh == 0)).sum() / ng) if ng else None})
        res["found_shares"] = t
    s = res["pooled"]
    fm = {k: (round(v["roc_auc"], 3) if v["roc_auc"] is not None else None) for k, v in res["per_photo_format"].items()}
    run.log(f"{name:46s} AUC {s['roc_auc']:.3f} {np.round(res['roc_auc_bootstrap_95'], 3).tolist() if res['roc_auc_bootstrap_95'] else ''} "
            f"logloss {s['log_loss']:.3f} | within format {fm}")
    return res


def cv_predict(X, prac, class_weight=None, mask=None):
    rows = []
    for k in sorted(prac["dev_fold"].unique()):
        tr = (prac["dev_fold"] != k).to_numpy()
        te = (prac["dev_fold"] == k).to_numpy()
        m, cut = fit_lr(X[tr], prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy(), class_weight)
        Xte = X[te] if mask is None else mask(X[te])
        p = m.predict_proba(Xte)[:, 1]
        d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].copy()
        d["p_bad"], d["pred_bad"] = p, (p >= cut).astype(int)
        rows.append(d)
    return pd.concat(rows)


def a03(run, prac, feats):
    out = {}
    for tag, X in feats.items():
        rows = []
        for ct in sorted(prac["cell_type"].unique()):
            tr = (prac["cell_type"] != ct).to_numpy()
            te = ~tr
            m, cut = fit_lr(X[tr], prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy())
            p = m.predict_proba(X[te])[:, 1]
            d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].copy()
            d["p_bad"], d["pred_bad"] = p, (p >= cut).astype(int)
            rows.append(d)
        P = pd.concat(rows)
        run.save_df(f"unseen_type_predictions_{tag}.csv", P)
        out[tag] = report(P, run, f"A03 unseen cell type | {tag}")
    return out


def a06(run, prac, feats):
    out = {}
    for tag, X in feats.items():
        for cw in (None, "balanced"):
            P = cv_predict(X, prac, class_weight=cw)
            run.save_df(f"predictions_{tag}_{cw}.csv", P)
            out[f"{tag}|{cw}"] = report(P, run, f"A06 weighting={cw} | {tag}")
    return out


def a07(run, prac, feats):
    num = ["log10_seeding_density", "hours_after_seeding", "day", "flow_rate_ul_per_min"]
    out = {}
    for tag, X in feats.items():
        fcols = [f"f{i}" for i in range(X.shape[1])]
        D = pd.concat([pd.DataFrame(X, columns=fcols), prac[num + ["cell_type"]].reset_index(drop=True)], axis=1)
        for variant in ("photo_only", "details_only", "both", "both_details_missing_at_test"):
            cols_num = (fcols if variant != "details_only" else []) + (num if variant != "photo_only" else [])
            cols_cat = [] if variant == "photo_only" else ["cell_type"]
            rows = []
            for k in sorted(prac["dev_fold"].unique()):
                tr = (prac["dev_fold"] != k).to_numpy()
                te = ~tr
                pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), cols_num),
                                         ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat)])
                ytr, gtr = prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy()
                m = make_pipeline(pre, LogisticRegressionCV(Cs=CS, cv=inner_cv(gtr, ytr), scoring="neg_log_loss", max_iter=5000, n_jobs=-1))
                m.fit(D[tr], ytr)
                Dte = D[te].copy()
                if variant == "both_details_missing_at_test":
                    Dte[num] = np.nan
                    Dte["cell_type"] = "missing"
                d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].copy()
                d["p_bad"] = m.predict_proba(Dte)[:, 1]
                rows.append(d)
            P = pd.concat(rows)
            run.save_df(f"predictions_{tag}_{variant}.csv", P)
            out[f"{tag}|{variant}"] = report(P, run, f"A07 {variant} | {tag}")
    return out


def a08(run, prac, oof, tag):
    d = oof[oof["method"] == tag].copy()
    if "photo_format" not in d:
        d = d.merge(prac[["image_id", "photo_format"]], on="image_id", how="left")
    res = {"none": d["p_bad"].to_numpy()}
    for name in ("platt", "isotonic"):
        cal = np.zeros(len(d))
        for k in sorted(d["dev_fold"].unique()):
            tr, te = (d["dev_fold"] != k).to_numpy(), (d["dev_fold"] == k).to_numpy()
            p_tr, y_tr = d.loc[tr, "p_bad"].to_numpy(), d.loc[tr, "label_bad"].to_numpy()
            if name == "platt":
                z = np.log(np.clip(p_tr, 1e-6, 1 - 1e-6) / (1 - np.clip(p_tr, 1e-6, 1 - 1e-6)))[:, None]
                lr = LogisticRegression(C=1e6).fit(z, y_tr)
                zt = np.log(np.clip(d.loc[te, "p_bad"], 1e-6, 1 - 1e-6) / (1 - np.clip(d.loc[te, "p_bad"], 1e-6, 1 - 1e-6))).to_numpy()[:, None]
                cal[te] = lr.predict_proba(zt)[:, 1]
            else:
                cal[te] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p_tr, y_tr).predict(d.loc[te, "p_bad"])
        res[name] = cal
    out = {}
    for name, p in res.items():
        dd = d.assign(p_bad=p)
        r = report(dd, run, f"A08 calibration={name} | {tag}")
        r["reliability"] = {nb: _reliability(dd["label_bad"].to_numpy(), p, nb) for nb in (5, 10, 20)}
        out[name] = r
        run.save_df(f"calibrated_{name}.csv", dd)
    return out


def _reliability(y, p, nbins):
    q = np.quantile(p, np.linspace(0, 1, nbins + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nbins - 1)
    rows = [{"bin": i, "n": int((b == i).sum()), "mean_pred": float(p[b == i].mean()), "share_bad": float(y[b == i].mean())}
            for i in range(nbins) if (b == i).any()]
    ece = sum(r["n"] * abs(r["mean_pred"] - r["share_bad"]) for r in rows) / len(y)
    return {"bins": rows, "ece": float(ece)}


def a09_a10(run, prac, oof, tags, exp):
    a01 = latest_run("A01_simple_baselines")
    stats = pd.read_csv(a01 / "oof_predictions.csv", dtype={"image_id": str})
    feats = pd.read_parquet(Path(ooc.OOC_DIR) / "features" / "photo_stats_v1.parquet") if (Path(ooc.OOC_DIR) / "features" / "photo_stats_v1.parquet").exists() else None
    wide = oof.pivot_table(index="image_id", columns="method", values="p_bad")
    base = prac.set_index("image_id")[["group", "cell_type", "photo_format", "label_bad", "dev_fold"]].join(wide, how="inner")
    t1, t2 = tags[0], tags[1]
    base["disagreement"] = (base[t1] - base[t2]).abs()
    base["average"] = (base[t1] + base[t2]) / 2
    # proposed score: learned on training folds from (p1, p2, disagreement), applied out of fold
    prop = np.zeros(len(base))
    for k in sorted(base["dev_fold"].unique()):
        tr, te = (base["dev_fold"] != k).to_numpy(), (base["dev_fold"] == k).to_numpy()
        Z = base[[t1, t2, "disagreement"]].to_numpy()
        prop[te] = LogisticRegression(max_iter=1000).fit(Z[tr], base.loc[tr, "label_bad"]).predict_proba(Z[te])[:, 1]
    base["proposed"] = prop
    if feats is not None and "fixed_lap_var" in feats.columns:
        fmap = feats.assign(image_id=feats["member"].str.rsplit("/", n=1).str[1].str.rsplit(".", n=1).str[0]).set_index("image_id")["fixed_lap_var"]
        base["blur_rule"] = -np.log(base.index.map(fmap).astype(float) + 1e-9)   # blurrier first (sign fixed in advance: blur is a listed defect)
    out = {}
    if exp == "A09":
        orders = [c for c in ["blur_rule", t1, t2, "average", "proposed"] if c in base]
        y = base["label_bad"].to_numpy()
        curves = {}
        for o in orders:
            curves[o] = C.review_curve(y, base[o].to_numpy())["catch_rate"]
        rng = np.random.default_rng(SEED)
        rand = np.stack([C.review_curve(y, rng.random(len(y)))["catch_rate"] for _ in range(200)])
        curves["random_mean"], curves["random_p05"], curves["random_p95"] = rand.mean(0), np.percentile(rand, 5, 0), np.percentile(rand, 95, 0)
        n = len(y)
        cdf = pd.DataFrame({k: v for k, v in curves.items()})
        cdf.insert(0, "fraction_reviewed", np.arange(n + 1) / n)
        run.save_df("catch_curves_all.csv", cdf)
        for o in orders:
            area = float(np.mean(curves[o][1:]))
            g = base["group"].to_numpy()
            vals = C.grouped_bootstrap(lambda idx: C.review_area(y[idx], base[o].to_numpy()[idx]) -
                                       C.review_area(y[idx], base[t1].to_numpy()[idx]), g, n_boot=1000, seed=SEED)
            out[o] = {"review_area": area, "minus_bad_probability_order_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                      "per_cell_type": {ct: C.review_area(d["label_bad"].to_numpy(), d[o].to_numpy()) for ct, d in base.groupby("cell_type")},
                      "per_photo_format": {f: C.review_area(d["label_bad"].to_numpy(), d[o].to_numpy()) for f, d in base.groupby("photo_format")}}
            run.log(f"A09 order={o:20s} review area {area:.3f}; difference vs {t1} 95%: {np.round(out[o]['minus_bad_probability_order_95'], 3).tolist()}")
        out["random_review_area"] = float(np.mean(curves["random_mean"][1:]))
    else:
        y = base["label_bad"].to_numpy()
        e1 = ((base[t1] >= 0.5).astype(int) != y)
        e2 = ((base[t2] >= 0.5).astype(int) != y)
        out["error_overlap"] = {"only_" + t1: int((e1 & ~e2).sum()), "only_" + t2: int((~e1 & e2).sum()), "both_wrong": int((e1 & e2).sum()),
                                "both_right": int((~e1 & ~e2).sum()), "note": "0.5 used only to tabulate error overlap; scores below are threshold-free"}
        for name in (t1, t2, "average", "proposed"):
            out[name] = report(base.reset_index().rename(columns={name: "p_bad"}).assign(group=base["group"].to_numpy()), run, f"A10 {name}")
        q = base["disagreement"].to_numpy()
        out["guard_curve"] = [{"flag_share": float(s), "error_rate_unflagged": float(e1[q < np.quantile(q, 1 - s)].mean()) if s < 1 else None}
                              for s in np.linspace(0, 0.9, 10)]
        run.log(f"A10 error overlap: {out['error_overlap']}")
    return out


def a11(run, prac, X, tag, repeats=5):
    out = []
    groups_all = prac["group"].to_numpy()
    for k in sorted(prac["dev_fold"].unique()):
        pool = (prac["dev_fold"] != k).to_numpy()
        te = ~pool
        pool_groups = np.unique(groups_all[pool])
        sizes = sorted({int(x) for x in np.unique(np.round(np.geomspace(4, len(pool_groups), 6)))})
        for rep in range(repeats):
            rng = np.random.default_rng(SEED + 31 * rep + k)
            order = rng.permutation(pool_groups)
            for strategy in ("random", "active"):
                chosen = list(order[:sizes[0]])
                for size in sizes:
                    if strategy == "random":
                        chosen = list(order[:size])
                    else:
                        while len(chosen) < size:
                            tr = np.isin(groups_all, chosen) & pool
                            if len(np.unique(prac.loc[tr, "label_bad"])) < 2:
                                chosen.append(next(gg for gg in order if gg not in chosen))
                                continue
                            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], prac.loc[tr, "label_bad"])
                            rest = [gg for gg in pool_groups if gg not in chosen]
                            unc = {gg: float(np.mean(1 - np.abs(2 * m.predict_proba(X[groups_all == gg])[:, 1] - 1))) for gg in rest}
                            chosen.append(max(unc, key=unc.get))
                    tr = np.isin(groups_all, chosen) & pool
                    y_tr = prac.loc[tr, "label_bad"].to_numpy()
                    if len(np.unique(y_tr)) < 2:
                        out.append({"fold": int(k), "repeat": rep, "strategy": strategy, "groups": size, "photos": int(tr.sum()), "auc": None})
                        continue
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y_tr)
                    p = m.predict_proba(X[te])[:, 1]
                    out.append({"fold": int(k), "repeat": rep, "strategy": strategy, "groups": size, "photos": int(tr.sum()),
                                "auc": C.roc_auc(prac.loc[te, "label_bad"], p), "log_loss": C.log_loss(prac.loc[te, "label_bad"], p)})
    df = pd.DataFrame(out)
    run.save_df(f"label_efficiency_{tag}.csv", df)
    summ = df.groupby(["strategy", "groups"])[["photos", "auc", "log_loss"]].agg(["mean", "std"]).round(3)
    run.log(f"A11 label efficiency ({tag}; regularisation fixed at C=0.1 for all sizes, recorded):\n{summ.to_string()}")
    flat = summ.copy()
    flat.columns = ["_".join(c) for c in flat.columns]
    return flat.reset_index().to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    a02, prac, feats, oof = load()
    best, ll = best_feature_set(oof)
    tags = sorted(ll, key=ll.get)
    for exp in a.exp.split(","):
        names = {"A03": "unseen_cell_types", "A06": "class_weighting", "A07": "photos_plus_details", "A08": "calibration",
                 "A09": "review_order", "A10": "two_models_guard", "A11": "label_efficiency"}
        run = Run(exp, names[exp], config={"a02_run": a02.name, "feature_sets": list(feats), "best_by_oof_log_loss": best,
                                           "oof_log_loss": ll, "photos": "practice only"})
        if exp == "A03":
            res = a03(run, prac, feats)
        elif exp == "A06":
            res = a06(run, prac, feats)
        elif exp == "A07":
            res = a07(run, prac, feats)
        elif exp == "A08":
            res = a08(run, prac, oof, best)
        elif exp in ("A09", "A10"):
            res = a09_a10(run, prac, oof, [tags[0], next(t for t in tags[1:] if t.split("@")[0] != tags[0].split("@")[0])], exp)
        elif exp == "A11":
            res = a11(run, prac, feats[best], best)
        run.save_json("results.json", res)
        run.finish("measured", notes=f"{exp} on A02 features ({a02.name}); practice only")


if __name__ == "__main__":
    main()
