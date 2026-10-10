"""A08 / A09 / A10 for the chip-photo candidates (improvements.md section 6), from saved development predictions only.

Candidates (all on the same 5 practice folds, same photos; saved out-of-fold predictions):
  whole, fixed_3x3, fixed_5x5, sizeaware_*, smallwhole_*   X01 small-photo run (x01_small.py)
  frozen_LR_reference                                       A02 frozen whole-photo model
  head_only / last_block                                    A02 light retraining (if its full run exists)
Simple reference orderings: A01 photo_stats (simple photo measurements), A01 format_only, and random order.

A08  reliability of the raw scores as probabilities: reliability bins (5, 10, 20), expected calibration error, Brier, log loss,
     overall and per photo format, plus Platt and isotonic corrections fitted inside training folds only. A displayed
     probability is NOT advertised unless the raw score is reliable in the weakest group; this only reports the evidence.
A09  full bad-photo catch curves from 0 to 100% of photos reviewed (overall, per photo format, per cell type) for each
     candidate against the simple orderings and random order (mean and 5-95% spread over 200 random orders); review area
     with paired group-resampled differences.
A10  error overlap between candidates (do saved failures differ?), and the disagreement guard curve for pairs; no
     combination is adopted here, because validating a complete rule needs a fresh test.
No cut-off is chosen here. Practice photos only.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, RUNS_ROOT
from ai4s.runlog import Run, latest_run

SEED = 20261005


def load_all(run):
    P = {}
    x01 = latest_run("X01_small_photos_and_formats")
    d = pd.read_csv(x01 / "oof_predictions.csv", dtype={"image_id": str})
    for m, g in d.groupby("method"):
        P[m] = g.set_index("image_id")
    a02 = pd.read_csv(latest_run("A02_frozen_features") / "oof_predictions.csv", dtype={"image_id": str})
    P["frozen_LR_reference"] = a02[a02["method"] == "vit_large_patch14_dinov2@518"].set_index("image_id")
    a01 = pd.read_csv(latest_run("A01_simple_baselines") / "oof_predictions.csv", dtype={"image_id": str})
    simple = {"simple_photo_stats": a01[a01["method"] == "photo_stats"].set_index("image_id"),
              "simple_format_only": a01[a01["method"] == "format_only"].set_index("image_id")}
    try:
        lr = pd.read_csv(latest_run("A02_light_retrain_full") / "oof_predictions.csv", dtype={"image_id": str})
        for m, g in lr.groupby("method"):
            P[m.replace("vit_large_dinov2@518_", "retrain_")] = g.set_index("image_id")
        run.log("light-retraining predictions included")
    except FileNotFoundError:
        run.log("light-retraining full run not available yet: candidates here exclude it")
    return P, simple


def reliability(y, p, nbins):
    q = np.quantile(p, np.linspace(0, 1, nbins + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nbins - 1)
    rows = [{"bin": i, "n": int((b == i).sum()), "mean_pred": float(p[b == i].mean()), "share_bad": float(y[b == i].mean())}
            for i in range(nbins) if (b == i).any()]
    return {"bins": rows, "ece": float(sum(r["n"] * abs(r["mean_pred"] - r["share_bad"]) for r in rows) / len(y))}


def calibrate(d, kind):
    out = np.zeros(len(d))
    p = np.clip(d["p_bad"].to_numpy(), 1e-6, 1 - 1e-6)
    for k in sorted(d["dev_fold"].unique()):
        tr, te = (d["dev_fold"] != k).to_numpy(), (d["dev_fold"] == k).to_numpy()
        if kind == "platt":
            z = np.log(p / (1 - p))[:, None]
            out[te] = LogisticRegression(C=1e6).fit(z[tr], d["label_bad"][tr]).predict_proba(z[te])[:, 1]
        else:
            out[te] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[tr], d["label_bad"][tr]).predict(p[te])
    return out


def main():
    run = Run("A08", "reliability_review_and_overlap", config={"also": ["A09", "A10"], "seed": SEED, "random_orders": 200, "photos": "practice only"})
    P, simple = load_all(run)
    ids = P["whole"].index
    base = P["whole"][["group", "cell_type", "photo_format", "label_bad", "dev_fold"]]
    y = base["label_bad"].to_numpy()
    res = {"A08": {}, "A09": {}, "A10": {}}
    # ---------------- A08
    for name, d in P.items():
        d = d.loc[ids]
        r = {"raw": {"summary": C.summary(y, d["p_bad"].to_numpy()), "reliability": {nb: reliability(y, d["p_bad"].to_numpy(), nb) for nb in (5, 10, 20)},
                     "per_photo_format": {f: {"ece_10": reliability(y[m], d.loc[m, "p_bad"].to_numpy(), 10)["ece"] if m.sum() >= 20 else None, "n": int(m.sum()),
                                              "brier": C.brier(y[m], d.loc[m, "p_bad"].to_numpy())}
                                          for f, m in ((f, (base["photo_format"] == f).to_numpy()) for f in sorted(base["photo_format"].unique()))}}}
        for kind in ("platt", "isotonic"):
            pc = calibrate(d.assign(label_bad=y), kind)
            r[kind] = {"summary": C.summary(y, pc), "ece_10": reliability(y, pc, 10)["ece"]}
        res["A08"][name] = r
        run.log(f"A08 {name:28s} raw ECE(10 bins) {r['raw']['reliability'][10]['ece']:.3f} brier {r['raw']['summary']['brier']:.3f} logloss {r['raw']['summary']['log_loss']:.3f} | "
                f"Platt logloss {r['platt']['summary']['log_loss']:.3f} ECE {r['platt']['ece_10']:.3f} | isotonic logloss {r['isotonic']['summary']['log_loss']:.3f} ECE {r['isotonic']['ece_10']:.3f}")
    # ---------------- A09
    orders = {**{k: v.loc[ids, "p_bad"].to_numpy() for k, v in P.items()}, **{k: v.loc[ids, "p_bad"].to_numpy() for k, v in simple.items()}}
    rng = np.random.default_rng(SEED)
    rand = np.stack([C.review_curve(y, rng.random(len(y)))["catch_rate"] for _ in range(200)])
    curves = [pd.DataFrame({"order": "random_mean", "subset": "ALL", "fraction_reviewed": np.arange(len(y) + 1) / len(y), "catch_rate": rand.mean(0)}),
              pd.DataFrame({"order": "random_p05", "subset": "ALL", "fraction_reviewed": np.arange(len(y) + 1) / len(y), "catch_rate": np.percentile(rand, 5, 0)}),
              pd.DataFrame({"order": "random_p95", "subset": "ALL", "fraction_reviewed": np.arange(len(y) + 1) / len(y), "catch_rate": np.percentile(rand, 95, 0)})]
    subsets = [("ALL", np.ones(len(y), bool))] + [(f"format={f}", (base["photo_format"] == f).to_numpy()) for f in sorted(base["photo_format"].unique())] \
        + [(f"type={t}", (base["cell_type"] == t).to_numpy()) for t in sorted(base["cell_type"].unique())]
    for name, s in orders.items():
        area = {}
        for sub, m in subsets:
            if m.sum() < 20 or len(np.unique(y[m])) < 2:
                continue
            c = C.review_curve(y[m], s[m])["catch_rate"]
            curves.append(pd.DataFrame({"order": name, "subset": sub, "fraction_reviewed": np.arange(len(c)) / (len(c) - 1), "catch_rate": c}))
            area[sub] = C.review_area(y[m], s[m])
        res["A09"][name] = {"review_area": area}
    ref = "simple_photo_stats"
    g = base["group"].to_numpy()
    for name in P:
        vals = C.grouped_bootstrap(lambda idx: C.review_area(y[idx], orders[name][idx]) - C.review_area(y[idx], orders[ref][idx]), g, n_boot=1000, seed=SEED)
        res["A09"][name]["minus_simple_photo_stats_95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        run.log(f"A09 {name:28s} review area {res['A09'][name]['review_area']['ALL']:.3f}; minus simple photo-stats order 95% {np.round(res['A09'][name]['minus_simple_photo_stats_95'], 3).tolist()}")
    res["A09"]["random_review_area"] = float(np.mean(rand.mean(0)[1:]))
    run.log(f"A09 random order review area {res['A09']['random_review_area']:.3f}; simple photo-stats {res['A09'][ref]['review_area']['ALL']:.3f}")
    run.save_df("catch_curves.csv", pd.concat(curves, ignore_index=True))
    # ---------------- A10
    names = list(P)
    for a_, b_ in itertools.combinations(names, 2):
        pa, pb = P[a_].loc[ids, "p_bad"].to_numpy(), P[b_].loc[ids, "p_bad"].to_numpy()
        ea, eb = (pa >= 0.5).astype(int) != y, (pb >= 0.5).astype(int) != y
        res["A10"][f"{a_}|{b_}"] = {"only_a_wrong": int((ea & ~eb).sum()), "only_b_wrong": int((~ea & eb).sum()), "both_wrong": int((ea & eb).sum()),
                                    "rank_correlation_of_scores": float(pd.Series(pa).corr(pd.Series(pb), method="spearman")),
                                    "note": "0.5 only tabulates error overlap; scores are threshold-free elsewhere"}
        avg = (pa + pb) / 2
        res["A10"][f"{a_}|{b_}"]["average_auc"] = C.roc_auc(y, avg)
        res["A10"][f"{a_}|{b_}"]["auc_a"], res["A10"][f"{a_}|{b_}"]["auc_b"] = C.roc_auc(y, pa), C.roc_auc(y, pb)
    key = "whole|fixed_5x5"
    if key in res["A10"]:
        run.log(f"A10 example {key}: {res['A10'][key]}")
    run.save_json("results.json", res)
    run.finish("measured", notes="A08/A09/A10 for chip candidates from saved development predictions; practice only")


if __name__ == "__main__":
    main()
