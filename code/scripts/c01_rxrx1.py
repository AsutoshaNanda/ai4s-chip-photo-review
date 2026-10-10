"""C01: can ready-made cell measurements (RxRx1 deep-learning embeddings, 128 numbers per image site) tell
treatments (siRNAs) apart across lab batches (experiments) never seen in training?

Data: RxRx1 (Recursion, CC-BY-NC-SA 4.0; research use only here, not re-shared, not part of the final submission
unless the licence question is cleared). Split: the provider's own train/test experiments (33 / 18 experiments,
all four cell types in both). Negative-control wells (no siRNA) are left out of the class list.
Caveat recorded: the provider does not say which images trained the embedding model, so the test experiments may
have been seen by it; results are an upper bound for "new batches".

Batch corrections compared (none uses any treatment label; each uses only one plate's or experiment's own data):
  none | per-experiment standardise | per-plate standardise | per-plate centre on its control wells
Classifiers (no tuning knobs): nearest class centre (cosine) and shrinkage LDA.
Reference for the batch gap: same experiments, other imaging site (train on site 1, test on site 2).
Scores: accuracy per site and per well (2 sites averaged), per cell type; differences vs "none" with a 95%
interval from resampling whole test experiments.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run

ROOT = DATA_ROOT / "rxrx1" / "rxrx1"
SEED = 20261006


def correct(X, meta, how):
    X = X.copy()
    if how == "none":
        return X
    key = meta["experiment"] if how == "per_experiment" else meta["experiment"] + "_" + meta["plate"].astype(str)
    for _, idx in meta.groupby(key).indices.items():
        if how == "plate_controls":
            ctrl = idx[meta["well_type"].to_numpy()[idx] != "treatment"]
            mu, sd = X[ctrl].mean(0), X[idx].std(0)
        else:
            mu, sd = X[idx].mean(0), X[idx].std(0)
        X[idx] = (X[idx] - mu) / np.where(sd > 0, sd, 1)
    return X


def centroid_predict(Xtr, ytr, Xte):
    classes = np.unique(ytr)
    Cn = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-9
    Z = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    S = Z @ Cn.T
    return classes, S


def main():
    run = Run("C01", "rxrx1_batches", config={"data": "RxRx1 dl-embeddings + metadata (CC-BY-NC-SA 4.0)",
                                             "split": "provider train/test experiments", "seed": SEED})
    meta = pd.read_csv(ROOT / "metadata.csv")
    emb = pd.read_csv(ROOT / "embeddings.csv")
    D = meta.merge(emb, on="site_id")
    D = D[D["well_type"] != "negative_control"].reset_index(drop=True)
    feats = [c for c in emb.columns if c.startswith("feature_")]
    run.log(f"{len(D)} sites, {D['sirna_id'].nunique()} siRNA classes, {len(feats)} features; "
            f"train experiments {D.loc[D.dataset == 'train', 'experiment'].nunique()}, test {D.loc[D.dataset == 'test', 'experiment'].nunique()}")
    rows, per_site = [], []
    for how in ("none", "per_experiment", "per_plate", "plate_controls"):
        X = correct(D[feats].to_numpy(np.float32), D, how)
        y = D["sirna_id"].to_numpy()
        for setting, tr, te in (("new_batches", (D.dataset == "train").to_numpy(), (D.dataset == "test").to_numpy()),
                                ("same_batches_other_site", ((D.dataset == "train") & (D.site == 1)).to_numpy(),
                                 ((D.dataset == "train") & (D.site == 2)).to_numpy())):
            for clf in ("centroid", "lda"):
                if clf == "centroid":
                    classes, S = centroid_predict(X[tr], y[tr], X[te])
                else:
                    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
                    classes, S = m.classes_, m.decision_function(X[te])
                pred = classes[S.argmax(1)]
                d = D.loc[te, ["site_id", "well_id", "experiment", "cell_type", "sirna_id"]].copy()
                d["correct"] = (pred == y[te]).astype(int)
                # well level: average the two sites' scores
                Sw = pd.DataFrame(S).groupby(d["well_id"].to_numpy()).mean()
                wl = d.groupby("well_id").first()
                wpred = classes[Sw.loc[wl.index].to_numpy().argmax(1)]
                wl["correct_well"] = (wpred == wl["sirna_id"].to_numpy()).astype(int)
                d["correction"], d["setting"], d["classifier"] = how, setting, clf
                per_site.append(d)
                r = {"correction": how, "setting": setting, "classifier": clf, "site_acc": float(d["correct"].mean()),
                     "well_acc": float(wl["correct_well"].mean()), "sites": int(te.sum())}
                for ct, g in d.groupby("cell_type"):
                    r[f"site_acc_{ct}"] = float(g["correct"].mean())
                rows.append(r)
                run.log(f"{how:15s} {setting:24s} {clf:8s} site acc {r['site_acc']:.3f} well acc {r['well_acc']:.3f} | "
                        + " ".join(f"{ct} {r[f'site_acc_{ct}']:.3f}" for ct in sorted(d['cell_type'].unique())))
    R = pd.DataFrame(rows)
    run.save_df("summary.csv", R)
    P = pd.concat(per_site, ignore_index=True)
    run.save_df("per_site_correct.csv", P[["site_id", "experiment", "cell_type", "correction", "setting", "classifier", "correct"]])
    # differences vs "none" on new batches, resampling whole test experiments
    rng = np.random.default_rng(SEED)
    nb = P[P.setting == "new_batches"]
    exps = nb["experiment"].unique()
    acc = nb.groupby(["classifier", "correction", "experiment"])["correct"].agg(["sum", "count"])
    out = {}
    for clf in ("centroid", "lda"):
        for how in ("per_experiment", "per_plate", "plate_controls"):
            diffs = []
            for _ in range(2000):
                pick = rng.choice(exps, len(exps))
                a = acc.loc[(clf, how)].loc[pick].sum()
                b = acc.loc[(clf, "none")].loc[pick].sum()
                diffs.append(a["sum"] / a["count"] - b["sum"] / b["count"])
            out[f"{clf}|{how}_minus_none"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
            run.log(f"{clf:8s} {how:15s} minus none (new batches), 95% over experiments: [{out[f'{clf}|{how}_minus_none'][0]:+.3f}, {out[f'{clf}|{how}_minus_none'][1]:+.3f}]")
    run.save_json("results.json", {"summary": R.to_dict(orient="records"), "differences_vs_none_95": out,
                                   "chance_accuracy": 1 / D["sirna_id"].nunique()})
    run.finish("measured", notes="C01 RxRx1 embeddings; provider split; research use only")


if __name__ == "__main__":
    main()
