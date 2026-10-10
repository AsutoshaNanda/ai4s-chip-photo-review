"""A03 repair (improvements.md section 3): unseen cell types with the evaluated photo groups kept out of training.

For each cell type: test = its practice photos; training = practice photos of OTHER cell types whose filename
group is not used by any test photo (the earlier A03 kept those shared groups in training).
Compared on the same held-out photos: whole-image reference (A02 vit_large dinov2 @518) and the retained X01
candidates (whole + 3x3 patches, whole + 5x5 patches). The earlier shared-group protocol is rerun alongside on
the same photos so the change is visible. Every type reports class counts and the training data left after
exclusion; a type whose training or test set lacks one class is marked unavailable, not scored.
Practice photos only; the final test stays locked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from a_suite import fit_lr, SEED
from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, RUNS_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, latest_run

A02_RUN = "20261006-012654_A02_frozen_features"
X01_RUNS = {"3x3": "20261006-103251_X01_whole_plus_patches", "5x5": "20261006-103757_X01_whole_plus_patches_grids4-5-6"}
WHOLE = "vit_large_patch14_dinov2@518"


def run_dir(name):
    for root in (RUNS_ROOT, CODE_ROOT / "runs"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(name)


def load():
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = sp[sp.role == "practice"].merge(fmt[["member", "photo_format"]], on="member").reset_index(drop=True)
    a02 = run_dir(A02_RUN)
    ids = pd.read_csv(a02 / "features_practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    whole = pd.DataFrame(np.load(a02 / f"features_practice_{WHOLE}.npy").astype(np.float32), index=ids).loc[prac["image_id"]].to_numpy()
    feats = {"whole": whole}
    for g, rn in X01_RUNS.items():
        T = np.load(run_dir(rn) / f"features_tiles_{g}_meanmax.npy").astype(np.float32)   # same row order as prac (a_views.py)
        feats[f"whole_plus_patches_{g}"] = np.concatenate([whole, T], axis=1)
    return prac, feats


def alignment_check(run, prac, feats):
    """Rerun the original 5-fold X01 comparison with these features; must reproduce the saved X01 AUCs."""
    out = {}
    for tag, X in feats.items():
        rows = []
        for k in sorted(prac["dev_fold"].unique()):
            tr, te = (prac["dev_fold"] != k).to_numpy(), (prac["dev_fold"] == k).to_numpy()
            m, _ = fit_lr(X[tr], prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy())
            rows.append(pd.DataFrame({"y": prac.loc[te, "label_bad"].to_numpy(), "p": m.predict_proba(X[te])[:, 1]}))
        d = pd.concat(rows)
        out[tag] = C.roc_auc(d["y"], d["p"])
        run.log(f"alignment check (original 5 folds) {tag:28s} AUC {out[tag]:.4f}")
    for g, rn in X01_RUNS.items():
        saved = pd.read_csv(run_dir(rn) / "oof_predictions.csv", dtype={"image_id": str})
        for meth in (f"whole_plus_patches_{g}", "whole"):
            s = saved[saved["method"] == meth]
            if len(s):
                out[f"saved_{rn}_{meth}"] = C.roc_auc(s["label_bad"], s["p_bad"])
                run.log(f"saved X01 run {rn} {meth:28s} AUC {out[f'saved_{rn}_{meth}']:.4f}")
    return out


def split_for(prac, ct, exclude_groups):
    te = (prac["cell_type"] == ct).to_numpy()
    tr = ~te
    if exclude_groups:
        tr &= ~prac["group"].isin(set(prac.loc[te, "group"])).to_numpy()
    return tr, te


def main():
    prac, feats = load()
    run = Run("A03", "unseen_cell_types_groups_excluded", config={
        "a02_run": A02_RUN, "x01_runs": X01_RUNS, "whole_features": WHOLE, "feature_sets": list(feats),
        "protocols": {"groups_excluded": "train = other cell types AND groups not used by any test photo",
                      "shared_groups_original": "train = all other cell types (earlier A03 rule), rerun for comparison"},
        "photos": "practice only", "bootstrap": "whole filename groups resampled, 1000 replicates, seed 20261005"})
    res = {"alignment_check": alignment_check(run, prac, feats), "support": {}, "protocols": {}}
    for ct in sorted(prac["cell_type"].unique()):
        tr, te = split_for(prac, ct, True)
        tr_old, _ = split_for(prac, ct, False)
        overlap = set(prac.loc[tr, "group"]) & set(prac.loc[te, "group"])
        assert not overlap, f"{ct}: groups shared after exclusion: {sorted(overlap)[:5]}"
        sup = {"test_photos": int(te.sum()), "test_bad": int(prac.loc[te, "label_bad"].sum()),
               "test_good": int((1 - prac.loc[te, "label_bad"]).sum()), "test_groups": int(prac.loc[te, "group"].nunique()),
               "train_photos_groups_excluded": int(tr.sum()), "train_bad_groups_excluded": int(prac.loc[tr, "label_bad"].sum()),
               "train_good_groups_excluded": int((1 - prac.loc[tr, "label_bad"]).sum()),
               "train_groups_groups_excluded": int(prac.loc[tr, "group"].nunique()),
               "train_photos_shared_groups_original": int(tr_old.sum()),
               "test_groups_also_in_original_training": len(set(prac.loc[te, "group"]) & set(prac.loc[tr_old, "group"])),
               "groups_shared_after_exclusion": 0}
        res["support"][ct] = sup
        run.log(f"{ct}: test {sup['test_photos']} photos ({sup['test_bad']} bad / {sup['test_good']} good, {sup['test_groups']} groups); "
                f"training left {sup['train_photos_groups_excluded']} photos ({sup['train_bad_groups_excluded']} bad / "
                f"{sup['train_good_groups_excluded']} good) vs {sup['train_photos_shared_groups_original']} under the earlier rule")
    for proto, excl in (("groups_excluded", True), ("shared_groups_original", False)):
        P = []
        for ct in sorted(prac["cell_type"].unique()):
            tr, te = split_for(prac, ct, excl)
            ytr, yte = prac.loc[tr, "label_bad"].to_numpy(), prac.loc[te, "label_bad"].to_numpy()
            base = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad"]].copy()
            if len(np.unique(ytr)) < 2:
                run.log(f"{proto} {ct}: UNAVAILABLE - training data left has one class only")
                res["support"][ct][f"{proto}_status"] = "unavailable: training has one class"
                continue
            for tag, X in feats.items():
                m, cut = fit_lr(X[tr], ytr, prac.loc[tr, "group"].to_numpy())
                p = m.predict_proba(X[te])[:, 1]
                P.append(base.assign(method=tag, p_bad=p, pred_bad=(p >= cut).astype(int), cutoff=cut))
            res["support"][ct][f"{proto}_status"] = "scored" if len(np.unique(yte)) == 2 else "unavailable for AUC: test has one class"
        P = pd.concat(P, ignore_index=True)
        run.save_df(f"predictions_{proto}.csv", P)
        out = {}
        for tag, d in P.groupby("method"):
            y, p = d["label_bad"].to_numpy(), d["p_bad"].to_numpy()
            out[tag] = {"pooled": C.summary(y, p), "per_cell_type": C.by_group(y, p, d["cell_type"].to_numpy()),
                        "per_photo_format": C.by_group(y, p, d["photo_format"].to_numpy())}
            run.log(f"{proto:24s} {tag:28s} pooled AUC {out[tag]['pooled']['roc_auc']:.3f} | per type "
                    + str({k: (round(v['roc_auc'], 3) if v['roc_auc'] is not None else 'n/a') for k, v in out[tag]['per_cell_type'].items()}))
        # paired: each X01 candidate minus whole, same held-out photos, whole groups resampled
        W = P[P.method == "whole"].set_index("image_id")
        for tag in [t for t in feats if t != "whole"]:
            Xp = P[P.method == tag].set_index("image_id").loc[W.index]
            y, g = W["label_bad"].to_numpy(), W["group"].to_numpy()
            pw, px = W["p_bad"].to_numpy(), Xp["p_bad"].to_numpy()
            pair = {}
            for scope, mask in [("pooled", np.ones(len(W), bool))] + [(ct, (W["cell_type"] == ct).to_numpy()) for ct in sorted(W["cell_type"].unique())]:
                yy, gg, a, b = y[mask], g[mask], pw[mask], px[mask]
                if len(np.unique(yy)) < 2:
                    pair[scope] = {"status": "unavailable: one class"}
                    continue
                def diff(idx, yy=yy, a=a, b=b):
                    ra, rb = C.roc_auc(yy[idx], b[idx]), C.roc_auc(yy[idx], a[idx])
                    return None if ra is None or rb is None else ra - rb
                vals = C.grouped_bootstrap(diff, gg, n_boot=1000, seed=SEED)
                pair[scope] = {"difference": float(C.roc_auc(yy, b) - C.roc_auc(yy, a)), "n_groups": int(len(np.unique(gg))),
                               "range_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None,
                               "bootstrap_values_used": int(len(vals))}
            out[f"paired_{tag}_minus_whole"] = pair
            run.log(f"{proto:24s} {tag} minus whole (pooled): {pair['pooled']}")
        res["protocols"][proto] = out
    run.save_json("results.json", res)
    run.finish("measured", notes="A03 repaired: evaluated groups excluded from training; whole vs X01 3x3/5x5; practice only")


if __name__ == "__main__":
    main()
