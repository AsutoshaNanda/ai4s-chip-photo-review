"""A05 confirmation (improvements.md sections 4 and 6): the promising training change, after the validity fixes.

The promising A05 change was "train_plus_all_changes" (flips, 90-degree turns, brightness x0.8 / x1.25 added to
TRAINING photos only; development gain about +0.005 AUC, many variants tried, so not confirmed). Here it is tested
  - on the whole-photo reference and on the retained X01 candidates (whole + 3x3, whole + 5x5 patches),
  - under the 5 practice folds AND under the repaired A03 rule (held-out cell type and its groups out of training),
paired on identical photos with the same classifier as X01. Training-only changes: test photos are untouched.

  --part features : GPU. Features of the changed copies (whole @518 and native-resolution 3x3 / 5x5 patches,
                    mean+max pooled), same extractor as a_views.py. Saved for the fit step.
  --part fit      : CPU. All fits, paired differences (whole groups resampled), per format / type, catch curves.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, OOC_DIR, RUNS_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, latest_run

SEED = 20261005
A02_RUN = "20261006-012654_A02_frozen_features"
TILE_RUNS = {3: "20261006-103251_X01_whole_plus_patches", 5: "20261006-103757_X01_whole_plus_patches_grids4-5-6"}
WHOLE = "vit_large_patch14_dinov2@518"
CHANGES = ["hflip", "vflip", "rot90", "rot270", "bright_x0.8", "bright_x1.25"]     # A05's "all changes" set


def run_dir(name):
    for root in (RUNS_ROOT, CODE_ROOT / "runs"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(name)


def practice():
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    return sp[sp.role == "practice"].merge(fmt[["member", "photo_format"]], on="member").reset_index(drop=True)


def change(m, name):
    if name == "hflip":
        return m[:, ::-1]
    if name == "vflip":
        return m[::-1, :]
    if name == "rot90":
        return np.rot90(m, 1)
    if name == "rot270":
        return np.rot90(m, 3)
    b = float(name.split("x")[1])
    return np.clip(m.astype(np.float32) * b, 0, 255).astype(np.uint8)


def features(run, prac):
    import torch
    from a_views import Extractor, tiles
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [np.load(OOC_DIR / "gray1036" / f"{i}.npy") for i in prac["image_id"]]
    ex = Extractor("vit_large_patch14_dinov2.lvd142m", 518, dev, run)
    for ch in CHANGES:
        changed = [np.ascontiguousarray(change(m, ch)) for m in imgs]
        parts = [ex(f"{ch}_whole", changed)]
        for n in (3, 5):
            T = ex(f"{ch}_tiles_{n}x{n}", [t for m in changed for t in tiles(m, n)], native=True).reshape(len(changed), n * n, -1)
            parts.append(np.concatenate([T.mean(1), T.max(1)], axis=1))
        np.save(run.path(f"features_{ch}_whole.npy"), parts[0].astype(np.float16))
        np.save(run.path(f"features_{ch}_tiles_3x3_meanmax.npy"), parts[1].astype(np.float16))
        np.save(run.path(f"features_{ch}_tiles_5x5_meanmax.npy"), parts[2].astype(np.float16))
    pd.Series(prac["image_id"]).to_csv(run.path("features_ids.csv"), index=False, header=["image_id"])
    run.save_json("feature_timing.json", ex.timing)


def fit(run, prac):
    from a_views import fit_eval
    a02 = run_dir(A02_RUN)
    ids = pd.read_csv(a02 / "features_practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    whole = pd.DataFrame(np.load(a02 / f"features_practice_{WHOLE}.npy").astype(np.float32), index=ids).loc[prac["image_id"]].to_numpy()
    base = {"whole": whole}
    for n, r in TILE_RUNS.items():
        base[f"x01_{n}x{n}"] = np.concatenate([whole, np.load(run_dir(r) / f"features_tiles_{n}x{n}_meanmax.npy").astype(np.float32)], axis=1)
    fr = latest_run("A05_confirm_features")
    fids = pd.read_csv(fr / "features_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    assert fids == prac["image_id"].tolist(), "changed-copy features are not in practice order"
    extra = {k: [] for k in base}
    for ch in CHANGES:
        w = np.load(fr / f"features_{ch}_whole.npy").astype(np.float32)
        extra["whole"].append(w)
        for n in (3, 5):
            extra[f"x01_{n}x{n}"].append(np.concatenate([w, np.load(fr / f"features_{ch}_tiles_{n}x{n}_meanmax.npy").astype(np.float32)], axis=1))
    preds = []
    # (1) the 5 practice folds
    for name, X in base.items():
        for with_changes in (False, True):
            P, _ = fit_eval(X, X, prac, f"{name}{'_plus_all_changes' if with_changes else ''}", extra_train=extra[name] if with_changes else None)
            preds.append(P.assign(protocol="practice_folds"))
            run.log(f"practice folds {P['method'].iloc[0]:32s} AUC {C.roc_auc(P['label_bad'], P['p_bad']):.4f}")
    # (2) repaired A03: held-out cell type and its groups out of training (fold column re-used as the held-out type)
    types = sorted(prac["cell_type"].unique())
    for ct in types:
        te = (prac["cell_type"] == ct).to_numpy()
        keep = te | ~prac["group"].isin(set(prac.loc[te, "group"])).to_numpy()
        sub = prac[keep].reset_index(drop=True).assign(dev_fold=lambda d: (d["cell_type"] == ct).astype(int))
        if sub.loc[sub.dev_fold == 0, "label_bad"].nunique() < 2:
            run.log(f"A03 rule {ct}: unavailable (training has one class)")
            continue
        for name, X in base.items():
            Xs = X[keep]
            for with_changes in (False, True):
                ex_ = [e[keep] for e in extra[name]] if with_changes else None
                P, _ = fit_eval(Xs, Xs, sub, f"{name}{'_plus_all_changes' if with_changes else ''}", extra_train=ex_, folds=[1])
                preds.append(P.assign(protocol="a03_groups_excluded", dev_fold=-1))
        run.log(f"A03 rule {ct}: done")
    P = pd.concat(preds, ignore_index=True)
    run.save_df("predictions.csv", P)
    res, curves = {}, []
    for (proto, meth), d in P.groupby(["protocol", "method"]):
        y, p = d["label_bad"].to_numpy(), d["p_bad"].to_numpy()
        res[f"{proto}|{meth}"] = {"pooled": C.summary(y, p), "per_photo_format": C.by_group(y, p, d["photo_format"].to_numpy()),
                                  "per_cell_type": C.by_group(y, p, d["cell_type"].to_numpy())}
        c = C.review_curve(y, p)["catch_rate"]
        curves.append(pd.DataFrame({"protocol": proto, "method": meth, "fraction_reviewed": np.arange(len(c)) / (len(c) - 1), "catch_rate": c}))
    for proto, D in P.groupby("protocol"):
        for name in base:
            a = D[D.method == name].set_index("image_id")
            b = D[D.method == f"{name}_plus_all_changes"].set_index("image_id").loc[a.index]
            y, g, pa, pb = a["label_bad"].to_numpy(), a["group"].to_numpy(), a["p_bad"].to_numpy(), b["p_bad"].to_numpy()
            out = {}
            scopes = [("pooled", np.ones(len(a), bool))] + [(f"type={t}", (a["cell_type"] == t).to_numpy()) for t in types] \
                + [(f"format={f}", (a["photo_format"] == f).to_numpy()) for f in sorted(a["photo_format"].unique())]
            for scope, msk in scopes:
                yy, gg, x1, x2 = y[msk], g[msk], pa[msk], pb[msk]
                if len(np.unique(yy)) < 2:
                    out[scope] = {"status": "unavailable: one class"}
                    continue
                def diff(idx, yy=yy, x1=x1, x2=x2):
                    r1, r2 = C.roc_auc(yy[idx], x2[idx]), C.roc_auc(yy[idx], x1[idx])
                    return None if r1 is None or r2 is None else r1 - r2
                vals = C.grouped_bootstrap(diff, gg, n_boot=2000, seed=SEED)
                out[scope] = {"difference": float(C.roc_auc(yy, x2) - C.roc_auc(yy, x1)), "n_groups": int(len(np.unique(gg))), "n_bad": int(yy.sum()),
                              "range_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None}
            res[f"paired|{proto}|{name}_plus_all_changes_minus_{name}"] = out
            run.log(f"{proto:22s} {name}: all changes minus none, pooled {out['pooled']}")
    run.save_df("review_curves.csv", pd.concat(curves, ignore_index=True))
    run.save_json("results.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["features", "fit"], required=True)
    a = ap.parse_args()
    prac = practice()
    run = Run("A05", f"confirm_{a.part}", config={"changes": CHANGES, "a02_run": A02_RUN, "tile_runs": TILE_RUNS,
                                                  "protocols": ["practice_folds", "a03_groups_excluded"], "photos": "practice only"})
    (features if a.part == "features" else fit)(run, prac)
    run.finish("measured", notes=f"A05 confirmation, part {a.part}; practice only; final test untouched")


if __name__ == "__main__":
    main()
