"""F01 / F02: ONE evaluation of the frozen candidates on the locked final tests, plus review-workload curves from the saved predictions.

Reads code/final_config.json (written by freeze_choice.py --write). Nothing is chosen or tuned here; every number is reported.
  --part chip_features : features of the final chip photos (whole photo @518 and native-resolution tiles for the frozen grid),
                         same extractor as a_views.py (GPU job; needs the grey long-side-1036 arrays of the final photos).
  --part chip_eval     : fit on ALL practice photos (same classifier and inner grouped CV as X01), predict the final photos.
                         Reported: AUC, average precision, Brier, log loss, per cell type, per photo format, group-resampled interval,
                         paired difference to the whole-image reference, reliability bins, and F02 catch curves (bad photos found vs
                         share of photos reviewed) against random order. Small-format photos are routed by the input gate.
  --part nucleus       : the 5 saved fold models of B02 and B03 (seeds and variants from the config), each with its v2 best-count and
                         best-outline settings, applied to the 50 final images; B01 as the simple reference. Mean and spread over
                         fold models, per-image tables, and F02 correction-workload curves (count error left vs images reviewed in
                         disagreement order) against random order.
The final test is opened ONCE for the frozen config. Re-running with another config is a new, declared evaluation.
"""
import argparse
import json
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


def run_dir(name):
    for root in (RUNS_ROOT, CODE_ROOT / "runs"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(name)


def chip_tables():
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    sp = sp.merge(fmt[["member", "photo_format"]], on="member")
    return sp[sp["role"] == "practice"].reset_index(drop=True), sp[sp["role"] == "final_test"].reset_index(drop=True)


def chip_features(run, cfg, prac, fin, roles):
    import torch
    from a_views import Extractor, tiles
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ex = Extractor("vit_large_patch14_dinov2.lvd142m", 518, dev, run)
    n = cfg["chip"]["grid"]
    for role, tab, prefix in (("final_test", fin, "final"), ("practice", prac, "practice")):
        if role not in roles:
            continue
        imgs = [np.load(OOC_DIR / "gray1036" / f"{i}.npy") for i in tab["image_id"]]
        whole = ex(f"{prefix}_whole", imgs)
        np.save(run.path(f"{prefix}_features_whole.npy"), whole.astype(np.float16))
        if n:
            T = ex(f"{prefix}_tiles_{n}x{n}", [t for m in imgs for t in tiles(m, n)], native=True).reshape(len(imgs), n * n, -1)
            np.save(run.path(f"{prefix}_features_tiles_{n}x{n}_meanmax.npy"), np.concatenate([T.mean(1), T.max(1)], axis=1).astype(np.float16))
        tab[["image_id"]].to_csv(run.path(f"{prefix}_ids.csv"), index=False)
    run.save_json("feature_timing.json", ex.timing)


def fit_predict_final(Xtr, ytr, gtr, Xte):
    from a01_simple_baselines import CS, inner_cv
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    splits = inner_cv(gtr, ytr)
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000, n_jobs=-1))
    model.fit(Xtr, ytr)
    c = float(model[-1].C_[0])
    oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)), Xtr, ytr, cv=splits, method="predict_proba")[:, 1]
    cut = C.choose_cutoff(ytr, oof)["cutoff"]
    return model.predict_proba(Xte)[:, 1], cut, c


def reliability(y, p, nb):
    q = np.quantile(p, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb - 1)
    rows = [{"bin": i, "n": int((b == i).sum()), "mean_pred": float(p[b == i].mean()), "share_bad": float(y[b == i].mean())} for i in range(nb) if (b == i).any()]
    return {"bins": rows, "ece": float(sum(r["n"] * abs(r["mean_pred"] - r["share_bad"]) for r in rows) / len(y))}


def chip_eval(run, cfg, prac, fin, fresh=None):
    """fresh: folder with practice_/final_ features computed in a clean rebuild (F03); None = saved development features."""
    from input_gate import chip_route, seen_formats
    cand = cfg["chip"]["candidate"]
    n = cfg["chip"]["grid"]
    fr = Path(fresh) if fresh else latest_run("F01_chip_features")
    assert (pd.read_csv(fr / "final_ids.csv", dtype={"image_id": str})["image_id"].tolist() == fin["image_id"].tolist())
    if fresh:
        assert (pd.read_csv(fr / "practice_ids.csv", dtype={"image_id": str})["image_id"].tolist() == prac["image_id"].tolist())
        wp = np.load(fr / "practice_features_whole.npy").astype(np.float32)
        tp = np.load(fr / f"practice_features_tiles_{n}x{n}_meanmax.npy").astype(np.float32) if n else None
    else:
        a02 = run_dir(A02_RUN)
        ids = pd.read_csv(a02 / "features_practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
        wp = pd.DataFrame(np.load(a02 / f"features_practice_{WHOLE}.npy").astype(np.float32), index=ids).loc[prac["image_id"]].to_numpy()
        tp = np.load(run_dir(TILE_RUNS[n]) / f"features_tiles_{n}x{n}_meanmax.npy").astype(np.float32) if n else None
    wf = np.load(fr / "final_features_whole.npy").astype(np.float32)
    sets = {"whole": (wp, wf)}
    if n:
        tf = np.load(fr / f"final_features_tiles_{n}x{n}_meanmax.npy").astype(np.float32)
        sets[cand] = (np.concatenate([wp, tp], 1), np.concatenate([wf, tf], 1))
    y, g = prac["label_bad"].to_numpy(), prac["group"].to_numpy()
    preds, info = {}, {}
    for name, (Xp, Xf) in sets.items():
        p, cut, c = fit_predict_final(Xp, y, g, Xf)
        preds[name], info[name] = p, {"C": c, "cutoff": cut, "train_photos": len(y)}
        run.log(f"{name}: classifier fitted on {len(y)} practice photos, C={c:g}")
    D = fin[["image_id", "group", "cell_type", "photo_format", "label_bad"]].copy()
    for k, v in preds.items():
        D[f"p_{k}"] = v
    _, seen, _ = seen_formats()
    D["route"] = [chip_route(f, seen, cfg["chip"]["small_rule"]) for f in D["photo_format"]]
    run.save_df("final_predictions.csv", D)
    res = {"config": cfg["chip"], "settings": info, "routes": D["route"].value_counts().to_dict(), "final_photos_all": len(D)}
    curves = []
    for scope, mask in (("automatic_only", (D["route"] == "automatic").to_numpy()), ("all_photos", np.ones(len(D), bool))):
        Ds = D[mask]
        yy, gg = Ds["label_bad"].to_numpy(), Ds["group"].to_numpy()
        rs = {"photos": int(len(Ds)), "bad": int(yy.sum()), "groups": int(Ds["group"].nunique())}
        rng = np.random.default_rng(SEED)
        rand = np.stack([C.review_curve(yy, rng.random(len(yy)))["catch_rate"] for _ in range(200)])
        curves.append(pd.DataFrame({"scope": scope, "order": "random_mean", "subset": "ALL", "fraction_reviewed": np.arange(len(yy) + 1) / len(yy), "catch_rate": rand.mean(0)}))
        rs["random_review_area"] = float(np.mean(rand.mean(0)[1:]))
        for k in preds:
            p = Ds[f"p_{k}"].to_numpy()
            vals = C.grouped_bootstrap(lambda idx: C.roc_auc(yy[idx], p[idx]), gg, n_boot=2000, seed=SEED)
            r = {"pooled": C.summary(yy, p), "auc_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                 "per_cell_type": C.by_group(yy, p, Ds["cell_type"].to_numpy()), "per_photo_format": C.by_group(yy, p, Ds["photo_format"].to_numpy()),
                 "reliability": {nb: reliability(yy, p, nb) for nb in (5, 10)}, "review_area": C.review_area(yy, p)}
            rs[k] = r
            run.log(f"FINAL [{scope}] {k}: AUC {r['pooled']['roc_auc']:.4f} {np.round(r['auc_95'], 3).tolist()}; AP {r['pooled']['avg_precision_bad']:.3f}; "
                    f"log loss {r['pooled']['log_loss']:.3f}; review area {r['review_area']:.3f} (random {rs['random_review_area']:.3f}); per type "
                    + str({t: (round(v['roc_auc'], 3) if v['roc_auc'] is not None else None) for t, v in r['per_cell_type'].items()}))
            for sub, m in [("ALL", np.ones(len(yy), bool))] + [(f"type={t}", (Ds["cell_type"] == t).to_numpy()) for t in sorted(Ds["cell_type"].unique())] \
                    + [(f"format={f}", (Ds["photo_format"] == f).to_numpy()) for f in sorted(Ds["photo_format"].unique())]:
                if m.sum() >= 5 and len(np.unique(yy[m])) == 2:
                    c_ = C.review_curve(yy[m], p[m])["catch_rate"]
                    curves.append(pd.DataFrame({"scope": scope, "order": k, "subset": sub, "fraction_reviewed": np.arange(len(c_)) / (len(c_) - 1), "catch_rate": c_}))
        if cand != "whole":
            pa, pb = Ds["p_whole"].to_numpy(), Ds[f"p_{cand}"].to_numpy()
            vals = C.grouped_bootstrap(lambda idx: C.roc_auc(yy[idx], pb[idx]) - C.roc_auc(yy[idx], pa[idx]), gg, n_boot=2000, seed=SEED)
            rs["paired_candidate_minus_whole"] = {"difference": float(C.roc_auc(yy, pb) - C.roc_auc(yy, pa)), "range_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
            run.log(f"FINAL [{scope}] paired {cand} minus whole: {rs['paired_candidate_minus_whole']}")
        res[scope] = rs
    run.save_df("catch_curves.csv", pd.concat(curves, ignore_index=True))
    run.save_json("results.json", res)


def nucleus_eval(run, cfg):
    import torch
    import b_unet as U
    from ai4s import bbbc039 as B
    from ai4s import metrics_seg as S
    from b_posts import fold_models
    from g03_speed_test import UNet
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    fin = sp[sp["role"] == "final_test"].reset_index(drop=True)
    X = [U.norm(B.read_image(s)) for s in fin["stem"]]
    L = [B.read_labels(s) for s in fin["stem"]]
    res, per_image, counts = {}, [], {}
    rows_variants = [(exp, "", seed) for exp in cfg["nucleus"]["models"] for seed in cfg["nucleus"]["seeds"]]
    ev = cfg["nucleus"]["extra_row"]
    rows_variants += [(exp, ev["variant"], ev["seed"]) for exp in cfg["nucleus"]["models"]]
    for exp, variant, seed in rows_variants:
        target, n_cls = ("fg", 2) if exp == "B02" else ("boundary", 3)
        fm = fold_models(exp, target, seed, "v2", variant)
        if sorted(fm) != [0, 1, 2, 3, 4]:
            run.log(f"{exp}{variant} seed {seed}: fold models or v2 settings incomplete ({sorted(fm)}); row not evaluated")
            continue
        for name in ("best_count", "best_outline"):
            summ, cnt = [], []
            for k, (d, post) in sorted(fm.items()):
                m = UNet(out_ch=n_cls)
                m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
                m = m.eval().to(U.DEV)
                probs = U.predict_prob(m, np.stack(X))
                s = post[name]
                rs = []
                for i, p in enumerate(probs):
                    r = S.score_image(L[i], U.to_objects(p, target, s["thr"], int(s["min_distance"]), int(s["min_size"])))
                    r.update({"stem": fin.loc[i, "stem"], "fold_model": k, "method": f"{exp}{variant}_seed{seed}_{name}"})
                    rs.append(r)
                summ.append(S.summarize(rs))
                cnt.append([r["n_pred"] for r in rs])
                per_image += [{k_: v for k_, v in r.items() if not k_.startswith(("tp@", "fp@", "fn@"))} for r in rs]
            key = f"{exp}{variant}_seed{seed}_{name}"
            res[key] = {mname: [float(np.mean([s_[mname] for s_ in summ])), float(np.std([s_[mname] for s_ in summ]))]
                        for mname in ("count_mae", "count_bias", "pooled_f1@0.50", "pooled_f1@0.75", "joined_ref", "split_pred", "missed", "false")}
            counts[key] = np.mean(cnt, 0)
            run.log(f"FINAL {key}: count MAE {res[key]['count_mae'][0]:.2f} (sd over fold models {res[key]['count_mae'][1]:.2f}), "
                    f"F1@0.50 {res[key]['pooled_f1@0.50'][0]:.3f}, F1@0.75 {res[key]['pooled_f1@0.75'][0]:.3f}")
    # B01 simple reference
    import b01_threshold_watershed as T
    b01 = latest_run("B01_cv_threshold_watershed")
    chosen = json.load(open(b01 / "chosen_settings_per_fold.json"))
    for name in ("best_count", "best_outline"):
        summ = []
        for k in sorted(chosen):
            rs = [S.score_image(L[i], T.segment(B.read_image(s), **chosen[k][name])) for i, s in enumerate(fin["stem"])]
            summ.append(S.summarize(rs))
        res[f"B01_{name}"] = {mname: [float(np.mean([s_[mname] for s_ in summ])), float(np.std([s_[mname] for s_ in summ]))]
                              for mname in ("count_mae", "count_bias", "pooled_f1@0.50", "pooled_f1@0.75")}
        run.log(f"FINAL B01_{name}: count MAE {res[f'B01_{name}']['count_mae'][0]:.2f}, F1@0.75 {res[f'B01_{name}']['pooled_f1@0.75'][0]:.3f}")
    pd.DataFrame(per_image).to_csv(run.path("final_per_image.csv"), index=False)
    # F02 correction-workload curves (fold-model-averaged counts): disagreement B02 vs B03 vs random
    ref = np.array([int(S._relabel(l).max()) for l in L])
    k2, k3 = "B02_seed0_best_count", "B03_seed0_best_count"
    if k2 in counts and k3 in counts:
        err = np.abs(counts[k2] - ref)
        order = np.argsort(-np.abs(counts[k2] - counts[k3]), kind="stable")
        rng = np.random.default_rng(SEED)
        curve = [{"reviewed": k, "error_left_disagreement_order": float(err[np.setdiff1d(np.arange(len(err)), order[:k])].sum() / err.sum()),
                  "error_left_random_mean": float(1 - k / len(err)), "error_left_perfect_order": float(np.sort(err)[:len(err) - k].sum() / err.sum())}
                 for k in range(len(err) + 1)]
        pd.DataFrame(curve).to_csv(run.path("workload_curve_B02_count.csv"), index=False)
        res["workload_note"] = "review share is a simulated workload from saved predictions, not measured time saved"
    run.save_json("results.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["chip_features", "chip_eval", "nucleus"])
    ap.add_argument("--config", default=str(CODE_ROOT / "final_config.json"))
    ap.add_argument("--roles", default="final_test", help="chip_features: comma list of final_test,practice (practice only for the clean rebuild)")
    ap.add_argument("--fresh", default="", help="chip_eval: folder with freshly computed practice_/final_ features (clean rebuild)")
    ap.add_argument("--tag", default="", help="suffix for the run name (clean rebuild uses 'clean')")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    run = Run("F01" if not a.tag else "F03", {"chip_features": "chip_features", "chip_eval": "chip_final_eval", "nucleus": "nucleus_final_eval"}[a.part] + (f"_{a.tag}" if a.tag else ""),
              config={"part": a.part, "frozen_config": cfg, "note": "locked final test opened once for this frozen config"})
    prac, fin = chip_tables()
    if a.part == "chip_features":
        chip_features(run, cfg, prac, fin, a.roles.split(","))
    elif a.part == "chip_eval":
        chip_eval(run, cfg, prac, fin, a.fresh or None)
    else:
        nucleus_eval(run, cfg)
    run.finish("measured", notes=f"F01 {a.part}; frozen config; final test")


if __name__ == "__main__":
    main()
