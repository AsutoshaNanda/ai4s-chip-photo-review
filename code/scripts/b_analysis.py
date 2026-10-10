"""B05 / B06 on saved B01-B03 results (Mac; no training). Practice images only, out-of-fold; test stays locked.

B05  Crowded images and tiny nuclei. For every method, each held-out practice image is segmented again with the
     settings that fold chose (B01: chosen grid setting; B02/B03: saved fold model + chosen post-processing).
     - Tiny nuclei: every reference nucleus gets its best overlap (IoU) with any prediction. Size groups are the
       deciles of nucleus area on the TRAINING folds of that fold. Share found is reported at every overlap level.
     - Crowding: per-image count error against the reference count (full table + training-fold quartiles).
     - Gallery: the worst images by count error, with reference and predicted outlines.
B06  Can a warning flag bad counts for a person to fix? Warning scores use no reference outlines:
       disagreement between two methods' counts, a blur score, a brightness score, random order (mean and
       spread over many orders), and the perfect order (ceiling, labelled as such).
     Curve: for every review amount (0..all images), the count error left if reviewed images are fixed.
     Summary: area under that curve; difference vs random with plate-grouped bootstrap interval.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from ai4s import bbbc039 as B
from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run, latest_run

RUNS = Path(__file__).resolve().parents[1] / "runs"
SEED = 20261006


OPTS = {"seed": 0, "posts": "orig"}


def fold_runs(exp, target):
    from b_posts import fold_models
    return fold_models(exp, target, OPTS["seed"], OPTS["posts"])


def predictors(prac):
    """{method name: function(stem, fold) -> predicted label image}. Only methods whose runs exist."""
    import b01_threshold_watershed as T
    from b_unet import load as _unused  # noqa: F401  (keeps b_unet import errors visible early)
    import b_unet as U
    from g03_speed_test import UNet
    preds = {}
    try:
        b01 = latest_run("B01_cv_threshold_watershed")
    except FileNotFoundError:
        b01 = None
    if b01 is not None and (b01 / "chosen_settings_per_fold.json").exists():
        chosen = json.load(open(b01 / "chosen_settings_per_fold.json"))
        for name in ("best_count", "best_outline"):
            preds[f"B01_{name}"] = (lambda stem, k, name=name: T.segment(B.read_image(stem), **chosen[str(k)][name]))
    for exp, target, n_cls in (("B02", "fg", 2), ("B03", "boundary", 3)):
        fr = fold_runs(exp, target)
        if sorted(fr) != [0, 1, 2, 3, 4]:
            continue
        models = {}
        for k, (d, _) in fr.items():
            m = UNet(out_ch=n_cls)
            m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
            models[k] = m.eval().to(U.DEV)
        cache = {}

        def prob(stem, k, models=models, cache=cache):
            if stem not in cache:
                cache[stem] = U.predict_prob(models[k], U.norm(B.read_image(stem))[None])[0]
            return cache[stem]
        for name in ("best_count", "best_outline"):
            preds[f"{exp}_{name}"] = (lambda stem, k, fr=fr, prob=prob, target=target, name=name:
                                      U.to_objects(prob(stem, k), target, fr[k][1][name]["thr"],
                                                   int(fr[k][1][name]["min_distance"]), int(fr[k][1][name]["min_size"])))
    return preds


def per_object(ref, pred):
    ref, pred = S._relabel(ref), S._relabel(pred)
    iou = S.iou_from_counts(S.overlap_counts(ref, pred))
    areas = np.bincount(ref.ravel())[1:]
    best = iou.max(1) if iou.shape[1] else np.zeros(len(areas))
    return areas, best


def b05(run, prac, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    levels = S.DEFAULT_IOU_LEVELS
    obj_rows, img_rows = [], []
    ref_cache = {s: B.read_labels(s) for s in prac["stem"]}
    for meth, fn in preds.items():
        for k in sorted(prac["cv_fold"].unique()):
            train_areas = np.concatenate([np.bincount(S._relabel(ref_cache[s]).ravel())[1:]
                                          for s in prac.loc[prac["cv_fold"] != k, "stem"]])
            dec = np.quantile(train_areas, np.linspace(0.1, 0.9, 9))
            train_counts = [int(S._relabel(ref_cache[s]).max()) for s in prac.loc[prac["cv_fold"] != k, "stem"]]
            q = np.quantile(train_counts, [0.25, 0.5, 0.75])
            for s in prac.loc[prac["cv_fold"] == k, "stem"]:
                ref = ref_cache[s]
                lab = fn(s, k)
                areas, best = per_object(ref, lab)
                for a, b in zip(areas, best):
                    obj_rows.append({"method": meth, "stem": s, "cv_fold": int(k), "area": int(a), "best_iou": float(b),
                                     "size_decile": int(np.searchsorted(dec, a, side="right"))})
                r = S.score_image(ref, lab)
                img_rows.append({"method": meth, "stem": s, "cv_fold": int(k), "n_ref": r["n_ref"], "n_pred": r["n_pred"],
                                 "count_error": r["count_error"], "abs_count_error": r["abs_count_error"],
                                 "f1@0.50": r["f1@0.50"], "crowding_quartile": int(np.searchsorted(q, r["n_ref"], side="right")) + 1})
        run.log(f"B05 {meth}: segmented {int((prac.shape[0]))} held-out images")
    O, I = pd.DataFrame(obj_rows), pd.DataFrame(img_rows)
    run.save_df("B05_per_object.csv", O)
    run.save_df("B05_per_image.csv", I)
    found = pd.DataFrame([{"method": m, "size_decile": d, "objects": len(g), "median_area": float(g["area"].median()),
                           **{f"found@{lv:.2f}": float((g["best_iou"] >= lv).mean()) for lv in levels}}
                          for (m, d), g in O.groupby(["method", "size_decile"])])
    run.save_df("B05_found_by_size_decile.csv", found)
    run.log("B05 share of reference nuclei found, by size decile (0 = smallest; training-fold deciles), at IoU 0.50 / 0.75 "
            "(all levels in B05_found_by_size_decile.csv):\n" +
            found.pivot(index="size_decile", columns="method", values="found@0.50").round(3).to_string() + "\n" +
            found.pivot(index="size_decile", columns="method", values="found@0.75").round(3).to_string())
    crowd = I.groupby(["method", "crowding_quartile"]).agg(images=("stem", "size"), count_mae=("abs_count_error", "mean"),
                                                          bias=("count_error", "mean"), f1_050=("f1@0.50", "mean")).round(3)
    run.log("B05 crowding (quartiles of reference count on training folds; 4 = most crowded):\n" + crowd.to_string())
    gal = run.path("B05_gallery")
    Path(gal).mkdir(exist_ok=True)
    for meth, d in I.groupby("method"):
        worst = d.sort_values("abs_count_error", ascending=False).head(6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (_, w) in zip(axes.ravel(), worst.iterrows()):
            img = B.read_image(w["stem"]).astype(np.float32)
            lo, hi = np.percentile(img, [1, 99.8])
            ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
            rb = find_boundaries(ref_cache[w["stem"]], mode="inner")
            pb = find_boundaries(preds[meth](w["stem"], int(w["cv_fold"])), mode="inner")
            ov = np.zeros(img.shape + (4,))
            ov[rb] = [0, 1, 0, 0.8]
            ov[pb] = [1, 0, 1, 0.8]
            ax.imshow(ov)
            ax.set_title(f"{w['stem'][:28]}\nref {w['n_ref']} / pred {w['n_pred']}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{meth}: worst 6 by count error (green = reference, magenta = predicted)")
        fig.tight_layout()
        fig.savefig(Path(gal) / f"{meth}.png", dpi=80)
        plt.close(fig)
    return {"found_by_size": found.to_dict(orient="records"), "crowding": crowd.reset_index().to_dict(orient="records")}


def leftover_curve(err, order):
    """Total abs count error left after fixing the first r images of `order`, for r = 0..n (as share of the start)."""
    e = err[order]
    left = e.sum() - np.concatenate([[0], np.cumsum(e)])
    return left / max(e.sum(), 1e-9)


def b06(run, prac, preds):
    target = next((m for m in ("B02_best_count", "B03_best_count", "B01_best_count") if m in preds), None)
    if target is None:
        raise SystemExit("B06 needs at least one method's saved runs")
    others = [m for m in preds if m.endswith("best_count") and m != target]
    rows = []
    for _, p in prac.iterrows():
        s, k = p["stem"], int(p["cv_fold"])
        img = B.read_image(s).astype(np.float32)
        ref = B.read_labels(s)
        lab = preds[target](s, k)
        r = {"stem": s, "plate": p["plate"], "cv_fold": k, "abs_err": abs(int(S._relabel(lab).max()) - int(S._relabel(ref).max())),
             "n_pred": int(S._relabel(lab).max()),
             "blur_warning": -float(ndi.laplace(img / max(img.mean(), 1e-6)).var()),     # less edge detail = more warning
             "median_intensity": float(np.median(img))}
        for o in others:
            r[f"disagree_{o}"] = abs(int(S._relabel(preds[o](s, k)).max()) - r["n_pred"])
        rows.append(r)
    D = pd.DataFrame(rows)
    # brightness warning: distance of the median from the training folds' median (no reference outlines used)
    D["brightness_warning"] = 0.0
    for k in D["cv_fold"].unique():
        med = D.loc[D["cv_fold"] != k, "median_intensity"].median()
        D.loc[D["cv_fold"] == k, "brightness_warning"] = (D.loc[D["cv_fold"] == k, "median_intensity"] - med).abs()
    run.save_df("B06_per_image_warnings.csv", D)
    err = D["abs_err"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    scores = {c: D[c].to_numpy(float) for c in D.columns if c.startswith(("disagree_", "blur_", "brightness_"))}
    scores["perfect_order_ceiling"] = err
    curves = {n: leftover_curve(err, np.argsort(-v, kind="stable")) for n, v in scores.items()}
    rand = np.stack([leftover_curve(err, rng.permutation(len(err))) for _ in range(1000)])
    curves["random_mean"] = rand.mean(0)
    curves["random_p2.5"], curves["random_p97.5"] = np.percentile(rand, 2.5, 0), np.percentile(rand, 97.5, 0)
    C = pd.DataFrame(curves)
    C.insert(0, "images_reviewed", np.arange(len(err) + 1))
    C.insert(1, "share_reviewed", C["images_reviewed"] / len(err))
    run.save_df("B06_leftover_error_curves.csv", C)
    area = {n: float(np.trapezoid(v, C["share_reviewed"])) for n, v in curves.items()}
    plates = D["plate"].to_numpy()
    up = np.unique(plates)
    diffs = {n: [] for n in scores}
    for _ in range(2000):
        pick = np.concatenate([np.flatnonzero(plates == u) for u in rng.choice(up, len(up))])
        e = err[pick]
        r_area = np.mean([np.trapezoid(leftover_curve(e, rng.permutation(len(e))), np.linspace(0, 1, len(e) + 1)) for _ in range(20)])
        for n, v in scores.items():
            diffs[n].append(np.trapezoid(leftover_curve(e, np.argsort(-v[pick], kind="stable")), np.linspace(0, 1, len(e) + 1)) - r_area)
    res = {"target_method": target, "area_left_error": area,
           "area_minus_random_95": {n: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] for n, d in diffs.items()}}
    run.log(f"B06 warning scores for {target} (area under 'error left' curve; lower = better; random {area['random_mean']:.3f}):")
    for n in scores:
        lo, hi = res["area_minus_random_95"][n]
        run.log(f"   {n:40s} area {area[n]:.3f}; minus random 95% [{lo:+.3f}, {hi:+.3f}]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["B05", "B06"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posts", default="orig", choices=["orig", "v2"], help="v2 = settings from the wider v2 clean-up search")
    a = ap.parse_args()
    OPTS.update(seed=a.seed, posts=a.posts)
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    names = {"B05": "crowding_tiny_nuclei", "B06": "warning_flags"}
    suffix = ("_posts_v2" if a.posts == "v2" else "") + (f"_seed{a.seed}" if a.seed else "")
    run = Run(a.exp, names[a.exp] + suffix, config={"images": "150 practice, out-of-fold", "test": "locked", "seed": a.seed, "posts": a.posts})
    preds = predictors(prac)
    run.log(f"methods available: {sorted(preds)}")
    res = b05(run, prac, preds) if a.exp == "B05" else b06(run, prac, preds)
    run.save_json("results.json", res)
    run.finish("measured", notes=f"{a.exp} on saved B01-B03 fold settings/models; methods {sorted(preds)}")


if __name__ == "__main__":
    main()
