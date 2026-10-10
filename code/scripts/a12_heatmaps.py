"""A12: do heat-map explanations show what the chip-photo AI really uses?

Set-up = A02's best measured model (frozen vit_large_patch14_dinov2 @518 + logistic regression, C per fold from
A02's fold_settings.json). Every practice photo is explained by the classifier of the fold that did NOT see it.
Practice photos only; the final test stays locked.

Map = hide one square of a grid (4x4 and 8x8) and record how much the "bad" score (logit) drops. Because the
classifier is linear on the features, one set of hidden-square features serves several classifiers:
  real      the trained classifier
  shuffled  same recipe trained on shuffled labels (data randomisation check)
  random    random weights of the same size (model randomisation check)
If real maps look like shuffled/random maps, the maps do not reflect what the AI learned.
Checks (all reported as full distributions):
  - rank similarity (Spearman) of real vs shuffled and real vs random maps
  - fill check: hiding with the photo's median grey vs with a blurred copy of the square (hiding can itself
    create a new "defect"; if the two fills disagree, the map partly reflects the fill, not the photo)
  - deletion check: hide the most important squares first vs random squares, at every step of 1/8 of the photo
  - border share: share of positive importance on the outer ring of squares (links to A04: edges vs cells)
Gallery: confidently right bad, right good, and wrong photos per fold, with maps.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr

from ai4s.paths import OOC_DIR, SPLITS_ROOT
from ai4s.runlog import Run, latest_run
from a_views import Extractor

GRIDS = [4, 8]
FILLS = ["median", "blur"]
DEL_STEPS = 8
SEED = 20261006
TAG = "vit_large_patch14_dinov2@518"


def boxes(h, w, n):
    ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
    return [(ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(n) for j in range(n)]


def hide(a, bxs, fill, blurred):
    b = a.copy()
    for y0, y1, x0, x1 in bxs:
        b[y0:y1, x0:x1] = np.uint8(np.median(a)) if fill == "median" else blurred[y0:y1, x0:x1]
    return b


def heads(Xtr, ytr, C, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    real = LogisticRegression(C=C, max_iter=5000).fit(Z, ytr)
    shuf = LogisticRegression(C=C, max_iter=5000).fit(Z, rng.permutation(ytr))
    w = rng.normal(size=real.coef_.shape[1])
    w *= np.linalg.norm(real.coef_) / np.linalg.norm(w)
    out = {"real": (real.coef_[0], real.intercept_[0]), "shuffled": (shuf.coef_[0], shuf.intercept_[0]), "random": (w, 0.0)}
    return sc, out


def logit(sc, head, F_):
    return sc.transform(F_.reshape(-1, F_.shape[-1])) @ head[0] + head[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="", help="comma list of practice folds to explain (default all)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fills", default="median,blur", help="hiding styles (user budget choice 6 Oct: median only for folds 2-4)")
    a = ap.parse_args()
    fills = a.fills.split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, side = ("resnet18.tv_in1k", 224) if a.smoke else ("vit_large_patch14_dinov2.lvd142m", 518)
    a02 = latest_run("A02_frozen_features")
    run = Run("A12", "heatmaps" + (f"_folds{a.folds.replace(',', '-')}" if a.folds else "") + ("_smoke" if a.smoke else ""),
              config={"model": f"{model_name}@{side}", "grids": GRIDS, "fills": fills, "deletion_steps": DEL_STEPS,
                      "a02_run": a02.name, "smoke": a.smoke})
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    prac = sp[sp["role"] == "practice"].reset_index(drop=True)
    ids_a02 = pd.read_csv(a02 / "features_practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    prac = prac.set_index("image_id").loc[ids_a02].reset_index()
    tag = "resnet18@224" if a.smoke else TAG
    Xall = np.load(a02 / f"features_practice_{tag}.npy").astype(np.float32)
    Cs = {int(f["fold"]): f["C"] for f in json.load(open(a02 / "fold_settings.json"))[tag]}
    oof = pd.read_csv(a02 / "oof_predictions.csv", dtype={"image_id": str})
    oof = oof[oof["method"] == tag].set_index("image_id")
    ex = Extractor(model_name, side, dev, run)
    ex.quiet = True                                   # thousands of small calls; totals saved in feature_timing.json
    rng = np.random.default_rng(SEED)
    folds = [int(f) for f in a.folds.split(",")] if a.folds else sorted(prac["dev_fold"].unique())
    rows, maps_out = [], {}
    for k in folds:
        tr = (prac["dev_fold"] != k).to_numpy()
        sc, H = heads(Xall[tr], prac.loc[tr, "label_bad"].to_numpy(), Cs[k], rng)
        te_idx = np.flatnonzero(~tr)
        if a.smoke:
            te_idx = te_idx[:6]
        t0 = time.perf_counter()
        for i in te_idx:
            iid = prac.loc[i, "image_id"]
            img = np.load(OOC_DIR / "gray1036" / f"{iid}.npy")
            blurred = ndi.gaussian_filter(img.astype(np.float32), 8).astype(np.uint8)
            base = {h: float(logit(sc, H[h], Xall[i][None])[0]) for h in H}
            r = {"image_id": iid, "dev_fold": int(k), "label_bad": int(prac.loc[i, "label_bad"]),
                 "p_bad_a02": float(oof.loc[iid, "p_bad"]), "cell_type": prac.loc[i, "cell_type"]}
            for n in GRIDS:
                bx = boxes(*img.shape, n)
                ring = np.array([(q // n in (0, n - 1)) or (q % n in (0, n - 1)) for q in range(n * n)])
                for fill in fills:
                    Fh = ex(f"hide_{n}_{fill}", [hide(img, [b], fill, blurred) for b in bx])
                    imp = {h: base[h] - logit(sc, H[h], Fh) for h in H}      # drop in "bad" logit when hidden
                    maps_out[f"{iid}|{n}|{fill}"] = np.stack([imp[h] for h in ("real", "shuffled", "random")]).astype(np.float16)
                    r[f"rho_real_shuffled_{n}_{fill}"] = float(spearmanr(imp["real"], imp["shuffled"])[0])
                    r[f"rho_real_random_{n}_{fill}"] = float(spearmanr(imp["real"], imp["random"])[0])
                    pos = np.clip(imp["real"], 0, None)
                    r[f"border_share_{n}_{fill}"] = float(pos[ring].sum() / max(pos.sum(), 1e-9))
                    r[f"border_area_share_{n}"] = float(ring.mean())
                if "blur" in fills:
                    r[f"rho_fill_median_vs_blur_{n}"] = float(spearmanr(maps_out[f"{iid}|{n}|median"][0].astype(float),
                                                                        maps_out[f"{iid}|{n}|blur"][0].astype(float))[0])
            # deletion check on the 8x8 grid, median fill: most important first vs random order
            n = GRIDS[-1]
            bx = boxes(*img.shape, n)
            imp = maps_out[f"{iid}|{n}|median"][0].astype(float)
            orders = {"important_first": np.argsort(-imp, kind="stable"), "random": rng.permutation(n * n)}
            steps = [int(round(s * n * n / DEL_STEPS)) for s in range(1, DEL_STEPS + 1)]
            for oname, order in orders.items():
                Fd = ex(f"delete_{oname}", [hide(img, [bx[q] for q in order[:m]], "median", blurred) for m in steps])
                drop = base["real"] - logit(sc, H["real"], Fd)
                for s, v in zip(steps, drop):
                    r[f"del_{oname}_{s}"] = float(v)
            rows.append(r)
        run.log(f"fold {k}: explained {len(te_idx)} photos in {time.perf_counter() - t0:.0f} s")
    D = pd.DataFrame(rows)
    run.save_df("per_photo.csv", D)
    np.savez_compressed(run.path("maps.npz"), **{k.replace("|", "__"): v for k, v in maps_out.items()})
    summ = {}
    for c in D.columns:
        if c.startswith(("rho_", "border_share", "del_")):
            v = D[c].dropna().to_numpy()
            summ[c] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    run.save_json("results.json", summ)
    for c in [c for c in summ if c.startswith(("rho_", "border_share"))]:
        run.log(f"{c:36s} median {summ[c]['median']:+.3f} (10-90%: {summ[c]['p10']:+.3f} .. {summ[c]['p90']:+.3f})")
    run.log("deletion (drop in 'bad' logit, median over photos), important-first vs random: " +
            ", ".join(f"{s}/{GRIDS[-1] ** 2}: {summ[f'del_important_first_{s}']['median']:+.2f} vs {summ[f'del_random_{s}']['median']:+.2f}"
                      for s in [int(round(q * GRIDS[-1] ** 2 / DEL_STEPS)) for q in range(1, DEL_STEPS + 1)]))
    run.save_json("feature_timing.json", ex.timing)
    gallery(run, D, maps_out)
    run.finish("measured", notes=f"A12 heat maps for folds {folds}; practice only")


def gallery(run, D, maps_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = D.assign(pred_bad=(D["p_bad_a02"] >= 0.5).astype(int))   # 0.5 only to pick gallery examples, not for scoring
    picks = pd.concat([D[(D.label_bad == 1) & (D.pred_bad == 1)].nlargest(2, "p_bad_a02"),
                       D[(D.label_bad == 0) & (D.pred_bad == 0)].nsmallest(2, "p_bad_a02"),
                       D[D.label_bad != D.pred_bad].assign(gap=lambda d: (d.p_bad_a02 - d.label_bad).abs()).nlargest(2, "gap")])
    n = GRIDS[-1]
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, (_, p) in zip(axes, picks.iterrows()):
        img = np.load(OOC_DIR / "gray1036" / f"{p['image_id']}.npy")
        m = maps_out[f"{p['image_id']}|{n}|median"].astype(float)
        row[0].imshow(img, cmap="gray")
        row[0].set_title(f"{p['image_id']} label {'bad' if p['label_bad'] else 'good'}, p_bad {p['p_bad_a02']:.2f}", fontsize=8)
        for ax, h, mm in zip(row[1:], ("real", "shuffled labels", "random weights"), m):
            lim = max(abs(mm).max(), 1e-9)
            ax.imshow(img, cmap="gray")
            ax.imshow(np.kron(mm.reshape(n, n), np.ones((img.shape[0] // n + 1, img.shape[1] // n + 1)))[:img.shape[0], :img.shape[1]],
                      cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.5)
            ax.set_title(h, fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(run.path("gallery.png"), dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
