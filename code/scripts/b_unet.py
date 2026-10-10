"""B02 / B03 / B04 (+ per-image facts for B05): small U-Net trained from scratch on BBBC039 practice images.

Same 5 plate-grouped folds as B01 (provider test images stay locked). For each outer fold: one of the 4
training folds is the inner validation fold (early stopping + post-processing settings); train on the other 3;
predict the outer fold. No training-image changes except random crops (B04 tests changed images separately).
  --target fg        (B02): background / nucleus; objects by distance-watershed
  --target boundary  (B03): background / nucleus inside / nucleus edge; objects grown from "inside" seeds
Post-processing settings chosen on the inner validation fold, two objectives kept (lowest count error; best
outline score averaged over all overlap levels). B04: each held-out image also scored after brightness,
noise and blur changes at several strengths (all levels reported). Models saved per fold.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed

from ai4s import bbbc039 as B
from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run
from g03_speed_test import UNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STRESS = {"brightness": [0.5, 0.75, 1.25, 1.5], "noise_sd": [0.02, 0.05, 0.1], "blur_sigma": [1.0, 2.0, 3.0]}


def norm(im):
    im = im.astype(np.float32)
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1.5).astype(np.float32)


def target_of(lab, kind):
    if kind == "fg":
        return (lab > 0).astype(np.int64)
    edge = find_boundaries(lab, mode="inner") & (lab > 0)
    t = (lab > 0).astype(np.int64)
    t[edge] = 2
    return t


def load(stems, kind):
    X = np.stack([norm(B.read_image(s)) for s in stems])
    Y = np.stack([target_of(B.read_labels(s), kind) for s in stems])
    L = [B.read_labels(s) for s in stems]
    return X, Y, L


def train(Xtr, Ytr, Xva, Yva, n_cls, seed, max_steps, patience_checks=6, noise_max=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1000)      # separate stream: crop sequence stays identical to unchanged training
    m = UNet(out_ch=n_cls).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    xv = torch.from_numpy(Xva[:, None, :512, :688]).to(DEV)
    yv = torch.from_numpy(Yva[:, :512, :688]).to(DEV)
    best, best_state, bad, hist = 1e9, None, 0, []
    for step in range(1, max_steps + 1):
        idx = rng.integers(0, len(Xtr), 16)
        ys, xs = rng.integers(0, Xtr.shape[1] - 256, 16), rng.integers(0, Xtr.shape[2] - 256, 16)
        xb = np.stack([Xtr[i, a:a + 256, b:b + 256] for i, a, b in zip(idx, ys, xs)])[:, None]
        yb = np.stack([Ytr[i, a:a + 256, b:b + 256] for i, a, b in zip(idx, ys, xs)])
        if noise_max > 0:      # training change (B04 follow-up): per-crop Gaussian noise, sd ~ Uniform(0, noise_max)
            sd = noise_rng.uniform(0, noise_max, (len(xb), 1, 1, 1))
            xb = np.clip(xb + noise_rng.normal(0, 1, xb.shape) * sd, 0, 1.5).astype(np.float32)
        m.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(m(torch.from_numpy(xb).to(DEV)), torch.from_numpy(yb).to(DEV))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 250 == 0:
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                vl = float(sum(F.cross_entropy(m(xv[i:i + 4]), yv[i:i + 4]).item() * len(xv[i:i + 4]) for i in range(0, len(xv), 4)) / len(xv))
            hist.append({"step": step, "train_loss": float(loss.item()), "val_loss": vl})
            if vl < best:
                best, best_state, bad = vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience_checks:
                    break
    m.load_state_dict(best_state)
    return m.eval(), hist


@torch.no_grad()
def predict_prob(m, X):
    out = []
    for i in range(0, len(X), 4):
        x = torch.from_numpy(np.pad(X[i:i + 4], ((0, 0), (0, 8), (0, 8)), mode="reflect")[:, None]).to(DEV)  # 528x704
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            p = torch.softmax(m(x).float(), 1)[:, :, :520, :696]
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def to_objects(prob, kind, thr, min_distance, min_size):
    if kind == "fg":
        fg = ndi.binary_fill_holes(prob[1] > thr)
        dist = ndi.distance_transform_edt(fg)
        lab_fg, _ = ndi.label(fg)
        pk = peak_local_max(dist, min_distance=min_distance, labels=lab_fg, exclude_border=False)
        mk = np.zeros(fg.shape, np.int32)
        mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        lab = watershed(-dist, mk, mask=fg)
    else:
        fg = (prob[1] + prob[2]) > thr
        seeds, _ = ndi.label(prob[1] > thr)
        lab = watershed(-prob[1], seeds, mask=fg)
    if min_size > 0:
        sizes = np.bincount(lab.ravel())
        small = np.isin(lab, np.flatnonzero(sizes <= min_size)) & (lab > 0)
        lab[small] = 0
    return lab


def choose_post(probs, labels, kind):
    grid = list(itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [2, 4, 6, 8, 10] if kind == "fg" else [0], [0, 10, 25, 50]))
    rows = []
    for thr, md, ms in grid:
        rs = [S.score_image(l, to_objects(p, kind, thr, md, ms)) for p, l in zip(probs, labels)]
        sm = S.summarize(rs)
        rows.append({"thr": thr, "min_distance": md, "min_size": ms, "count_mae": sm["count_mae"],
                     "mean_f1_levels": float(np.mean([sm[f"mean_image_f1@{lv:.2f}"] for lv in S.DEFAULT_IOU_LEVELS]))})
    g = pd.DataFrame(rows)
    pick = {"best_count": g.sort_values(["count_mae", "mean_f1_levels"], ascending=[True, False]).iloc[0].to_dict(),
            "best_outline": g.sort_values(["mean_f1_levels", "count_mae"], ascending=[False, True]).iloc[0].to_dict()}
    return pick, g


def stressed(x, kind, level):
    if kind == "brightness":
        return np.clip(x * level, 0, 1.5).astype(np.float32)
    if kind == "noise_sd":
        return np.clip(x + np.random.default_rng(0).normal(0, level, x.shape), 0, 1.5).astype(np.float32)
    return ndi.gaussian_filter(x, level).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="fg", choices=["fg", "boundary"])
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train_noise_max", type=float, default=0.0,
                    help="B04 follow-up training change: per-crop noise sd ~ U(0, this). 0 = unchanged training")
    ap.add_argument("--folds", default="", help="comma list of outer folds to run (default all); join with b_unet_combine.py")
    a = ap.parse_args()
    exp = "B02" if a.target == "fg" else "B03"
    tag = f"_folds{a.folds.replace(',', '-')}" if a.folds else ""
    variant = "_noisetrain" if a.train_noise_max > 0 else ""
    run = Run(exp, f"unet_{a.target}{variant}_seed{a.seed}{tag}", config={"train_noise_max": a.train_noise_max,"folds_run": a.folds or "all", "target": a.target, "max_steps": a.max_steps, "seed": a.seed,
                                                           "folds": "5 plate-grouped (splits/bbbc039_splits.csv)", "stress": STRESS,
                                                           "device": DEV, "early_stopping": "inner validation loss every 250 steps"})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    if a.smoke:
        prac = prac.groupby("cv_fold").head(3).reset_index(drop=True)
        a.max_steps = 500
    n_cls = 2 if a.target == "fg" else 3
    folds = sorted(prac["cv_fold"].unique())
    X, Y, L = load(prac["stem"].tolist(), a.target)
    oof = {"best_count": [], "best_outline": []}
    stress_rows, info = [], []
    run_folds = [int(f) for f in a.folds.split(",")] if a.folds else folds
    for k in run_folds:
        inner = folds[(folds.index(k) + 1) % len(folds)]
        te = (prac["cv_fold"] == k).to_numpy()
        va = (prac["cv_fold"] == inner).to_numpy()
        tr = ~te & ~va
        t0 = time.time()
        m, hist = train(X[tr], Y[tr], X[va], Y[va], n_cls, a.seed, a.max_steps, noise_max=a.train_noise_max)
        train_s = time.time() - t0
        torch.save(m.state_dict(), run.path(f"model_fold{k}.pt"))
        pick, grid = choose_post(predict_prob(m, X[va]), [L[i] for i in np.flatnonzero(va)], a.target)
        grid.to_csv(run.path(f"post_grid_fold{k}.csv"), index=False)
        t1 = time.time()
        pte = predict_prob(m, X[te])
        pred_s = (time.time() - t1) / max(te.sum(), 1)
        info.append({"fold": int(k), "inner_val_fold": int(inner), "train_seconds": train_s, "predict_s_per_image": pred_s,
                     "steps_run": hist[-1]["step"] if hist else 0, "history": hist, "post": pick})
        for name, s in pick.items():
            for i, p in zip(np.flatnonzero(te), pte):
                lab = to_objects(p, a.target, s["thr"], int(s["min_distance"]), int(s["min_size"]))
                r = S.score_image(L[i], lab)
                areas = np.bincount(L[i].ravel())[1:]
                r.update({"stem": prac.loc[i, "stem"], "plate": prac.loc[i, "plate"], "cv_fold": int(k),
                          "ref_median_area": float(np.median(areas)) if len(areas) else 0.0})
                oof[name].append(r)
        s = pick["best_outline"]
        for kind, levels in STRESS.items():
            for lv in levels:
                pst = predict_prob(m, np.stack([stressed(X[i], kind, lv) for i in np.flatnonzero(te)]))
                for i, p in zip(np.flatnonzero(te), pst):
                    r = S.score_image(L[i], to_objects(p, a.target, s["thr"], int(s["min_distance"]), int(s["min_size"])))
                    stress_rows.append({"stem": prac.loc[i, "stem"], "cv_fold": int(k), "stress": kind, "level": lv,
                                        "count_error": r["count_error"], "abs_count_error": r["abs_count_error"],
                                        "f1@0.50": r["f1@0.50"], "f1@0.75": r["f1@0.75"]})
        run.log(f"fold {k}: trained {train_s:.0f} s, steps {info[-1]['steps_run']}, post {pick['best_outline']}")
    run.save_json("fold_info.json", info)
    results = {}
    for name, rows in oof.items():
        df = pd.DataFrame(rows)
        run.save_df(f"oof_full_{name}.csv", df)
        run.save_df(f"oof_per_image_{name}.csv", df.drop(columns=[c for c in df.columns if c.startswith(("tp@", "fp@", "fn@"))]))
        res = {"all": S.summarize(rows), "per_fold": {int(k): S.summarize(d.to_dict("records")) for k, d in df.groupby("cv_fold")}}
        results[name] = res
        aa = res["all"]
        run.log(f"{exp} {name}: count MAE {aa['count_mae']:.2f} (bias {aa['count_bias']:.2f}); pooled F1 0.25/0.50/0.75/0.90 "
                f"{aa['pooled_f1@0.25']:.3f}/{aa['pooled_f1@0.50']:.3f}/{aa['pooled_f1@0.75']:.3f}/{aa['pooled_f1@0.90']:.3f}; "
                f"joined {aa['joined_ref']} split {aa['split_pred']} missed {aa['missed']} false {aa['false']}")
    st = pd.DataFrame(stress_rows)
    run.save_df("B04_stress_per_image.csv", st)
    if len(st):
        agg = st.groupby(["stress", "level"]).agg(count_mae=("abs_count_error", "mean"), bias=("count_error", "mean"),
                                                   f1_050=("f1@0.50", "mean"), f1_075=("f1@0.75", "mean")).round(3)
        run.log("B04 stress tests (best-outline settings):\n" + agg.to_string())
        results["B04_stress"] = agg.reset_index().to_dict(orient="records")
    run.save_json("results.json", results)
    run.finish("measured", notes=f"{exp} U-Net ({a.target}) 5-fold OOF on practice; B04 stress included; test locked")


if __name__ == "__main__":
    main()
