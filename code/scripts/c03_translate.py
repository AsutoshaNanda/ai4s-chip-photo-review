"""C03: is predicting the glowing (lamin B1) image from the plain DIC photo useful for a real measurement?

Data: c03_prep.py output (40 Allen stacks, 5 depth planes each, 0.34 um pixels). Groups = wells (B02..B05);
4 folds, each holds out one whole well; the next well (in order) is the inner check set for early stopping;
training uses the other two wells. Research use only; Allen data is not re-shared.

Route A (predict the picture): U-Net regresses the lamin image from DIC; nuclei are then measured on the
          predicted image with exactly the reference measurement (c03_prep.nuclei_from_lamin).
Route B (measure directly): U-Net labels each pixel of the DIC photo as background / nucleus inside / envelope
          ring, trained on the reference measurement; nuclei = connected "inside" regions.
Reference: the same measurement on the real lamin image.
Scores on held-out wells (all 5 planes and the clearest plane z* separately): nucleus count error, F1 at every
overlap level, joined/split/missed/false; route A also gets pixel correlation with the real image.
Route A minus route B: 95% interval from resampling held-out stacks.
"""
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

from ai4s import metrics_seg as S
from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run
from c03_prep import nuclei_from_lamin
from g03_speed_test import UNet

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 20261006
CROP, BATCH, EVERY, PATIENCE, MAX_STEPS = 192, 16, 200, 6, 12000   # v2: cap raised (v1 fold 2 hit 6000 while still improving)


def pnorm(a):
    lo, hi = np.percentile(a, [1, 99.8])
    return ((a - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def ring_target(lab):
    from skimage.segmentation import find_boundaries
    t = (lab > 0).astype(np.int64)
    t[find_boundaries(lab, mode="inner") & (lab > 0)] = 2
    return t


def train(route, Xtr, Ytr, Xva, Yva, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = UNet(out_ch=1 if route == "A" else 3).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = (lambda o, y: F.mse_loss(o[:, 0], y)) if route == "A" else (lambda o, y: F.cross_entropy(o, y))
    xv = torch.from_numpy(Xva[:, None]).to(DEV)
    yv = torch.from_numpy(Yva).to(DEV)
    best, state, bad, hist = 1e9, None, 0, []
    H, W = Xtr.shape[1:]
    for step in range(1, MAX_STEPS + 1):
        idx = rng.integers(0, len(Xtr), BATCH)
        ys, xs = rng.integers(0, H - CROP + 1, BATCH), rng.integers(0, W - CROP + 1, BATCH)
        xb = np.stack([Xtr[i, a:a + CROP, b:b + CROP] for i, a, b in zip(idx, ys, xs)])[:, None]
        yb = np.stack([Ytr[i, a:a + CROP, b:b + CROP] for i, a, b in zip(idx, ys, xs)])
        if rng.random() < 0.5:
            xb, yb = xb[..., ::-1].copy(), yb[..., ::-1].copy()
        m.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(m(torch.from_numpy(xb).to(DEV)), torch.from_numpy(yb).to(DEV))
        loss.backward()
        opt.step()
        if step % EVERY == 0:
            m.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_fn(m(xv[i:i + 4]), yv[i:i + 4]).item() for i in range(0, len(xv), 4)]))
            hist.append({"step": step, "train_loss": float(loss.item()), "val_loss": vl})
            if vl < best:
                best, state, bad = vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    m.load_state_dict(state)
    return m.eval(), hist


@torch.no_grad()
def predict(m, X):
    return np.concatenate([m(torch.from_numpy(X[i:i + 4, None]).to(DEV)).float().cpu().numpy() for i in range(0, len(X), 4)])


def objects_route_b(out):
    cls = out.argmax(0)
    lab, _ = ndi.label(cls == 1)
    return lab.astype(np.int32)


def drop_small(lab, min_size):
    if min_size <= 0:
        return lab
    sizes = np.bincount(lab.ravel())
    keep = sizes > min_size
    keep[0] = False
    out, _ = ndi.label(keep[lab])
    return out.astype(np.int32)



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="c03_slices.npz", help="prepared slices file in data/allen (input-only depth: c03_slices_inputonly.npz)")
    a = ap.parse_args()
    run = Run("C03", "dic_to_lamin_v3" + ("_inputonly" if "inputonly" in a.slices else ""), config={"slices": a.slices, "version": "v3: input-only depth option, clean-up range extended at the upper edge; v2 settings otherwise", "device": DEV, "crop": CROP, "batch": BATCH, "check_every": EVERY,
                                             "patience_checks": PATIENCE, "max_steps": MAX_STEPS, "seed": SEED})
    d = np.load(DATA_ROOT / "allen" / a.slices)
    dic, lam, labels, wells, zi = d["dic"], d["lamin"], d["labels"], d["wells"], d["zstar_index"]
    n, P = dic.shape[:2]
    Xn = np.stack([[pnorm(dic[i, p]) for p in range(P)] for i in range(n)])
    Ln = np.stack([[pnorm(lam[i, p]) for p in range(P)] for i in range(n)])
    Tb = np.stack([[ring_target(labels[i, p]) for p in range(P)] for i in range(n)])
    order = sorted(np.unique(wells))
    rows, hist_all = [], {}
    for k, w in enumerate(order):
        inner = order[(k + 1) % len(order)]
        te, va = wells == w, wells == inner
        tr = ~te & ~va
        flat = lambda a, msk: a[msk].reshape(-1, *a.shape[2:])
        t0 = time.time()
        mA, hA = train("A", flat(Xn, tr), flat(Ln, tr), flat(Xn, va), flat(Ln, va), SEED + k)
        mB, hB = train("B", flat(Xn, tr), flat(Tb, tr), flat(Xn, va), flat(Tb, va), SEED + k)
        hist_all[w] = {"A": hA, "B": hB}
        torch.save(mA.state_dict(), run.path(f"routeA_fold{k}.pt"))
        torch.save(mB.state_dict(), run.path(f"routeB_fold{k}.pt"))
        # clean-up rule (same for both routes): drop objects <= min_size px; min_size chosen on the CHECK well by
        # count error, from a grid spanning 0 .. the training wells' 25th-percentile reference nucleus area
        tr_areas = np.concatenate([np.bincount(labels[i, p].ravel())[1:] for i in np.flatnonzero(tr) for p in range(P)])
        grid = np.unique(np.linspace(0, np.percentile(tr_areas[tr_areas > 0], 25), 11).astype(int))
        va_raw = {"A": [], "B": []}
        for i in np.flatnonzero(va):
            pa, pb = predict(mA, Xn[i])[:, 0], predict(mB, Xn[i])
            for p in range(P):
                va_raw["A"].append((nuclei_from_lamin(pa[p]), labels[i, p]))
                va_raw["B"].append((objects_route_b(pb[p]), labels[i, p]))
        min_size, edge_log = {}, {}
        area_max = int(tr_areas.max())                     # valid limit: above the largest training nucleus nothing is left
        for route_key, pairs in va_raw.items():
            g_list = [int(v) for v in grid]
            step = max(int(g_list[-1] - g_list[-2]), 1) if len(g_list) > 1 else 1
            errs = {g_: np.mean([abs(int(drop_small(l, g_).max()) - int(r.max())) for l, r in pairs]) for g_ in g_list}
            # improvements.md (C03): check clean-up range boundaries; an upper-edge choice extends the range by the same step
            while min(errs, key=errs.get) == max(errs) and max(errs) + step <= area_max:
                g_new = max(errs) + step
                errs[g_new] = np.mean([abs(int(drop_small(l, g_new).max()) - int(r.max())) for l, r in pairs])
            min_size[route_key] = int(min(errs, key=errs.get))
            edge_log[route_key] = {"searched": sorted(errs), "on_upper_edge_after_extension": min_size[route_key] == max(errs)}
        run.log(f"fold {k}: clean-up min size chosen on check well {inner}: {min_size}; ranges {edge_log}")
        run.log(f"fold {k} (test well {w}, check well {inner}): trained in {time.time() - t0:.0f} s; steps A {hA[-1]['step']} B {hB[-1]['step']}")
        for i in np.flatnonzero(te):
            pa = predict(mA, Xn[i])[:, 0]
            pb = predict(mB, Xn[i])
            for p in range(P):
                ref = labels[i, p]
                la_raw, lb_raw = nuclei_from_lamin(pa[p]), objects_route_b(pb[p])
                la, lb = drop_small(la_raw, min_size["A"]), drop_small(lb_raw, min_size["B"])
                r_corr = float(np.corrcoef(pa[p].ravel(), Ln[i, p].ravel())[0, 1])
                for route, lab in (("A_predict_picture", la), ("B_direct", lb), ("A_no_cleanup", la_raw), ("B_no_cleanup", lb_raw)):
                    r = S.score_image(ref, lab)
                    rows.append({"stack": str(d["files"][i]), "well": w, "plane": p, "is_zstar": int(p == zi[i]), "route": route,
                                 "pixel_corr": r_corr if route.startswith("A") else np.nan, **r})
    D = pd.DataFrame(rows)
    run.save_df("per_plane.csv", D.drop(columns=[c for c in D.columns if c.startswith(("tp@", "fp@", "fn@"))]))
    run.save_json("training_history.json", hist_all)
    res = {}
    for scope, dd in (("all_planes", D), ("zstar_plane", D[D.is_zstar == 1])):
        for route, g in dd.groupby("route"):
            s = S.summarize(g.to_dict("records"))
            res[f"{scope}|{route}"] = s
            run.log(f"{scope:12s} {route:18s} planes {s['images']:4d} | count MAE {s['count_mae']:.2f} bias {s['count_bias']:+.2f} | "
                    f"F1 0.50/0.75 {s['pooled_f1@0.50']:.3f}/{s['pooled_f1@0.75']:.3f} | joined {s['joined_ref']} split {s['split_pred']} "
                    f"missed {s['missed']} false {s['false']}" + (f" | pixel corr {g['pixel_corr'].mean():.3f}" if route.startswith("A") else ""))
        a = dd[dd.route == "A_predict_picture"].groupby("stack")["abs_count_error"].mean()
        b = dd[dd.route == "B_direct"].groupby("stack")["abs_count_error"].mean().loc[a.index]
        rng = np.random.default_rng(SEED)
        diffs = [float((a.iloc[ix] - b.iloc[ix]).mean()) for ix in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        res[f"{scope}|A_minus_B_count_MAE_95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        run.log(f"{scope}: route A minus route B count MAE, 95% over stacks: {np.round(res[f'{scope}|A_minus_B_count_MAE_95'], 2).tolist()}")
    run.save_json("results.json", res)
    run.finish("measured", notes="C03 DIC->lamin, 4 well-held-out folds; research use only")


if __name__ == "__main__":
    main()
