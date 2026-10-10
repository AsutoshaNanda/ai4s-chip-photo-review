"""A02 part 2 (improvements.md section 4): light retraining of the big image model vs its frozen version.

"Light retraining" as defined in the experiment playbook (A02): only the FINAL block of the model is trained
(plus its final normalisation and a one-number output layer); every earlier block stays frozen. Model and photo
preparation are the same as A02's best frozen set-up: vit_large_patch14_dinov2.lvd142m at long side 518, grey
photos copied to 3 channels. Because the earlier blocks are frozen and no training-time image changes are used,
their output is computed once per photo and cached; training then runs the last block only. This gives exactly
the same model outputs as running the whole network, at a fraction of the cost (checked in --mode timing).

Two variants, trained by the same procedure on the same cached inputs:
  head_only    final block frozen, only the output layer trained  (frozen control, same training procedure)
  last_block   final block + final normalisation + output layer trained  (light retraining)
The saved A02 frozen logistic-regression predictions remain the main reference (same 5 practice folds).

Settings (learning speed, number of passes over the data) are chosen inside each training fold only, from
grouped inner validation (lowest mean inner log loss). A learning speed chosen at the edge of the searched range
triggers one more step past that edge; a best pass count equal to the maximum triggers a longer run. Yes/no
cut-off from the inner validation predictions at the chosen setting. Practice photos only; final test locked.

  --mode timing : cache inputs for all practice photos, check equality with the full model, time a few passes
                  and record peak GPU memory. No results are produced.
  --mode full   : the comparison on all 5 practice folds.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from a01_simple_baselines import inner_cv
from a02_frozen import summarise, target_shape
from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, OOC_DIR, RUNS_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, latest_run

MODEL, SIDE = "vit_large_patch14_dinov2.lvd142m", 518
SEED = 20261005
try:
    _BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
except TypeError:                                        # older torch: no emulation flag
    _BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
DT = torch.bfloat16 if _BF16 else torch.float16


def practice():
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    return sp[sp["role"] == "practice"].merge(fmt[["member", "photo_format"]], on="member").reset_index(drop=True)


def load_model(dev):
    import timm
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def prep(arrays, shape, m, dev):
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=dev)[:, None, None]
    std = torch.tensor(cfg["std"], device=dev)[:, None, None]
    x = torch.stack([torch.from_numpy(a) for a in arrays]).float().div(255)[:, None].to(dev)
    x = F.interpolate(x, size=shape, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
    return (x - mean) / std


@torch.no_grad()
def cache_inputs(m, ids, dev, run):
    """Output of all blocks except the last, per photo (kept on the GPU), grouped by token shape."""
    arrays = {i: np.load(OOC_DIR / "gray1036" / f"{i}.npy") for i in ids}
    by_shape = {}
    for i, a in arrays.items():
        by_shape.setdefault(target_shape(*a.shape, SIDE), []).append(i)
    cache, t0 = {}, time.perf_counter()
    for shape, members in by_shape.items():
        for b in range(0, len(members), 32):
            chunk = members[b:b + 32]
            with torch.autocast("cuda", dtype=DT):
                x = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(prep([arrays[i] for i in chunk], shape, m, dev)))))
                for blk in m.blocks[:-1]:
                    x = blk(x)
            for i, v in zip(chunk, x.to(DT)):
                cache[i] = v
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    run.log(f"cached last-block inputs for {len(ids)} photos in {secs:.1f} s; shapes {list(by_shape)}; "
            f"GPU memory used {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return cache, arrays, by_shape, secs


class Top(torch.nn.Module):
    """Last block + final norm + output layer, starting from the pretrained weights."""
    def __init__(self, m, train_block):
        super().__init__()
        import copy
        self.block, self.norm = copy.deepcopy(m.blocks[-1]), copy.deepcopy(m.norm)
        self.head = torch.nn.Linear(m.num_features, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.train_block = train_block
        for p in list(self.block.parameters()) + list(self.norm.parameters()):
            p.requires_grad_(train_block)

    def forward(self, x):
        return self.head(self.norm(self.block(x))[:, 0]).squeeze(1)    # class token, as the frozen A02 features


def batches(ids, cache, shape_of, bs, rng=None):
    """Batches of photos with the same token shape."""
    groups = {}
    for i in ids:
        groups.setdefault(shape_of[i], []).append(i)
    out = []
    for members in groups.values():
        members = list(members)
        if rng is not None:
            rng.shuffle(members)
        out += [members[b:b + bs] for b in range(0, len(members), bs)]
    if rng is not None:
        rng.shuffle(out)
    return out


@torch.no_grad()
def predict(top, ids, cache, shape_of, bs=64):
    top.eval()
    out = {}
    for chunk in batches(ids, cache, shape_of, bs):
        with torch.autocast("cuda", dtype=DT):
            z = top(torch.stack([cache[i] for i in chunk]))
        out.update(dict(zip(chunk, torch.sigmoid(z.float()).cpu().numpy())))
    return np.array([out[i] for i in ids])


def train(m, train_block, tr_ids, y, cache, shape_of, lr, epochs, bs, seed, eval_ids=None, eval_y=None):
    """Returns the trained top and the eval log loss after every pass (if eval photos given)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    top = Top(m, train_block).cuda()
    opt = torch.optim.AdamW([p for p in top.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=DT == torch.float16)     # float16 (T4) needs loss scaling; bfloat16 does not
    yv = dict(zip(tr_ids, y))
    curve = []
    for _ in range(epochs):
        top.train()
        for chunk in batches(tr_ids, cache, shape_of, bs, rng):
            with torch.autocast("cuda", dtype=DT):
                z = top(torch.stack([cache[i] for i in chunk]))
            loss = F.binary_cross_entropy_with_logits(z.float(), torch.tensor([yv[i] for i in chunk], dtype=torch.float32, device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if eval_ids is not None:
            curve.append(C.log_loss(eval_y, predict(top, eval_ids, cache, shape_of)))
    return top, curve


def choose(m, train_block, ids, y, groups, cache, shape_of, lrs, max_epochs, bs, n_inner, run, label, max_seconds):
    """Grouped inner validation inside the training fold: mean inner log loss for each (lr, passes)."""
    splits = inner_cv(groups, y)[:n_inner]
    lrs = sorted(lrs)
    table = {}
    t_start = time.perf_counter()
    unresolved = []

    def score(lr, E):
        curves, preds = [], []
        for j, (a, b) in enumerate(splits):
            top, cur = train(m, train_block, [ids[t] for t in a], y[a], cache, shape_of, lr, E, bs, SEED + j,
                             [ids[t] for t in b], y[b])
            curves.append(cur)
            preds.append((b, predict(top, [ids[t] for t in b], cache, shape_of)))
        table[lr] = np.mean(curves, axis=0)
        return preds

    while True:
        for lr in lrs:
            if lr not in table or len(table[lr]) < max_epochs:
                score(lr, max_epochs)
        best_lr = min(lrs, key=lambda lr: table[lr].min())
        best_E = int(np.argmin(table[best_lr])) + 1
        edge = best_lr in (lrs[0], lrs[-1])
        if (best_E == max_epochs or edge) and time.perf_counter() - t_start > max_seconds:
            unresolved.append(f"stopped by the per-fold time guard ({max_seconds} s) with best at an edge: lr {best_lr:g}, passes {best_E}/{max_epochs}")
            run.log(f"{label}: " + unresolved[-1])
            break
        if best_E == max_epochs:
            run.log(f"{label}: best pass count equals the maximum {max_epochs}; doubling it")
            max_epochs *= 2
            table.clear()
            continue
        if edge:
            step = lrs[1] / lrs[0]
            new = lrs[0] / step if best_lr == lrs[0] else lrs[-1] * step
            if new in table or not (1e-9 < new < 1.0):
                break
            run.log(f"{label}: best learning speed {best_lr:g} is at the edge of {lrs}; adding {new:g}")
            lrs = sorted(lrs + [new])
            continue
        break
    # inner predictions at the chosen setting, for the yes/no cut-off
    preds = score(best_lr, best_E)
    inner_p = np.zeros(len(ids))
    seen = np.zeros(len(ids), bool)
    for b, p in preds:
        inner_p[b], seen[b] = p, True
    cut = C.choose_cutoff(y[seen], inner_p[seen])["cutoff"]
    return {"lr": best_lr, "passes": best_E, "cutoff": cut, "searched_lrs": lrs, "max_passes": max_epochs,
            "boundary_unresolved": unresolved, "inner_log_loss": {f"{k:g}": [round(float(v), 5) for v in c] for k, c in table.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["timing", "full", "combine", "a03"], required=True)
    ap.add_argument("--combine_runs", default="", help="combine mode: comma list of partial run-name tags to join")
    ap.add_argument("--lrs", default="1e-6,1e-5,1e-4,1e-3", help="starting learning-speed grid (extended at an edge)")
    ap.add_argument("--max-passes", type=int, default=30, help="starting maximum passes (doubled if the best equals it)")
    ap.add_argument("--inner", type=int, default=1, help="number of grouped inner validation splits used for choosing")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-seconds", type=float, default=1800, help="resource guard per fold and variant for range extension (not a result threshold)")
    ap.add_argument("--folds", default="", help="comma list of practice folds (default all)")
    a = ap.parse_args()
    dev = "cuda"
    lrs = [float(v) for v in a.lrs.split(",")]
    if a.mode == "combine":
        return combine(a.combine_runs.split(","))
    run = Run("A02", f"light_retrain_{a.mode}" + (f"_folds{a.folds.replace(',', '-')}" if a.folds else ""), config={
        "model": f"{MODEL}@{SIDE}", "trained": "last block + final norm + output layer (light retraining); head_only control",
        "start_lrs": lrs, "time_guard_seconds": a.max_seconds, "start_max_passes": a.max_passes, "inner_splits": a.inner, "batch": a.batch,
        "optimizer": "AdamW, weight decay 0", "precision": str(DT), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "seed": SEED, "photos": "practice only"})
    prac = practice()
    ids = prac["image_id"].tolist()
    m = load_model(dev)
    torch.cuda.reset_peak_memory_stats()
    cache, arrays, by_shape, cache_secs = cache_inputs(m, ids, dev, run)
    shape_of = {i: tuple(cache[i].shape) for i in ids}

    if a.mode == "a03":
        return a03_mode(a, run, prac, ids, m, cache, shape_of)
    if a.mode == "timing":
        # 1) equality: cached path + pretrained last block == the full model's frozen features
        some = by_shape[next(iter(by_shape))][:8]
        with torch.no_grad(), torch.autocast("cuda", dtype=DT):
            full = m(prep([arrays[i] for i in some], next(iter(by_shape)), m, dev)).float()
            part = m.norm(m.blocks[-1](torch.stack([cache[i] for i in some])))[:, 0].float()
        diff = float((full - part).abs().max() / full.abs().max())
        run.log(f"equality check: largest relative difference between full model and cached path = {diff:.2e}")
        # 2) speed and memory of a few passes, both variants, on fold 0 training photos
        tr = (prac["dev_fold"] != sorted(prac["dev_fold"].unique())[0]).to_numpy()
        tr_ids, y = [ids[t] for t in np.flatnonzero(tr)], prac.loc[tr, "label_bad"].to_numpy()
        timing = {"cache_seconds": cache_secs, "photos": len(ids), "equality_rel_diff": diff}
        for variant, tb in (("head_only", False), ("last_block", True)):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train(m, tb, tr_ids, y, cache, shape_of, 1e-4, 3, a.batch, SEED)
            torch.cuda.synchronize()
            s = (time.perf_counter() - t0) / 3
            timing[variant] = {"seconds_per_pass": s, "photos_per_pass": len(tr_ids),
                               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            run.log(f"{variant}: {s:.2f} s per pass over {len(tr_ids)} photos; peak GPU memory {timing[variant]['peak_gpu_gb']:.1f} GB")
        run.save_json("timing_measured.json", timing)
        run.finish("measured", notes="A02 light retraining: timing and equality check only (no results)")
        return

    folds = [int(f) for f in a.folds.split(",")] if a.folds else sorted(prac["dev_fold"].unique())
    rows, settings = [], {}
    for k in folds:
        tr, te = (prac["dev_fold"] != k).to_numpy(), (prac["dev_fold"] == k).to_numpy()
        tr_ids, te_ids = [ids[t] for t in np.flatnonzero(tr)], [ids[t] for t in np.flatnonzero(te)]
        y, g = prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy()
        for variant, tb in (("head_only", False), ("last_block", True)):
            t0 = time.perf_counter()
            s = choose(m, tb, tr_ids, y, g, cache, shape_of, lrs, a.max_passes, a.batch, a.inner, run, f"fold {k} {variant}", a.max_seconds)
            top, _ = train(m, tb, tr_ids, y, cache, shape_of, s["lr"], s["passes"], a.batch, SEED)
            p = predict(top, te_ids, cache, shape_of)
            s["seconds"] = time.perf_counter() - t0
            settings[f"{k}|{variant}"] = s
            run.log(f"fold {k} {variant}: lr {s['lr']:g}, passes {s['passes']}, {s['seconds']:.0f} s")
            d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].copy()
            d["method"], d["p_bad"], d["pred_bad"], d["cutoff"] = f"vit_large_dinov2@518_{variant}", p, (p >= s["cutoff"]).astype(int), s["cutoff"]
            rows.append(d)
        run.save_json("fold_settings.json", settings)
        pd.concat(rows, ignore_index=True).to_csv(run.path("oof_predictions_partial.csv"), index=False)
    P = pd.concat(rows, ignore_index=True)
    run.save_df("oof_predictions.csv", P)
    if a.folds and sorted(folds) != sorted(prac["dev_fold"].unique()):
        run.finish("measured", notes=f"A02 light retraining, folds {folds} only; join with --mode combine")
        return
    finish_results(run, P)


def a03_mode(a, run, prac, ids, m, cache, shape_of):
    """Repaired A03 rule for the light-retrained candidate: hold out one cell type, remove every filename group used by
    that type from training, train with FIXED settings taken from the 5 practice-fold searches (most common learning speed,
    median number of passes; chosen from training-side validation only, recorded), predict the held-out type."""
    import json
    full = [p for p in sorted((RUNS_ROOT, CODE_ROOT / "runs"), key=str) for p in (p.glob("*_A02_light_retrain_full") if p.exists() else [])]
    assert full, "run --mode combine first (needs the combined A02 light-retraining run)"
    settings = json.load(open(sorted(full)[-1] / "fold_settings.json"))
    fixed = {}
    for variant in ("head_only", "last_block"):
        ss = [v for k, v in settings.items() if k.endswith("|" + variant)]
        lrs = [v["lr"] for v in ss]
        fixed[variant] = {"lr": max(set(lrs), key=lrs.count), "passes": int(np.median([v["passes"] for v in ss])), "from_folds": len(ss)}
    run.log(f"fixed settings from the practice-fold searches: {fixed}")
    rows = []
    for ct in sorted(prac["cell_type"].unique()):
        te = (prac["cell_type"] == ct).to_numpy()
        tr = ~te & ~prac["group"].isin(set(prac.loc[te, "group"])).to_numpy()
        ytr = prac.loc[tr, "label_bad"].to_numpy()
        if len(np.unique(ytr)) < 2:
            run.log(f"{ct}: unavailable (training has one class)")
            continue
        tr_ids, te_ids = [ids[t] for t in np.flatnonzero(tr)], [ids[t] for t in np.flatnonzero(te)]
        for variant, tb in (("head_only", False), ("last_block", True)):
            top, _ = train(m, tb, tr_ids, ytr, cache, shape_of, fixed[variant]["lr"], fixed[variant]["passes"], a.batch, SEED)
            p = predict(top, te_ids, cache, shape_of)
            d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad"]].copy()
            d["method"], d["p_bad"], d["protocol"] = f"vit_large_dinov2@518_{variant}", p, "a03_groups_excluded"
            rows.append(d)
        run.log(f"{ct}: trained on {int(tr.sum())} photos, tested on {int(te.sum())}")
    P = pd.concat(rows, ignore_index=True)
    run.save_df("predictions_a03.csv", P)
    ref = pd.read_csv(latest_run("A03_unseen_cell_types_groups_excluded") / "predictions_groups_excluded.csv", dtype={"image_id": str})
    ref = ref[ref["method"] == "whole"].set_index("image_id")
    res = {"fixed_settings": fixed}
    for variant in ("head_only", "last_block"):
        d = P[P["method"].str.endswith(variant)].set_index("image_id")
        r = ref.loc[d.index]
        y, g, pa, pb = d["label_bad"].to_numpy(), d["group"].to_numpy(), r["p_bad"].to_numpy(), d["p_bad"].to_numpy()
        vals = C.grouped_bootstrap(lambda idx: C.roc_auc(y[idx], pb[idx]) - C.roc_auc(y[idx], pa[idx]), g, n_boot=1000, seed=SEED)
        res[variant] = {"pooled": C.summary(y, pb), "per_cell_type": C.by_group(y, pb, d["cell_type"].to_numpy()),
                        "per_photo_format": C.by_group(y, pb, d["photo_format"].to_numpy()),
                        "minus_frozen_LR": {"difference": float(C.roc_auc(y, pb) - C.roc_auc(y, pa)),
                                            "range_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}}
        run.log(f"A03 rule, {variant}: pooled AUC {res[variant]['pooled']['roc_auc']:.3f}; minus frozen LR {res[variant]['minus_frozen_LR']}")
    run.save_json("results.json", res)
    run.finish("measured", notes="A02 light retraining under the repaired A03 rule; fixed settings from practice folds")


def finish_results(run, P):
    res = summarise(P, run)
    ref_all = pd.read_csv(latest_run("20261006-012654_A02_frozen_features") / "oof_predictions.csv", dtype={"image_id": str})
    both = pd.concat([P, ref_all[ref_all["method"] == "vit_large_patch14_dinov2@518"].assign(method="frozen_LR_reference")], ignore_index=True)
    curves = []                                          # bad-photo catch curves, overall and per format / cell type
    for meth, d in both.groupby("method"):
        for col in ("ALL", "photo_format", "cell_type"):
            for key, dd in ([("ALL", d)] if col == "ALL" else list(d.groupby(col))):
                c = C.review_curve(dd["label_bad"].to_numpy(), dd["p_bad"].to_numpy())["catch_rate"]
                curves.append(pd.DataFrame({"method": meth, "subset": f"{col}={key}", "fraction_reviewed": np.arange(len(c)) / (len(c) - 1), "catch_rate": c}))
    run.save_df("review_curves.csv", pd.concat(curves, ignore_index=True))
    # paired with the saved A02 frozen logistic-regression reference on identical photos
    ref = pd.read_csv(latest_run("20261006-012654_A02_frozen_features") / "oof_predictions.csv", dtype={"image_id": str})
    ref = ref[ref["method"] == "vit_large_patch14_dinov2@518"].set_index("image_id")
    for variant in ("head_only", "last_block"):
        d = P[P["method"].str.endswith(variant)].set_index("image_id")
        r = ref.loc[d.index]
        yy, gg = d["label_bad"].to_numpy(), d["group"].to_numpy()
        a_, b_ = r["p_bad"].to_numpy(), d["p_bad"].to_numpy()
        vals = C.grouped_bootstrap(lambda idx: C.roc_auc(yy[idx], b_[idx]) - C.roc_auc(yy[idx], a_[idx]), gg, n_boot=2000, seed=SEED)
        res[f"paired_{variant}_minus_frozen_LR"] = {"difference": float(C.roc_auc(yy, b_) - C.roc_auc(yy, a_)),
                                                     "range_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
        run.log(f"{variant} minus saved frozen reference (same photos): {res[f'paired_{variant}_minus_frozen_LR']}")
    run.save_json("results.json", res)
    run.finish("measured", notes="A02 light retraining (last block) vs frozen; practice folds; final test untouched")


def combine(tags):
    parts = [latest_run(t) for t in tags]
    P = pd.concat([pd.read_csv(p / "oof_predictions.csv", dtype={"image_id": str}) for p in parts], ignore_index=True)
    assert P.groupby("method")["image_id"].nunique().nunique() == 1 and not P.duplicated(["method", "image_id"]).any()
    import json
    settings = {}
    for p in parts:
        settings.update(json.load(open(p / "fold_settings.json")))
    run = Run("A02", "light_retrain_full", config={"joined_from": [p.name for p in parts]})
    run.save_json("fold_settings.json", settings)
    run.save_df("oof_predictions.csv", P)
    finish_results(run, P)


if __name__ == "__main__":
    main()
