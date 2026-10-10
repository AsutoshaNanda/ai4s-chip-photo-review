"""A04 / A05 / X01: what part of the photo matters, safe training changes, and zoomed patches.

All three reuse A02's best measured set-up (frozen vit_large_patch14_dinov2 at long side 518, logistic regression
with grouped inner CV) on the same 5 practice folds, so every result is paired with A02 on identical photos.
Practice photos only; the final test stays locked.

A04  whole photo vs centre crop vs edges-only (centre filled with the photo's median grey), over a RANGE of centre
     sizes (no single size assumed). Crop-position check: classifier trained on centred crops, tested on crops
     shifted by a range of offsets (curve of score vs shift). If a centre crop loses score that the edges keep,
     the label depends on something outside the centre (or on the background).
A05  safe changes added to TRAINING photos only (flips, 90-degree turns, brightness): each change alone and all
     together, tested on the untouched photos. No blur (blur can be why a photo is "bad"). Frozen-feature
     version: the classifier is trained on features of the changed copies; the big model itself is not retrained.
X01  whole photo vs whole + zoomed patches (grid of 2x2 and 3x3 tiles at native resolution). Patch features are
     pooled per photo (mean and max), so patches never get the photo's label on their own; scoring per photo.
     (Plan: adding sample details only if A07 shows they help; A07 showed no gain, so not added.)
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

from ai4s import metrics_cls as C
from ai4s.paths import OOC_DIR, SPLITS_ROOT
from ai4s.runlog import Run, latest_run
from a02_frozen import summarise, target_shape

CENTRE_FRACS = [0.25, 0.5, 0.75]
SHIFTS = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]     # as a share of the photo side, applied to the 0.5 centre crop
GRIDS = [2, 3]
BRIGHT = [0.8, 1.25]


class Extractor:
    def __init__(self, model_name, long_side, dev, run):
        import timm
        kw = {"dynamic_img_size": True} if "vit" in model_name else {}
        self.m = timm.create_model(model_name, pretrained=True, num_classes=0, **kw).to(dev).eval()
        cfg = self.m.pretrained_cfg
        self.mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device=dev)[:, None, None]
        self.std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device=dev)[:, None, None]
        self.dev, self.side, self.run, self.timing, self.quiet = dev, long_side, run, {}, False

    @torch.no_grad()
    def __call__(self, name, arrays, native=False):
        """arrays: list of 2-D uint8 images. Resized so the long side = self.side (or kept native, rounded to 14)."""
        by_shape = {}
        for j, a in enumerate(arrays):
            shp = target_shape(*a.shape, max(a.shape) if native else self.side)
            by_shape.setdefault((a.shape, shp), []).append(j)       # same raw size AND same target size per batch
        out = [None] * len(arrays)
        t0 = time.perf_counter()
        for (_, shape), members in by_shape.items():
            for b in range(0, len(members), 32):
                chunk = members[b:b + 32]
                x = torch.stack([torch.from_numpy(np.ascontiguousarray(arrays[j])) for j in chunk]).float().div(255)[:, None].to(self.dev)
                x = F.interpolate(x, size=shape, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
                x = ((x - self.mean) / self.std).clamp(-10, 10)
                with torch.autocast(self.dev, dtype=torch.float16, enabled=(self.dev == "cuda")):
                    f = self.m(x).float().cpu().numpy()
                for j, v in zip(chunk, f):
                    out[j] = v
        if self.dev == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        t = self.timing.setdefault(name, {"seconds": 0.0, "images": 0, "calls": 0})
        t["seconds"] += secs
        t["images"] += len(arrays)
        t["calls"] += 1
        if not self.quiet:
            self.run.log(f"features {name}: {len(arrays)} images in {secs:.1f} s")
        return np.stack(out)


def centre_box(h, w, frac, dy=0.0, dx=0.0):
    ch, cw = int(round(h * frac)), int(round(w * frac))
    y0 = int(round((h - ch) / 2 + dy * h))
    x0 = int(round((w - cw) / 2 + dx * w))
    y0, x0 = min(max(y0, 0), h - ch), min(max(x0, 0), w - cw)
    return y0, x0, ch, cw


def crop(a, frac, dy=0.0, dx=0.0):
    y0, x0, ch, cw = centre_box(*a.shape, frac, dy, dx)
    return a[y0:y0 + ch, x0:x0 + cw]


def edges_only(a, frac):
    y0, x0, ch, cw = centre_box(*a.shape, frac)
    b = a.copy()
    b[y0:y0 + ch, x0:x0 + cw] = np.uint8(np.median(a))
    return b


def tiles(a, n):
    h, w = a.shape
    ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
    return [a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] for i in range(n) for j in range(n)]


def fit_eval(Xtr_list, Xte, prac, tag, extra_train=None, folds=None):
    """Logistic regression as in A02. Xtr_list: features used for TRAINING rows (aligned with prac);
    extra_train: list of extra feature arrays (changed copies of the same photos, same labels/groups), training only."""
    from a01_simple_baselines import CS, inner_cv
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, info = [], []
    extra_train = extra_train or []
    for k in (folds if folds is not None else sorted(prac["dev_fold"].unique())):
        tr, te = (prac["dev_fold"] != k).to_numpy(), (prac["dev_fold"] == k).to_numpy()
        ytr, gtr = prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy()
        Xa = np.concatenate([Xtr_list[tr]] + [e[tr] for e in extra_train])
        ya = np.tile(ytr, 1 + len(extra_train))
        ga = np.tile(gtr, 1 + len(extra_train))
        splits = inner_cv(ga, ya)          # copies share their photo's group, so they never cross inner folds
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000, n_jobs=-1))
        model.fit(Xa, ya)
        c = float(model[-1].C_[0])
        oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)), Xa, ya,
                                cv=splits, method="predict_proba")[:, 1]
        n0 = len(ytr)
        cut = C.choose_cutoff(ytr, oof[:n0])["cutoff"]    # cut-off from the untouched training photos only
        p = model.predict_proba(Xte[te])[:, 1]
        d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].copy()
        d["method"], d["p_bad"], d["pred_bad"], d["cutoff"] = tag, p, (p >= cut).astype(int), cut
        rows.append(d)
        info.append({"fold": int(k), "C": c, "cutoff": cut})
    return pd.concat(rows), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["A04", "A05", "X01"])
    ap.add_argument("--smoke", action="store_true", help="tiny model and 200 photos, only to catch crashes")
    ap.add_argument("--grids", default="", help="X01 only: comma list of grid sizes (default 2,3)")
    a = ap.parse_args()
    grids = [int(g) for g in a.grids.split(",")] if a.grids else GRIDS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, side = ("resnet18.tv_in1k", 224) if a.smoke else ("vit_large_patch14_dinov2.lvd142m", 518)
    names = {"A04": "crop_vs_edges", "A05": "safe_changes_frozen", "X01": "whole_plus_patches"}
    run = Run(a.exp, names[a.exp] + (f"_grids{a.grids.replace(',', '-')}" if a.grids else "") + ("_smoke" if a.smoke else ""), config={
        "model": f"{model_name}@{side}", "device": dev, "photos": "practice only", "centre_fracs": CENTRE_FRACS,
        "shifts": SHIFTS, "grids": grids, "brightness": BRIGHT, "smoke": a.smoke})
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    g01 = latest_run("G01_ooc_check")
    fmt = pd.read_csv(g01 / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = sp[sp["role"] == "practice"].merge(fmt[["member", "photo_format"]], on="member").reset_index(drop=True)
    if a.smoke:
        prac = prac.groupby("dev_fold", group_keys=False).sample(40, random_state=0).reset_index(drop=True)
    ids = prac["image_id"].tolist()
    imgs = [np.load(OOC_DIR / "gray1036" / f"{i}.npy") for i in ids]
    ex = Extractor(model_name, side, dev, run)
    whole = ex("whole", imgs)
    preds, info = [], {}

    def add(P, inf):
        preds.append(P)
        info[P["method"].iloc[0]] = inf

    add(*fit_eval(whole, whole, prac, "whole"))
    if a.exp == "A04":
        for f in CENTRE_FRACS:
            Xc = ex(f"centre_{f}", [crop(m, f) for m in imgs])
            add(*fit_eval(Xc, Xc, prac, f"centre_{f}"))
            Xe = ex(f"edges_{f}", [edges_only(m, f) for m in imgs])
            add(*fit_eval(Xe, Xe, prac, f"edges_only_outside_{f}"))
            if f == 0.5:
                X0 = Xc
        for s in SHIFTS:
            if s == 0.0:
                continue
            for axis in ("x", "y"):
                Xs = ex(f"shift_{axis}{s}", [crop(m, 0.5, dy=s if axis == "y" else 0, dx=s if axis == "x" else 0) for m in imgs])
                add(*fit_eval(X0, Xs, prac, f"train_centre_0.5_test_shift_{axis}{s:+.2f}"))
    elif a.exp == "A05":
        changes = {"hflip": lambda m: m[:, ::-1], "vflip": lambda m: m[::-1, :], "rot90": lambda m: np.rot90(m, 1),
                   "rot270": lambda m: np.rot90(m, 3)}
        for b in BRIGHT:
            changes[f"bright_x{b}"] = lambda m, b=b: np.clip(m.astype(np.float32) * b, 0, 255).astype(np.uint8)
        feats = {k: ex(k, [fn(m) for m in imgs]) for k, fn in changes.items()}
        for k, Xk in feats.items():
            add(*fit_eval(whole, whole, prac, f"train_plus_{k}", extra_train=[Xk]))
        add(*fit_eval(whole, whole, prac, "train_plus_flips_turns", extra_train=[feats[k] for k in ("hflip", "vflip", "rot90", "rot270")]))
        add(*fit_eval(whole, whole, prac, "train_plus_all_changes", extra_train=list(feats.values())))
    else:
        for n in grids:
            T = ex(f"tiles_{n}x{n}", [t for m in imgs for t in tiles(m, n)], native=True).reshape(len(imgs), n * n, -1)
            pooled = np.concatenate([T.mean(1), T.max(1)], axis=1)
            np.save(run.path(f"features_tiles_{n}x{n}_meanmax.npy"), pooled.astype(np.float16))
            add(*fit_eval(pooled, pooled, prac, f"patches_{n}x{n}_only"))
            both = np.concatenate([whole, pooled], axis=1)
            add(*fit_eval(both, both, prac, f"whole_plus_patches_{n}x{n}"))
    P = pd.concat(preds, ignore_index=True)
    run.save_df("oof_predictions.csv", P)
    run.save_json("fold_settings.json", info)
    run.save_json("feature_timing.json", ex.timing)
    run.save_json("results.json", summarise(P, run))
    run.finish("measured", notes=f"{a.exp} on practice folds ({model_name}@{side}); final test untouched")


if __name__ == "__main__":
    main()
