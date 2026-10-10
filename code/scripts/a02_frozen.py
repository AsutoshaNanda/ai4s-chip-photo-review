"""A02 part 1: pretrained image models used "frozen" (they only read features; a simple classifier sits on top).

Models (licences recorded in G01 model-file check): resnet18.tv_in1k (small; torchvision ImageNet weights, BSD-3)
and vit_large_patch14_dinov2.lvd142m (big modern model; DINOv2, Apache-2.0). Each at a smaller and a larger
long side (224/448 and 224/518), aspect ratio kept, all photos in one format (grayscale, prepared on Modal).
Only PRACTICE photos are processed; final-test photos stay untouched until Thursday (F01).

Classifier: logistic regression with scaling, regularisation chosen by grouped inner folds (log loss), yes/no
cut-off chosen on the training fold's inner out-of-fold scores. Same 5 practice folds as A01, so results are
paired with A01 photo measurements on identical photos. Reported overall, per cell type and per photo format.
"""
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

from ai4s import metrics_cls as C
from ai4s.paths import OOC_DIR, SPLITS_ROOT
from ai4s.runlog import Run, latest_run

MODELS = [("resnet18.tv_in1k", 224), ("resnet18.tv_in1k", 448),
          ("vit_large_patch14_dinov2.lvd142m", 224), ("vit_large_patch14_dinov2.lvd142m", 518)]
SEED = 20261005


def target_shape(h, w, long_side):
    s = long_side / max(h, w)
    nh, nw = h * s, w * s
    return int(round(nh / 14) * 14), int(round(nw / 14) * 14)


@torch.no_grad()
def extract(model_name, long_side, ids, dev, run):
    import timm
    kw = {"dynamic_img_size": True} if "vit" in model_name else {}
    m = timm.create_model(model_name, pretrained=True, num_classes=0, **kw).to(dev).eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device=dev)[:, None, None]
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device=dev)[:, None, None]
    arrays = {i: np.load(OOC_DIR / "gray1036" / f"{i}.npy") for i in ids}
    by_shape = {}
    for i, a in arrays.items():
        by_shape.setdefault(target_shape(*a.shape, long_side), []).append(i)
    feats = {}
    t0 = time.perf_counter()
    for shape, members in by_shape.items():
        for b in range(0, len(members), 32):
            chunk = members[b:b + 32]
            x = torch.stack([torch.from_numpy(arrays[i]) for i in chunk]).float().div(255)[:, None].to(dev)
            x = F.interpolate(x, size=shape, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
            x = (x - mean) / std
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                f = m(x).float().cpu().numpy()
            feats.update(dict(zip(chunk, f)))
    if dev == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    run.log(f"{model_name}@{long_side}: {len(ids)} photos in {secs:.1f} s ({len(ids) / secs:.1f}/s), dim {len(next(iter(feats.values())))}, shapes {list(by_shape)}")
    return np.stack([feats[i] for i in ids]), secs


def evaluate(X, prac, run, tag):
    from a01_simple_baselines import CS, inner_cv
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, info = [], []
    for k in sorted(prac["dev_fold"].unique()):
        tr, te = (prac["dev_fold"] != k).to_numpy(), (prac["dev_fold"] == k).to_numpy()
        ytr, gtr = prac.loc[tr, "label_bad"].to_numpy(), prac.loc[tr, "group"].to_numpy()
        splits = inner_cv(gtr, ytr)
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000, n_jobs=-1))
        model.fit(X[tr], ytr)
        c = float(model[-1].C_[0])
        oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)), X[tr], ytr,
                                cv=splits, method="predict_proba")[:, 1]
        cut = C.choose_cutoff(ytr, oof)["cutoff"]
        p = model.predict_proba(X[te])[:, 1]
        d = prac.loc[te, ["image_id", "group", "cell_type", "photo_format", "label_bad", "dev_fold"]].copy()
        d["method"], d["p_bad"], d["pred_bad"], d["cutoff"] = tag, p, (p >= cut).astype(int), cut
        rows.append(d)
        info.append({"fold": int(k), "C": c, "cutoff": cut})
    return pd.concat(rows), info


def summarise(P, run):
    out = {}
    for name, d in P.groupby("method"):
        y, p, g = d["label_bad"].to_numpy(), d["p_bad"].to_numpy(), d["group"].to_numpy()
        res = {"pooled": C.summary(y, p), "per_cell_type": C.by_group(y, p, d["cell_type"].to_numpy()),
               "per_photo_format": C.by_group(y, p, d["photo_format"].to_numpy())}
        vals = C.grouped_bootstrap(lambda idx: C.roc_auc(y[idx], p[idx]), g, n_boot=2000, seed=SEED)
        res["roc_auc_bootstrap_95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out[name] = res
        fmt = {f: (round(v["roc_auc"], 3) if v["roc_auc"] is not None else None) for f, v in res["per_photo_format"].items()}
        run.log(f"{name:42s} AUC {res['pooled']['roc_auc']:.3f} {np.round(res['roc_auc_bootstrap_95'], 3).tolist()} "
                f"AP {res['pooled']['avg_precision_bad']:.3f} logloss {res['pooled']['log_loss']:.3f} | within format {fmt}")
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = Run("A02", "frozen_features", config={"models": MODELS, "device": dev, "photos": "practice only",
                                                 "classifier": "logistic regression, C by grouped inner CV (log loss)"})
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    g01 = latest_run("G01_ooc_check")
    fmt = pd.read_csv(g01 / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = sp[sp["role"] == "practice"].merge(fmt[["member", "photo_format"]], on="member").reset_index(drop=True)
    ids = prac["image_id"].tolist()
    preds, info, timing = [], {}, {}
    for model_name, side in MODELS:
        X, secs = extract(model_name, side, ids, dev, run)
        tag = f"{model_name.split('.')[0]}@{side}"
        np.save(run.path(f"features_practice_{tag}.npy"), X.astype(np.float16))
        timing[tag] = {"seconds": secs, "photos_per_s": len(ids) / secs}
        P, inf = evaluate(X, prac, run, tag)
        preds.append(P)
        info[tag] = inf
    pd.Series(ids).to_csv(run.path("features_practice_ids.csv"), index=False, header=["image_id"])
    P = pd.concat(preds, ignore_index=True)
    run.save_df("oof_predictions.csv", P)
    run.save_json("fold_settings.json", info)
    run.save_json("feature_timing.json", timing)
    run.save_json("results.json", summarise(P, run))
    run.finish("measured", notes="A02 frozen features on practice folds; final test untouched")


if __name__ == "__main__":
    main()
