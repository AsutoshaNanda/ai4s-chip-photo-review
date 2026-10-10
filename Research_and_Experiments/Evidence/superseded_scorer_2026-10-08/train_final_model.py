"""Train the shipped chip-photo scoring model (frozen config: whole photo + 3x3 native-resolution tiles, frozen DINOv2 ViT-L @518,
logistic regression with grouped inner CV) on ALL development photos and save it with the list of camera formats seen in development.

Training features: the freshly computed development features of the clean rebuild (F03), so training and inference use the SAME
feature extractor (a_views.Extractor). Output: model_x01_3x3.joblib (+ .json with formats and settings) next to this repo's root
(or the folder given with --out)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a01_simple_baselines import CS, inner_cv
from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, SPLITS_ROOT
from ai4s.runlog import latest_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT))
    a = ap.parse_args()
    cfg = json.load(open(CODE_ROOT / "final_config.json"))
    assert cfg["chip"]["candidate"] == "fixed_3x3"
    fr = latest_run("F03_chip_features_clean")
    ids = pd.read_csv(fr / "practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str}).set_index("image_id").loc[ids]
    fmt = pd.read_csv(latest_run("G01_ooc_check_clean") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    sp = sp.reset_index().merge(fmt[["member", "photo_format"]], on="member")
    X = np.concatenate([np.load(fr / "practice_features_whole.npy"), np.load(fr / "practice_features_tiles_3x3_meanmax.npy")], 1).astype(np.float32)
    y, g = sp["label_bad"].to_numpy(), sp["group"].to_numpy()
    splits = inner_cv(g, y)
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=splits, scoring="neg_log_loss", max_iter=5000, n_jobs=-1))
    model.fit(X, y)
    c = float(model[-1].C_[0])
    oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)), X, y, cv=splits, method="predict_proba")[:, 1]
    cut = C.choose_cutoff(y, oof)["cutoff"]
    meta = {"candidate": "fixed_3x3", "train_photos": int(len(y)), "C": c, "yes_no_cutoff_from_inner_cv": cut,
            "automatic_formats": sorted(set(sp["photo_format"]) - {"RGB/640x480"}), "thin_support_formats": ["RGB/640x480"],
            "small_rule": cfg["chip"]["small_rule"], "feature_model": "vit_large_patch14_dinov2.lvd142m", "long_side": 518, "tiles": "3x3, native resolution, mean+max",
            "training_set_note": "all 2,397 development photos; the 675 locked final photos were not used"}
    out = Path(a.out)
    joblib.dump(model, out / "model_x01_3x3.joblib")
    (out / "model_x01_3x3.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
