"""F03: with FRESHLY computed practice features (clean rebuild), redo the 5-fold development AUC for the whole-image model and the
frozen candidate, the numbers recorded in A02 / X01 (0.8433 / 0.8751 for 3x3). Fresh features only; no saved features."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from a_views import fit_eval
from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, latest_run


def main():
    cfg = json.load(open(CODE_ROOT / "final_config.json"))
    n = cfg["chip"]["grid"]
    run = Run("F03", "practice_folds_check_clean", config={"candidate": cfg["chip"]["candidate"]})
    fr = latest_run("F03_chip_features_clean")
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str, "prefix": str})
    fmt = pd.read_csv(latest_run("G01_ooc_check_clean") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    prac = sp[sp["role"] == "practice"].merge(fmt[["member", "photo_format"]], on="member").reset_index(drop=True)
    ids = pd.read_csv(fr / "practice_ids.csv", dtype={"image_id": str})["image_id"].tolist()
    assert ids == prac["image_id"].tolist()
    w = np.load(fr / "practice_features_whole.npy").astype(np.float32)
    out = {}
    P, _ = fit_eval(w, w, prac, "whole")
    out["whole"] = C.roc_auc(P["label_bad"], P["p_bad"])
    if n:
        t = np.load(fr / f"practice_features_tiles_{n}x{n}_meanmax.npy").astype(np.float32)
        X = np.concatenate([w, t], 1)
        P, _ = fit_eval(X, X, prac, cfg["chip"]["candidate"])
        out[cfg["chip"]["candidate"]] = C.roc_auc(P["label_bad"], P["p_bad"])
    run.log(f"clean-rebuild 5-fold practice AUC: {out}")
    run.save_json("results.json", out)
    run.finish("measured", notes="clean rebuild: practice-fold AUC from fresh features")


if __name__ == "__main__":
    main()
