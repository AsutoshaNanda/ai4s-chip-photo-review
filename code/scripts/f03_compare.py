"""F03: compare the clean-rebuild outputs with the saved records. Differences only; no pass mark is invented.
Usage: python scripts/f03_compare.py  (after the F03 outputs were copied to code/runs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s.runlog import Run, latest_run


def main():
    run = Run("F03", "compare_clean_vs_saved", config={})
    out = {}
    try:
        steps = json.load(open(next(Path(__file__).resolve().parents[1].glob("kaggle/out/f03/f03_steps.json"))))
        out["steps"] = steps
        run.log("steps (seconds, ok): " + str([(s["step"], s["seconds"], s["ok"]) for s in steps]))
    except StopIteration:
        run.log("f03_steps.json not found")
    new = json.load(open(latest_run("F03_chip_final_eval_clean") / "results.json"))
    old = json.load(open(latest_run("F01_chip_final_eval") / "results.json"))
    cmp = {}
    for scope in ("automatic_only", "all_photos"):
        for k in [k for k in new[scope] if isinstance(new[scope][k], dict) and "pooled" in new[scope][k]]:
            a, b = new[scope][k]["pooled"], old[scope][k]["pooled"]
            cmp[f"{scope}|{k}"] = {m: {"clean": a[m], "saved": b[m], "difference": a[m] - b[m]} for m in ("roc_auc", "avg_precision_bad", "brier", "log_loss", "review_area")}
            run.log(f"final chip [{scope}] {k}: AUC clean {a['roc_auc']:.5f} vs saved {b['roc_auc']:.5f} (diff {a['roc_auc'] - b['roc_auc']:+.5f}); log loss diff {a['log_loss'] - b['log_loss']:+.5f}")
    out["final_chip"] = cmp
    try:
        pc = json.load(open(latest_run("F03_practice_folds_check_clean") / "results.json"))
        saved = {"whole": 0.8433, "fixed_3x3": 0.8751, "fixed_5x5": 0.8775}          # A02 / X01 records (saved logs, 4 decimals)
        out["practice_folds"] = {k: {"clean": v, "saved_record": saved.get(k), "difference": (v - saved[k]) if k in saved else None} for k, v in pc.items()}
        run.log(f"practice 5-fold AUC clean vs record: {out['practice_folds']}")
    except FileNotFoundError:
        run.log("practice-fold check output missing")
    root = Path(__file__).resolve().parents[1] / "runs"
    saved_dir = root / "20261006-101000_B03_unet_boundary_seed0_folds0"
    clean_dirs = sorted((Path(__file__).resolve().parents[1] / "kaggle" / "out" / "f03" / "code" / "runs").glob("*_B03_unet_boundary_seed0_folds0"))
    if saved_dir.exists() and clean_dirs and (clean_dirs[-1] / "results.json").exists():
        o, n_ = saved_dir, clean_dirs[-1]
        ro, rn = json.load(open(o / "results.json")), json.load(open(n_ / "results.json"))
        nuc = {}
        for name in ("best_count", "best_outline"):
            a_, b_ = rn[name]["all"], ro[name]["all"]
            nuc[name] = {m: {"clean": a_[m], "saved": b_[m], "difference": a_[m] - b_[m]} for m in ("count_mae", "count_bias", "pooled_f1@0.50", "pooled_f1@0.75")}
            run.log(f"nucleus B03 fold 0 [{name}]: count MAE clean {a_['count_mae']:.3f} vs saved {b_['count_mae']:.3f}; F1@0.75 {a_['pooled_f1@0.75']:.3f} vs {b_['pooled_f1@0.75']:.3f}")
        out["nucleus_fold0_retrain"] = {"saved_run": o.name, "clean_run": n_.name, "comparison": nuc,
                                        "note": "GPU training is not bit-identical; differences are reported, not judged"}
    else:
        run.log("clean nucleus retrain run not found")
    run.save_json("results.json", out)
    run.finish("measured", notes="clean rebuild vs saved records")


if __name__ == "__main__":
    main()
