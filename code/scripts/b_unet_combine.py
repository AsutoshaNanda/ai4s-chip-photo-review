"""Join per-fold b_unet.py runs (--folds) into one 5-fold out-of-fold result, same summary as a single full run.

  python scripts/b_unet_combine.py --exp B02 --target fg       (looks for runs/*_B02_unet_fg_seed0_folds*)
Fails loudly if any of the 5 outer folds is missing or appears twice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ai4s import metrics_seg as S
from ai4s.runlog import Run

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    parts = sorted(d for d in RUNS.glob(f"*_{a.exp}_unet_{a.target}_seed{a.seed}_folds*") if (d / "results.json").exists())
    run = Run(a.exp, f"unet_{a.target}_seed{a.seed}_combined", config={"parts": [d.name for d in parts]})
    results = {}
    for name in ("best_count", "best_outline"):
        df = pd.concat([pd.read_csv(d / f"oof_full_{name}.csv") for d in parts], ignore_index=True)
        got = sorted(df["cv_fold"].unique().tolist())
        if got != [0, 1, 2, 3, 4] or df["stem"].duplicated().any():
            run.finish("failed", notes=f"folds found {got}; duplicated images {int(df['stem'].duplicated().sum())}")
            raise SystemExit(f"incomplete or duplicated folds: {got}")
        rows = df.to_dict("records")
        res = {"all": S.summarize(rows), "per_fold": {int(k): S.summarize(d.to_dict("records")) for k, d in df.groupby("cv_fold")}}
        results[name] = res
        run.save_df(f"oof_per_image_{name}.csv", df.drop(columns=[c for c in df.columns if c.startswith(("tp@", "fp@", "fn@"))]))
        aa = res["all"]
        run.log(f"{a.exp} {name}: count MAE {aa['count_mae']:.2f} (bias {aa['count_bias']:.2f}); pooled F1 0.25/0.50/0.75/0.90 "
                f"{aa['pooled_f1@0.25']:.3f}/{aa['pooled_f1@0.50']:.3f}/{aa['pooled_f1@0.75']:.3f}/{aa['pooled_f1@0.90']:.3f}; "
                f"joined {aa['joined_ref']} split {aa['split_pred']} missed {aa['missed']} false {aa['false']}")
    st = pd.concat([pd.read_csv(d / "B04_stress_per_image.csv") for d in parts], ignore_index=True)
    run.save_df("B04_stress_per_image.csv", st)
    agg = st.groupby(["stress", "level"]).agg(count_mae=("abs_count_error", "mean"), bias=("count_error", "mean"),
                                               f1_050=("f1@0.50", "mean"), f1_075=("f1@0.75", "mean")).round(3)
    run.log("B04 stress tests (best-outline settings):\n" + agg.to_string())
    results["B04_stress"] = agg.reset_index().to_dict(orient="records")
    run.save_json("results.json", results)
    run.finish("measured", notes=f"{a.exp} combined from {len(parts)} per-fold runs; test locked")


if __name__ == "__main__":
    main()
