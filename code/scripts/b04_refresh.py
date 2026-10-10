"""B04 refresh (improvements.md sections 3, 5 and 6): stress curves for the exact retained B02/B03 settings.

The earlier B04 used the original clean-up settings. Here every held-out practice image (out of fold, same saved fold
models) is scored clean and at every recorded stress level (b_unet.STRESS, unchanged) with BOTH retained v2
settings per fold (best count and best outline, from the wider v2 clean-up search). Also used for the planned
noise training change (--variant _noisetrain) so both are scored the same way. Synthetic stress, not real outside data.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import b_unet as U
from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run
from b_posts import fold_models
from g03_speed_test import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["B02", "B03"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="")
    a = ap.parse_args()
    target, n_cls = ("fg", 2) if a.exp == "B02" else ("boundary", 3)
    fm = fold_models(a.exp, target, a.seed, "v2", a.variant)
    run = Run("B04", f"stress_refresh_{a.exp}{a.variant}_seed{a.seed}", config={
        "exp": a.exp, "seed": a.seed, "variant": a.variant, "stress": U.STRESS, "settings": "v2 best_count and best_outline per fold",
        "models": {k: v[0].name for k, v in fm.items()}, "settings_source": {k: v[1].get("source") for k, v in fm.items()}})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    X, _, L = U.load(prac["stem"].tolist(), target)
    levels = [("clean", 0.0)] + [(k, lv) for k, lvs in U.STRESS.items() for lv in lvs]
    rows = []
    for k, (d, post) in sorted(fm.items()):
        m = UNet(out_ch=n_cls)
        m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
        m = m.eval().to(U.DEV)
        te = np.flatnonzero(prac["cv_fold"] == k)
        for kind, lv in levels:
            xs = X[te] if kind == "clean" else np.stack([U.stressed(X[i], kind, lv) for i in te])
            P = U.predict_prob(m, xs)
            for i, p in zip(te, P):
                for name in ("best_count", "best_outline"):
                    s = post[name]
                    r = S.score_image(L[i], U.to_objects(p, target, s["thr"], int(s["min_distance"]), int(s["min_size"])))
                    rows.append({"stem": prac.loc[i, "stem"], "cv_fold": int(k), "setting": name, "stress": kind, "level": lv,
                                 "count_error": r["count_error"], "abs_count_error": r["abs_count_error"],
                                 "f1@0.50": r["f1@0.50"], "f1@0.75": r["f1@0.75"]})
        run.log(f"fold {k} done ({len(te)} images x {len(levels)} conditions)")
    D = pd.DataFrame(rows)
    run.save_df("stress_per_image.csv", D)
    agg = D.groupby(["setting", "stress", "level"]).agg(images=("stem", "size"), count_mae=("abs_count_error", "mean"),
                                                         bias=("count_error", "mean"), worst_abs_count_error=("abs_count_error", "max"),
                                                         f1_050=("f1@0.50", "mean"), f1_075=("f1@0.75", "mean")).round(3).reset_index()
    run.save_df("stress_summary.csv", agg)
    run.log("stress curves (every level; clean first):\n" + agg.to_string(index=False))
    run.save_json("results.json", agg.to_dict(orient="records"))
    run.finish("measured", notes=f"B04 refresh with v2 settings: {a.exp}{a.variant} seed {a.seed}")


if __name__ == "__main__":
    main()
