"""B06 follow-up (improvements.md B06 row): do B02 and B03 fail on the SAME images, and what does the disagreement warning miss?

From saved per-image results of the retained v2 settings (count setting, held-out practice images, seeds 0 and 1):
  - Spearman correlation of the two models' absolute count errors;
  - overlap curve: for every review size k, the share of B02's k worst images that are also among B03's k worst
    (chance level = k / N);
  - warning miss curve: after flagging the f most-disagreeing images (disagreement = |count B02 - count B03|), the share of
    B02's total absolute error that sits in images NOT flagged, split into images where both models are wrong in the same
    direction (shared failure) and the rest. Full curves for f from 0 to 1; no cut-off chosen.
Practice images only; no model run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s.runlog import Run, latest_run


def main():
    run = Run("B06", "shared_failures", config={"settings": "v2 best_count", "seeds": [0, 1]})
    res = {}
    for seed, (t2, t3) in {0: ("B02_unet_fg_repost_v2", "B03_unet_boundary_repost_v2"),
                           1: ("B02_unet_fg_repost_v2_seed1", "B03_unet_boundary_repost_v2_seed1")}.items():
        d2 = pd.read_csv(_exact(t2) / "oof_per_image_best_count.csv")
        d3 = pd.read_csv(_exact(t3) / "oof_per_image_best_count.csv")
        d = d2.merge(d3, on="stem", suffixes=("_b02", "_b03"))
        e2, e3 = d["count_error_b02"].to_numpy(), d["count_error_b03"].to_numpy()
        a2, a3 = np.abs(e2), np.abs(e3)
        n = len(d)
        rho = float(pd.Series(a2).corr(pd.Series(a3), method="spearman"))
        o2, o3 = np.argsort(-a2, kind="stable"), np.argsort(-a3, kind="stable")
        overlap = [{"k": k, "share_also_in_other_worst_k": float(len(set(o2[:k]) & set(o3[:k])) / k), "chance": k / n} for k in range(1, n + 1)]
        dis = np.abs(d["n_pred_b02"].to_numpy() - d["n_pred_b03"].to_numpy()) if "n_pred_b02" in d else np.abs(e2 - e3)
        order = np.argsort(-dis, kind="stable")
        same_dir = (np.sign(e2) == np.sign(e3)) & (a2 > 0) & (a3 > 0)
        tot = a2.sum()
        miss = []
        for k in range(0, n + 1):
            un = np.ones(n, bool)
            un[order[:k]] = False
            miss.append({"flagged": k, "fraction_flagged": k / n, "error_left_share": float(a2[un].sum() / tot),
                         "shared_failure_share_left": float(a2[un & same_dir].sum() / tot)})
        pd.DataFrame(overlap).to_csv(run.path(f"overlap_curve_seed{seed}.csv"), index=False)
        pd.DataFrame(miss).to_csv(run.path(f"warning_miss_curve_seed{seed}.csv"), index=False)
        res[f"seed{seed}"] = {"images": n, "spearman_abs_errors": rho, "images_both_wrong_same_direction": int(same_dir.sum()),
                              "share_of_b02_error_in_shared_failures": float(a2[same_dir].sum() / tot)}
        run.log(f"seed {seed}: {n} images; Spearman of absolute count errors B02 vs B03 = {rho:.3f}; "
                f"both wrong in the same direction on {int(same_dir.sum())} images carrying {100 * a2[same_dir].sum() / tot:.0f}% of B02's total error")
        half = n // 2
        run.log(f"seed {seed}: top-{half} overlap {overlap[half - 1]['share_also_in_other_worst_k']:.2f} (chance {overlap[half - 1]['chance']:.2f}); "
                f"error left unflagged after flagging {half} most-disagreeing images: {miss[half]['error_left_share']:.2f} of B02's total "
                f"(of which shared failures {miss[half]['shared_failure_share_left']:.2f})")
    run.save_json("results.json", res)
    run.finish("measured", notes="B02/B03 shared failures; saved per-image results only")


def _exact(tag):
    root = Path(__file__).resolve().parents[1] / "runs"
    hits = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_" + tag))
    return hits[-1]


if __name__ == "__main__":
    main()
