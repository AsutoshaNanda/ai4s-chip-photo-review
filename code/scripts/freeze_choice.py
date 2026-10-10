"""Freeze the chip-photo candidate by a rule written BEFORE the light-retraining results were known (7 Oct 2026, 15:00 IST).

Rule (mechanical; no manual override):
  Candidates, ordered by measured cost per photo:  whole  <  fixed_3x3  <  fixed_5x5  <  light-retrained last block.
  Primary evidence: pooled AUC under the repaired A03 rule (held-out cell type AND its filename groups removed from training),
  the hardest development test, on identical photos.
  Start with `whole`. Move to the next more expensive candidate only if its paired difference to the CURRENT choice has a
  95% range (whole filename groups resampled, 1000 draws, seed 20261005) that excludes zero in its favour; otherwise stay.
  The light-retrained candidate is considered only if its full run and its A03-rule run both exist.
  Weakest-group results (per cell type, per format) and measured running time are printed with the decision as context;
  they do not override the rule.
Small photos: no small-photo rule is validated (6 bad photos), so the gate's small_rule is "manual_review" (improvements.md fallback).
Nucleus: B02 and B03 are both retained as separate candidates (improvements.md: preserve alternatives); the noise-trained
variant is evaluated as an extra labelled row. No selection among nucleus rows is made on the final test.
Writes code/final_config.json and Research_and_Experiments/18_Freeze_Decision.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s import metrics_cls as C
from ai4s.paths import CODE_ROOT, RUNS_ROOT
from ai4s.runlog import latest_run

SEED = 20261005
COST_ORDER = ["whole", "fixed_3x3", "fixed_5x5", "retrain_last_block"]


def a03_predictions():
    P = {}
    d = pd.read_csv(latest_run("A03_unseen_cell_types_groups_excluded") / "predictions_groups_excluded.csv", dtype={"image_id": str})
    ren = {"whole": "whole", "whole_plus_patches_3x3": "fixed_3x3", "whole_plus_patches_5x5": "fixed_5x5"}
    for m, g in d.groupby("method"):
        P[ren[m]] = g.set_index("image_id")
    try:
        r = latest_run("A02_light_retrain_a03")
        d = pd.read_csv(r / "predictions_a03.csv", dtype={"image_id": str})
        P["retrain_last_block"] = d[d["method"].str.endswith("last_block")].set_index("image_id")
    except FileNotFoundError:
        pass
    return P


def _md(rows):
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def paired(a, b, ids):
    y = a.loc[ids, "label_bad"].to_numpy()
    g = a.loc[ids, "group"].to_numpy()
    pa, pb = a.loc[ids, "p_bad"].to_numpy(), b.loc[ids, "p_bad"].to_numpy()
    vals = C.grouped_bootstrap(lambda idx: C.roc_auc(y[idx], pb[idx]) - C.roc_auc(y[idx], pa[idx]), g, n_boot=1000, seed=SEED)
    return float(C.roc_auc(y, pb) - C.roc_auc(y, pa)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    write = "--write" in sys.argv                      # without --write: a dry run that only prints
    P = a03_predictions()
    ids = P["whole"].index
    for k in P:
        ids = ids.intersection(P[k].index)
    current, steps = "whole", []
    for cand in COST_ORDER[1:]:
        if cand not in P:
            steps.append(f"- {cand}: not available (no A03-rule predictions) - not considered")
            continue
        diff, rng = paired(P[current], P[cand], ids)
        move = rng[0] > 0
        steps.append(f"- {cand} minus {current}: {diff:+.4f}, 95% range [{rng[0]:+.4f}, {rng[1]:+.4f}] -> {'MOVE to ' + cand if move else 'stay with ' + current}")
        if move:
            current = cand
    y = P["whole"].loc[ids, "label_bad"].to_numpy()
    table = []
    for k in P:
        d = P[k].loc[ids]
        pt = {t: C.roc_auc(g["label_bad"], g["p_bad"]) for t, g in d.groupby("cell_type") if g["label_bad"].nunique() == 2}
        table.append({"candidate": k, "pooled_AUC_A03_rule": round(C.roc_auc(y, d["p_bad"].to_numpy()), 4),
                      "weakest_cell_type": min(pt, key=pt.get), "weakest_type_AUC": round(min(pt.values()), 3)})
    cfg = {"frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"), "rule": "scripts/freeze_choice.py docstring",
           "chip": {"candidate": current, "small_rule": "manual_review", "whole_features": "vit_large_patch14_dinov2.lvd142m@518 (A02)",
                    "grid": {"whole": 0, "fixed_3x3": 3, "fixed_5x5": 5}.get(current), "classifier": "logistic regression, grouped inner CV (as A02/X01)"},
           "nucleus": {"models": ["B02", "B03"], "seeds": [0, 1], "posts": "v2", "settings": ["best_count", "best_outline"],
                       "extra_row": {"variant": "_noisetrain", "seed": 0}},
           "final_sets": {"chip": "ooc_splits role final_test (675 photos)", "nucleus": "bbbc039_splits role final_test (50 images)"}}
    md = ["# Freeze decision (7 Oct 2026)", "", "Mechanical rule fixed before the light-retraining results were known: see the docstring of `code/scripts/freeze_choice.py`.", "",
          "## Chip photos", "", *steps, "", f"**Frozen chip-photo candidate: `{current}`.** Small 640×480 photos: manual review (no validated rule).", "",
          "Context (repaired A03 rule, same photos):", "", _md(table), "",
          "## Nucleus", "", "B02 and B03 both stay as separate candidates, each with its best-count and best-outline settings (v2 search), seeds 0 and 1; the noise-trained variant (seed 0) is an extra labelled row. No selection among them on the final test.", "",
          "## Not decided here", "", "Whether C01/C03 appear in the submission, and the code licence: user decisions (see `Evidence/2026-10-07_release_rights_check.md`)."]
    if write:
        (CODE_ROOT / "final_config.json").write_text(json.dumps(cfg, indent=1))
        Path(CODE_ROOT.parent / "Research_and_Experiments" / "18_Freeze_Decision.md").write_text("\n".join(md))
    print("\n".join(md) + ("" if write else "\n\n(dry run: nothing written)"))


if __name__ == "__main__":
    main()
