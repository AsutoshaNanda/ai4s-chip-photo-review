"""Which saved fold models and clean-up settings a B02/B03 analysis uses.

posts="orig": settings chosen during training (b_unet.py, original narrow search; fold_info.json).
posts="v2":   settings chosen by the wider v2 search (b_repost.py) on the same fold models: re-derived from that run's
              post_grid_fold{k}.csv with exactly b_repost.py's rule (lowest count error, then best outline; and the reverse).
"""
import json
from pathlib import Path

import pandas as pd

from ai4s.paths import CODE_ROOT, RUNS_ROOT

ROOTS = [RUNS_ROOT, CODE_ROOT / "runs"]


def _dirs(pattern):
    found = {}
    for root in ROOTS:
        if root.exists():
            for d in root.glob(pattern):
                found.setdefault(d.name, []).append(d)
    return found


def _with(name, found, fname):
    return next((d for d in found[name] if (d / fname).exists()), found[name][0])


def _is_smoke(dirs):
    """A crash-check run (b_unet.py --smoke: at most 500 training steps on a few images). Judged from the training
    steps in its log; if no fold finished its log line, from the number of images in its per-image results."""
    import re
    for d in dirs:
        f = d / "log.txt"
        steps = [int(x) for x in re.findall(r"steps (\d+)", f.read_text())] if f.exists() else []
        if steps:
            return max(steps) <= 500
    for d in dirs:
        f = d / "oof_per_image_best_count.csv"
        if f.exists():
            return len(pd.read_csv(f)) < 150
    return False


def fold_models(exp, target, seed=0, posts="orig", variant=""):
    """{fold: (folder holding model_fold{k}.pt, {"best_count": {...}, "best_outline": {...}})}"""
    out = {}
    # all folds in one folder, then one folder per fold (per-fold folders win: they hold the seed-0 models used so far)
    for pattern in (f"*_{exp}_unet_{target}{variant}_seed{seed}", f"*_{exp}_unet_{target}{variant}_seed{seed}_folds?"):
        found = _dirs(pattern)
        for name in sorted(found):
            if pattern.endswith(f"seed{seed}") and _is_smoke(found[name]):
                continue
            for k in range(5):
                if any((d / f"model_fold{k}.pt").exists() for d in found[name]):
                    d = _with(name, found, f"model_fold{k}.pt")
                    info = _with(name, found, "fold_info.json") / "fold_info.json"
                    posts_k = {int(f["fold"]): f["post"] for f in json.load(open(info))} if info.exists() else {}
                    out[k] = (d, posts_k.get(k))
    if posts == "v2":
        tag = f"*_{exp}_unet_{target}{variant}_repost_v2" + (f"_seed{seed}" if seed else "")
        reps = sorted(n for n in _dirs(tag))
        assert reps, f"no v2 clean-up run matching {tag}"
        rep = _with(reps[-1], _dirs(tag), "post_grid_fold0.csv")
        for k in list(out):
            G = pd.read_csv(rep / f"post_grid_fold{k}.csv")
            out[k] = (out[k][0], {"best_count": G.sort_values(["count_mae", "mean_f1_levels"], ascending=[True, False]).iloc[0].to_dict(),
                                  "best_outline": G.sort_values(["mean_f1_levels", "count_mae"], ascending=[False, True]).iloc[0].to_dict(),
                                  "source": rep.name})
    return out
