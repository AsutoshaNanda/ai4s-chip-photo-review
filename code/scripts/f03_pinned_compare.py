"""F03 (improvements.md): pinned-package repeat of four earlier Kaggle re-post runs.
The earlier B02/B03 clean-up searches for seed 1 and the extended-threshold search ran under Kaggle's default packages
(scikit-image 0.25.2); the repeat uses scikit-image 0.26.0 (the version pinned in requirements.txt). Same saved fold models,
same grid, same rule, no training. This compares each new run with the earlier one: package versions, grid, picks per fold,
out-of-fold summary numbers, per-image results and the full post-grid tables.
Usage: python scripts/f03_pinned_compare.py NEW_DIR [OLD_DIR]   (default OLD_DIR = code/runs)"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("B02", "unet_fg_repost_v2_seed1"), ("B03", "unet_boundary_repost_v2_seed1"),
         ("B03", "unet_boundary_repost_v2_thr_extended"), ("B03", "unet_boundary_repost_v2_seed1_thr_extended")]
NAME = re.compile(r"^(\d{8}-\d{6})_(B0[23])_(.+)$")


def runs_by_key(root):
    out = {}
    for d in sorted(Path(root).rglob("results.json")):
        m = NAME.match(d.parent.name)
        if m and (m.group(2), m.group(3)) in PAIRS:
            out.setdefault((m.group(2), m.group(3)), []).append(d.parent)
    return out


def picks(run):
    return {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"fold (\d): picks (\{.*?\}); on range edge",
                                                                       (run / "log.txt").read_text())}


def num_diff(a, b):
    ks = [k for k in a if isinstance(a[k], (int, float)) and k in b]
    return max(abs(a[k] - b[k]) for k in ks) if ks else float("nan"), len(ks)


def compare(old, new):
    r = {"old_run": old.name, "new_run": new.name}
    eo, en = json.loads((old / "env.json").read_text()), json.loads((new / "env.json").read_text())
    r["versions_differing"] = {k: [eo["versions"].get(k), en["versions"].get(k)] for k in sorted(set(eo["versions"]) | set(en["versions"]))
                               if eo["versions"].get(k) != en["versions"].get(k)}
    r["grid_identical"] = json.loads((old / "config.json").read_text()).get("grid") == json.loads((new / "config.json").read_text()).get("grid")
    po, pn = picks(old), picks(new)
    r["picks_per_fold_identical"] = po == pn and len(po) == 5
    r["picks_differing_folds"] = sorted(k for k in set(po) | set(pn) if po.get(k) != pn.get(k))
    ro, rn = json.loads((old / "results.json").read_text()), json.loads((new / "results.json").read_text())
    for obj in ("best_count", "best_outline"):
        d, n = num_diff(ro[obj], rn[obj])
        r[f"{obj}_max_abs_diff_over_{n}_numbers"] = d
        r[f"{obj}_headline_old_new"] = {k: [ro[obj].get(k), rn[obj].get(k)] for k in ("count_mae", "pooled_f1@0.75", "pooled_f1@0.90", "mean_image_f1@0.75")}
    r["edges_identical"] = ro.get("edges") == rn.get("edges")
    for obj in ("best_count", "best_outline"):
        a, b = pd.read_csv(old / f"oof_per_image_{obj}.csv"), pd.read_csv(new / f"oof_per_image_{obj}.csv")
        same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
        nums = a.select_dtypes("number").columns
        r[f"per_image_{obj}"] = {"same_shape": same_shape, "exactly_equal": bool(same_shape and a.equals(b)),
                                 "max_abs_diff": float(np.abs(a[nums].to_numpy() - b[nums].to_numpy()).max()) if same_shape else None}
    grid = {}
    for k in range(5):
        a, b = pd.read_csv(old / f"post_grid_fold{k}.csv"), pd.read_csv(new / f"post_grid_fold{k}.csv")
        ok = a.shape == b.shape and list(a.columns) == list(b.columns)
        cols = [c for c in ("count_mae", "mean_f1_levels") if c in a]
        grid[k] = {"same_shape": ok, "max_abs_diff": float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max()) if ok else None,
                   "settings_differing": int((a.loc[:, ["thr", "min_distance", "min_size"]].ne(b.loc[:, ["thr", "min_distance", "min_size"]])).any(axis=1).sum()) if ok else None}
    r["post_grid_per_fold"] = grid
    r["all_identical"] = bool(r["picks_per_fold_identical"] and r["grid_identical"] and r["edges_identical"]
                              and all(v["exactly_equal"] for k, v in r.items() if k.startswith("per_image_"))
                              and all(v["same_shape"] and v["max_abs_diff"] == 0 and v["settings_differing"] == 0 for v in grid.values())
                              and r["best_count_max_abs_diff_over_%d_numbers" % num_diff(ro["best_count"], rn["best_count"])[1]] == 0
                              and r["best_outline_max_abs_diff_over_%d_numbers" % num_diff(ro["best_outline"], rn["best_outline"])[1]] == 0)
    return r


def main():
    new_root = Path(sys.argv[1])
    old_root = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs"
    new, old = runs_by_key(new_root), runs_by_key(old_root)
    out = {}
    for key in PAIRS:
        label = f"{key[0]} {key[1]}"
        if key not in new:
            out[label] = {"error": "no new run found"}
            continue
        o = [d for d in old.get(key, []) if d.parent != new_root and d not in new[key]]
        if len(o) != 1:
            out[label] = {"error": f"expected exactly one earlier run, found {[d.name for d in o]}"}
            continue
        n = sorted(new[key])[-1]
        out[label] = compare(o[0], n)
    dest = new_root / "f03_pinned_comparison.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    for label, r in out.items():
        print("==", label)
        if "error" in r:
            print("  ", r["error"])
            continue
        print("   old:", r["old_run"], "| new:", r["new_run"])
        print("   package versions that differ:", r["versions_differing"] or "none")
        print("   grid identical:", r["grid_identical"], "| picks identical in all 5 folds:", r["picks_per_fold_identical"], r["picks_differing_folds"] or "")
        for obj in ("best_count", "best_outline"):
            key2 = [k for k in r if k.startswith(f"{obj}_max_abs_diff")][0]
            print(f"   {obj}: largest difference over all summary numbers = {r[key2]}; old/new headline {r[obj + '_headline_old_new']}")
        print("   per-image tables exactly equal:", {o: r[f'per_image_{o}']['exactly_equal'] for o in ("best_count", "best_outline")},
              "| post-grid max diff per fold:", {k: v["max_abs_diff"] for k, v in r["post_grid_per_fold"].items()})
        print("   ALL IDENTICAL:", r["all_identical"])
    print("saved", dest)


if __name__ == "__main__":
    main()
