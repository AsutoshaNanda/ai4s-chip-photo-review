"""B01 re-run as required: simple counting judged on 5 rotating plate-grouped folds of the 150 practice images
(provider training + validation); the provider's 50 test images stay locked. Adds the playbook's
"crowded-image errors".

For each fold: settings are chosen on the other 4 folds, keeping two objectives (lowest count error; best
outline score averaged over all overlap levels; no single score assumed), then scored on the held-out fold.
Result: out-of-fold scores for all 150 practice images. Crowding groups use quartiles of the reference
nucleus count computed on the training folds only; the per-image table allows a full curve instead.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run, n_workers
from b01_threshold_watershed import GRID, eval_one, grid_one_image

KEYS = ["sigma", "otsu_factor", "min_distance", "min_size"]
COLS = ["stem"] + KEYS + ["abs_err", "err", "mean_f1_levels", "tp50", "fp50", "fn50"]


def as_params(row):
    return {"sigma": int(row["sigma"]), "otsu_factor": float(row["otsu_factor"]),
            "min_distance": int(row["min_distance"]), "min_size": int(row["min_size"])}


def main():
    run = Run("B01", "cv_threshold_watershed", config={"grid": GRID, "folds": "splits/bbbc039_splits.csv (5 plate-grouped)",
                                                        "objectives": ["lowest count MAE", "best mean F1 over all IoU levels"]})
    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    workers = n_workers()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for chunk in ex.map(grid_one_image, prac["stem"].tolist()) for r in chunk]
    g = pd.DataFrame(rows, columns=COLS)
    g.to_parquet(run.path("grid_per_image.parquet"), index=False)
    run.log(f"grid scored for {g['stem'].nunique()} practice images x {len(g) // g['stem'].nunique()} settings ({workers} workers)")
    g = g.merge(prac[["stem", "cv_fold"]], on="stem")

    chosen, oof = {}, {"best_count": [], "best_outline": []}
    for k in sorted(prac["cv_fold"].unique()):
        tr = g[g["cv_fold"] != k]
        agg = tr.groupby(KEYS).agg(count_mae=("abs_err", "mean"), mean_f1_levels=("mean_f1_levels", "mean")).reset_index()
        picks = {"best_count": agg.sort_values(["count_mae", "mean_f1_levels"], ascending=[True, False]).iloc[0],
                 "best_outline": agg.sort_values(["mean_f1_levels", "count_mae"], ascending=[False, True]).iloc[0]}
        chosen[int(k)] = {n: as_params(r) for n, r in picks.items()}
        held = prac[prac["cv_fold"] == k]
        train_counts = prac.loc[prac["cv_fold"] != k, "n_objects"]
        edges = np.quantile(train_counts, [0.25, 0.5, 0.75])
        for name, r in picks.items():
            with ProcessPoolExecutor(max_workers=workers) as ex:
                out = list(ex.map(eval_one, [(s, as_params(r)) for s in held["stem"]]))
            for res, _ in out:
                res["cv_fold"] = int(k)
                res["crowding_group"] = ["fewest 25%", "25-50%", "50-75%", "most 25%"][int(np.searchsorted(edges, res["n_ref"], side="right"))]
                oof[name].append(res)
    run.save_json("chosen_settings_per_fold.json", chosen)

    results = {}
    for name, rows in oof.items():
        df = pd.DataFrame(rows)
        run.save_df(f"oof_per_image_{name}.csv", df.drop(columns=[c for c in df.columns if c.startswith(("tp@", "fp@", "fn@"))]))
        res = {"all_150": S.summarize(rows),
               "per_fold": {int(k): S.summarize(d.to_dict("records")) for k, d in df.groupby("cv_fold")},
               "per_crowding_group": {gname: S.summarize(d.to_dict("records")) for gname, d in df.groupby("crowding_group")},
               "seconds_per_image_median": float(df["seconds"].median())}
        results[name] = res
        a = res["all_150"]
        run.log(f"{name}: count MAE {a['count_mae']:.2f} (bias {a['count_bias']:.2f}); pooled F1 at 0.25/0.50/0.75/0.90: "
                f"{a['pooled_f1@0.25']:.3f}/{a['pooled_f1@0.50']:.3f}/{a['pooled_f1@0.75']:.3f}/{a['pooled_f1@0.90']:.3f}; "
                f"joined {a['joined_ref']}, split {a['split_pred']}, missed {a['missed']}, false {a['false']}")
        for gname, s in res["per_crowding_group"].items():
            run.log(f"   {name} {gname:10s}: images {s['images']}, count MAE {s['count_mae']:.2f}, bias {s['count_bias']:.2f}, F1@0.50 {s['pooled_f1@0.50']:.3f}")
    run.save_json("results.json", results)
    run.finish("measured", notes="B01 on 5 plate-grouped folds of 150 practice images; test locked; crowding groups added")


if __name__ == "__main__":
    main()
