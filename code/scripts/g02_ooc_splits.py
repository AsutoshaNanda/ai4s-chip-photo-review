"""G02 for chip photos: lock the final-test photos BEFORE any training, then build practice (development) folds.

Rules (decided from the G01 evidence, recorded in splits/README_ooc.md):
- Grouping unit = filename prefix (an UNCONFIRMED clue, probably an acquisition date). Photos that are exact
  pixel duplicates or have identical DCT hashes are merged into one group, so copies never sit on both sides.
- Final-test size = the dataset authors' own test share (656 of 3,072 = 21.4%), taken from the source,
  not invented. Whole groups only.
- Selection: StratifiedGroupKFold over (cell type, label) with a recorded seed; among its folds, the one that
  covers both classes in the most cell types is chosen (ties: closest to 21.4%). This uses only labels and
  counts, never model results.
- Practice data = everything else, split into grouped, stratified rotating folds (cross-validation).
The original published split is kept as a column for reference only.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ai4s.paths import RUNS_ROOT, SPLITS_ROOT
from ai4s.runlog import Run

SEED = 20261005
N_DEV_FOLDS = 5
AUTHORS_TEST_SHARE = 656 / 3072


def latest_run(tag):
    runs = sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir() and tag in p.name)
    return runs[-1]


def union_groups(df, pairs):
    parent = {g: g for g in df["prefix"].unique()}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pid = dict(zip(df["image_id"], df["prefix"]))
    merged = []
    for a, b in pairs:
        ra, rb = find(pid[a]), find(pid[b])
        if ra != rb:
            parent[rb] = ra
            merged.append((pid[a], pid[b]))
    return df["prefix"].map(find), merged


def composition(d):
    return d.groupby(["cell_type", "label_bad"]).size().unstack(fill_value=0).rename(columns={0: "good", 1: "bad"})


def main():
    g01 = latest_run("G01_ooc_check")
    df = pd.read_csv(g01 / "per_photo.csv", dtype={"prefix": str, "image_id": str})
    summ = json.load(open(g01 / "summary.json"))
    run = Run("G02", "ooc_splits", config={"g01_run": g01.name, "seed": SEED, "n_dev_folds": N_DEV_FOLDS,
                                           "final_test_share_target": AUTHORS_TEST_SHARE})

    pairs = []
    for grp in summ.get("exact_pixel_duplicate_groups", []):
        pairs += [(grp[0], x) for x in grp[1:]]
    pairs += [tuple(p) for p in summ.get("hash_identical_pairs_list", [])]
    df["group"], merged = union_groups(df, pairs)
    run.log(f"groups: {df['prefix'].nunique()} prefixes -> {df['group'].nunique()} groups after merging duplicates; merges={merged}")

    strat = df["cell_type"] + "_" + df["label_bad"].astype(str)
    n_folds_test = int(round(1 / AUTHORS_TEST_SHARE))
    sgk = StratifiedGroupKFold(n_splits=n_folds_test, shuffle=True, random_state=SEED)
    candidates = []
    for k, (_, te) in enumerate(sgk.split(df, strat, df["group"])):
        sub = df.iloc[te]
        comp = composition(sub)
        both = int(((comp.get("good", 0) > 0) & (comp.get("bad", 0) > 0)).sum())
        candidates.append({"fold": k, "n": len(te), "share": len(te) / len(df), "cell_types_with_both_classes": both,
                           "cell_types_present": int(sub["cell_type"].nunique()), "idx": te})
    best = sorted(candidates, key=lambda c: (-c["cell_types_with_both_classes"], abs(c["share"] - AUTHORS_TEST_SHARE)))[0]
    run.save_json("final_test_candidates.json", [{k: v for k, v in c.items() if k != "idx"} for c in candidates])
    df["role"] = "practice"
    df.loc[df.index[best["idx"]], "role"] = "final_test"

    dev = df[df["role"] == "practice"].copy()
    sgk2 = StratifiedGroupKFold(n_splits=N_DEV_FOLDS, shuffle=True, random_state=SEED)
    df["dev_fold"] = -1
    for k, (_, te) in enumerate(sgk2.split(dev, strat[dev.index], dev["group"])):
        df.loc[dev.index[te], "dev_fold"] = k

    # checks
    test_groups = set(df.loc[df.role == "final_test", "group"])
    dev_groups = set(df.loc[df.role == "practice", "group"])
    assert not (test_groups & dev_groups), "a group is on both sides"
    fold_groups = df[df.role == "practice"].groupby("dev_fold")["group"].apply(set)
    for a in fold_groups.index:
        for b in fold_groups.index:
            if a < b:
                assert not (fold_groups[a] & fold_groups[b]), "a group is in two practice folds"

    cols = ["image_id", "prefix", "group", "cell_type", "label_bad", "published_split", "role", "dev_fold", "member"]
    out = df[cols].sort_values("image_id").reset_index(drop=True)
    ids_text = "\n".join(out.loc[out.role == "final_test", "image_id"])
    lock = {"final_test_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
            "final_test_n": int((out.role == "final_test").sum()), "practice_n": int((out.role == "practice").sum()),
            "chosen_candidate": {k: v for k, v in best.items() if k != "idx"}}
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLITS_ROOT / "ooc_splits.csv", index=False)
    json.dump(lock, open(SPLITS_ROOT / "ooc_final_test_lock.json", "w"), indent=2)
    run.save_df("ooc_splits.csv", out)
    run.save_json("lock.json", lock)

    report = {"final_test_composition": composition(out[out.role == "final_test"]).to_dict(),
              "practice_composition": composition(out[out.role == "practice"]).to_dict(),
              "practice_fold_sizes": out[out.role == "practice"]["dev_fold"].value_counts().sort_index().to_dict(),
              "practice_fold_bad_share": out[out.role == "practice"].groupby("dev_fold")["label_bad"].mean().round(3).to_dict(),
              "final_test_groups": sorted(test_groups), "n_groups_final": len(test_groups), "n_groups_practice": len(dev_groups),
              "published_split_of_final_test": out[out.role == "final_test"]["published_split"].value_counts().to_dict()}
    run.save_json("split_report.json", report)
    run.log(f"lock: {lock}")
    run.log(f"report: {report}")
    (SPLITS_ROOT / "README_ooc.md").write_text(
        "# Chip-photo splits (locked " + run.started + ")\n\n" + (__doc__ or "") +
        f"\nFinal-test photos: {lock['final_test_n']}; practice photos: {lock['practice_n']}.\n"
        f"SHA-256 of the sorted final-test id list: {lock['final_test_ids_sha256']}\n"
        "The final-test photos are opened once, on Thursday (F01). Do not change this file after training starts.\n")
    run.finish("measured", notes="final test locked; practice folds built")


if __name__ == "__main__":
    main()
