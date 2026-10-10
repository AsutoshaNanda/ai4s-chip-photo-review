"""G02 for BBBC039: lock the provider's 50 test images as the final test; split the 150 practice images
(provider training + validation) into 5 rotating folds grouped by plate (cross-validation), so that ideas are
not judged on one small 50-image practice set. Plates appear in every provider split (D49), so the final test
shares plates with practice images: a recorded limitation, not fixed here."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.model_selection import GroupKFold

from ai4s import bbbc039 as B
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run

SEED = 20261005
N_FOLDS = 5


def main():
    run = Run("G02", "bbbc039_splits", config={"seed": SEED, "n_folds": N_FOLDS})
    sp, plates = B.split_lists(), B.plates()
    rows = [{"stem": s, "plate": plates[s], "provider_split": k} for k, v in sp.items() for s in v]
    df = pd.DataFrame(rows)
    df["role"] = df["provider_split"].map({"test": "final_test", "training": "practice", "validation": "practice"})
    df["cv_fold"] = -1
    prac = df[df.role == "practice"]
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(gkf.split(prac, groups=prac["plate"])):
        df.loc[prac.index[te], "cv_fold"] = k
    df["n_objects"] = [int(B.read_labels(s).max()) for s in df["stem"]]
    ids_text = "\n".join(sorted(df.loc[df.role == "final_test", "stem"]))
    lock = {"final_test_n": int((df.role == "final_test").sum()), "practice_n": int((df.role == "practice").sum()),
            "final_test_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest()}
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS_ROOT / "bbbc039_splits.csv", index=False)
    json.dump(lock, open(SPLITS_ROOT / "bbbc039_final_test_lock.json", "w"), indent=2)
    run.save_df("bbbc039_splits.csv", df)
    rep = {"lock": lock,
           "fold_sizes": df[df.role == "practice"]["cv_fold"].value_counts().sort_index().to_dict(),
           "fold_objects": df[df.role == "practice"].groupby("cv_fold")["n_objects"].sum().to_dict(),
           "plates_per_fold": df[df.role == "practice"].groupby("cv_fold")["plate"].nunique().to_dict(),
           "final_test_objects": int(df.loc[df.role == "final_test", "n_objects"].sum())}
    run.save_json("split_report.json", rep)
    run.log(rep)
    run.finish("measured", notes="BBBC039 final test locked (provider test); 5 plate-grouped practice folds")


if __name__ == "__main__":
    main()
