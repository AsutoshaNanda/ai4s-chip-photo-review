"""G02: test the scoring code on fake answers before any real use.
Segmentation: perfect, empty, both-empty, false-only, merged pair, split object, partial overlap, relabelled ids,
and a real BBBC039 mask scored against itself. Classification: perfect, all-wrong, constant, one-class group,
cut-off choice, review curve, grouped bootstrap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ai4s import bbbc039 as B
from ai4s import metrics_cls as C
from ai4s import metrics_seg as S
from ai4s.runlog import Run

results = []


def check(name, cond, detail=""):
    results.append({"test": name, "passed": bool(cond), "detail": str(detail)})
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def seg_tests():
    ref = np.zeros((40, 60), int)
    ref[5:15, 5:15] = 1
    ref[5:15, 15:25] = 2          # touches object 1
    ref[25:35, 40:55] = 3
    lv = S.DEFAULT_IOU_LEVELS

    r = S.score_image(ref, ref)
    check("perfect: F1=1 at every level", all(r[f"f1@{x:.2f}"] == 1.0 for x in lv))
    check("perfect: no count error / joins / splits", r["count_error"] == 0 and r["join_events"] == 0 and r["split_events"] == 0)

    r = S.score_image(ref, np.zeros_like(ref))
    check("empty prediction: F1=0, all missed", r["f1@0.50"] == 0 and r["missed"] == 3 and r["count_error"] == -3)

    z = np.zeros_like(ref)
    r = S.score_image(z, z)
    check("both empty: F1=1", r["f1@0.50"] == 1.0 and r["n_ref"] == 0)

    r = S.score_image(z, ref)
    check("reference empty, prediction not: F1=0, all false", r["f1@0.50"] == 0 and r["false"] == 3)

    merged = ref.copy()
    merged[merged == 2] = 1
    r = S.score_image(ref, merged)
    check("merged pair: 1 join event joining 2 nuclei", r["join_events"] == 1 and r["joined_ref"] == 2, r)
    check("merged pair: count error -1", r["count_error"] == -1)

    split = ref.copy()
    split[25:35, 48:55] = 4       # object 3 cut into two predictions
    r = S.score_image(ref, split)
    check("split object: 1 split event made of 2 predictions", r["split_events"] == 1 and r["split_pred"] == 2, r)

    shifted = np.zeros_like(ref)
    shifted[7:17, 7:17] = 1       # overlaps object 1 with IoU 64/136
    r = S.score_image(ref[:, :15] * (ref[:, :15] == 1), shifted[:, :15] * 0 + (shifted[:, :15] > 0))
    f1s = [S.score_image(np.where(ref == 1, 1, 0), shifted)[f"f1@{x:.2f}"] for x in lv]
    check("partial overlap: F1 never increases as the level gets stricter", all(a >= b for a, b in zip(f1s, f1s[1:])), f1s)
    iou = 64 / 136
    expect = [1.0 if x <= iou else 0.0 for x in lv]
    check("partial overlap: matched exactly when IoU >= level", f1s == expect, f"IoU={iou:.3f}")

    perm = np.zeros_like(ref)
    perm[ref == 1], perm[ref == 2], perm[ref == 3] = 17, 5, 9
    a, b = S.score_image(ref, ref), S.score_image(ref, perm)
    check("ids can be any numbers (relabelling gives the same scores)", a == b)

    stem = B.split_lists()["training"][0]
    lab = B.read_labels(stem)
    r = S.score_image(lab, lab)
    check("real BBBC039 mask vs itself: perfect", r["f1@0.95"] == 1.0 and r["count_error"] == 0, f"objects={r['n_ref']}")


def cls_tests():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    p = y.astype(float)
    s = C.summary(y, p)
    check("perfect: AUC=1, AP=1, Brier=0", s["roc_auc"] == 1.0 and s["avg_precision_bad"] == 1.0 and s["brier"] == 0.0)
    rc = C.review_curve(y, p)
    k = int(y.sum())
    check("perfect: all bad caught after reviewing exactly n_bad photos", rc["catch_rate"][k] == 1.0 and rc["catch_rate"][k - 1] < 1.0)

    s = C.summary(y, 1 - p)
    check("all wrong: AUC=0, Brier=1", s["roc_auc"] == 0.0 and s["brier"] == 1.0)

    s = C.summary(y, np.full(len(y), 0.3))
    check("constant score: AUC=0.5", s["roc_auc"] == 0.5)

    s = C.summary(np.zeros(5, int), np.linspace(0, 1, 5))
    check("one-class group: AUC and AP unavailable (None), not 0", s["roc_auc"] is None and s["avg_precision_bad"] is None)

    cut = C.choose_cutoff(y, p)
    check("cut-off chosen on practice data separates perfectly", cut["balanced_accuracy"] == 1.0, cut)

    groups = np.array(list("aabbccddee"))
    vals = C.grouped_bootstrap(lambda idx: C.brier(y[idx], p[idx]), groups, n_boot=200, seed=1)
    check("grouped bootstrap runs and keeps perfect Brier at 0", len(vals) == 200 and np.allclose(vals, 0))

    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 4000)
    area = C.review_area(yy, rng.random(4000))
    check("random scores: review area near 0.5", abs(area - 0.5) < 0.05, f"{area:.3f}")


def main():
    run = Run("G02", "scoring_tests")
    seg_tests()
    cls_tests()
    run.save_json("tests.json", results)
    n_fail = sum(not r["passed"] for r in results)
    run.log(f"{len(results) - n_fail}/{len(results)} tests passed")
    run.finish("measured" if n_fail == 0 else "failed", notes=f"{n_fail} failing tests")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
