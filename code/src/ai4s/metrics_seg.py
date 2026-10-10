"""Scoring for nucleus outlines and counts (B experiments).

No single overlap cut-off is assumed: object matching is reported at every requested overlap level (IoU),
and joined/split/missed/false objects are defined by best-overlap rules that need no threshold.

Definitions (plain words):
- IoU of two objects = shared area / combined area.
- A match pairs one reference nucleus with one predicted nucleus whose IoU is at least the chosen level.
  We make the largest possible number of matches, and among equal numbers prefer larger total IoU.
- F1 = 2*TP / (2*TP + FP + FN); both empty -> 1, only one empty -> 0.
- Joined (merge): several reference nuclei have the same predicted object as their best overlap.
- Split: several predicted objects have the same reference nucleus as their best overlap.
- Missed: reference nucleus touched by no prediction. False: prediction touching no reference nucleus.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_IOU_LEVELS = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))  # the full range is reported, not one value


def overlap_counts(ref, pred):
    """Pixel counts shared between every reference id (rows) and prediction id (cols); row/col 0 = background."""
    r = ref.ravel().astype(np.int64)
    p = pred.ravel().astype(np.int64)
    nr, npd = int(r.max()), int(p.max())
    counts = np.bincount(r * (npd + 1) + p, minlength=(nr + 1) * (npd + 1)).reshape(nr + 1, npd + 1)
    return counts


def iou_from_counts(counts):
    inter = counts[1:, 1:].astype(np.float64)
    area_r = counts[1:, :].sum(1).astype(np.float64)
    area_p = counts[:, 1:].sum(0).astype(np.float64)
    union = area_r[:, None] + area_p[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def match_at(iou, level):
    """Largest number of one-to-one pairs with IoU >= level; ties broken by larger total IoU."""
    if iou.size == 0:
        return 0
    ok = iou >= level
    if not ok.any():
        return 0
    big = min(iou.shape) + 1.0
    cost = -(ok * big + iou * ok)
    rows, cols = linear_sum_assignment(cost)
    return int(ok[rows, cols].sum())


def structure_errors(counts):
    """Threshold-free joined/split/missed/false counts from best-overlap rules."""
    inter = counts[1:, 1:]
    nr, npd = inter.shape
    out = {"joined_ref": 0, "join_events": 0, "split_pred": 0, "split_events": 0, "missed": 0, "false": 0}
    if nr == 0 or npd == 0:
        out["missed"], out["false"] = nr, npd
        return out
    touched_r = inter.sum(1) > 0
    touched_p = inter.sum(0) > 0
    out["missed"] = int((~touched_r).sum())
    out["false"] = int((~touched_p).sum())
    best_p = np.where(touched_r, inter.argmax(1), -1)
    best_r = np.where(touched_p, inter.argmax(0), -1)
    vals, n = np.unique(best_p[best_p >= 0], return_counts=True)
    out["join_events"] = int((n > 1).sum())
    out["joined_ref"] = int(n[n > 1].sum())
    vals, n = np.unique(best_r[best_r >= 0], return_counts=True)
    out["split_events"] = int((n > 1).sum())
    out["split_pred"] = int(n[n > 1].sum())
    return out


def score_image(ref, pred, levels=DEFAULT_IOU_LEVELS):
    """All scores for one image. ref/pred are label images (0 = background, 1..n = objects, ids need not be dense)."""
    ref = _relabel(ref)
    pred = _relabel(pred)
    counts = overlap_counts(ref, pred)
    iou = iou_from_counts(counts)
    nr, npd = iou.shape
    row = {"n_ref": nr, "n_pred": npd, "count_error": npd - nr, "abs_count_error": abs(npd - nr)}
    for lv in levels:
        tp = match_at(iou, lv)
        fp, fn = npd - tp, nr - tp
        f1 = 1.0 if (nr == 0 and npd == 0) else 2 * tp / (2 * tp + fp + fn)
        row[f"tp@{lv:.2f}"], row[f"fp@{lv:.2f}"], row[f"fn@{lv:.2f}"], row[f"f1@{lv:.2f}"] = tp, fp, fn, f1
    row.update(structure_errors(counts))
    return row


def summarize(rows, levels=DEFAULT_IOU_LEVELS):
    """Dataset-level summary: count MAE and bias, mean per-image F1 and pooled F1 at every level."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {"images": int(len(df)), "count_mae": float(df["abs_count_error"].mean()),
           "count_bias": float(df["count_error"].mean()),
           "ref_objects": int(df["n_ref"].sum()), "pred_objects": int(df["n_pred"].sum())}
    for k in ["joined_ref", "join_events", "split_pred", "split_events", "missed", "false"]:
        out[k] = int(df[k].sum())
    for lv in levels:
        tp, fp, fn = (int(df[f"{x}@{lv:.2f}"].sum()) for x in ("tp", "fp", "fn"))
        out[f"pooled_f1@{lv:.2f}"] = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        out[f"mean_image_f1@{lv:.2f}"] = float(df[f"f1@{lv:.2f}"].mean())
    return out


def _relabel(lab):
    """Make ids 1..n (keeps background 0)."""
    lab = np.asarray(lab)
    ids = np.unique(lab)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros(lab.shape, np.int32)
    if ids[-1] == len(ids) and ids[0] == 1:
        return lab.astype(np.int32)
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]
