"""Scoring for chip-photo quality (A experiments). Label convention: bad = 1 (positive), good = 0.

No single decision cut-off is assumed:
- threshold-free scores: ROC AUC, average precision (bad class), Brier score, log loss;
- threshold-dependent scores (balanced accuracy, false-pass rate) are given as full curves over every cut-off;
  when one cut-off is needed, choose_cutoff() picks it on practice data and the choice is recorded;
- review curves cover every review amount from 0 to all images.
Groups with only one class get "unavailable" (None) for scores that need both classes, never 0.
"""
import numpy as np


def _clip(p, eps=1e-12):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)


def brier(y, p):
    y = np.asarray(y, float)
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def log_loss(y, p):
    y = np.asarray(y, float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def average_precision(y, s):
    y = np.asarray(y)
    if y.sum() == 0:
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def cutoff_curve(y, s):
    """For every distinct cut-off: share of bad caught (sensitivity), share of good passed (specificity),
    balanced accuracy, false-pass rate (bad predicted good / all bad)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    cuts = np.unique(s)
    rows = []
    nb, ng = y.sum(), (1 - y).sum()
    for c in np.concatenate([cuts, [np.inf]]):
        pred_bad = s >= c
        tp = int((pred_bad & (y == 1)).sum())
        tn = int((~pred_bad & (y == 0)).sum())
        sens = tp / nb if nb else None
        spec = tn / ng if ng else None
        bal = (sens + spec) / 2 if (sens is not None and spec is not None) else None
        rows.append({"cutoff": float(c), "sensitivity_bad": sens, "specificity_good": spec,
                     "balanced_accuracy": bal, "false_pass_rate": (1 - sens) if sens is not None else None})
    return rows


def choose_cutoff(y_practice, s_practice, rule="max_balanced_accuracy"):
    """Pick one cut-off from practice data only (recorded with the rule used)."""
    curve = [r for r in cutoff_curve(y_practice, s_practice) if r["balanced_accuracy"] is not None]
    if not curve:
        return None
    if rule == "max_balanced_accuracy":
        best = max(curve, key=lambda r: (r["balanced_accuracy"], -abs(r["cutoff"])))
        return {"rule": rule, **best}
    raise ValueError(rule)


def review_curve(y, s, tie_break_seed=0):
    """Order images by score (highest = review first); ties broken by a recorded random seed.
    For every number reviewed k = 0..N: share of all bad photos caught, and bad rate among the unreviewed."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    rng = np.random.default_rng(tie_break_seed)
    order = np.lexsort((rng.random(len(s)), -s))
    caught = np.concatenate([[0], np.cumsum(y[order])])
    n, nb = len(y), y.sum()
    k = np.arange(n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        catch = caught / nb if nb else np.full(n + 1, np.nan)
        residual = np.where(n - k > 0, (nb - caught) / np.maximum(n - k, 1), np.nan)
    return {"k": k, "fraction_reviewed": k / n, "catch_rate": catch, "residual_bad_rate": residual}


def review_area(y, s, tie_break_seed=0):
    """Average catch rate over all review amounts (1.0 = perfect ordering; random ordering ~0.5)."""
    c = review_curve(y, s, tie_break_seed)["catch_rate"]
    return float(np.nanmean(c[1:])) if np.isfinite(c).any() else None


def summary(y, p):
    """Threshold-free summary for one set of predictions."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    return {"n": int(len(y)), "n_bad": int(y.sum()), "n_good": int((1 - y).sum()),
            "roc_auc": roc_auc(y, p), "avg_precision_bad": average_precision(y, p),
            "brier": brier(y, p), "log_loss": log_loss(y, p), "review_area": review_area(y, p)}


def by_group(y, p, groups):
    """Same summary for each group (e.g. cell type), with counts; scores needing both classes may be None."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    return {str(g): summary(y[groups == g], p[groups == g]) for g in np.unique(groups)}


def grouped_bootstrap(stat_fn, groups, n_boot=2000, seed=0):
    """Resample whole groups with replacement; returns the bootstrap values of stat_fn(index_array).
    n_boot only sets numerical precision of the interval; it is reported with every result."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        v = stat_fn(idx)
        if v is not None:
            vals.append(v)
    return np.asarray(vals, float)
