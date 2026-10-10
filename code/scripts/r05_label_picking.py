"""R05: can an RL agent choose which chip photos to label better than random or simple smart picking?

Same set-up as A11 so the curves are directly comparable: units are photo groups (whole groups are labelled
together), A02's best features, logistic regression with C fixed at 0.1 (A11's recorded choice), the same label
budgets (number of groups, geometric steps from 4 to the whole pool) and the same 4 random starting groups per repeat.
Practice photos only; the final test's labels stay hidden.

For each outer practice fold k (one job per fold, --fold):
  pool = groups of the other practice folds; scored on fold k at every budget.
  random       groups in random order (A11's order for that repeat)
  uncertainty  next group = highest mean uncertainty (A11 "active")
  rl           a learned picking rule. Each candidate group is described by 5 numbers (mean uncertainty, mean
               predicted "bad" chance, photo count, distance to the nearest already-labelled group, progress);
               the agent scores candidates with a linear rule and picks by softmax. Trained by REINFORCE
               (trial and error) ONLY inside the pool: every training episode splits the pool's groups at random
               into a picking pool and a hidden check set; reward = gain in check-set AUC after each pick.
               At test time the agent picks greedily from the whole pool; fold k is never seen in training.
Learning curve (reward per training episode) is saved so convergence can be judged, not assumed.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from a_suite import SEED, best_feature_set, load
from ai4s import metrics_cls as C
from ai4s.runlog import Run

C_FIXED = 0.1          # same as A11 (recorded there), so curves are comparable
N_START = 4            # same 4 random starting groups as A11


def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_FIXED, max_iter=3000)).fit(X, y)


class Pool:
    def __init__(self, X, y, groups, pool_groups):
        self.X, self.y, self.groups = X, y, groups
        self.pool_groups = list(pool_groups)
        self.idx = {g: np.flatnonzero(groups == g) for g in pool_groups}
        self.gmean = {g: X[self.idx[g]].mean(0) / max(np.linalg.norm(X[self.idx[g]].mean(0)), 1e-9) for g in pool_groups}

    def rows(self, chosen):
        return np.concatenate([self.idx[g] for g in chosen])

    def model(self, chosen):
        r = self.rows(chosen)
        return fit(self.X[r], self.y[r]) if len(np.unique(self.y[r])) == 2 else None

    def state(self, m, chosen, rest, progress):
        G = np.stack([self.gmean[g] for g in chosen])
        feats = []
        for g in rest:
            p = m.predict_proba(self.X[self.idx[g]])[:, 1] if m is not None else np.full(len(self.idx[g]), 0.5)
            feats.append([np.mean(1 - np.abs(2 * p - 1)), p.mean(), np.log(len(self.idx[g])),
                          1 - float((G @ self.gmean[g]).max()), progress])
        F = np.array(feats)
        sd = F.std(0)
        F[:, :4] = (F[:, :4] - F[:, :4].mean(0)) / np.where(sd[:4] > 0, sd[:4], 1)   # compare candidates with each other
        return F


def auc_on(m, X, y):
    return 0.5 if m is None else C.roc_auc(y, m.predict_proba(X)[:, 1])


def pick_sequence(P, start, size_max, how, theta=None, rng=None):
    """Yields the chosen list after each pick. how in {random(order given), uncertainty, rl_greedy}."""
    chosen = list(start)
    while len(chosen) < size_max:
        rest = [g for g in P.pool_groups if g not in chosen]
        m = P.model(chosen)
        if how == "uncertainty" and m is not None:
            F = P.state(m, chosen, rest, len(chosen) / len(P.pool_groups))
            chosen.append(rest[int(np.argmax(F[:, 0]))])
        elif how == "rl_greedy" and m is not None:
            F = P.state(m, chosen, rest, len(chosen) / len(P.pool_groups))
            chosen.append(rest[int(np.argmax(F @ theta))])
        else:                                   # random, or a model is not possible yet (one class only)
            chosen.append(rest[int(rng.integers(len(rest)))])
        yield chosen


def train_agent(X, y, groups, pool_groups, episodes, run, rng, lr=0.05):
    theta, base, curve = np.zeros(5), None, []
    for ep in range(episodes):
        g = rng.permutation(pool_groups)
        n_check = max(2, len(g) // 4)
        check, pick_from = g[:n_check], g[n_check:]
        Pe = Pool(X, y, groups, pick_from)
        cr = np.isin(groups, check)
        chosen = list(pick_from[:N_START])
        m = Pe.model(chosen)
        prev = auc_on(m, X[cr], y[cr])
        grads, rewards = [], []
        steps = int(rng.integers(N_START + 1, len(pick_from)))          # varied episode lengths = varied budgets
        while len(chosen) < steps:
            rest = [q for q in Pe.pool_groups if q not in chosen]
            if m is None:
                chosen.append(rest[int(rng.integers(len(rest)))])
                m = Pe.model(chosen)
                prev = auc_on(m, X[cr], y[cr])
                continue
            F = Pe.state(m, chosen, rest, len(chosen) / len(Pe.pool_groups))
            z = F @ theta
            pr = np.exp(z - z.max())
            pr /= pr.sum()
            a = int(rng.choice(len(rest), p=pr))
            grads.append(F[a] - pr @ F)
            chosen.append(rest[a])
            m = Pe.model(chosen)
            now = auc_on(m, X[cr], y[cr])
            rewards.append(now - prev)
            prev = now
        if rewards:
            ret = np.cumsum(np.array(rewards)[::-1])[::-1]      # reward still to come after each pick
            total = float(ret[0])
            base = total if base is None else 0.9 * base + 0.1 * total
            theta += lr * sum(gr * (r - base * (len(ret) - i) / len(ret)) for i, (gr, r) in enumerate(zip(grads, ret)))
            curve.append({"episode": ep, "picks": len(rewards), "total_auc_gain": total, "theta": theta.round(4).tolist()})
    return theta, pd.DataFrame(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.episodes, a.repeats = 3, 1
    a02, prac, feats, oof = load()
    best, _ = best_feature_set(oof)
    X, y, groups = feats[best], prac["label_bad"].to_numpy(), prac["group"].to_numpy()
    k = a.fold
    run = Run("R05", f"label_picking_fold{k}" + ("_smoke" if a.smoke else ""),
              config={"features": best, "a02_run": a02.name, "C_fixed": C_FIXED, "start_groups": N_START,
                      "episodes": a.episodes, "repeats": a.repeats, "fold": k, "smoke": a.smoke,
                      "policy": "linear softmax over 5 candidate-group numbers, REINFORCE with moving-average baseline"})
    pool_mask = (prac["dev_fold"] != k).to_numpy()
    te = ~pool_mask
    pool_groups = np.unique(groups[pool_mask])
    sizes = sorted({int(x) for x in np.unique(np.round(np.geomspace(N_START, len(pool_groups), 6)))})
    rng = np.random.default_rng(SEED + 1000 + k)
    t0 = time.perf_counter()
    theta, curve = train_agent(X, y, groups, pool_groups, a.episodes, run, rng)
    train_s = time.perf_counter() - t0
    run.save_df("training_curve.csv", curve)
    q = max(1, len(curve) // 5)
    run.log(f"fold {k}: trained {a.episodes} episodes in {train_s:.0f} s; mean AUC gain per episode, first vs last fifth: "
            f"{curve['total_auc_gain'][:q].mean():+.3f} vs {curve['total_auc_gain'][-q:].mean():+.3f}; weights {theta.round(3).tolist()} "
            f"(uncertainty, p_bad, log size, distance, progress)")
    P = Pool(X, y, groups, pool_groups)
    rows = []
    for rep in range(a.repeats):
        order_rng = np.random.default_rng(SEED + 31 * rep + k)        # A11's starting order for this repeat
        order = order_rng.permutation(pool_groups)
        start = list(order[:N_START])
        for how in ("random", "uncertainty", "rl_greedy"):
            seq_rng = np.random.default_rng(SEED + 7 * rep + k)
            if how == "random":
                seqs = {s: list(order[:s]) for s in sizes}
            else:
                seqs = {N_START: start} if N_START in sizes else {}
                for ch in pick_sequence(P, start, sizes[-1], how, theta, seq_rng):
                    if len(ch) in sizes:
                        seqs[len(ch)] = list(ch)
            for s in sizes:
                r = P.rows(seqs[s])
                m = fit(X[r], y[r]) if len(np.unique(y[r])) == 2 else None
                p = m.predict_proba(X[te])[:, 1] if m is not None else None
                rows.append({"fold": k, "repeat": rep, "strategy": how, "groups": s, "photos": len(r),
                             "auc": C.roc_auc(y[te], p) if p is not None else None,
                             "log_loss": C.log_loss(y[te], p) if p is not None else None})
    D = pd.DataFrame(rows)
    run.save_df("budget_curves.csv", D)
    summ = D.groupby(["strategy", "groups"])[["photos", "auc", "log_loss"]].agg(["mean", "std"]).round(3)
    run.log(f"R05 fold {k} (scored on fold {k}; pool = other folds):\n{summ.to_string()}")
    flat = summ.copy()
    flat.columns = ["_".join(c) for c in flat.columns]
    run.save_json("results.json", {"theta": theta.tolist(), "train_seconds": train_s, "summary": flat.reset_index().to_dict(orient="records")})
    run.finish("measured", notes=f"R05 fold {k}; practice only, final test hidden")


if __name__ == "__main__":
    main()
