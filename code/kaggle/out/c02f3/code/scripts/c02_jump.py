"""C02: does re-scaling JUMP cell-profile tables remove lab-batch noise but keep the biology?

Data: JUMP-CP cpg0016 ORF profiles "profiles_wellpos_cc_var_mad_outlier" (provider's per-plate normalised,
interpretable features, before any batch correction), 1.8 GB, BSD-3 (jump-cellpainting/datasets), plus the
provider's plate and ORF metadata. Downloaded into storage by this script if missing.
ORF data = one lab site (source_4), 13 batches. Folds: batches split into 5 groups (seeded); each fold holds out
one group of batches; everything is learned on the other batches only.

Compared on held-out batches:
  raw        the table as provided
  sphering   whitening (ZCA on correlation) learned on the TRAINING batches' negative-control wells; strength
             (eps) is not fixed: a range is scored and the one chosen on training batches (inner check: same
             metric, leave-one-training-batch-out) is reported, with the full curve saved.
Metrics (held-out batches, cosine similarity):
  batch_info   share of wells whose nearest other well comes from the same batch (lower = less batch noise),
               with the share expected by chance
  replicate_mAP for each treated well with a same-gene well in ANOTHER held-out batch: average precision of
               ranking same-gene wells above all other wells from other batches (higher = biology kept)
"""
import gzip
import io
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run

J = DATA_ROOT / "jump"
PROFILE_URL = ("https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/"
               "profiles_assembled/ORF/v1.0a/profiles_wellpos_cc_var_mad_outlier.parquet")
META = "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/"
EPS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEED = 20261006


def fetch(run):
    J.mkdir(parents=True, exist_ok=True)
    f = J / "orf_profiles_wellpos_cc_var_mad_outlier.parquet"
    if not f.exists() or f.stat().st_size < 1_800_000_000:
        t0 = time.time()
        urllib.request.urlretrieve(PROFILE_URL, f)
        run.log(f"downloaded profiles {f.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    for m in ("plate.csv.gz", "orf.csv.gz"):
        if not (J / m).exists():
            urllib.request.urlretrieve(META + m, J / m)
    return f


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def batch_info(X, batch, query):
    """Share of query wells whose nearest other well (among ALL wells) is from the same batch; chance = that
    batch's share of the other wells."""
    Z = unit(X)
    q = np.flatnonzero(query)
    same, chance = [], []
    for s in range(0, len(q), 2000):
        qi = q[s:s + 2000]
        S = Z[qi] @ Z.T
        S[np.arange(len(qi)), qi] = -np.inf
        same.append(batch[S.argmax(1)] == batch[qi])
    sizes = pd.Series(batch).value_counts()
    chance = float(np.mean([(sizes[batch[i]] - 1) / (len(batch) - 1) for i in q]))
    return float(np.concatenate(same).mean()), chance


def replicate_map(X, gene, batch, query):
    """Query wells search ALL wells from other batches; AP of finding same-gene wells."""
    Z = unit(X)
    aps = []
    q = np.flatnonzero(query)
    for s in range(0, len(q), 1000):
        qi = q[s:s + 1000]
        S = Z[qi] @ Z.T
        for row, i in zip(S, qi):
            other = batch != batch[i]
            pos = other & (gene == gene[i])
            if not pos.any():
                continue
            sc = row[other]
            lab = pos[other]
            order = np.argsort(-sc, kind="stable")
            hits = lab[order]
            ranks = np.flatnonzero(hits) + 1
            aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
    return float(np.mean(aps)) if aps else float("nan"), len(aps)


def sphere_fit(C, eps):
    mu, sd = C.mean(0), C.std(0) + 1e-9
    Zc = (C - mu) / sd
    cov = np.cov(Zc, rowvar=False)
    w, V = np.linalg.eigh(cov)
    W = V @ np.diag(1.0 / np.sqrt(np.clip(w, 0, None) + eps)) @ V.T
    return mu, sd, W


def sphere_apply(X, fit):
    mu, sd, W = fit
    return ((X - mu) / sd) @ W


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1, help="one fold only (default all 5)")
    a = ap.parse_args()
    run = Run("C02", "jump_batch_rescaling" + (f"_fold{a.fold}" if a.fold >= 0 else ""), config={"fold": a.fold, "profiles": PROFILE_URL, "eps_range": EPS, "seed": SEED,
                                                     "licence": "BSD-3-Clause (jump-cellpainting/datasets)"})
    f = fetch(run)
    P = pd.read_parquet(f)
    plate = pd.read_csv(J / "plate.csv.gz")
    orf = pd.read_csv(J / "orf.csv.gz", usecols=["Metadata_JCP2022", "Metadata_Symbol", "Metadata_pert_type"])
    P = P.merge(plate[["Metadata_Plate", "Metadata_Batch"]], on="Metadata_Plate", how="left").merge(orf, on="Metadata_JCP2022", how="left")
    feats = [c for c in P.columns if not c.startswith("Metadata_")]
    X = P[feats].to_numpy(np.float32)
    keep = np.isfinite(X).all(0)
    X = X[:, keep]
    batch = P["Metadata_Batch"].astype(str).to_numpy()
    gene = P["Metadata_JCP2022"].astype(str).to_numpy()
    ptype = P["Metadata_pert_type"].fillna("unknown").to_numpy()
    run.log(f"{len(P)} wells, {X.shape[1]} usable features (of {len(feats)}), {len(np.unique(batch))} batches, "
            f"pert types {pd.Series(ptype).value_counts().to_dict()}")
    rng = np.random.default_rng(SEED)
    batches = rng.permutation(np.unique(batch))
    groups = np.array_split(batches, 5)
    rows = []
    for k, hold in enumerate(groups):
        if a.fold >= 0 and k != a.fold:
            continue
        te = np.isin(batch, hold)
        tr = ~te
        Xtr, Xte = X[tr], X[te]
        nc = tr & (ptype == "negcon")
        # inner choice of eps on training batches: leave-one-training-batch-out replicate mAP (same metric)
        inner = {}
        tr_batches = np.unique(batch[tr])
        for eps in EPS:
            scores = []
            for b in rng.permutation(tr_batches)[:4]:          # 4 training batches (seeded) to bound time; recorded
                fit_b = sphere_fit(X[nc & (batch != b)], eps)
                sel = tr & (batch != b)
                Zb = sphere_apply(X[sel], fit_b)
                qb = (batch[sel] == b) & (ptype[sel] == "trt")
                scores.append(replicate_map(np.vstack([Zb, sphere_apply(X[batch == b], fit_b)]),
                                            np.concatenate([gene[sel], gene[batch == b]]),
                                            np.concatenate([batch[sel], batch[batch == b]]),
                                            np.concatenate([np.zeros(sel.sum(), bool), (ptype[batch == b] == "trt")]))[0])
            inner[eps] = float(np.nanmean(scores))
        eps_pick = max(inner, key=inner.get)
        query = te & (ptype == "trt")
        for name, Z in [("raw", X)] + [(f"sphering_eps{eps:g}", sphere_apply(X, sphere_fit(X[nc], eps))) for eps in EPS]:
            bi, ch = batch_info(Z, batch, te)
            mp, nq = replicate_map(Z, gene, batch, query)
            rows.append({"fold": k, "held_out_batches": ";".join(hold), "method": name, "batch_info": bi, "batch_chance": ch,
                         "replicate_mAP": mp, "queries": nq, "chosen_on_training": name == f"sphering_eps{eps_pick:g}"})
            run.log(f"fold {k} {name:22s} batch info {bi:.3f} (chance {ch:.3f}) | replicate mAP {mp:.4f} ({nq} queries)"
                    + ("  <- chosen on training batches" if rows[-1]["chosen_on_training"] else ""))
        run.log(f"fold {k}: inner choice {inner} -> eps {eps_pick:g}")
    R = pd.DataFrame(rows)
    run.save_df("per_fold.csv", R)
    chosen = R[R.chosen_on_training | (R.method == "raw")].assign(method=lambda d: np.where(d.method == "raw", "raw", "sphering_chosen"))
    summ = chosen.groupby("method")[["batch_info", "batch_chance", "replicate_mAP"]].agg(["mean", "std"]).round(4)
    run.log("C02 summary over the folds run here (held-out batches):\n" + summ.to_string())
    run.log("full curve over eps:\n" + R.groupby("method")[["batch_info", "replicate_mAP"]].mean().round(4).to_string())
    flat = summ.copy()
    flat.columns = ["_".join(c) for c in flat.columns]
    run.save_json("results.json", {"summary": flat.reset_index().to_dict(orient="records"), "inner_note":
                                   "eps chosen per fold by leave-one-training-batch-out on 4 seeded training batches; queries search all other training batches"})
    run.finish("measured", notes="C02 JUMP ORF raw vs sphering learned on training batches")


if __name__ == "__main__":
    main()
