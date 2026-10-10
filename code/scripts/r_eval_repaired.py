"""Repaired evaluation (protocol v2) of the SAVED R02, R03 and R04 policies. No training, no retuning.

Why: the RL scoring review (Evidence/2026-10-07_RL_scoring_review.md; both points re-checked in the code) found that the
original evaluation (a) reported errors against the NEXT target at target changes and (b) did not give different controllers
the same target schedule under a shared seed. This script scores every saved policy and both tuned rule-based controllers in
`DropletEnvV2` (src/ai4s/rl/droplet_env_v2.py), where the error uses the target in force and each scenario seed fixes the
plant, the whole target schedule and the measurement-noise random numbers for every controller.

Scenarios: 50 NEW scenario seeds (100000-100049; the original test used 10000-10049, tuning 30000-30009, validation
20000-20019, the first confirmation 50000-50049), in both condition ranges (TRAIN ranges, harder UNTOUCHED ranges); R04 also
under its stress tests (missing inputs 10% / 30%, measurement noise x2 / x4, TRAIN ranges). All five training seeds of every
variant are kept; the best seed is never selected.

Uncertainty: each scenario is one independent sample. For a PPO variant the five training seeds are averaged inside each
scenario before comparing; differences are paired by scenario and the 95% ranges come from resampling whole scenarios
(10,000 draws, fixed seed). Time steps and training seeds are never treated as independent samples.
R02 PPO_all, R03 ordinary and R04 inputs_all turned out to be the SAME five trained networks (training with a fixed seed is
deterministic; checked weight by weight in `identical_policies` of results.json), so they are three views of one policy set, not
three independent batches: they give identical scores here and are evaluated once per study only for transparency.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import r_control as R
from ai4s.paths import CODE_ROOT, RUNS_ROOT
from ai4s.rl.controllers import InverseModel
from ai4s.rl.droplet_env import OBS_SETS
from ai4s.rl.droplet_env_v2 import DropletEnvV2
from ai4s.runlog import Run, n_workers

SCENARIOS = list(range(100_000, 100_050))
SEED = 20261008
METRICS = ["d_err_mean", "d_err_settled_mean", "rate_err_mean", "violation_share"]
CONDS = {"TRAIN": (R.TRAIN, 1.0, 0.0), "UNTOUCHED": (R.UNTOUCHED, 1.0, 0.0),
         "missing_0.1": (R.TRAIN, 1.0, 0.1), "missing_0.3": (R.TRAIN, 1.0, 0.3),
         "noise_x2": (R.TRAIN, 2.0, 0.0), "noise_x4": (R.TRAIN, 4.0, 0.0)}


def find(pattern, probe):
    for root in (RUNS_ROOT, CODE_ROOT / "runs"):
        for d in sorted(root.glob(pattern)):
            if (d / probe).exists() or (d / probe.removesuffix(".zip")).exists():     # Kaggle unpacks .zip files inside datasets
                return d
    raise FileNotFoundError(f"{pattern} containing {probe}")


def zip_path(zip_file):
    """Path of a saved policy as a real .zip; if a dataset host unpacked it into a folder, pack that folder again."""
    z = Path(zip_file)
    if z.is_file():
        return str(z)
    folder = z.with_suffix("")
    if folder.is_dir():
        import shutil
        import tempfile
        return shutil.make_archive(str(Path(tempfile.mkdtemp()) / folder.name), "zip", root_dir=folder)
    raise FileNotFoundError(zip_file)


def specs(smoke=False):
    out = []
    r02 = find("*_R02_pi_vs_ppo", "rule_based_gains.json")
    gains = json.load(open(r02 / "rule_based_gains.json"))
    base = ["TRAIN", "UNTOUCHED"]
    for kind in ("PI", "FF+PI"):
        out.append(dict(study="R02", variant=kind, seed=-1, kind="rule", gains=gains[kind], obs_set="minimal", conds=base))
    seeds = [0] if smoke else range(5)
    for s in seeds:
        out.append(dict(study="R02", variant="PPO_all", seed=s, kind="ppo", zip=str(r02 / f"PPO_all_seed{s}.zip"),
                        vec=str(r02 / f"PPO_all_seed{s}_vecnormalize.pkl"), obs_set="all", conds=base))
    r03 = find("*_R03_staged_vs_ordinary_ppo", "ordinary_seed0.zip")
    for v in ("ordinary", "staged", "nominal_only"):
        for s in seeds:
            out.append(dict(study="R03", variant=v, seed=s, kind="ppo", zip=str(r03 / f"{v}_seed{s}.zip"),
                            vec=str(r03 / f"{v}_seed{s}_vecnormalize.pkl"), obs_set="all", conds=base))
    r04 = find("*_R04_input_sets", "inputs_all_seed0.zip")
    for o in OBS_SETS:
        for s in seeds:
            out.append(dict(study="R04", variant=f"inputs_{o}", seed=s, kind="ppo", zip=str(r04 / f"inputs_{o}_seed{s}.zip"),
                            vec=str(r04 / f"inputs_{o}_seed{s}_vecnormalize.pkl"), obs_set=o,
                            conds=base + ["missing_0.1", "missing_0.3", "noise_x2", "noise_x4"]))
    return out


def run_episode_v2(env, policy, seed, mask_prob=0.0, rng=None):
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    hold = env.cfg["target_hold_steps"]
    d_err, r_err, settled, viol, ret, t, done = [], [], [], 0, 0.0, 0, False
    while not done:
        a = policy(obs, env, rng, mask_prob)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        d_err.append(info["d_err_um"])
        r_err.append(info["rate_err_rel"])
        viol += int(info["violation"])
        if (t % hold) >= hold // 2:
            settled.append(info["d_err_um"])
        t += 1
        done = term or trunc
    return {"return": ret, "d_err_mean": float(np.mean(d_err)), "d_err_settled_mean": float(np.mean(settled)),
            "rate_err_mean": float(np.mean(r_err)), "violation_share": viol / t}


def eval_policy(args):
    spec, scenarios = args
    import torch
    torch.set_num_threads(1)
    if spec["kind"] == "rule":
        inv = InverseModel() if spec["variant"] == "FF+PI" else None
        pol = R.make_pi(spec["variant"], tuple(spec["gains"]), inv)
    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        model = PPO.load(zip_path(spec["zip"]), device="cpu")
        venv = VecNormalize.load(spec["vec"], DummyVecEnv([lambda: DropletEnvV2(obs_set=spec["obs_set"], **R.TRAIN)]))
        venv.training, venv.norm_reward = False, False
        pol = R._ppo_policy(model, venv)
    rows = []
    for cname in spec["conds"]:
        cond, extra, mask = CONDS[cname]
        env = DropletEnvV2(obs_set=spec["obs_set"], extra_noise=extra, **cond)
        for sc in scenarios:
            rng = np.random.default_rng([sc, 99]) if mask else None
            rows.append({"study": spec["study"], "variant": spec["variant"], "training_seed": spec["seed"], "condition": cname,
                         "scenario": sc, **run_episode_v2(env, pol, sc, mask, rng)})
    return rows


def boot_ci(diff, n_boot=10_000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def comparisons(variants):
    c = []
    for a in ("PPO_all",):
        for b in ("PI", "FF+PI"):
            c.append(("R02", a, b))
    c += [("R03", "staged", "ordinary"), ("R03", "nominal_only", "ordinary")]
    for o in OBS_SETS:
        if o != "camera":
            c.append(("R04", f"inputs_{o}", "inputs_camera"))
    c.append(("R04", "inputs_all", "inputs_minimal"))
    return [x for x in c if x[1] in variants and x[2] in variants]


def check_identical(run):
    """Are R02 PPO_all, R03 ordinary and R04 inputs_all the same networks and input scaling for each training seed?"""
    import pickle
    import torch
    from stable_baselines3 import PPO
    r02, r03, r04 = (find("*_R02_pi_vs_ppo", "rule_based_gains.json"), find("*_R03_staged_vs_ordinary_ppo", "ordinary_seed0.zip"),
                     find("*_R04_input_sets", "inputs_all_seed0.zip"))
    out = {}
    for s in range(5):
        sd = [PPO.load(zip_path(p), device="cpu").policy.state_dict() for p in (r02 / f"PPO_all_seed{s}.zip", r03 / f"ordinary_seed{s}.zip", r04 / f"inputs_all_seed{s}.zip")]
        vec = [pickle.load(open(p, "rb")) for p in (r02 / f"PPO_all_seed{s}_vecnormalize.pkl", r03 / f"ordinary_seed{s}_vecnormalize.pkl", r04 / f"inputs_all_seed{s}_vecnormalize.pkl")]
        same_w = all(all(torch.equal(sd[0][k], sd[i][k]) for k in sd[0]) for i in (1, 2))
        same_s = all(np.allclose(vec[0].obs_rms.mean, v.obs_rms.mean) and np.allclose(vec[0].obs_rms.var, v.obs_rms.var) for v in vec[1:])
        out[f"seed{s}"] = {"weights_identical": bool(same_w), "input_scaling_identical": bool(same_s)}
    run.log(f"R02 PPO_all / R03 ordinary / R04 inputs_all identical networks: {out}")
    return out


def analyse(D, run):
    res = {"by_variant": {}, "paired": []}
    G = D.groupby(["variant", "condition", "scenario"])[METRICS].mean().reset_index()      # seeds averaged within a scenario
    variants = set(D["variant"])
    for (v, cname), g in G.groupby(["variant", "condition"]):
        res["by_variant"][f"{v}|{cname}"] = {m: {"mean": float(g[m].mean()), "median": float(g[m].median()),
                                                 "p95": float(np.percentile(g[m], 95)), "worst_scenario": float(g[m].max())} for m in METRICS}
    rows = []
    for study, a, b in comparisons(variants):
        for cname in sorted(set(D["condition"])):
            ga = G[(G.variant == a) & (G.condition == cname)].set_index("scenario").sort_index()
            gb = G[(G.variant == b) & (G.condition == cname)].set_index("scenario").sort_index()
            if len(ga) == 0 or len(gb) == 0 or not ga.index.equals(gb.index):
                continue
            for m in METRICS:
                diff = (ga[m] - gb[m]).to_numpy()
                lo, hi = boot_ci(diff)
                rows.append({"comparison": study, "A": a, "B": b, "condition": cname, "metric": m, "mean_A": float(ga[m].mean()),
                             "mean_B": float(gb[m].mean()), "difference_A_minus_B": float(diff.mean()), "range_95_low": lo,
                             "range_95_high": hi, "share_scenarios_A_better": float((diff < 0).mean()),
                             "worst_scenario_for_A": float(diff.max()), "scenarios": int(len(diff))})
    P = pd.DataFrame(rows)
    res["paired"] = rows
    S = D.groupby(["study", "variant", "training_seed", "condition"])[METRICS].agg(["mean", "max"]).round(4)
    S.columns = ["_".join(c) for c in S.columns]
    S = S.reset_index()
    return res, P, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 scenarios, one training seed per variant: crash check only")
    a = ap.parse_args()
    scen = SCENARIOS[:2] if a.smoke else SCENARIOS
    sp = specs(a.smoke)
    run = Run("R02", "repaired_evaluation_v2" + ("_smoke" if a.smoke else ""), config={
        "protocol": "v2: error against target in force; separate streams for plant, target schedule, noise (common random numbers)",
        "scenarios": [scen[0], scen[-1], len(scen)], "conditions": CONDS and {k: {"extra_noise": v[1], "missing_prob": v[2]} for k, v in CONDS.items()},
        "policies": len(sp), "applies_to": ["R02", "R03", "R04"], "no_training": True, "bootstrap": "whole scenarios, 10000 draws, seed 20261008"})
    run.log(f"{len(sp)} saved policies / controllers; {len(scen)} scenarios each")
    with ProcessPoolExecutor(n_workers()) as ex:
        rows = [r for part in ex.map(eval_policy, [(s, scen) for s in sp]) for r in part]
    D = pd.DataFrame(rows)
    run.save_df("episodes.csv", D)
    res, P, S = analyse(D, run)
    res["identical_policies"] = check_identical(run)
    run.save_df("paired_differences.csv", P)
    run.save_df("summary_by_policy.csv", S)
    run.save_json("results.json", res)
    show = P[(P.metric == "d_err_mean") & (P.condition.isin(["TRAIN", "UNTOUCHED"]))]
    for _, r in show.iterrows():
        run.log(f"{r.comparison:11s} {r.A:16s} minus {r.B:16s} [{r.condition:9s}] diameter error: {r.mean_A:.3f} vs {r.mean_B:.3f}  "
                f"diff {r.difference_A_minus_B:+.3f} [{r.range_95_low:+.3f}, {r.range_95_high:+.3f}]  A better in {100 * r.share_scenarios_A_better:.0f}% of scenarios")
    run.finish("measured", notes="repaired R02-R04 evaluation (protocol v2) of saved policies; no training")


if __name__ == "__main__":
    main()
