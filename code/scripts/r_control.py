"""R02, R03, R04: droplet control on the environment built from the real-data-fitted model (R01).
Practice environment; NOT proof about real chips.

Condition sets (design choices recorded here; values from the paper are marked):
  TRAIN     : pump lag tau ~ U(0.5, 5) s (not in paper, swept); mismatch diameter +/-3% and rate +/-20% (paper's
              reported model accuracy); diameter measurement sd 2 um (paper's +/-2 um); droplet-count noise (physics).
  NOMINAL   : tau 0.5 s, no mismatch, no measurement noise (R03 stage 1).
  UNTOUCHED : evaluation only, outside training ranges: tau ~ U(5, 15) s, mismatch +/-5% and +/-30%, diameter sd 3 um.
R02: PI and FF+PI (tuned on TRAIN by random search) vs ordinary PPO (TRAIN, all inputs).
R03: ordinary PPO (TRAIN) vs staged PPO (half NOMINAL then half TRAIN; same total steps) vs NOMINAL-only PPO.
R04: PPO with each input set (minimal / pumps / camera / history / all), same steps; stress tests at evaluation:
     inputs missing (replaced by the training average after scaling) and extra measurement noise.
Scaling (VecNormalize) is learned during training only and frozen for evaluation. No future information in inputs.
Every evaluation reports full distributions (mean, median, 95th percentile, worst episode), not a pass mark.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s.rl.controllers import FFPI, PI, InverseModel
from ai4s.rl.droplet_env import OBS_SETS, DropletEnv
from ai4s.runlog import Run, n_workers

TRAIN = dict(tau_range=(0.5, 5.0), mismatch_d=0.03, mismatch_rate=0.20, noise_d_um=2.0, rate_noise="poisson")
NOMINAL = dict(tau_range=(0.5, 0.5), mismatch_d=0.0, mismatch_rate=0.0, noise_d_um=0.0, rate_noise="relative", noise_rate_rel=0.0)
UNTOUCHED = dict(tau_range=(5.0, 15.0), mismatch_d=0.05, mismatch_rate=0.30, noise_d_um=3.0, rate_noise="poisson")
TEST_SEEDS = list(range(10_000, 10_050))      # never used for tuning or training
VAL_SEEDS = list(range(20_000, 20_020))       # learning curves / PI tuning check
TUNE_SEEDS = list(range(30_000, 30_010))      # PI tuning


def run_episode(env, policy, seed, mask_prob=0.0, rng=None):
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    hold = env.cfg["target_hold_steps"]
    d_err, r_err, viol, settled_d = [], [], 0, []
    done, t, ret = False, 0, 0.0
    while not done:
        a = policy(obs, env, rng, mask_prob)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        d_err.append(info["d_err_um"])
        r_err.append(info["rate_err_rel"])
        viol += int(info["violation"])
        if (t % hold) >= hold // 2:
            settled_d.append(info["d_err_um"])
        t += 1
        done = term or trunc
    return {"return": ret, "d_err_mean": float(np.mean(d_err)), "d_err_settled_mean": float(np.mean(settled_d)),
            "rate_err_mean": float(np.mean(r_err)), "violation_share": viol / t}


def summarize(rows):
    df = pd.DataFrame(rows)
    out = {}
    for c in ["d_err_mean", "d_err_settled_mean", "rate_err_mean", "violation_share", "return"]:
        v = df[c].to_numpy()
        out[c] = {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                  "worst": float(v.max() if c != "return" else v.min())}
    out["episodes"] = len(df)
    return out


class PIPolicy:
    def __init__(self, ctrl):
        self.c = ctrl

    def reset(self):
        self.c.reset()

    def __call__(self, obs, env, rng, mask_prob):
        return self.c.act(float(obs[0]), float(obs[1]), env)


def make_pi(kind, k, inverse=None):
    return PIPolicy(PI(*k) if kind == "PI" else FFPI(inverse, *k))


def tune_one(args):
    kind, k, cond = args
    inv = InverseModel() if kind == "FF+PI" else None
    env = DropletEnv(obs_set="minimal", **cond)
    pol = make_pi(kind, k, inv)
    rets = [run_episode(env, pol, s)["return"] for s in TUNE_SEEDS]
    return kind, k, float(np.mean(rets))


def tune_controllers(workers, run, n_candidates=400):
    rng = np.random.default_rng(0)
    cands = []
    for kind in ("PI", "FF+PI"):
        for _ in range(n_candidates):
            k = tuple(float(x) for x in 10 ** rng.uniform([-2, -3, -2, -3], [0.5, -0.5, 0.5, -0.5]))
            cands.append((kind, k, TRAIN))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(tune_one, cands, chunksize=4))
    best = {}
    for kind, k, score in res:
        if kind not in best or score > best[kind][1]:
            best[kind] = (k, score)
    run.log(f"rule-based controllers tuned on TRAIN ({n_candidates} random gain sets each): {best}")
    return {kind: v[0] for kind, v in best.items()}


def ppo_train_eval(args):
    """Train one PPO run (stages = list of (cond, steps)); evaluate on TEST/UNTOUCHED (+ stress for R04)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    tag, obs_set, stages, seed, out_dir, stress = args
    t0 = time.time()
    venv = VecNormalize(DummyVecEnv([lambda: DropletEnv(obs_set=obs_set, **stages[0][0])]), norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", venv, n_steps=2048, batch_size=256, seed=seed, device="cpu", verbose=0)
    curve, done_steps = [], 0
    for cond, steps in stages:
        venv.envs[0].__init__(obs_set=obs_set, **cond)          # switch conditions, same normaliser and policy
        chunk = 50_000
        while steps > 0:
            n = min(chunk, steps)
            model.learn(total_timesteps=n, reset_num_timesteps=False)
            steps -= n
            done_steps += n
            curve.append({"steps": done_steps, **summarize([run_episode(DropletEnv(obs_set=obs_set, **TRAIN),
                          _ppo_policy(model, venv), s) for s in VAL_SEEDS[:8]])["d_err_mean"]})
    venv.training = False
    pol = _ppo_policy(model, venv)
    rng = np.random.default_rng(seed)
    res = {"tag": tag, "obs_set": obs_set, "seed": seed, "train_seconds": time.time() - t0, "env_steps": done_steps,
           "curve": curve,
           "test_TRAIN": summarize([run_episode(DropletEnv(obs_set=obs_set, **TRAIN), pol, s) for s in TEST_SEEDS]),
           "test_UNTOUCHED": summarize([run_episode(DropletEnv(obs_set=obs_set, **UNTOUCHED), pol, s) for s in TEST_SEEDS])}
    if stress:
        for mp in (0.1, 0.3):
            res[f"missing_{mp}"] = summarize([run_episode(DropletEnv(obs_set=obs_set, **TRAIN), pol, s, mp, rng) for s in TEST_SEEDS])
        for xn in (2.0, 4.0):
            res[f"noise_x{xn:g}"] = summarize([run_episode(DropletEnv(obs_set=obs_set, extra_noise=xn, **TRAIN), pol, s) for s in TEST_SEEDS])
    model.save(os.path.join(out_dir, f"{tag}_seed{seed}.zip"))
    venv.save(os.path.join(out_dir, f"{tag}_seed{seed}_vecnormalize.pkl"))
    return res


def _ppo_policy(model, venv):
    def act(obs, env, rng, mask_prob):
        o = venv.normalize_obs(np.asarray(obs, np.float32)[None])
        if mask_prob and rng is not None:
            o = np.where(rng.random(o.shape) < mask_prob, 0.0, o)    # missing input -> training average
        a, _ = model.predict(o, deterministic=True)
        return a[0]
    return act


def eval_rule_based(kind, k):
    inv = InverseModel() if kind == "FF+PI" else None
    out = {}
    for name, cond in (("test_TRAIN", TRAIN), ("test_UNTOUCHED", UNTOUCHED)):
        env = DropletEnv(obs_set="minimal", **cond)
        pol = make_pi(kind, k, inv)
        out[name] = summarize([run_episode(env, pol, s) for s in TEST_SEEDS])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="R02,R03,R04")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.seeds = 4096, 1
        global TEST_SEEDS, VAL_SEEDS, TUNE_SEEDS
        TEST_SEEDS, VAL_SEEDS, TUNE_SEEDS = TEST_SEEDS[:2], VAL_SEEDS[:2], TUNE_SEEDS[:2]
    workers = n_workers()
    seeds = list(range(a.seeds))
    common = {"TRAIN": TRAIN, "NOMINAL": NOMINAL, "UNTOUCHED": UNTOUCHED, "steps_per_ppo_run": a.steps, "seeds": seeds,
              "test_episodes": len(TEST_SEEDS), "note": "practice environment from a real-data-fitted model; not proof on real chips"}
    for exp in a.exp.split(","):
        run = Run(exp, {"R02": "pi_vs_ppo", "R03": "staged_vs_ordinary_ppo", "R04": "input_sets"}[exp], config=common)
        jobs = []
        if exp == "R02":
            gains = tune_controllers(workers, run, n_candidates=20 if a.smoke else 400)
            run.save_json("rule_based_gains.json", gains)
            rule = {kind: eval_rule_based(kind, k) for kind, k in gains.items()}
            run.save_json("rule_based_results.json", rule)
            jobs = [("PPO_all", "all", [(TRAIN, a.steps)], s, str(run.dir), False) for s in seeds]
        elif exp == "R03":
            jobs = ([("ordinary", "all", [(TRAIN, a.steps)], s, str(run.dir), False) for s in seeds] +
                    [("staged", "all", [(NOMINAL, a.steps // 2), (TRAIN, a.steps - a.steps // 2)], s, str(run.dir), False) for s in seeds] +
                    [("nominal_only", "all", [(NOMINAL, a.steps)], s, str(run.dir), False) for s in seeds])
        elif exp == "R04":
            jobs = [(f"inputs_{o}", o, [(TRAIN, a.steps)], s, str(run.dir), True) for o in OBS_SETS for s in seeds]
        with ProcessPoolExecutor(min(workers, max(1, len(jobs)))) as ex:
            results = list(ex.map(ppo_train_eval, jobs))
        run.save_json("ppo_results.json", results)
        table = []
        for r in results:
            for k in [k for k in r if k.startswith(("test_", "missing_", "noise_"))]:
                table.append({"variant": r["tag"], "seed": r["seed"], "condition": k, **{f"{m}_{s}": r[k][m][s]
                              for m in ("d_err_mean", "d_err_settled_mean", "rate_err_mean", "violation_share") for s in ("mean", "p95", "worst")}})
        df = pd.DataFrame(table)
        run.save_df("ppo_summary_table.csv", df)
        if exp == "R02":
            for kind, res in rule.items():
                for cond, s in res.items():
                    run.log(f"{kind:6s} {cond}: diameter error mean {s['d_err_mean']['mean']:.2f} um (p95 {s['d_err_mean']['p95']:.2f}), "
                            f"rate error {s['rate_err_mean']['mean']:.3f}, out-of-range share {s['violation_share']['mean']:.3f}")
        if len(df):
            agg = df.groupby(["variant", "condition"])[["d_err_mean_mean", "rate_err_mean_mean", "violation_share_mean"]].agg(["mean", "std"])
            run.log("\n" + agg.round(3).to_string())
        run.finish("measured", notes=f"{exp}: practice environment; see config for conditions")


if __name__ == "__main__":
    main()
