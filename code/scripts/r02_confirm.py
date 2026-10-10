"""R02 confirmation (improvements.md section 6): frozen protocol on fresh conditions.

Nothing is retrained or retuned: the saved R02 PPO policies (5 training seeds, frozen input scaling) and the saved
rule-based gains (PI, FF+PI) are evaluated exactly as in R02, on FRESH episode seeds never used before
(same number of test episodes as R02: 50 per condition family; TRAIN-range and UNTOUCHED conditions).
Reported: full distributions (mean, median, 95th percentile, worst episode) of diameter error, rate error and
out-of-range share, plus paired per-episode differences (same fresh episode) learned minus each rule-based controller.
Practice simulator built from published fitted equations (R01); not proof on real chips.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import r_control as R
from ai4s.rl.droplet_env import DropletEnv
from ai4s.runlog import Run, latest_run

FRESH_SEEDS = list(range(50_000, 50_050))       # never used before (R02 test 10_000.., val 20_000.., tune 30_000..)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    src = latest_run("R02_pi_vs_ppo")
    gains = json.load(open(src / "rule_based_gains.json"))
    run = Run("R02", "confirm_fresh_conditions", config={"source_run": src.name, "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
                                                         "episodes_per_condition": len(FRESH_SEEDS), "gains": gains,
                                                         "note": "frozen R02 protocol; practice simulator, not real chips"})
    conds = {"TRAIN_range": R.TRAIN, "UNTOUCHED": R.UNTOUCHED}
    rows = []
    for kind, k in gains.items():
        inv = R.InverseModel() if kind == "FF+PI" else None
        for cname, cond in conds.items():
            env = DropletEnv(obs_set="minimal", **cond)
            pol = R.make_pi(kind, tuple(k), inv)
            for s in FRESH_SEEDS:
                rows.append({"controller": kind, "seed_train": -1, "condition": cname, "episode_seed": s, **R.run_episode(env, pol, s)})
    import shutil
    import tempfile
    found = sorted({p.name.removesuffix(".zip") for p in src.glob("PPO_all_seed*") if not p.name.endswith(".pkl")})
    assert found, f"no saved PPO policies in {src}"
    run.log(f"saved PPO policies: {found}")
    for stem in found:
        ts = int(stem.split("seed")[1])
        zf = src / f"{stem}.zip"
        if not zf.is_file():                         # Kaggle unpacks .zip files inside datasets: pack the folder again
            tmp = Path(tempfile.mkdtemp())
            zf = Path(shutil.make_archive(str(tmp / stem), "zip", root_dir=src / stem))
        model = PPO.load(zf, device="cpu")
        for cname, cond in conds.items():
            venv = VecNormalize.load(str(src / f"PPO_all_seed{ts}_vecnormalize.pkl"), DummyVecEnv([lambda cond=cond: DropletEnv(obs_set="all", **cond)]))
            venv.training, venv.norm_reward = False, False
            pol = R._ppo_policy(model, venv)
            env = DropletEnv(obs_set="all", **cond)
            for s in FRESH_SEEDS:
                rows.append({"controller": "PPO_all", "seed_train": ts, "condition": cname, "episode_seed": s, **R.run_episode(env, pol, s)})
        run.log(f"PPO seed {ts} evaluated")
    D = pd.DataFrame(rows)
    run.save_df("episodes.csv", D)
    res = {}
    for (ctrl, ts, cname), d in D.groupby(["controller", "seed_train", "condition"]):
        res[f"{ctrl}|seed{ts}|{cname}"] = R.summarize(d.to_dict("records"))
    for cname in conds:
        ppo = D[(D.controller == "PPO_all") & (D.condition == cname)].groupby("episode_seed")[["d_err_mean", "rate_err_mean", "violation_share"]].mean()
        for kind in gains:
            rb = D[(D.controller == kind) & (D.condition == cname)].set_index("episode_seed")[["d_err_mean", "rate_err_mean", "violation_share"]]
            diff = (ppo - rb.loc[ppo.index])
            res[f"paired|{cname}|PPO_mean_over_seeds_minus_{kind}"] = {c: {"mean": float(diff[c].mean()), "median": float(diff[c].median()),
                                                                          "share_episodes_ppo_better": float((diff[c] < 0).mean()),
                                                                          "worst_for_ppo": float(diff[c].max())} for c in diff.columns}
            run.log(f"{cname}: PPO (mean of 5 seeds) minus {kind}, diameter error: mean {diff['d_err_mean'].mean():+.3f} um, "
                    f"PPO better in {100 * (diff['d_err_mean'] < 0).mean():.0f}% of fresh episodes; worst {diff['d_err_mean'].max():+.3f}")
    agg = D.groupby(["controller", "condition"])[["d_err_mean", "rate_err_mean", "violation_share"]].agg(["mean", "max"]).round(3)
    run.log("fresh-condition results (mean and worst episode):\n" + agg.to_string())
    run.save_json("results.json", res)
    run.finish("measured", notes="R02 frozen protocol on fresh episode seeds; practice simulator")


if __name__ == "__main__":
    main()
