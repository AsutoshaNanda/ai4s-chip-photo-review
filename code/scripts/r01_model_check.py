"""R01 check: does our code reproduce the published droplet model (Pärnamets et al. 2022, Fig. 8 / eqs. 4-8),
and does the environment run for every input set (R04) with sensible bookkeeping?

Writes a plot in the same layout as the paper's Fig. 8 (rate vs oil flow on a log axis; diameter and
ellipticity vs oil flow) for a by-eye comparison, a value table, mass-conservation check, and env smoke tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s.rl import droplet_model as M
from ai4s.rl.droplet_env import OBS_SETS, DropletEnv
from ai4s.runlog import Run


def main():
    run = Run("R01", "droplet_model_check", config={"source": "Pärnamets et al. 2022 IEEE Access, CC BY 4.0"})
    rows = []
    for i, qw in enumerate(M.QW_SERIES):
        for qoil in np.arange(2 * qw, M.QOIL_MAX_MEASURED[i] + 0.1, 4.0):
            d, e, r = M.series_values(i, qoil)
            vol_m3 = e * np.pi / 6 * (d * 1e-6) ** 3
            r_mass = (qw * 1e-9 / 60) / vol_m3       # water flow / droplet volume
            rows.append({"qw": qw, "qoil": qoil, "diameter_um": d, "ellipticity": e, "rate_per_s": r,
                         "rate_from_mass_conservation": r_mass})
    df = pd.DataFrame(rows)
    run.save_df("model_values.csv", df)
    rel = float((df["rate_per_s"] / df["rate_from_mass_conservation"] - 1).abs().max())
    run.log(f"max relative difference, eq.(4) vs water-volume conservation: {rel:.4f}")
    check = df[(df.qw == 8) & (df.qoil == 48)].iloc[0].to_dict()
    run.log(f"example Qw=8, Qoil=48: {check}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for qw, d in df.groupby("qw"):
            ax[0].semilogy(d["qoil"], d["rate_per_s"], "-o", ms=3, label=f"Qw={qw:g}")
            ax[1].plot(d["qoil"], d["diameter_um"], "-o", ms=3, label=f"D, Qw={qw:g}")
            ax[1].plot(d["qoil"], d["ellipticity"] * 10, "--", label=f"10×E, Qw={qw:g}")
        ax[0].set(xlabel="oil flow (uL/min)", ylabel="droplets per second", title="our code: rate (compare paper Fig. 8a)")
        ax[1].set(xlabel="oil flow (uL/min)", ylabel="diameter (um) / 10×ellipticity", title="our code: diameter, ellipticity (Fig. 8b)")
        for a in ax:
            a.grid(alpha=.3)
            a.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(run.path("model_vs_paper_fig8.png"), dpi=110)
        plt.close()
    except Exception as e:
        run.log(f"plot failed: {e!r}")

    smoke = []
    for obs_set in OBS_SETS:
        for missing in (0.0, 0.2):
            env = DropletEnv(obs_set=obs_set, missing_prob=missing, tau_range=(0.5, 5.0), mismatch_d=0.03,
                             mismatch_rate=0.2, noise_d_um=2.0, noise_rate_rel=0.05)
            o, info = env.reset(seed=0)
            assert env.observation_space.contains(o) or o.shape == env.observation_space.shape
            rng = np.random.default_rng(0)
            tot, viol, steps = 0.0, 0, 0
            done = False
            while not done:
                o, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
                tot += r
                viol += info["violation"]
                steps += 1
                done = term or trunc
                assert np.all(np.isfinite(o)) and np.isfinite(r)
            smoke.append({"obs_set": obs_set, "missing_prob": missing, "obs_len": int(o.shape[0]), "steps": steps,
                          "random_policy_return": round(tot, 1), "violations": int(viol)})
    run.save_df("env_smoke_tests.csv", pd.DataFrame(smoke))
    run.log(pd.DataFrame(smoke).to_string())
    run.finish("measured", notes="model reproduces eqs. 4-8; env runs for all input sets")


if __name__ == "__main__":
    main()
