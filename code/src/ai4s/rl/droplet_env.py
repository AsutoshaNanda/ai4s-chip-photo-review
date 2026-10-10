"""Droplet-control environment for R02-R04 (gymnasium API).

Task: two pumps (water Qw, oil Qoil, in uL/min) feed a flow-focusing chip. Keep the droplet diameter D and
the droplets-per-second rate near targets that change during an episode, without leaving the operating range
that was actually measured in the source paper.

Built from the published model in droplet_model.py (Pärnamets et al. 2022, CC BY 4.0). It is a PRACTICE
ENVIRONMENT BASED ON A MODEL FITTED TO REAL MEASUREMENTS. It is not proof about real chips.

Where every number comes from:
- steady-state outputs, operating range, ~3% diameter / 5-20% rate model accuracy, +/-2 um diameter
  measurement accuracy: from the paper.
- pump response speed: NOT in the paper. tau_range is swept and reported; nothing claims a true value.
- control interval, episode length, target changes, action step sizes, reward weights: benchmark design
  choices, recorded in every run's config and varied in sensitivity checks.

Observation sets (R04, "minimal inputs vs multiple global features"), all built from information available
at the current step only (no future targets):
  minimal  : errors to the targets (diameter, log-rate)
  pumps    : minimal + current pump set-points
  camera   : minimal + camera ("global image") measurements: diameter, ellipticity, log-rate, plus targets
  history  : minimal + the last `history_len` actions and errors
  all      : everything above
Stress tests for R04: missing_prob (each input replaced by 0 = "missing" with this chance, plus a missing
flag per input) and extra_noise (multiplies measurement noise).
Variation for R03 (staged training): per-episode pump lag, model mismatch, measurement noise.
"""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import droplet_model as M

DEFAULTS = dict(
    dt=1.0,                    # s per control step (design choice)
    episode_steps=300,         # design choice
    target_hold_steps=100,     # targets change every this many steps (design choice)
    tau_range=(0.5, 0.5),      # s, first-order pump lag; swept (not in the paper)
    mismatch_d=0.0,            # +/- fraction on diameter (paper: ~3% model accuracy)
    mismatch_rate=0.0,         # +/- fraction on rate (paper: 5-10% central, ~20% overall)
    noise_d_um=0.0,            # measurement noise sd on diameter (paper: +/-2 um accuracy)
    noise_rate_rel=0.0,        # relative measurement noise sd on rate (used when rate_noise="relative")
    rate_noise="relative",     # "poisson": counting noise of droplets in one control step (physics, no guess)
    obs_set="minimal",
    history_len=4,
    missing_prob=0.0,
    extra_noise=1.0,
    step_qw=1.0,               # max set-point change per step, uL/min
    step_qoil=4.0,             # max set-point change per step, uL/min (paper's oil steps were 4 uL/min)
    d_scale_um=2.0,            # error scale = paper's diameter measurement accuracy
    rate_scale=0.10,           # error scale on log-rate ~ 10% (paper's central-region accuracy)
    w_violation=10.0,          # reward weight for leaving the measured range (design choice; sensitivity-checked)
    w_action=0.01,             # small smoothness weight (design choice)
)

OBS_SETS = ("minimal", "pumps", "camera", "history", "all")


class DropletEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        assert self.cfg["obs_set"] in OBS_SETS
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n = self._raw_obs_len()
        if self.cfg["missing_prob"] > 0:
            n *= 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.rng = np.random.default_rng()

    # ---------- helpers ----------
    def _raw_obs_len(self):
        k = self.cfg["history_len"]
        return {"minimal": 2, "pumps": 4, "camera": 2 + 5, "history": 2 + 4 * k,
                "all": 2 + 2 + 5 + 4 * k}[self.cfg["obs_set"]]

    def _plant(self, qw, qoil):
        """True outputs at the actual flows; flows are kept inside the measured range for model evaluation."""
        lo, hi = M.qoil_limits(qw)
        d, e, rate = M.steady_state(qw, float(np.clip(qoil, lo, hi)))
        return d * self.k_d, e, rate * self.k_rate

    def _sample_operating_point(self):
        qw = self.rng.uniform(M.QW_SERIES[0], M.QW_SERIES[-1])
        lo, hi = M.qoil_limits(qw)
        return qw, self.rng.uniform(lo, hi)

    def _new_targets(self):
        qw, qoil = self._sample_operating_point()
        d, _, rate = self._plant(qw, qoil)       # reachable under this episode's true plant
        self.target = (d, rate)

    def _measure(self):
        c = self.cfg
        d, e, rate = self._plant(*self.q_actual)
        sd = c["noise_d_um"] * c["extra_noise"]
        sr = c["noise_rate_rel"] * c["extra_noise"]
        dm = d + self.rng.normal(0, sd) if sd > 0 else d
        if c["rate_noise"] == "poisson":
            counted = self.rng.poisson(rate * c["dt"]) / c["dt"]
            rm = max(1e-3, rate + (counted - rate) * c["extra_noise"])
        else:
            rm = rate * max(1e-3, 1 + self.rng.normal(0, sr)) if sr > 0 else rate
        em = e
        return (d, e, rate), (dm, em, rm)

    def _errors(self, d, rate):
        td, tr = self.target
        return np.array([(td - d) / self.cfg["d_scale_um"], (np.log(tr) - np.log(rate)) / self.cfg["rate_scale"]])

    def _obs(self, meas):
        c = self.cfg
        dm, em, rm = meas
        err = self._errors(dm, rm)
        parts = [err]
        s = c["obs_set"]
        if s in ("pumps", "all"):
            parts.append(np.array([(self.q_set[0] - 8.0) / 4.0, (self.q_set[1] - 48.0) / 40.0]))
        if s in ("camera", "all"):
            parts.append(np.array([(dm - 60.0) / 15.0, em - 1.0, (np.log(rm) - 6.5),
                                   (self.target[0] - 60.0) / 15.0, (np.log(self.target[1]) - 6.5)]))
        if s in ("history", "all"):
            parts.append(np.concatenate([np.ravel(a) for a in self.hist_actions] + [np.ravel(x) for x in self.hist_errors]))
        o = np.concatenate(parts).astype(np.float32)
        if c["missing_prob"] > 0:
            miss = self.rng.random(o.shape) < c["missing_prob"]
            o = np.concatenate([np.where(miss, 0.0, o), miss.astype(np.float32)]).astype(np.float32)
        return o

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.tau = self.rng.uniform(*c["tau_range"])
        self.k_d = 1 + self.rng.uniform(-c["mismatch_d"], c["mismatch_d"])
        self.k_rate = 1 + self.rng.uniform(-c["mismatch_rate"], c["mismatch_rate"])
        self.q_set = np.array(self._sample_operating_point())
        self.q_actual = self.q_set.copy()
        self.t = 0
        self._new_targets()
        k = c["history_len"]
        self.hist_actions = deque([np.zeros(2)] * k, maxlen=k)
        self.hist_errors = deque([np.zeros(2)] * k, maxlen=k)
        _, meas = self._measure()
        self.hist_errors.append(self._errors(meas[0], meas[2]))
        return self._obs(meas), {"tau": self.tau, "k_d": self.k_d, "k_rate": self.k_rate}

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, float), -1, 1)
        self.q_set = self.q_set + a * np.array([c["step_qw"], c["step_qoil"]])
        self.q_set[0] = np.clip(self.q_set[0], M.QW_SERIES[0], M.QW_SERIES[-1])
        self.q_set[1] = np.clip(self.q_set[1], M.QOIL_MIN_FACTOR * M.QW_SERIES[0], M.QOIL_MAX_MEASURED[-1])
        alpha = 1 - np.exp(-c["dt"] / max(self.tau, 1e-6))
        self.q_actual = self.q_actual + alpha * (self.q_set - self.q_actual)
        violation = not M.in_measured_region(*self.q_actual)
        (d, e, rate), meas = self._measure()
        err_true = self._errors(d, rate)
        reward = -float(np.abs(err_true).sum()) - c["w_violation"] * violation - c["w_action"] * float((a ** 2).sum())
        self.t += 1
        if self.t % c["target_hold_steps"] == 0:
            self._new_targets()
        self.hist_actions.append(a)
        self.hist_errors.append(self._errors(meas[0], meas[2]))
        info = {"d_true": d, "rate_true": rate, "d_err_um": abs(self.target[0] - d),
                "rate_err_rel": abs(rate / self.target[1] - 1), "violation": violation,
                "qw": self.q_actual[0], "qoil": self.q_actual[1]}
        return self._obs(meas), reward, False, self.t >= c["episode_steps"], info
