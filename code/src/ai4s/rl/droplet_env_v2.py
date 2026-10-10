"""Droplet-control EVALUATION environment, protocol v2 (7 Oct 2026) for the repaired R02-R04 comparison.

The training environment (droplet_env.py) is left untouched: saved policies were trained in it and its training reward
was already computed against the target in force. Two problems in how it REPORTED and PAIRED evaluation episodes
(found in the RL scoring review, `Evidence/2026-10-07_RL_scoring_review.md`, and re-checked directly in the code):

1. At a target change (steps 100, 200) the reported diameter and rate errors were measured against the NEXT target, because
   the target was replaced before the info dict was filled in. Here the reported error always uses the target that was in
   force when the action was applied (`target_used`).
2. One random-number stream served both measurement noise and later targets, and the Poisson draw consumes a different number
   of random numbers at different flow rates, so the same episode seed gave different target sequences to different controllers.
   Here every episode seed is split into three independent streams: (a) plant parameters and starting pumps, (b) the whole
   target schedule (generated at reset, depends only on the plant parameters), (c) measurement noise, drawn through the inverse
   cumulative distribution with exactly two uniform numbers per measurement, so every controller gets identical targets,
   identical plant and identical noise random numbers.

Everything else (plant, pump lag, observations, reward, limits) is copied from droplet_env.py without change.
"""
from collections import deque

import numpy as np
from scipy import stats

from . import droplet_model as M
from .droplet_env import DropletEnv


class DropletEnvV2(DropletEnv):
    def reset(self, seed=None, options=None):
        c = self.cfg
        plant_ss, target_ss, noise_ss = np.random.SeedSequence(seed).spawn(3)
        r_plant, r_target = np.random.default_rng(plant_ss), np.random.default_rng(target_ss)
        self.rng_noise = np.random.default_rng(noise_ss)
        self.rng = r_plant
        self.tau = self.rng.uniform(*c["tau_range"])
        self.k_d = 1 + self.rng.uniform(-c["mismatch_d"], c["mismatch_d"])
        self.k_rate = 1 + self.rng.uniform(-c["mismatch_rate"], c["mismatch_rate"])
        self.q_set = np.array(self._sample_operating_point())
        self.q_actual = self.q_set.copy()
        self.t = 0
        self.rng = r_target                                    # target stream only: fixed schedule for the whole episode
        blocks = -(-c["episode_steps"] // c["target_hold_steps"])
        self.schedule = []
        for _ in range(blocks):
            qw, qoil = self._sample_operating_point()
            d, _, rate = self._plant(qw, qoil)                 # reachable under this episode's true plant, as in the original
            self.schedule.append((d, rate))
        self.target = self.schedule[0]
        self.rng = r_plant
        k = c["history_len"]
        self.hist_actions = deque([np.zeros(2)] * k, maxlen=k)
        self.hist_errors = deque([np.zeros(2)] * k, maxlen=k)
        _, meas = self._measure()
        self.hist_errors.append(self._errors(meas[0], meas[2]))
        return self._obs(meas), {"tau": self.tau, "k_d": self.k_d, "k_rate": self.k_rate}

    def _measure(self):
        c = self.cfg
        d, e, rate = self._plant(*self.q_actual)
        u_d, u_r = np.clip(self.rng_noise.random(2), 1e-12, 1 - 1e-12)   # always exactly two draws
        sd = c["noise_d_um"] * c["extra_noise"]
        dm = d + stats.norm.ppf(u_d) * sd if sd > 0 else d
        if c["rate_noise"] == "poisson":
            counted = stats.poisson.ppf(u_r, rate * c["dt"]) / c["dt"]
            rm = max(1e-3, rate + (counted - rate) * c["extra_noise"])
        else:
            sr = c["noise_rate_rel"] * c["extra_noise"]
            rm = rate * max(1e-3, 1 + stats.norm.ppf(u_r) * sr) if sr > 0 else rate
        return (d, e, rate), (dm, e, rm)

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
        target_used = self.target                              # the target in force when this action was applied
        err_true = self._errors(d, rate)
        reward = -float(np.abs(err_true).sum()) - c["w_violation"] * violation - c["w_action"] * float((a ** 2).sum())
        self.t += 1
        if self.t % c["target_hold_steps"] == 0 and self.t < c["episode_steps"]:
            self.target = self.schedule[self.t // c["target_hold_steps"]]
        self.hist_actions.append(a)
        self.hist_errors.append(self._errors(meas[0], meas[2]))
        info = {"d_true": d, "rate_true": rate, "d_err_um": abs(target_used[0] - d),
                "rate_err_rel": abs(rate / target_used[1] - 1), "violation": violation,
                "qw": self.q_actual[0], "qoil": self.q_actual[1], "target_used": target_used}
        return self._obs(meas), reward, False, self.t >= c["episode_steps"], info
