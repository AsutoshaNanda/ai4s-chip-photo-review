"""Rule-based reference controllers for R02 (fair, tuned on the same training conditions as PPO).

PI:      two feedback loops. Oil flow corrects the diameter error (more oil -> smaller droplets); water flow
         corrects the rate error (more water -> more droplets). Classic, needs no model.
FF+PI:   feedforward from the published model run backwards (find the pump set-points that give the targets
         under the nominal model, by a fine grid over the measured range), plus the PI loops for corrections.
Both act through the same action interface as PPO (set-point steps in [-1, 1], same step limits).
"""
import numpy as np

from . import droplet_model as M


class PI:
    def __init__(self, kp_d, ki_d, kp_r, ki_r):
        self.k = (kp_d, ki_d, kp_r, ki_r)
        self.reset()

    def reset(self):
        self.i_d = 0.0
        self.i_r = 0.0

    def act(self, err_d, err_r, env=None):
        kp_d, ki_d, kp_r, ki_r = self.k
        self.i_d = float(np.clip(self.i_d + err_d, -50, 50))   # anti-windup clamp on the integral state
        self.i_r = float(np.clip(self.i_r + err_r, -50, 50))
        a_oil = -(kp_d * err_d + ki_d * self.i_d)              # diameter too small (err>0) -> less oil
        a_water = kp_r * err_r + ki_r * self.i_r               # rate too low (err>0) -> more water
        return np.clip(np.array([a_water, a_oil]), -1, 1)


class InverseModel:
    """Grid over the measured operating range; returns the set-points whose nominal outputs best match targets."""

    def __init__(self, n_qw=81, n_qoil=161):
        pts = []
        for qw in np.linspace(M.QW_SERIES[0], M.QW_SERIES[-1], n_qw):
            lo, hi = M.qoil_limits(qw)
            for qoil in np.linspace(lo, hi, n_qoil):
                d, _, r = M.steady_state(qw, qoil)
                pts.append((qw, qoil, d, np.log(r)))
        self.pts = np.array(pts)

    def setpoints(self, d_target, rate_target, d_scale, rate_scale):
        e = ((self.pts[:, 2] - d_target) / d_scale) ** 2 + ((self.pts[:, 3] - np.log(rate_target)) / rate_scale) ** 2
        j = int(np.argmin(e))
        return self.pts[j, 0], self.pts[j, 1]


class FFPI:
    def __init__(self, inverse, kp_d, ki_d, kp_r, ki_r):
        self.inv = inverse
        self.pi = PI(kp_d, ki_d, kp_r, ki_r)
        self.k = self.pi.k

    def reset(self):
        self.pi.reset()
        self.offset = np.zeros(2)

    def act(self, err_d, err_r, env):
        c = env.cfg
        qw_t, qoil_t = self.inv.setpoints(env.target[0], env.target[1], c["d_scale_um"], c["rate_scale"])
        corr = self.pi.act(err_d, err_r)                          # PI correction in step units
        self.offset = self.offset + corr * np.array([c["step_qw"], c["step_qoil"]]) * 0.5
        want = np.array([qw_t, qoil_t]) + self.offset
        step = (want - env.q_set) / np.array([c["step_qw"], c["step_qoil"]])
        return np.clip(step, -1, 1)
