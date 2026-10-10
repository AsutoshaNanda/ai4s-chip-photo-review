"""Steady-state droplet model fitted to real chip measurements (R01 environment basis).

Source: K. Pärnamets, A. Udal, A. Koel, T. Pardy, N. Gyimah, T. Rang, "Compact Empirical Model for Droplet
Generation in a Lab-on-Chip Cytometry System", IEEE Access 10:127708-127717 (2022), doi:10.1109/ACCESS.2022.3226623,
licensed CC BY 4.0. Equations (4)-(8) and Table 1 are used exactly as published.

What the source measured: a PDMS flow-focusing chip (84 um x 100 um channel), water droplets in mineral oil with
2% surfactant, syringe pumps, high-speed camera; water flow Qw = 4, 8, 12 uL/min; oil flow from 2*Qw up to
60 (Qw=4), 64 (Qw=8) and 88 (Qw=12) uL/min. Reported accuracy: 5-10% for droplet rate in the central region,
about 20% over the whole area; diameters about 3%; diameter measurement about +/-2 um.

What the source does NOT give: how fast the chip responds to pump changes (dynamics). That is handled in
droplet_env.py as an explicitly swept, reported parameter, not a guessed constant.
"""
import numpy as np

QW_SERIES = np.array([4.0, 8.0, 12.0])        # uL/min (Table 1)
E8 = np.array([2.6, 3.6, 4.15])               # ellipticity at Qoil = 8 uL/min (Table 1)
QE = np.array([9.9, 10.4, 12.4])              # uL/min, exponential decay of ellipticity (Table 1)
QD = np.array([157.0, 196.0, 248.0])          # uL/min, linear decay of diameter (Table 1)
CHANNEL_UM = 84.0                             # eq. (5)
DPS_REF, QW_REF, D_REF = 557.0, 6.0, 70.0     # eq. (4)

# Oil-flow range actually measured per water series (text and Figs. 6, 7, 9)
QOIL_MIN_FACTOR = 2.0                         # Qoil >= 2*Qw
QOIL_MAX_MEASURED = np.array([60.0, 64.0, 88.0])  # Qw = 4, 8, 12; dps above 64 at Qw=12 was extrapolated by the authors


def series_values(i, qoil):
    """Equations (4)-(6) for water series i (0, 1, 2)."""
    qoil = np.asarray(qoil, float)
    d = CHANNEL_UM * (1.0 - qoil / QD[i])
    e = 1.0 + (E8[i] - 1.0) * np.exp(-(qoil - 8.0) / QE[i])
    dps = DPS_REF * (QW_SERIES[i] / QW_REF) * (1.0 / e) * (D_REF / d) ** 3
    return d, e, dps


def steady_state(qw, qoil, c_ml=1.0):
    """Droplet diameter (um), ellipticity, and droplets per second for any Qw in [4, 12] uL/min.
    Linear interpolation between the measured water series, as in equations (7)-(8); c_ml is the
    paper's real-time correction factor (1 = the published model)."""
    qw = float(np.clip(qw, QW_SERIES[0], QW_SERIES[-1]))
    j = 0 if qw <= QW_SERIES[1] else 1
    a = (QW_SERIES[j + 1] - qw) / (QW_SERIES[j + 1] - QW_SERIES[j])
    d0, e0, r0 = series_values(j, qoil)
    d1, e1, r1 = series_values(j + 1, qoil)
    return a * d0 + (1 - a) * d1, a * e0 + (1 - a) * e1, c_ml * (a * r0 + (1 - a) * r1)


def qoil_limits(qw):
    """Measured operating range of oil flow for a given water flow (interpolated between series)."""
    qw = float(np.clip(qw, QW_SERIES[0], QW_SERIES[-1]))
    return QOIL_MIN_FACTOR * qw, float(np.interp(qw, QW_SERIES, QOIL_MAX_MEASURED))


def in_measured_region(qw, qoil):
    lo, hi = qoil_limits(qw)
    return (QW_SERIES[0] <= qw <= QW_SERIES[-1]) and (lo <= qoil <= hi)
