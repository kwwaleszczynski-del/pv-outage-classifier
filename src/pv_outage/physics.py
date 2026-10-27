"""Electrical model of the simulated plant: single-diode substrings, bypass diodes, MPPT inputs, inverters.

A module is three substrings (one bypass diode each) described by the CEC single-diode model. A string is
``n_sub`` substrings in series; a substring can be fully lit, shaded (beam blocked, it then sees only the diffuse
light), or bypassed by a shorted diode (about 0 V). Strings on one MPPT input share its voltage; the input sits at
the global maximum of the summed power curve, or somewhere else when it has failed or when the inverter has to
shed DC power (clipping at its AC rating, an export limit, a curtailment set-point). The inverter converts DC to
AC with the PVWatts efficiency curve.

Two paths keep this fast:
- uniform strings (the vast majority of steps): maximum power point from a lookup table over irradiance and cell
  temperature, built once with pvlib's single-diode solver;
- everything else: string I-V curves on a current grid (pvlib ``v_from_i``), inverted to I(V) on a common voltage
  grid with one flattened ``np.interp`` call, then searched for the operating point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODULE = "Jinko_Solar__Co___Ltd_JKM400M_72L"  # 400 W, 72 cells, 3 bypass diodes
BYPASS_DROP = -0.5  # V across a conducting bypass diode


@dataclass
class Module:
    alpha_sc: float
    a_ref: float
    I_L_ref: float
    I_o_ref: float
    R_sh_ref: float
    R_s: float
    Adjust: float
    p_stc: float
    gamma_pdc: float  # 1/K

    @classmethod
    def from_cec(cls, name: str = MODULE) -> "Module":
        import pvlib

        m = pvlib.pvsystem.retrieve_sam("CECMod")[name]
        return cls(float(m.alpha_sc), float(m.a_ref), float(m.I_L_ref), float(m.I_o_ref), float(m.R_sh_ref),
                   float(m.R_s), float(m.Adjust), float(m.STC), float(m.gamma_r) / 100.0)

    def substring_params(self, g, t_cell):
        """Single-diode parameters of one substring (a third of the module's cells)."""
        import pvlib

        il, i0, rs, rsh, nvth = pvlib.pvsystem.calcparams_cec(
            np.asarray(g, dtype=float), np.asarray(t_cell, dtype=float), self.alpha_sc, self.a_ref,
            self.I_L_ref, self.I_o_ref, self.R_sh_ref, self.R_s, self.Adjust)
        return (np.asarray(il, dtype=float), np.asarray(i0, dtype=float), np.asarray(rs, dtype=float) / 3,
                np.asarray(rsh, dtype=float) / 3, np.asarray(nvth, dtype=float) / 3)


class MppTable:
    """Maximum power point of one uniformly lit substring on a grid of irradiance and cell temperature."""

    def __init__(self, module: Module, g_max: float = 1500.0, dg: float = 5.0, t_min: float = -30.0,
                 t_max: float = 85.0, dt: float = 1.0):
        import pvlib

        self.g = np.arange(0.0, g_max + dg, dg)
        self.t = np.arange(t_min, t_max + dt, dt)
        gg, tt = np.meshgrid(np.maximum(self.g, 1e-3), self.t, indexing="ij")
        il, i0, rs, rsh, nvth = module.substring_params(gg.ravel(), tt.ravel())
        out = pvlib.pvsystem.singlediode(il, i0, rs, rsh, nvth, method="newton")
        shape = gg.shape
        self.i_mp = np.array(out["i_mp"], dtype=float).reshape(shape)
        self.v_mp = np.array(out["v_mp"], dtype=float).reshape(shape)
        self.v_oc = np.array(out["v_oc"], dtype=float).reshape(shape)
        self.i_mp[0], self.v_mp[0], self.v_oc[0] = 0.0, 0.0, 0.0
        self.dg, self.dt, self.t_min = dg, dt, t_min

    def lookup(self, g, t_cell):
        """(i_mp, v_mp, v_oc) of a substring, bilinear in irradiance and temperature."""
        g = np.clip(np.asarray(g, dtype=float), 0, self.g[-1] - 1e-9)
        t = np.clip(np.asarray(t_cell, dtype=float), self.t_min, self.t[-1] - 1e-9)
        x, y = g / self.dg, (t - self.t_min) / self.dt
        i, j = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - i, y - j
        res = []
        for tab in (self.i_mp, self.v_mp, self.v_oc):
            res.append(tab[i, j] * (1 - fx) * (1 - fy) + tab[i + 1, j] * fx * (1 - fy)
                       + tab[i, j + 1] * (1 - fx) * fy + tab[i + 1, j + 1] * fx * fy)
        return res


def _flat_interp(x_rows: np.ndarray, xp_rows: np.ndarray, fp_rows: np.ndarray) -> np.ndarray:
    """Row-wise np.interp for 2-D arrays (xp increasing within each row) via one call on offset rows."""
    n = xp_rows.shape[0]
    span = np.nanmax(np.abs(xp_rows)) * 4 + np.nanmax(np.abs(x_rows)) * 4 + 1.0
    off = (np.arange(n) * span)[:, None]
    out = np.interp((x_rows + off).ravel(), (xp_rows + off).ravel(), fp_rows.ravel())
    return out.reshape(x_rows.shape)


def string_iv(module: Module, g_full, g_shade, t_cell, n_sub: int, n_shaded, n_bypassed, n_points: int = 160):
    """I-V curves of strings, shape (N, n_points): current grid and string voltage.

    ``n_shaded`` substrings see ``g_shade``, ``n_bypassed`` are shorted (0 V), the rest see ``g_full``. Every
    substring has a bypass diode, so its voltage never drops below ``BYPASS_DROP``.
    """
    g_full = np.asarray(g_full, dtype=float)
    n = g_full.shape[0]
    il_f, i0_f, rs_f, rsh_f, nv_f = module.substring_params(np.maximum(g_full, 1e-3), t_cell)
    il_s, i0_s, rs_s, rsh_s, nv_s = module.substring_params(np.maximum(np.asarray(g_shade, float), 1e-3), t_cell)
    # current grid: uniform plus clustering below each knee (full and shaded short-circuit currents)
    u = np.linspace(0.0, 1.0, n_points - 60)
    knee = 1.0 - np.geomspace(0.25, 1e-4, 30)
    ratio = np.clip(il_s / np.maximum(il_f, 1e-9), 0.0, 1.0)
    grid = np.concatenate([np.broadcast_to(u, (n, u.size)), np.broadcast_to(knee, (n, 30)),
                           np.clip(ratio[:, None] * knee[None, :], 0, 1)], axis=1)
    grid.sort(axis=1)
    cur = grid * il_f[:, None]
    vf = np.asarray(_v_from_i(cur, il_f, i0_f, rs_f, rsh_f, nv_f)).clip(min=BYPASS_DROP)
    vs = np.asarray(_v_from_i(cur, il_s, i0_s, rs_s, rsh_s, nv_s)).clip(min=BYPASS_DROP)
    n_shaded = np.asarray(n_shaded, dtype=float)[:, None]
    n_bypassed = np.asarray(n_bypassed, dtype=float)[:, None]
    volt = (n_sub - n_shaded - n_bypassed) * vf + n_shaded * vs
    return cur, volt


def _v_from_i(cur, il, i0, rs, rsh, nv):
    import pvlib

    return pvlib.pvsystem.v_from_i(cur, il[:, None], i0[:, None], rs[:, None], rsh[:, None], nv[:, None])


def current_at(volt_grid: np.ndarray, cur: np.ndarray, volt: np.ndarray) -> np.ndarray:
    """I(V) of each string on ``volt_grid`` (N, G) from its I-V curve (current increasing, voltage decreasing)."""
    xp = volt[:, ::-1]
    fp = cur[:, ::-1]
    xp = np.maximum.accumulate(xp, axis=1)  # guard against tiny non-monotonic noise
    i = _flat_interp(volt_grid, xp, fp)
    return np.where(volt_grid > xp[:, -1:], 0.0, i)


def operating_points(curves: list[tuple[np.ndarray, np.ndarray]], v_factor: np.ndarray, p_fraction: np.ndarray,
                     n_grid: int = 256):
    """Operating point of N MPPT inputs, each with the strings given in ``curves`` (list over strings).

    v_factor: NaN for a working tracker, else the input is stuck at ``v_factor`` x its true MPP voltage.
    p_fraction: share of the input's maximum power the inverter lets it deliver (1 = MPP); a fraction below 1
    moves the operating point to the right of the MPP (higher voltage, lower current), as inverters do when they
    clip or follow a set-point.
    Returns (v_op, i_op per string (list), p_mpp).
    """
    v_max = np.max([v[:, 0] for _, v in curves], axis=0)  # open-circuit voltage at zero current
    frac = np.linspace(0.0, 1.0, n_grid)
    vg = v_max[:, None] * frac[None, :]
    i_each = [current_at(vg, c, v) for c, v in curves]
    p = vg * np.sum(i_each, axis=0)
    k = np.argmax(p, axis=1)
    rows = np.arange(len(k))
    # parabolic refinement of the maximum on the voltage grid
    km, kp = np.clip(k - 1, 0, n_grid - 1), np.clip(k + 1, 0, n_grid - 1)
    y0, y1, y2 = p[rows, km], p[rows, k], p[rows, kp]
    denom = y0 - 2 * y1 + y2
    shift = np.where((km != k) & (kp != k) & (denom < 0), 0.5 * (y0 - y2) / np.where(denom < 0, denom, -1), 0.0)
    dv = v_max * (frac[1] - frac[0])
    v_mpp = vg[rows, k] + shift * dv
    p_mpp = np.maximum(y1 - 0.25 * (y0 - y2) * shift, 0.0)

    v_op = v_mpp.copy()
    stuck = np.isfinite(v_factor)
    v_op[stuck] = v_factor[stuck] * v_mpp[stuck]
    shed = (p_fraction < 1.0 - 1e-6) & ~stuck
    if shed.any():
        target = p_fraction[shed] * p_mpp[shed]
        ps = p[shed]
        right = np.arange(n_grid)[None, :] > k[shed][:, None]
        below = right & (ps <= target[:, None])
        j = np.where(below.any(axis=1), np.argmax(below, axis=1), n_grid - 1)
        r = np.arange(len(j))
        p_hi, p_lo = ps[r, j - 1], ps[r, j]
        w = np.clip((p_hi - target) / np.maximum(p_hi - p_lo, 1e-12), 0, 1)
        v_op[shed] = vg[shed][r, j - 1] + w * (vg[shed][r, j] - vg[shed][r, j - 1])
    i_op = [current_at(v_op[:, None], c, v)[:, 0] for c, v in curves]
    return v_op, i_op, p_mpp


def pvwatts_ac(p_dc, pac0: float, eta_nom: float = 0.98):
    """PVWatts inverter: AC power for DC power (same units as pac0), capped at pac0."""
    import pvlib

    return np.asarray(pvlib.inverter.pvwatts(np.asarray(p_dc, dtype=float), pac0 / eta_nom, eta_nom),
                      dtype=float).clip(0, pac0)


def dc_for_ac(p_ac, pac0: float, eta_nom: float = 0.98) -> np.ndarray:
    """Inverse of ``pvwatts_ac`` below the rating (DC input needed for a given AC output)."""
    pdc = np.linspace(0, pac0 / eta_nom * 1.05, 4001)
    pac = pvwatts_ac(pdc, pac0, eta_nom)
    pac = np.maximum.accumulate(pac)
    return np.interp(np.asarray(p_ac, dtype=float), pac, pdc)
