"""Per device and time step: the residuals and flags every detector and classifier is built on.

Primary signal per device: AC power for inverters and the plant meter, DC power (current x MPPT input voltage)
for strings, or the string current when no voltage is reported. Power, not current: a tracker stuck below the
MPP voltage raises the current while the power falls. Everything below works on the long format alone; plant
metadata (location, orientation, DC/AC ratings) is optional and only adds the clear-sky checks and rated values.

- expected model (PVWatts): signal = k_d x POA/1000 x (1 + gamma (T_mod - 25)), capped at the AC rating, where
  k_d is a per-device calibration: the median ratio per 50 W/m2 irradiance bin (absorbs low-light and
  incidence-angle behaviour), one value ("scalar"), or one per month x hour of day ("hour_month", for plants
  whose devices face other ways than the irradiance sensor; the peer baseline then gets the same bins). Outage
  steps are left out of the median;
- peer reference: the leave-one-out median of the calibrated siblings (strings of one inverter, inverters of
  one plant; a pair of twins references each other). Needs no irradiance sensor and cancels clouds, which hit all
  siblings alike;
- voltage against the siblings and against the device's own typical voltage at that irradiance and module
  temperature (an inverter that sheds power, or sits idle at open circuit, runs above its MPP voltage);
- data quality: missing values, frozen values, implausible irradiance sensor (ratio to clear sky at steps where
  production says the sun is out: its day level, and its shape within the day);
- sun-position recurrence: how often a device underperformed on other days with the sun in the same patch of
  sky (fixed obstructions);
- context: plant-wide share of low inverters, cold modules (snow), status codes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pv_outage.schema import Panel

LIGHT_POA = 50.0  # W/m2: a step counts for detection and availability above this irradiance
CAL_QUANTILE = 0.5  # calibration and peer baselines use this quantile of the device's ratios
F32 = np.float32


@dataclass
class Features:
    panel: Panel
    light: np.ndarray  # (T, D) bool
    y: np.ndarray  # primary signal (kW or A)
    y_kw: np.ndarray  # AC power equivalent, kW
    is_amp: np.ndarray  # (D,) True where the signal is a current (strings without voltage)
    k: np.ndarray  # (T, D) calibration of the expected model
    y_exp: np.ndarray  # expected signal
    r_exp: np.ndarray
    ref_peer: np.ndarray  # peer reference of the signal
    r_peer: np.ndarray
    r_v: np.ndarray  # voltage vs siblings
    r_vself: np.ndarray  # voltage vs own typical at this irradiance / temperature
    zero: np.ndarray
    missing: np.ndarray
    frozen: np.ndarray
    near_pac: np.ndarray
    parent_limited: np.ndarray
    status_bad: np.ndarray
    recur: np.ndarray
    frac_low: np.ndarray  # share of siblings with r_exp < -0.1
    sensor_dev: np.ndarray  # (T,) day level of the irradiance sensor vs clear sky, clear steps only (NaN otherwise)
    sensor_shape: np.ndarray  # (T,) the sensor vs clear sky relative to its own day level (shadow on the sensor)
    cold: np.ndarray  # (T,) module temperature below 2 degC within the last 24 h
    sun_el: np.ndarray
    sun_az: np.ndarray
    kc: np.ndarray  # (T,) POA / clear-sky POA (NaN without metadata)
    pac: np.ndarray  # (D,) AC rating or high percentile
    mppt_group: np.ndarray  # (D,) group id of strings sharing an MPPT input, -1 otherwise
    v_ref: np.ndarray  # (T, D) sibling median voltage (strings) for power conversion
    poa: np.ndarray
    tmod: np.ndarray
    info: dict = field(default_factory=dict)


def _loo_median(vals: np.ndarray, valid: np.ndarray, min_n: int = 2) -> np.ndarray:
    """Leave-one-out median across columns (siblings), NaN where fewer than ``min_n`` others are valid."""
    v = np.where(valid, vals, np.nan)
    out = np.full(v.shape, np.nan, dtype=float)
    n = v.shape[1]
    for j in range(n):
        others = np.delete(v, j, axis=1)
        cnt = np.isfinite(others).sum(axis=1)
        with np.errstate(all="ignore"):
            med = np.nanmedian(others, axis=1) if others.shape[1] else np.full(v.shape[0], np.nan)
        out[:, j] = np.where(cnt >= min_n, med, np.nan)
    return out


def _bin_calibration(ratio: np.ndarray, ok: np.ndarray, time: pd.DatetimeIndex, mode: str,
                     poa: np.ndarray | None = None, q: float = CAL_QUANTILE) -> np.ndarray:
    """Per-device calibration factor (T, D): one value, one per irradiance bin or one per month x hour.

    The value is the median ratio over steps that are not outages (ratio above 30 % of the device's 80th
    percentile), so long outages never become a device's reference. A partial fault that lasts more than about
    half of the analysed period does: analyse a longer period than the faults you look for.
    """
    T, D = ratio.shape
    r = np.where(ok, ratio, np.nan)
    with np.errstate(all="ignore"):
        hi = np.nanquantile(r, 0.8, axis=0) if np.isfinite(r).any() else np.full(D, np.nan)
        r = np.where(r > 0.3 * hi[None, :], r, np.nan)
        glob = np.nanquantile(r, q, axis=0) if np.isfinite(r).any() else np.full(D, np.nan)
    if mode == "scalar":
        return np.broadcast_to(glob, (T, D)).astype(float)
    if mode == "poa":  # irradiance bins of 50 W/m2 (low-light and incidence-angle behaviour)
        key = np.clip(np.nan_to_num(poa) // 50, 0, 30).astype(int)
    else:
        slot = np.asarray(time.hour) * 4 + np.asarray(time.minute) // 15  # quarter-hour of the day
        key = np.broadcast_to(((np.asarray(time.month) - 1) * 96 + slot)[:, None], (T, D))
    out = np.full((T, D), np.nan)
    for j in range(D):
        s = pd.Series(r[:, j])
        val = s.groupby(key[:, j]).quantile(q)
        cnt = s.groupby(key[:, j]).count()
        val = val.where(cnt >= 20)
        out[:, j] = val.reindex(key[:, j]).to_numpy()
    return np.where(np.isfinite(out), out, glob[None, :])


def compute(panel: Panel, calibration: str = "poa", meta: dict | None = None) -> Features:
    meta = {**panel.meta, **(meta or {})}
    T, D = len(panel.time), panel.n
    lv = panel.devices["level"].to_numpy()
    parent = panel.devices["parent"].to_numpy()
    dev_ids = list(panel.devices.index)
    is_str = lv == "string"
    d = panel.data

    poa = panel.inherited("poa")
    tmod = panel.inherited("t_mod")
    poa_p = np.nanmedian(poa, axis=1) if D else np.zeros(T)
    light = np.nan_to_num(poa) > LIGHT_POA
    # without irradiance at a device's step (logger down at the plant), fall back to the sun
    sun_el, sun_az, kc, poa_cs = _sun(panel, meta, poa_p)
    no_poa = ~np.isfinite(poa)
    if np.isfinite(sun_el).any():
        light = np.where(no_poa, (sun_el > 10)[:, None], light)

    # strings: DC power where the MPPT voltage is reported (a stuck tracker raises the current), else current
    p_dc = d["i_dc"] * d["v_dc"] / 1000.0
    has_v = np.array([np.isfinite(d["v_dc"][light[:, j], j]).mean() > 0.5 if light[:, j].any() else False
                      for j in range(D)])
    is_amp = is_str & ~has_v
    y = np.where(is_str[None, :], np.where(has_v[None, :], p_dc, d["i_dc"]), d["p_ac"])
    y = np.where(~is_str[None, :] & ~np.isfinite(y), p_dc * 0.98, y)
    # a device that never reports its signal (a plant row without a meter) has no data to miss
    reports = np.array([np.isfinite(y[light[:, j], j]).mean() > 0.05 if light[:, j].any() else False
                        for j in range(D)])
    missing = light & ~np.isfinite(y) & reports[None, :]
    frozen = _frozen(d, poa_p, light)
    valid = light & ~missing & ~frozen

    # AC rating: metadata or a high percentile of the device's own output
    pac = np.full(D, np.inf)
    for j in range(D):
        if not is_str[j]:
            pj = (meta.get("pac_kw") or {}).get(dev_ids[j])
            yy = y[valid[:, j], j]
            pac[j] = float(pj) if pj else (np.nanpercentile(yy, 99.9) if yy.size else np.inf)
    near_pac = (y >= 0.97 * pac[None, :]) & np.isfinite(pac)[None, :]

    gamma = float(meta.get("gamma_pdc", -0.004))
    tfac = np.where(is_amp[None, :], 1 + 0.0005 * (np.nan_to_num(tmod, nan=25) - 25),
                    1 + gamma * (np.nan_to_num(tmod, nan=25) - 25))
    model = np.nan_to_num(poa) / 1000.0 * tfac
    ymax = np.array([np.nanpercentile(y[valid[:, j], j], 99) if valid[:, j].any() else np.nan for j in range(D)])
    clean = valid & (np.nan_to_num(poa) > 300) & (y > 0.05 * ymax[None, :]) & ~near_pac
    clean_k = clean if calibration == "scalar" else valid & (np.nan_to_num(poa) > 60) & (y > 0.01 * ymax[None, :])
    with np.errstate(all="ignore"):
        k = _bin_calibration(y / model, clean_k, panel.time, calibration, poa)
    y_exp = k * model
    y_exp = np.where(np.isfinite(pac)[None, :], np.minimum(y_exp, pac[None, :]), y_exp)

    parent_limited = np.zeros((T, D), bool)
    for j in np.flatnonzero(parent >= 0):
        parent_limited[:, j] = near_pac[:, parent[j]]
    with np.errstate(all="ignore"):
        r_exp = np.where(valid & (y_exp > 0.02 * np.nanmax(y_exp, axis=0)), y / y_exp - 1, np.nan)
    r_exp = np.where(parent_limited, np.nan, r_exp)

    # peer reference among siblings
    ref_peer = np.full((T, D), np.nan)
    r_v = np.full((T, D), np.nan)
    v = d["v_dc"]
    groups = panel.devices.groupby(["parent", "level"]).indices
    v_ref = np.full((T, D), np.nan)
    with np.errstate(all="ignore"):
        xn = y / k
    for (_, _), idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        min_n = 1 if len(idx) == 2 else 2  # a twin pair: each is the other's reference
        med = _loo_median(xn[:, idx], valid[:, idx], min_n)
        ref_peer[:, idx] = med * k[:, idx]
        vv = _loo_median(v[:, idx], valid[:, idx] & (np.nan_to_num(v[:, idx]) > 0), min_n)
        r_v[:, idx] = np.where(valid[:, idx], v[:, idx] / vv - 1, np.nan)
        v_ref[:, idx] = vv
    with np.errstate(all="ignore"):
        ratio = np.where(valid & (ref_peer > 0.02 * np.nanmax(ref_peer, axis=0)), y / ref_peer, np.nan)
    if calibration == "hour_month":  # devices that face another way than their siblings
        base = _bin_calibration(ratio, np.isfinite(ratio) & (np.nan_to_num(poa) > 200), panel.time, calibration)
        ratio = ratio / base
        ref_peer = ref_peer * base
    r_peer = ratio - 1

    zero = valid & (y < 0.02 * np.maximum(y_exp, 0.05 * ymax[None, :])) & (np.nan_to_num(poa) > 100)
    status = d["status"]
    status_bad = np.isfinite(status) & (status != 0)
    r_vself = _vself(v, poa, tmod, valid & ~zero & ~near_pac & (np.nan_to_num(r_peer) > -0.05))

    # siblings low on the expected model (plant-wide context)
    frac_low = np.full((T, D), np.nan)
    low = (np.nan_to_num(r_exp, nan=0) < -0.1) | zero
    for (_, _), idx in groups.items():
        idx = np.asarray(idx)
        cnt = valid[:, idx].sum(axis=1)
        share = (low[:, idx] & valid[:, idx]).sum(axis=1) / np.maximum(cnt, 1)
        frac_low[:, idx] = np.where(cnt > 0, share, np.nan)[:, None]
    # the plant row gets the share of its inverters
    for j in np.flatnonzero(lv == "plant"):
        kids = np.flatnonzero((parent == j) & (lv == "inverter"))
        if len(kids):
            frac_low[:, j] = frac_low[:, kids[0]]

    y_kw = np.where(is_amp[None, :], y * np.where(np.isfinite(v_ref), v_ref, v) / 1000.0 * 0.98,
                    np.where(is_str[None, :], y * 0.98, y))
    recur = _recurrence(r_peer, valid & (np.nan_to_num(kc, nan=0.8)[:, None] > 0.6), sun_az, sun_el, panel.time)
    sensor_dev, sensor_shape = _sensor_check(panel, poa_p, poa_cs, y, lv, valid, sun_el)
    tm = pd.Series(np.nanmedian(tmod, axis=1), index=panel.time)
    cold = (tm.rolling("24h", min_periods=1).min() < 2.0).to_numpy()

    mppt_group = _mppt_groups(panel, v, valid)
    return Features(panel=panel, light=light, y=y, y_kw=y_kw, is_amp=is_amp, k=k, y_exp=y_exp, r_exp=r_exp.astype(F32),
                    ref_peer=ref_peer, r_peer=r_peer.astype(F32), r_v=r_v.astype(F32),
                    r_vself=r_vself.astype(F32), zero=zero, missing=missing, frozen=frozen, near_pac=near_pac,
                    parent_limited=parent_limited, status_bad=status_bad, recur=recur.astype(F32),
                    frac_low=frac_low.astype(F32), sensor_dev=sensor_dev, sensor_shape=sensor_shape, cold=cold,
                    sun_el=sun_el,
                    sun_az=sun_az, kc=kc, pac=pac, mppt_group=mppt_group, v_ref=v_ref, poa=poa, tmod=tmod,
                    info={"calibration": calibration})


def _sun(panel: Panel, meta: dict, poa_p: np.ndarray):
    T = len(panel.time)
    if "lat" not in meta or "lon" not in meta:
        nan = np.full(T, np.nan)
        return nan, nan, nan, nan
    import pvlib

    loc = pvlib.location.Location(meta["lat"], meta["lon"])
    mid = panel.time + pd.Timedelta(hours=panel.step_h / 2)
    sp = loc.get_solarposition(mid)
    el = sp["apparent_elevation"].to_numpy()
    az = sp["azimuth"].to_numpy()
    kc = np.full(T, np.nan)
    poa_cs = np.full(T, np.nan)
    if "tilt" in meta and "azimuth" in meta:
        cs = loc.get_clearsky(mid, solar_position=sp)
        poa = pvlib.irradiance.get_total_irradiance(meta["tilt"], meta["azimuth"], sp["apparent_zenith"], az,
                                                    cs["dni"], cs["ghi"], cs["dhi"])
        poa_cs = np.nan_to_num(np.asarray(poa["poa_global"], dtype=float))
        with np.errstate(all="ignore"):
            kc = np.where(poa_cs > 50, poa_p / poa_cs, np.nan)
    return el, az, kc, poa_cs


def _frozen(d: dict, poa_p: np.ndarray, light: np.ndarray, window: int = 4) -> np.ndarray:
    """All reported channels unchanged for ``window`` steps while the irradiance moved."""
    T, D = d["p_ac"].shape
    same = np.ones((T, D), bool)
    any_ch = np.zeros((T, D), bool)
    for ch in ("p_ac", "i_dc", "v_dc"):
        x = d[ch]
        fin = np.isfinite(x)
        dx = np.vstack([np.full((1, D), np.nan), np.diff(x, axis=0)])
        same &= ~fin | (dx == 0)
        any_ch |= fin
    same &= any_ch
    run = pd.DataFrame(same.astype(np.int8)).rolling(window, min_periods=window).sum().to_numpy() >= window
    pp = pd.Series(poa_p)
    moving = ((pp.rolling(window, min_periods=2).max() - pp.rolling(window, min_periods=2).min())
              > np.maximum(0.02 * pp.rolling(window, min_periods=2).mean(), 5)).to_numpy()
    nonzero = np.zeros((T, D), bool)
    for ch in ("p_ac", "i_dc"):
        nonzero |= np.nan_to_num(d[ch]) != 0
    hit = run & moving[:, None] & nonzero  # repeated zeros are an outage as often as a frozen logger
    # mark the whole window, not just its last step
    out = hit.copy()
    for s in range(1, window):
        out[:-s] |= hit[s:]
    return out & light


def _vself(v, poa, tmod, ok):
    """Voltage relative to the device's median at the same irradiance (100 W/m2) and module temperature (5 K)."""
    T, D = v.shape
    out = np.full((T, D), np.nan)
    gb = np.clip(np.nan_to_num(poa) // 100, 0, 13).astype(int)
    tb = np.clip((np.nan_to_num(tmod, nan=25) + 30) // 5, 0, 25).astype(int)
    key = gb * 32 + tb
    for j in range(D):
        if not np.isfinite(v[:, j]).any():
            continue
        s = pd.Series(np.where(ok[:, j], v[:, j], np.nan))
        med = s.groupby(key[:, j]).median()
        ref = med.reindex(key[:, j]).to_numpy()
        with np.errstate(all="ignore"):
            out[:, j] = np.where((ref > 0) & (np.nan_to_num(poa[:, j]) > LIGHT_POA), v[:, j] / ref - 1, np.nan)
    return out


def _recurrence(r_peer, ok, sun_az, sun_el, time, thr=-0.03, min_days=3):
    """Share of distinct days a device ran low with the sun in the same 5 deg x 3 deg cell, minus its typical share."""
    T, D = r_peer.shape
    out = np.zeros((T, D))
    if not np.isfinite(sun_az).any():
        return out
    cell = (np.clip(sun_az // 5, 0, 71) * 40 + np.clip(sun_el // 3, 0, 39)).astype(int)
    day = np.asarray((time.normalize() - time[0].normalize()).days)
    for j in range(D):
        m = ok[:, j] & np.isfinite(r_peer[:, j])
        if m.sum() < 50:
            continue
        df = pd.DataFrame({"cell": cell[m], "day": day[m], "low": r_peer[m, j] < thr})
        g = df.groupby(["cell", "day"])["low"].mean() > 0.5
        per_cell = g.groupby(level=0).agg(["mean", "size"])
        frac = per_cell["mean"].where(per_cell["size"] >= min_days)
        typical = float(np.nanmedian(frac)) if frac.notna().any() else 0.0
        val = (frac - typical).reindex(cell).to_numpy()
        out[:, j] = np.nan_to_num(val)
    return out


def _sensor_check(panel, poa_p, poa_cs, y, lv, valid, sun_el):
    """Irradiance sensor vs clear sky at steps where production says the sun is out (NaN elsewhere).

    A step qualifies on a day whose production follows the clear-sky shape (smooth, near the best days of the
    month) and when its own production is within 5 % of that day's level. Soiling lowers production, not the
    sensor; a drifting or shaded sensor moves its ratio to clear sky while production stays put."""
    T = len(panel.time)
    out = np.full(T, np.nan)
    if not np.isfinite(poa_cs).any():
        return out, out.copy()
    inv = np.flatnonzero(lv == "inverter")
    if not len(inv):
        return out, out.copy()
    prod = np.nansum(np.where(valid[:, inv], y[:, inv], np.nan), axis=1)
    complete = valid[:, inv].all(axis=1)
    with np.errstate(all="ignore"):
        shape = np.where(complete & (poa_cs > 150) & (sun_el > 12), prod / poa_cs, np.nan)
        ratio = np.where(np.isfinite(shape), poa_p / poa_cs, np.nan)
    day = pd.Series(panel.time.normalize(), index=np.arange(T))
    df = pd.DataFrame({"day": day.to_numpy(), "shape": shape})
    g = df.groupby("day")["shape"]
    lvl = g.transform("median").to_numpy()
    rough = (df.assign(d=np.abs(pd.Series(shape).diff()))
             .groupby("day")["d"].transform("median").to_numpy())
    cover = g.transform("count").to_numpy()
    p90 = pd.Series(lvl).groupby(day.to_numpy()).first().rolling(31, center=True, min_periods=5).quantile(0.9)
    p90 = p90.reindex(day.to_numpy()).to_numpy()
    with np.errstate(all="ignore"):
        clear_day = (cover >= 12) & (rough / lvl < 0.02) & (lvl > 0.8 * p90)
        clear_step = clear_day & (np.abs(shape / lvl - 1) < 0.05)  # production itself says: full sun
        dev_step = np.where(clear_step, ratio, np.nan)
    if not np.isfinite(dev_step).any():
        return out, out.copy()
    level = pd.Series(dev_step).groupby(day.to_numpy()).transform("median").to_numpy()
    shape_dev = dev_step / level - 1  # within the day: haze cancels, a shadow on the sensor does not
    day_dev = level / np.nanmedian(level) - 1  # day level vs the period: drift, but haze moves it too
    return day_dev, shape_dev


def _mppt_groups(panel: Panel, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Strings of one inverter that report the same voltage almost always share an MPPT input."""
    lv = panel.devices["level"].to_numpy()
    parent = panel.devices["parent"].to_numpy()
    grp = np.full(panel.n, -1)
    gid = 0
    for p in np.unique(parent[lv == "string"]):
        idx = np.flatnonzero((parent == p) & (lv == "string"))
        assigned = set()
        for a in idx:
            if a in assigned:
                continue
            members = [a]
            for b in idx:
                if b <= a or b in assigned:
                    continue
                m = valid[:, a] & valid[:, b] & (np.nan_to_num(v[:, a]) > 0)
                if m.sum() > 20 and np.mean(np.abs(v[m, a] - v[m, b]) <= 1e-6 * np.abs(v[m, a]) + 1e-9) > 0.9:
                    members.append(b)
            if len(members) > 1:
                for mbr in members:
                    grp[mbr] = gid
                    assigned.add(mbr)
                gid += 1
    return grp
