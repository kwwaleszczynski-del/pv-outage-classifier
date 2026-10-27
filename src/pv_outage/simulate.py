"""A simulated PV plant with injected outages, faults and data problems, and the exact truth behind them.

Plant: 10 inverters of 100 kW AC, each with 12 strings of 26 x 400 W modules on six MPPT inputs (1.25 MWp,
DC/AC 1.25), fixed tilt 30 deg facing south, real ERA5 weather downscaled to 15 minutes (see ``weather``).

Physics the detectors never see: incidence-angle losses, single-diode substrings with bypass diodes, MPPT
operating points (including how an inverter sheds power when it clips or follows a set-point), Faiman cell
temperature, static string mismatch, cloud-driven spatial noise between inverters, a point pyranometer that sees
different clouds than the array, soiling washed off by real rain (Kimber), snow cover and sliding from real
snowfall (Marion / NREL).

Every event is applied in its own layer, and the plant is solved once per layer:

    healthy -> + soiling -> + snow -> + shading -> + device faults -> + curtailment and grid outages = actual

The true energy lost to an event is the difference between its layer and the one before, on the event's
inverters, inside its window. Data problems (communication loss, frozen values, irradiance sensor faults) only
change the measurements, never the energy.

Usage:
    raw = weather.fetch_era5(51.9, 19.5, 2021, 2024, Path("data/era5"))
    sim = simulate(raw, year=2023, seed=11)
    sim.long.to_parquet("plant.parquet"); sim.truth.to_csv("truth.csv")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pv_outage import physics
from pv_outage.taxonomy import CATEGORY, CLASSES
from pv_outage.weather import downscale

log = logging.getLogger(__name__)

STATUS = {"normal": 0, "limited": 1, "grid_fault": 2, "inverter_fault": 3}
LAYERS = ["healthy", "soiling", "snow", "shading", "device", "external"]
LAYER_OF = {"soiling": 1, "snow": 2, "partial_shading": 3, "inverter_trip": 4, "string_open": 4,
            "bypass_short": 4, "mppt_failure": 4, "curtailment": 5, "grid_outage": 5}


@dataclass
class PlantConfig:
    lat: float = 51.9
    lon: float = 19.5
    tilt: float = 30.0
    azimuth: float = 180.0
    n_inverters: int = 10
    strings_per_inverter: int = 12
    strings_per_mppt: int = 2
    modules_per_string: int = 26
    pac0_kw: float = 100.0
    meter_loss: float = 0.015
    # events per plant-year
    n_trip_short: int = 6
    n_trip_long: int = 4
    n_string_open: int = 8
    n_bypass: int = 5
    n_mppt: int = 4
    n_shading: int = 2
    n_curtail: int = 12
    n_grid: int = 4
    n_comm: int = 8
    n_frozen: int = 5
    n_sensor: int = 2

    @property
    def n_strings(self) -> int:
        return self.n_inverters * self.strings_per_inverter

    @property
    def n_mppts(self) -> int:
        return self.n_strings // self.strings_per_mppt

    @property
    def n_sub(self) -> int:
        return 3 * self.modules_per_string


@dataclass
class SimResult:
    long: pd.DataFrame
    truth: pd.DataFrame
    wx: pd.DataFrame
    meta: dict
    labels: np.ndarray  # (time, device) class index into CLASSES, -1 = normal
    label_event: np.ndarray  # (time, device) event_id of the truth table, -1 = none
    device_ids: list
    clip: np.ndarray  # (time, inverter) healthy clipping at the AC rating
    p_healthy: np.ndarray  # (time, inverter) AC power without any event, kW
    p_actual: np.ndarray  # (time, inverter) true AC power, kW
    layers_p: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------- names
def device_names(cfg: PlantConfig):
    inv = [f"INV{i + 1:02d}" for i in range(cfg.n_inverters)]
    strings = [f"{inv[s // cfg.strings_per_inverter]}-S{s % cfg.strings_per_inverter + 1:02d}"
               for s in range(cfg.n_strings)]
    return "PLANT", inv, strings


# ----------------------------------------------------------------------------------------------- layout
def layout(cfg: PlantConfig, rng: np.random.Generator) -> dict:
    n_i, n_s = cfg.n_inverters, cfg.n_strings
    road = rng.choice(n_i - 2) + np.arange(3)  # three neighbouring inverters next to a gravel road
    base = rng.uniform(0.0008, 0.0014, n_i)
    rate = base.copy()
    rate[road] += rng.uniform(0.003, 0.005, 3)
    return {
        "mismatch": np.clip(1 + rng.normal(0, 0.006, n_s), 0.98, 1.02),
        "inv_gain": 1 + rng.normal(0, 0.004, n_i),
        "inv_dt": rng.normal(0, 0.7, n_i),
        "row_pos": rng.uniform(0, 1, n_s),  # string position on the table, 0 = bottom edge (snow stays longest)
        "slide": rng.uniform(0.75, 1.25, n_i),
        "soil_rate": rate, "road": road,
    }


# ----------------------------------------------------------------------------------------------- events
def _window(rng, time, day_mask, dur_h, daylight_start=True, months=None, hours=None):
    """Random [start, end) index window of ``dur_h`` hours."""
    step_h = (time[1] - time[0]) / pd.Timedelta("1h")
    ok = day_mask.copy() if daylight_start else np.ones(len(time), bool)
    if months is not None:
        ok &= np.isin(time.month, months)
    if hours is not None:
        ok &= np.isin(time.hour, hours)
    cand = np.flatnonzero(ok)
    i0 = int(rng.choice(cand))
    return i0, min(len(time), i0 + max(1, int(round(dur_h / step_h))))


def schedule(cfg: PlantConfig, wx: pd.DataFrame, plant: dict, rng: np.random.Generator) -> list[dict]:
    """Scheduled events (soiling and snow come from the weather and are added later)."""
    time = wx.index
    plant_id, inv, strings = device_names(cfg)
    day = (wx["sun_el"] > 8).to_numpy()
    busy = {i: [] for i in range(cfg.n_inverters)}  # device faults never overlap on one inverter
    events = []

    def free(i, a, b, margin=96):
        return all(b + margin <= s or a >= e + margin for s, e in busy[i])

    def place(cls, dur_h, pick_inv=None, daylight_start=True, months=None, hours=None, tries=500):
        for _ in range(tries):
            i = int(rng.integers(cfg.n_inverters)) if pick_inv is None else pick_inv
            a, b = _window(rng, time, day, dur_h(), daylight_start, months, hours)
            if free(i, a, b):
                busy[i].append((a, b))
                return i, a, b
        raise RuntimeError(f"could not place {cls}")

    # shading: an obstruction next to 1-3 strings of one inverter, active for a season or for good
    for k in range(cfg.n_shading):
        i = int(rng.integers(cfg.n_inverters))
        first = int(rng.integers(0, cfg.strings_per_inverter - 2))
        n_str = int(rng.integers(1, 4))
        east = rng.random() < 0.5
        seasonal = k % 2 == 0
        if seasonal:
            a = int(np.searchsorted(time, time[0] + pd.Timedelta(days=int(rng.integers(110, 150)))))
            b = int(np.searchsorted(time, time[0] + pd.Timedelta(days=int(rng.integers(250, 290)))))
        else:
            a = int(np.searchsorted(time, time[0] + pd.Timedelta(days=int(rng.integers(20, 120)))))
            b = len(time)
        busy[i].append((a, b))
        sidx = [i * cfg.strings_per_inverter + first + j for j in range(n_str)]
        events.append(dict(cls="partial_shading", a=a, b=b, devices=[strings[s] for s in sidx], strings=sidx,
                           inverters=[i], n_shaded=[int(rng.integers(3, 13)) for _ in sidx],
                           az_c=float(rng.uniform(80, 120) if east else rng.uniform(240, 280)),
                           half_width=float(rng.uniform(15, 30)), el_top=float(rng.uniform(15, 30)),
                           transmission=float(rng.uniform(0.05, 0.3)), status=0,
                           note=("tree, in leaf" if seasonal else "pole / structure")))

    for _ in range(cfg.n_trip_long):
        i, a, b = place("inverter_trip", lambda: rng.uniform(24, 24 * 7), daylight_start=False)
        events.append(dict(cls="inverter_trip", a=a, b=b, devices=[inv[i]], inverters=[i],
                           status=STATUS["inverter_fault"] if rng.random() < 0.6 else 0, note="long outage"))
    for _ in range(cfg.n_trip_short):
        i, a, b = place("inverter_trip", lambda: rng.uniform(0.75, 6))
        events.append(dict(cls="inverter_trip", a=a, b=b, devices=[inv[i]], inverters=[i],
                           status=STATUS["inverter_fault"] if rng.random() < 0.6 else 0, note="trip and reset"))
    for _ in range(cfg.n_string_open):
        i, a, b = place("string_open", lambda: rng.uniform(2 * 24, 25 * 24), daylight_start=False)
        s = i * cfg.strings_per_inverter + int(rng.integers(cfg.strings_per_inverter))
        events.append(dict(cls="string_open", a=a, b=b, devices=[strings[s]], strings=[s], inverters=[i],
                           status=0, note="blown fuse"))
    for _ in range(cfg.n_bypass):
        i, a, b = place("bypass_short", lambda: rng.uniform(10 * 24, 60 * 24), daylight_start=False)
        s = i * cfg.strings_per_inverter + int(rng.integers(cfg.strings_per_inverter))
        nb = int(rng.integers(2, 7))
        events.append(dict(cls="bypass_short", a=a, b=b, devices=[strings[s]], strings=[s], inverters=[i],
                           n_bypassed=nb, status=0, note=f"{nb} shorted bypass diode(s)"))
    for _ in range(cfg.n_mppt):
        i, a, b = place("mppt_failure", lambda: rng.uniform(2 * 24, 20 * 24), daylight_start=False)
        m = i * (cfg.strings_per_inverter // cfg.strings_per_mppt) + int(
            rng.integers(cfg.strings_per_inverter // cfg.strings_per_mppt))
        ss = [m * cfg.strings_per_mppt + j for j in range(cfg.strings_per_mppt)]
        vf = float(rng.uniform(0.62, 0.85))
        events.append(dict(cls="mppt_failure", a=a, b=b, devices=[strings[s] for s in ss], strings=ss, mppt=m,
                           inverters=[i], v_factor=vf, status=0, note=f"tracker stuck at {vf:.2f} x Vmpp"))

    # plant-wide external events never overlap each other
    taken = []

    def place_plant(dur_h, months=None, hours=None, daylight_start=True):
        for _ in range(500):
            a, b = _window(rng, time, day, dur_h(), daylight_start, months, hours)
            if all(b + 8 <= s or a >= e + 8 for s, e in taken):
                taken.append((a, b))
                return a, b
        raise RuntimeError("could not place plant event")

    for _ in range(cfg.n_curtail):
        a, b = place_plant(lambda: rng.uniform(1, 5), months=range(3, 10), hours=range(8, 13))
        sp = 0.0 if rng.random() < 0.35 else float(np.round(rng.uniform(0.2, 0.7), 2))
        events.append(dict(cls="curtailment", a=a, b=b, devices=[plant_id], inverters=list(range(cfg.n_inverters)),
                           setpoint=sp, status=STATUS["limited"], status_p=0.8,
                           note=f"export set-point {sp:.0%} of AC rating"))
    for _ in range(cfg.n_grid):
        a, b = place_plant(lambda: rng.uniform(0.25, 8), months=range(1, 13))
        events.append(dict(cls="grid_outage", a=a, b=b, devices=[plant_id], inverters=list(range(cfg.n_inverters)),
                           status=STATUS["grid_fault"], status_p=0.9, note="no grid voltage"))

    # data problems (any time, independent of the above)
    for k in range(cfg.n_comm):
        whole = k < 3
        a, b = _window(rng, time, day, rng.uniform(1, 72), daylight_start=False)
        if whole:
            events.append(dict(cls="comm_loss", a=a, b=b, devices=[plant_id], inverters=list(range(cfg.n_inverters)),
                               status=0, note="plant logger offline"))
        else:
            i = int(rng.integers(cfg.n_inverters))
            events.append(dict(cls="comm_loss", a=a, b=b, devices=[inv[i]], inverters=[i], status=0,
                               note="inverter communication lost"))
    for k in range(cfg.n_frozen):
        i = int(rng.integers(cfg.n_inverters))
        a, b = _window(rng, time, day, rng.uniform(2, 48), daylight_start=k >= 2)
        events.append(dict(cls="frozen_data", a=a, b=b, devices=[inv[i]], inverters=[i], status=0,
                           note="logger repeats the last values"))
    for k in range(cfg.n_sensor):
        if k % 2 == 0:
            a, b = _window(rng, time, day, rng.uniform(21 * 24, 56 * 24), daylight_start=False)
            d = float(rng.choice([-1, 1]) * rng.uniform(0.07, 0.15))
            events.append(dict(cls="sensor_fault", a=a, b=b, devices=[plant_id], inverters=[], kind="drift",
                               drift=d, status=0, note=f"pyranometer drift to {d:+.0%}"))
        else:
            a, b = _window(rng, time, day, rng.uniform(30 * 24, 90 * 24), daylight_start=False, months=range(3, 9))
            events.append(dict(cls="sensor_fault", a=a, b=b, devices=[plant_id], inverters=[], kind="shade",
                               az_c=float(rng.uniform(95, 140)), half_width=float(rng.uniform(8, 15)),
                               el_top=float(rng.uniform(15, 30)), status=0, note="pyranometer shaded in the morning"))
    return events


# ----------------------------------------------------------------------------------------------- weather effects
def soiling_loss(cfg: PlantConfig, wx: pd.DataFrame, plant: dict) -> np.ndarray:
    """(time, inverter) soiling loss fraction from real rain (Kimber, 5 mm/day cleans, 2 days damp)."""
    import pvlib

    rain_h = wx["precip_mm"].resample("1h").sum()
    out = np.zeros((len(wx), cfg.n_inverters))
    for i in range(cfg.n_inverters):
        s = pvlib.soiling.kimber(rain_h, cleaning_threshold=5, soiling_loss_rate=plant["soil_rate"][i],
                                 grace_period=2, max_soiling=0.25)
        out[:, i] = s.reindex(wx.index, method="ffill").fillna(0).to_numpy()
    return out


def snow_cover(cfg: PlantConfig, wx: pd.DataFrame, plant: dict) -> np.ndarray:
    """(time, inverter) snow-covered fraction of the table (Marion / NREL sliding model, real snowfall)."""
    import pvlib

    snow_h = wx["snow_cm"].resample("1h").sum()
    poa_h = wx["poa"].resample("1h").mean()
    t_h = wx["t_air"].resample("1h").mean()
    out = np.zeros((len(wx), cfg.n_inverters))
    for i in range(cfg.n_inverters):
        c = pvlib.snow.coverage_nrel(snow_h, poa_h, t_h, cfg.tilt, threshold_snowfall=0.25,
                                     slide_amount_coefficient=0.197 * plant["slide"][i])
        out[:, i] = c.reindex(wx.index, method="ffill").fillna(0).to_numpy()
    return out


def _episodes(mask: np.ndarray, min_len: int = 1):
    """[start, end) runs of True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def weather_events(cfg, wx, soil, cover) -> list[dict]:
    plant_id, inv, _ = device_names(cfg)
    events = []
    # soiling episodes: loss >= 3 % until the next cleaning rain; inverters in the same episode are one event
    eps = []
    for i in range(cfg.n_inverters):
        dirty = soil[:, i] > 1e-4
        for a, b in _episodes(dirty):
            hit = np.flatnonzero(soil[a:b, i] >= 0.03)
            if len(hit):
                eps.append((a + hit[0], b, i))
    eps.sort()
    used = [False] * len(eps)
    for k, (a, b, i) in enumerate(eps):
        if used[k]:
            continue
        group, lo, hi = [i], a, b
        for k2 in range(k + 1, len(eps)):
            a2, b2, i2 = eps[k2]
            if not used[k2] and a2 < hi and b2 > lo and abs(b2 - b) <= 4 * 24 * 2:  # same cleaning rain
                group.append(i2)
                used[k2] = True
                lo, hi = min(lo, a2), max(hi, b2)
        used[k] = True
        whole = len(group) >= 0.8 * cfg.n_inverters
        events.append(dict(cls="soiling", a=lo, b=hi, devices=[plant_id] if whole else [inv[j] for j in sorted(group)],
                           inverters=sorted(group), status=0,
                           note=f"max loss {soil[lo:hi, sorted(group)].max():.1%}, washed by rain"))
    # snow: from first to last covered step, plant-wide
    covered = cover.max(axis=1) > 0.01
    for a, b in _episodes(covered):
        events.append(dict(cls="snow", a=a, b=b, devices=[plant_id], inverters=list(range(cfg.n_inverters)),
                           status=0, note=f"snowfall {wx['snow_cm'].iloc[max(0, a - 8):b].sum():.1f} cm"))
    return events


# ----------------------------------------------------------------------------------------------- electrical solve
@dataclass
class LayerInput:
    g_full: np.ndarray  # (T, S) irradiance on lit substrings
    g_shade: np.ndarray  # (T, S) irradiance on shaded substrings
    n_shaded: np.ndarray  # (T, S)
    n_bypassed: np.ndarray  # (T, S)
    open_: np.ndarray  # (T, S) bool
    mppt_vf: np.ndarray  # (T, M) NaN = working tracker
    inv_off: np.ndarray  # (T, I) bool
    p_lim: np.ndarray  # (T, I) AC limit kW


def solve(cfg: PlantConfig, inp: LayerInput, t_cell: np.ndarray, module: physics.Module, table: physics.MppTable):
    """Operating point of every string, MPPT input and inverter. Returns dict of arrays."""
    T, S = inp.g_full.shape
    spm, spi = cfg.strings_per_mppt, cfg.strings_per_inverter
    M, I = S // spm, cfg.n_inverters
    inv_of_s = np.arange(S) // spi
    tc_s = t_cell[:, inv_of_s]
    i_mp, v_mp, v_oc = table.lookup(inp.g_full, tc_s)
    n_eff = cfg.n_sub - inp.n_bypassed
    i_s = np.where(inp.open_, 0.0, i_mp)
    v_s = n_eff * v_mp
    voc_s = n_eff * v_oc

    # uniform MPPT inputs: voltage = mean of the working strings' MPP voltages
    work = ~inp.open_
    vsum = (v_s * work).reshape(T, M, spm).sum(axis=2)
    nwork = work.reshape(T, M, spm).sum(axis=2)
    v_m = np.where(nwork > 0, vsum / np.maximum(nwork, 1), voc_s.reshape(T, M, spm).mean(axis=2))
    complex_m = ((inp.n_shaded > 0) | (inp.n_bypassed > 0)).reshape(T, M, spm).any(axis=2) | np.isfinite(inp.mppt_vf)
    complex_m &= inp.g_full.reshape(T, M, spm).max(axis=2) > 1.0
    i_out = i_s.copy()
    p_m = (i_s.reshape(T, M, spm) * v_m[:, :, None]).sum(axis=2)

    def curves_for(tt, mm):
        cols = (mm[:, None] * spm + np.arange(spm)[None, :])
        cur_l = []
        for j in range(spm):
            sj = cols[:, j]
            cur, volt = physics.string_iv(module, inp.g_full[tt, sj], inp.g_shade[tt, sj], tc_s[tt, sj],
                                          cfg.n_sub, inp.n_shaded[tt, sj], inp.n_bypassed[tt, sj])
            cur = np.where(inp.open_[tt, sj][:, None], 0.0, cur)
            cur_l.append((cur, volt))
        return cols, cur_l

    pmpp_m = p_m.copy()
    tt, mm = np.nonzero(complex_m)
    for c0 in range(0, len(tt), 20000):
        t_, m_ = tt[c0:c0 + 20000], mm[c0:c0 + 20000]
        cols, cl = curves_for(t_, m_)
        v_op, i_op, p_mpp = physics.operating_points(cl, inp.mppt_vf[t_, m_], np.ones(len(t_)))
        v_m[t_, m_] = v_op
        for j in range(spm):
            i_out[t_, cols[:, j]] = i_op[j]
        p_m[t_, m_] = v_op * np.sum(i_op, axis=0)
        pmpp_m[t_, m_] = p_mpp

    p_dc = p_m.reshape(T, I, M // I).sum(axis=2) / 1000.0  # kW
    p_ac = physics.pvwatts_ac(p_dc, cfg.pac0_kw)
    limit = np.minimum(inp.p_lim, cfg.pac0_kw)
    limited = (p_ac > limit + 1e-6) & ~inp.inv_off
    if limited.any():
        target_dc = physics.dc_for_ac(limit[limited], cfg.pac0_kw)
        frac = np.clip(target_dc / np.maximum(p_dc[limited], 1e-9), 0, 1)
        lt, li = np.nonzero(limited)
        mpi = M // I
        # every MPPT input of a limited inverter delivers the same share of its maximum power
        rows_t = np.repeat(lt, mpi)
        rows_m = (li[:, None] * mpi + np.arange(mpi)[None, :]).ravel()
        rows_f = np.repeat(frac, mpi)
        for c0 in range(0, len(rows_t), 20000):
            t_, m_, f_ = rows_t[c0:c0 + 20000], rows_m[c0:c0 + 20000], rows_f[c0:c0 + 20000]
            cols, cl = curves_for(t_, m_)
            v_op, i_op, _ = physics.operating_points(cl, inp.mppt_vf[t_, m_], f_)
            v_m[t_, m_] = v_op
            for j in range(spm):
                i_out[t_, cols[:, j]] = i_op[j]
            p_m[t_, m_] = v_op * np.sum(i_op, axis=0)
        p_dc = p_m.reshape(T, I, M // I).sum(axis=2) / 1000.0
        p_ac = np.minimum(physics.pvwatts_ac(p_dc, cfg.pac0_kw), np.where(limited, limit, np.inf))

    # inverter off: open circuit
    off_s = inp.inv_off[:, inv_of_s]
    i_out[off_s] = 0.0
    voc_m = voc_s.reshape(T, M, spm).mean(axis=2)
    off_m = inp.inv_off[:, np.arange(M) // (M // I)]
    v_m = np.where(off_m, voc_m, v_m)
    p_ac = np.where(inp.inv_off, 0.0, p_ac)
    p_dc = np.where(inp.inv_off, 0.0, p_dc)
    dark = inp.g_full.reshape(T, M, spm).max(axis=2) < 2.0
    v_m = np.where(dark, 0.0, v_m)
    clip = (p_ac >= cfg.pac0_kw * 0.999) & ~inp.inv_off
    p_m = np.where(off_m, 0.0, p_m)
    return {"p_ac": p_ac, "p_dc": p_dc, "i_str": i_out, "v_mppt": v_m, "clip": clip, "p_mppt": p_m / 1000.0}


# ----------------------------------------------------------------------------------------------- main entry
def simulate(raw_weather: pd.DataFrame, year: int, seed: int, cfg: PlantConfig | None = None) -> SimResult:
    import pvlib

    cfg = cfg or PlantConfig()
    rng = np.random.default_rng(seed)
    wx = downscale(raw_weather, cfg.lat, cfg.lon, cfg.tilt, cfg.azimuth, seed=seed, year=year)
    time = wx.index
    T, I, S, M = len(time), cfg.n_inverters, cfg.n_strings, cfg.n_mppts
    plant = layout(cfg, rng)
    module, table = physics.Module.from_cec(), None
    table = physics.MppTable(module)
    plant_id, inv_ids, str_ids = device_names(cfg)
    inv_of_s = np.arange(S) // cfg.strings_per_inverter

    # irradiance reaching the array, with cloud-driven spatial noise between inverters
    aoi = pvlib.irradiance.aoi(cfg.tilt, cfg.azimuth, 90 - wx["sun_el"].to_numpy(), wx["sun_az"].to_numpy())
    iam = np.nan_to_num(pvlib.iam.physical(aoi), nan=0.0)
    beam = wx["poa_beam"].to_numpy() * iam
    diffuse = wx["poa_diffuse"].to_numpy() * 0.97
    kt = np.nan_to_num(wx["kt"].clip(0, 1.3).to_numpy(), nan=1.0)
    vi = pd.Series(kt).diff().abs().rolling(4, min_periods=1).mean().fillna(0).to_numpy()
    cloud = np.clip(vi / 0.1, 0, 1)
    e = rng.standard_normal((T, I))
    ar = np.empty((T, I))
    ar[0] = e[0]
    for t in range(1, T):
        ar[t] = 0.5 * ar[t - 1] + np.sqrt(0.75) * e[t]
    f_inv = (1 + (0.003 + 0.02 * cloud)[:, None] * ar) * plant["inv_gain"][None, :]
    f_str = f_inv[:, inv_of_s] * plant["mismatch"][None, :] * (1 + 0.002 * rng.standard_normal((T, S)))
    t_cell = (pvlib.temperature.faiman(wx["poa"].to_numpy(), wx["t_air"].to_numpy(), wx["wind"].to_numpy())
              .reshape(-1, 1) + plant["inv_dt"][None, :])

    events = schedule(cfg, wx, plant, rng)
    soil = soiling_loss(cfg, wx, plant)
    cover = snow_cover(cfg, wx, plant)
    events = weather_events(cfg, wx, soil, cover) + events
    sun_az, sun_el = wx["sun_az"].to_numpy(), wx["sun_el"].to_numpy()

    def shading_mask(ev):
        x = (sun_az - ev["az_c"]) / ev["half_width"]
        top = ev["el_top"] * np.clip(1 - x ** 2, 0, None)
        m = (np.abs(x) < 1) & (sun_el > 0) & (sun_el < top)
        out = np.zeros(T, bool)
        out[ev["a"]:ev["b"]] = m[ev["a"]:ev["b"]]
        return out

    # build layers cumulatively
    soil_s = soil[:, inv_of_s]
    covered = (cover[:, inv_of_s] > plant["row_pos"][None, :])
    snow_f = np.where(covered, 0.03, 1.0)
    g_lit = (beam + diffuse)[:, None] * f_str
    g_dif = (diffuse[:, None] * f_str)
    zeros_i = np.zeros((T, S), dtype=np.int16)
    base = LayerInput(g_full=g_lit, g_shade=g_dif, n_shaded=zeros_i, n_bypassed=zeros_i.copy(),
                      open_=np.zeros((T, S), bool), mppt_vf=np.full((T, M), np.nan),
                      inv_off=np.zeros((T, I), bool), p_lim=np.full((T, I), np.inf))
    inputs = [base]
    l1 = LayerInput(**{**base.__dict__, "g_full": g_lit * (1 - soil_s), "g_shade": g_dif * (1 - soil_s)})
    inputs.append(l1)
    l2 = LayerInput(**{**l1.__dict__, "g_full": l1.g_full * snow_f, "g_shade": l1.g_shade * snow_f})
    inputs.append(l2)
    n_sh = zeros_i.copy()
    g_sh = l2.g_shade.copy()
    for ev in events:
        if ev["cls"] == "partial_shading":
            m = shading_mask(ev)
            ev["active"] = m
            for s, k in zip(ev["strings"], ev["n_shaded"]):
                n_sh[m, s] = k
                g_sh[m, s] = l2.g_shade[m, s] + ev["transmission"] * (l2.g_full[m, s] - l2.g_shade[m, s])
    l3 = LayerInput(**{**l2.__dict__, "n_shaded": n_sh, "g_shade": g_sh})
    inputs.append(l3)
    n_byp, open_, vf, off = zeros_i.copy(), np.zeros((T, S), bool), np.full((T, M), np.nan), np.zeros((T, I), bool)
    for ev in events:
        a, b = ev["a"], ev["b"]
        if ev["cls"] == "bypass_short":
            n_byp[a:b, ev["strings"][0]] = ev["n_bypassed"]
        elif ev["cls"] == "string_open":
            open_[a:b, ev["strings"][0]] = True
        elif ev["cls"] == "mppt_failure":
            vf[a:b, ev["mppt"]] = ev["v_factor"]
        elif ev["cls"] == "inverter_trip":
            off[a:b, ev["inverters"][0]] = True
    l4 = LayerInput(**{**l3.__dict__, "n_bypassed": n_byp, "open_": open_, "mppt_vf": vf, "inv_off": off})
    inputs.append(l4)
    off5, lim = off.copy(), np.full((T, I), np.inf)
    for ev in events:
        a, b = ev["a"], ev["b"]
        if ev["cls"] == "curtailment":
            lim[a:b, :] = ev["setpoint"] * cfg.pac0_kw
        elif ev["cls"] == "grid_outage":
            off5[a:b, :] = True
    l5 = LayerInput(**{**l4.__dict__, "inv_off": off5, "p_lim": lim})
    inputs.append(l5)

    layers = []
    for name, inp in zip(LAYERS, inputs):
        layers.append(solve(cfg, inp, t_cell, module, table))
        log.debug("layer %s solved", name)
    act = layers[-1]

    truth = truth_table(cfg, events, layers, wx, time)
    # the temperature sensor sits on a module of the first inverter; under snow it stays near freezing
    cov0 = cover[:, 0]
    t_sensor = np.where(cov0 > 0.5, np.minimum(t_cell[:, 0], 0.5), t_cell[:, 0])
    meas, status = measure(cfg, act, wx, t_sensor, events, rng)
    devs = [plant_id] + inv_ids + str_ids
    labels, label_event = step_labels(cfg, truth, events, layers, wx, devs)
    long = to_long(cfg, meas, status, time)
    meta = {"lat": cfg.lat, "lon": cfg.lon, "tilt": cfg.tilt, "azimuth": cfg.azimuth, "year": year, "seed": seed,
            "pdc_kw": {d: cfg.strings_per_inverter * cfg.modules_per_string * module.p_stc / 1000 for d in inv_ids}
            | {d: cfg.modules_per_string * module.p_stc / 1000 for d in str_ids}
            | {plant_id: cfg.n_strings * cfg.modules_per_string * module.p_stc / 1000},
            "pac_kw": {d: cfg.pac0_kw for d in inv_ids} | {plant_id: cfg.pac0_kw * I * (1 - cfg.meter_loss)},
            "gamma_pdc": module.gamma_pdc}
    return SimResult(long=long, truth=truth, wx=wx, meta=meta, labels=labels, label_event=label_event, device_ids=devs,
                     clip=layers[0]["clip"] & act["clip"], p_healthy=layers[0]["p_ac"], p_actual=act["p_ac"],
                     layers_p=[ly["p_ac"] for ly in layers])


# ----------------------------------------------------------------------------------------------- truth
def truth_table(cfg, events, layers, wx, time) -> pd.DataFrame:
    step_h = (time[1] - time[0]) / pd.Timedelta("1h")
    light = (wx["poa"] > 20).to_numpy()
    rows = []
    for k, ev in enumerate(events):
        a, b = ev["a"], ev["b"]
        cls = ev["cls"]
        inv = ev["inverters"]
        if cls in LAYER_OF and inv:
            L = LAYER_OF[cls]
            before = layers[L - 1]["p_ac"][a:b][:, inv].sum(axis=1)
            after = layers[L]["p_ac"][a:b][:, inv].sum(axis=1)
            loss = np.clip(before - after, 0, None)
            if "strings" in ev:  # visible on the string's MPPT input even when the inverter clips it away
                mp = sorted({s_ // cfg.strings_per_mppt for s_ in ev["strings"]})
                b_m = layers[L - 1]["p_mppt"][a:b][:, mp].sum(axis=1)
                d_m = np.clip(b_m - layers[L]["p_mppt"][a:b][:, mp].sum(axis=1), 0, None)
                eff = light[a:b] & (d_m > 0.01 * np.maximum(b_m, 1e-9)) & (d_m > 0.02)
            else:
                eff = light[a:b] & (loss > 0.005 * np.maximum(before, 1e-9)) & (loss > 0.05)
            loss_kwh = float(loss.sum() * step_h)
            ref_kwh = float(before.sum() * step_h)
        else:
            eff = light[a:b]
            loss_kwh, ref_kwh = 0.0, 0.0
        ev["effect"] = eff
        rows.append(dict(event_id=k, cls=cls, category=CATEGORY[cls], device_id=ev["devices"][0],
                         devices=";".join(ev["devices"]), level=_level(ev["devices"][0]),
                         start=time[a], end=time[b - 1] + (time[1] - time[0]),
                         first_effect=time[a + int(np.argmax(eff))] if eff.any() else pd.NaT,
                         effect_hours=float(eff.sum() * step_h), true_loss_kwh=loss_kwh,
                         ref_energy_kwh=ref_kwh, note=ev.get("note", "")))
    df = pd.DataFrame(rows)
    # a loss event without any visible effect (a trip after sunset, a set-point above what clouds allowed)
    # cannot be detected by anything and is dropped
    df = df[(df["category"] == "data") | (df["effect_hours"] >= 0.25)]
    return df.sort_values("start").reset_index(drop=True)


def _level(dev: str) -> str:
    return "plant" if dev == "PLANT" else ("string" if "-S" in dev else "inverter")


def step_labels(cfg, truth, events, layers, wx, devs):
    """(time, device) class index and event id of the event affecting each device at each step (-1 = none).

    Scope: the event's devices and their descendants; string events also reach their inverter. A device in scope
    is affected at a daylight step (POA > 50 W/m2) when its own power drops between the event's layer and the one
    before: strings by > 1 % (and 5 W), inverters by > 0.5 % (and 50 W). Data problems affect their whole window.
    A more specific event (string over inverter over plant) wins, and data problems win over the rest.
    Events dropped from the truth table (no visible effect) label nothing.
    """
    T = len(wx)
    D = len(devs)
    lab = np.full((T, D), -1, dtype=np.int16)
    eid = np.full((T, D), -1, dtype=np.int32)
    depth = np.full((T, D), -1, dtype=np.int8)
    prio = np.full((T, D), -1, dtype=np.int8)
    prio_rank = {"environment": 0, "external": 1, "fault": 2, "data": 3}
    light = (wx["poa"] > 50).to_numpy()
    spi, spm, I = cfg.strings_per_inverter, cfg.strings_per_mppt, cfg.n_inverters
    kept = set(truth["event_id"])
    for k, ev in enumerate(events):
        if k not in kept:
            continue
        cls = ev["cls"]
        a, b = ev["a"], ev["b"]
        scope = []  # (column, depth of the event's own level, kind, index)
        for d in ev["devices"]:
            lv = _level(d)
            if cls == "sensor_fault":  # only the plant row carries the irradiance sensor
                scope += [(0, 0, "plant", 0)]
            elif lv == "plant":
                scope += [(0, 0, "plant", 0)] + [(1 + i, 0, "inv", i) for i in range(I)]
                scope += [(1 + I + s_, 0, "str", s_) for s_ in range(cfg.n_strings)]
            elif lv == "inverter":
                i = int(d[3:5]) - 1
                scope += [(1 + i, 1, "inv", i)] + [(1 + I + i * spi + s_, 1, "str", i * spi + s_) for s_ in range(spi)]
            else:
                s_ = devs.index(d) - 1 - I
                scope += [(1 + I + s_, 2, "str", s_), (1 + s_ // spi, 2, "inv", s_ // spi)]
        L = LAYER_OF.get(cls)
        for col, dep, kind, idx in set(scope):
            if L is None:
                eff = light[a:b].copy()
            else:
                prev, cur = layers[L - 1], layers[L]
                if kind == "str":
                    m = idx // spm
                    pp = prev["i_str"][a:b, idx] * prev["v_mppt"][a:b, m] / 1000
                    pc = cur["i_str"][a:b, idx] * cur["v_mppt"][a:b, m] / 1000
                    eff = light[a:b] & (pp - pc > np.maximum(0.01 * pp, 0.005))
                elif kind == "inv":
                    pp, pc = prev["p_ac"][a:b, idx], cur["p_ac"][a:b, idx]
                    eff = light[a:b] & (pp - pc > np.maximum(0.005 * pp, 0.05))
                else:
                    pp, pc = prev["p_ac"][a:b].sum(axis=1), cur["p_ac"][a:b].sum(axis=1)
                    eff = light[a:b] & (pp - pc > np.maximum(0.005 * pp, 0.05))
            sl = slice(a, b)
            pr = prio_rank[CATEGORY[cls]]
            cur_d, cur_p = depth[sl, col], prio[sl, col]
            if pr == 3:  # data problems override everything
                win = eff
            else:  # the more specific event wins, then the higher category
                win = eff & (cur_p < 3) & ((dep > cur_d) | ((dep == cur_d) & (pr >= cur_p)))
            lab[sl, col] = np.where(win, CLASSES.index(cls), lab[sl, col])
            eid[sl, col] = np.where(win, k, eid[sl, col])
            depth[sl, col] = np.where(win, dep, cur_d)
            prio[sl, col] = np.where(win, pr, cur_p)
    return lab, eid


# ----------------------------------------------------------------------------------------------- measurement
def measure(cfg, act, wx, t_sensor, events, rng):
    """SCADA view of the true plant: sensor noise, status codes, then the data problems."""
    T, I, S = act["p_ac"].shape[0], cfg.n_inverters, cfg.n_strings
    M = cfg.n_mppts
    kt = np.nan_to_num(wx["kt"].clip(0, 1.3).to_numpy(), nan=1.0)
    vi = pd.Series(kt).diff().abs().rolling(4, min_periods=1).mean().fillna(0).to_numpy()
    cloud = np.clip(vi / 0.1, 0, 1)

    p_inv = act["p_ac"] * (1 + 0.003 * rng.standard_normal((T, I)))
    i_s = act["i_str"] * (1 + 0.008 * rng.standard_normal((T, S))) + 0.01 * rng.standard_normal((T, S))
    i_s = np.where(act["i_str"] > 0, i_s.clip(0), 0.0)
    v_m = act["v_mppt"] * (1 + 0.002 * rng.standard_normal((T, M)))
    v_s = np.repeat(v_m, cfg.strings_per_mppt, axis=1)
    i_inv = act["i_str"].reshape(T, I, -1).sum(axis=2) * (1 + 0.004 * rng.standard_normal((T, I)))
    v_inv = act["v_mppt"].reshape(T, I, -1).mean(axis=2) * (1 + 0.002 * rng.standard_normal((T, I)))
    p_plant = act["p_ac"].sum(axis=1) * (1 - cfg.meter_loss) * (1 + 0.002 * rng.standard_normal(T))
    poa = wx["poa"].to_numpy() * (1 + (0.01 + 0.04 * cloud) * rng.standard_normal(T))
    tmod = t_sensor - 2.0 + 0.8 * rng.standard_normal(T)
    status = np.zeros((T, I))
    sun_az, sun_el = wx["sun_az"].to_numpy(), wx["sun_el"].to_numpy()
    beam = wx["poa_beam"].to_numpy()

    for ev in events:
        a, b, cls = ev["a"], ev["b"], ev["cls"]
        if ev.get("status") and rng.random() < ev.get("status_p", 1.0):
            status[a:b, ev["inverters"]] = ev["status"]
        if cls == "sensor_fault":
            if ev["kind"] == "drift":
                ramp = np.linspace(0, ev["drift"], b - a)
                poa[a:b] *= 1 + ramp
            else:
                x = (sun_az[a:b] - ev["az_c"]) / ev["half_width"]
                m = (np.abs(x) < 1) & (sun_el[a:b] < ev["el_top"] * np.clip(1 - x ** 2, 0, None)) & (sun_el[a:b] > 0)
                poa[a:b] = np.where(m, poa[a:b] - 0.9 * beam[a:b], poa[a:b])
    poa = poa.clip(0)

    chans = {"p_inv": p_inv, "i_inv": i_inv, "v_inv": v_inv, "i_s": i_s, "v_s": v_s, "status": status,
             "p_plant": p_plant[:, None], "poa": poa[:, None], "tmod": tmod[:, None]}
    spi = cfg.strings_per_inverter
    for ev in events:
        a, b, cls = ev["a"], ev["b"], ev["cls"]
        if cls not in ("comm_loss", "frozen_data"):
            continue
        whole = ev["devices"][0] == "PLANT"
        invs = ev["inverters"]
        strs = np.concatenate([np.arange(i * spi, (i + 1) * spi) for i in invs])
        for name, arr in chans.items():
            cols = (strs if name in ("i_s", "v_s") else invs if arr.shape[1] == I else ([0] if whole else []))
            if len(cols) == 0:
                continue
            if cls == "comm_loss":
                arr[a:b, cols] = np.nan
            else:
                arr[a:b, cols] = arr[a, cols][None, :]
    return chans, status


def to_long(cfg, meas, status, time) -> pd.DataFrame:
    """Measurements in the tool's long format (see ``schema``)."""
    plant_id, inv_ids, str_ids = device_names(cfg)
    T, I, S = len(time), cfg.n_inverters, cfg.n_strings
    nan_t = np.full(T, np.nan)
    blocks = [pd.DataFrame({"timestamp": time, "device_id": plant_id, "parent_id": "", "level": "plant",
                            "p_ac": meas["p_plant"][:, 0], "i_dc": nan_t, "v_dc": nan_t,
                            "poa": meas["poa"][:, 0], "t_mod": meas["tmod"][:, 0], "status": nan_t})]
    tt = np.repeat(time.to_numpy(), I)
    blocks.append(pd.DataFrame({"timestamp": tt, "device_id": np.tile(inv_ids, T), "parent_id": plant_id,
                                "level": "inverter", "p_ac": meas["p_inv"].ravel(), "i_dc": meas["i_inv"].ravel(),
                                "v_dc": meas["v_inv"].ravel(), "poa": np.nan, "t_mod": np.nan,
                                "status": meas["status"].ravel()}))
    tt = np.repeat(time.to_numpy(), S)
    parents = [s.split("-")[0] for s in str_ids]
    blocks.append(pd.DataFrame({"timestamp": tt, "device_id": np.tile(str_ids, T), "parent_id": np.tile(parents, T),
                                "level": "string", "p_ac": np.nan, "i_dc": meas["i_s"].ravel(),
                                "v_dc": meas["v_s"].ravel(), "poa": np.nan, "t_mod": np.nan, "status": np.nan}))
    out = pd.concat(blocks, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for c in ("p_ac", "i_dc", "v_dc", "poa", "t_mod", "status"):
        out[c] = out[c].astype("float32")
    return out
