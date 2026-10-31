"""Facts about the PVDAQ Las Vegas campus that several repositories quote: the course of the inverter outages and
the timestamp alignment of the channels. Writes results/pvdaq_site_facts.json, quoted in PVDAQ_SITE_NOTES.md.

Times are local standard time (UTC-8) as logged. Off-grid stamps are kept for the error codes (the logger writes
some records a few seconds after the quarter-hour) and dropped for the clock check.

Clock check: on clear days, the shift that best aligns a channel with a clear-sky plane-of-array curve, as the
minutes to add to the stamp to get the instant the value represents (a 15-minute mean labelled with the end of its
interval gives -7.5, with the start +7.5, a centred mean or an instantaneous sample 0). An orientation error of a
few degrees moves this by minutes, so the offset between two channels measured side by side is the firmer number.

Usage:
    python -m pv_outage site-facts
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from pv_outage.real import META, TZ, UNITS

log = logging.getLogger(__name__)

# (unit, first and last day searched); A and D1 are the plain outages, B is the 2017 gap
OUTAGES = [("A", "2016-05-25", "2016-07-20"), ("B", "2017-08-15", "2017-10-05"), ("D1", "2017-10-10", "2017-12-15")]
SUN = 200.0          # W/m2: plane-of-array irradiance above which an inverter at zero is plainly down
ALIVE = 0.01         # share of the DC rating above which an inverter produces
GRID = np.arange(-15.0, 15.01, 0.5)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix + "__")]
    return df[cols[0]].astype(float) if cols else pd.Series(np.nan, index=df.index)


def load_days(raw_dir: Path, system: int, first: str, last: str) -> pd.DataFrame:
    frames = []
    for day in pd.date_range(first, last, freq="D"):
        path = Path(raw_dir) / f"system_{system}" / f"system_{system}__date_{day:%Y_%m_%d}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df.index = pd.to_datetime(df.pop("measured_on"))
    return df[~df.index.duplicated()].sort_index()


def chronology(p_kw: pd.Series, poa: pd.Series, code: pd.Series, pdc_kw: float) -> dict:
    """Last output, stop in sun, start of missing inverter data, first error code and return of output."""
    alive = p_kw > ALIVE * pdc_kw
    # missing inverter data for at least an hour (single event records carry only the error code)
    missing = p_kw.isna() & poa.notna()
    gap = missing[::-1].rolling(4, min_periods=4).sum()[::-1] == 4
    t_blank = gap.index[gap.to_numpy()][0] if gap.any() else None
    stopped = (poa > SUN) & (p_kw <= ALIVE * pdc_kw)
    if t_blank is not None:
        stopped &= stopped.index < t_blank
    t_stop = stopped.index[stopped.to_numpy()][0] if stopped.any() else None
    t_down = t_stop if t_stop is not None else t_blank
    t_last = alive.index[alive.to_numpy() & (alive.index < t_down)][-1]
    back = alive & (alive.index > t_down)
    t_back = back.index[back.to_numpy()][0]
    codes = code[(code.index > t_last) & (code.index < t_back) & (code.fillna(0) != 0)]

    def days(a, b):
        return None if a is None or b is None else round((b - a) / pd.Timedelta(days=1), 2)

    return {
        "last_output": str(t_last), "stop_in_sun": None if t_stop is None else str(t_stop),
        "zero_output_logged_days": days(t_stop, t_blank),
        "inverter_data_missing_from": None if t_blank is None else str(t_blank),
        "inverter_data_missing_days": days(t_blank, t_back),
        "first_error_code": None if codes.empty else {"time": str(codes.index[0]), "code": int(codes.iloc[0])},
        "first_error_code_after_stop_days": days(t_stop, codes.index[0] if len(codes) else None),
        "error_codes": sorted(int(c) for c in codes.unique()),
        "output_back": str(t_back), "days_without_output": days(t_last, t_back),
    }


def _sec(ix) -> np.ndarray:
    return ((ix - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(float)


def _best_shift(x: np.ndarray, models: np.ndarray) -> tuple[float, float]:
    """Shift of the grid that fits x best with one scale factor (parabola through the best point)."""
    k = (models @ x) / np.maximum(np.einsum("ij,ij->i", models, models), 1e-9)
    cost = np.sqrt(((x[None, :] - k[:, None] * models) ** 2).mean(1)) / np.maximum(k * models.mean(1), 1e-9)
    b = int(np.argmin(cost))
    shift = GRID[b]
    if 0 < b < len(GRID) - 1:
        den = cost[b - 1] - 2 * cost[b] + cost[b + 1]
        if den > 0:
            shift += 0.5 * (cost[b - 1] - cost[b + 1]) / den * (GRID[1] - GRID[0])
    return float(shift), float(cost[b])


def clock_offsets(raw_dir: Path, years=(2016, 2017), array_tilt: float = 11.2, sensor_tilt: float = 5.0,
                  clear_nrmse: float = 0.03, min_elevation: float = 15.0) -> dict:
    """Clock offsets of the met-station POA and of system 34's AC power on clear days (see module docstring)."""
    df = load_days(raw_dir, 34, f"{min(years)}-01-01", f"{max(years)}-12-31")
    df = df[df.index.second == 0]
    idx = df.index.tz_localize(TZ)
    poa = _col(df, "poa_irradiance").to_numpy()
    ac = (_col(df, "ac_power_hw") * 0.1).to_numpy()  # hW -> kW (W only from August 2018)
    loc = pvlib.location.Location(META["lat"], META["lon"], TZ, 620)
    fine = pd.date_range(idx[0].floor("D"), idx[-1].ceil("D"), freq="1min", tz=TZ)
    sp = loc.get_solarposition(fine)
    cs = loc.get_clearsky(fine, model="ineichen", solar_position=sp)
    t, tf = _sec(idx), _sec(fine)
    elev = np.interp(t, tf, sp["apparent_elevation"].to_numpy())

    def shifted_models(tilt):
        g = pvlib.irradiance.get_total_irradiance(tilt, 180.0, sp["apparent_zenith"], sp["azimuth"], cs["dni"],
                                                  cs["ghi"], cs["dhi"])["poa_global"].clip(lower=0)
        g = g.rolling(15, center=True, min_periods=1).mean().to_numpy()  # a quarter-hour mean around each minute
        return np.stack([np.interp(t + s * 60.0, tf, g) for s in GRID])

    day = idx.tz_localize(None).normalize()
    ok = (elev > min_elevation) & np.isfinite(poa) & np.isfinite(ac) & (ac < 0.95 * 135.0)  # no clipping
    m_sensor, m_array = shifted_models(sensor_tilt), shifted_models(array_tilt)
    rows = []
    for d in day.unique():
        sel = ok & (day == d)
        if sel.sum() < 20:
            continue
        s_poa, c_poa = _best_shift(poa[sel], m_sensor[:, sel])
        s_ac, _ = _best_shift(ac[sel], m_array[:, sel])
        # AC against the measured POA of the same day, no model involved
        all_day = (day == d) & np.isfinite(poa)
        direct = np.stack([np.interp(t[sel] + s * 60.0, t[all_day], poa[all_day]) for s in GRID])
        s_rel, _ = _best_shift(ac[sel], direct)
        rows.append({"day": d, "poa": s_poa, "ac": s_ac, "ac_vs_poa": s_rel, "nrmse_poa": c_poa})
    r = pd.DataFrame(rows)
    clear = r[r["nrmse_poa"] < clear_nrmse]

    def stats(s):
        return {"median_min": round(float(s.median()), 1), "q25_min": round(float(s.quantile(0.25)), 1),
                "q75_min": round(float(s.quantile(0.75)), 1)}

    return {"clear_days": int(len(clear)), "years": list(years), "sensor_tilt_deg": sensor_tilt,
            "array_tilt_deg": array_tilt, "azimuth_deg": 180.0,
            "poa_vs_clear_sky": stats(clear["poa"]), "ac_vs_clear_sky": stats(clear["ac"]),
            "ac_vs_measured_poa": stats(clear["ac_vs_poa"])}


def run(data_dir: Path, results_dir: Path) -> dict:
    raw = Path(data_dir) / "pvdaq"
    out = {"outages": {}}
    for unit, first, last in OUTAGES:
        system, p_col, _, _, code_col = UNITS[unit]
        df = load_days(raw, system, first, last)
        out["outages"][unit] = {"system": system, **chronology(_col(df, p_col) * 0.1, _col(df, "poa_irradiance"),
                                                              _col(df, code_col), META["pdc_kw"][unit])}
        log.info("%s: %s", unit, out["outages"][unit])
    out["clock"] = clock_offsets(raw)
    log.info("clock: %s", out["clock"])
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "pvdaq_site_facts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
