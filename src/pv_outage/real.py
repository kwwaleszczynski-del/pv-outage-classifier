"""Real data: the rooftop fleet of one school campus in Las Vegas from NREL PVDAQ, as five inverters of one plant.

Source: Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D. (2021). Photovoltaic Data
Acquisition (PVDAQ) Public Datasets. Open Energy Data Initiative (OEDI), NREL. https://doi.org/10.25984/1846021,
CC BY 4.0. One CSV per system and day in a public bucket; 15-minute values in local standard time (UTC-8).

Units: buildings A (system 34, 146.6 kWp, south, 11 deg), B (1276, 68.5 kWp, south, 5 deg), D (1278, 171.4 kWp,
two inverters D1 and D2) and the gymnasium G (35, 121.7 kWp, facing west), one met station on site. Inverter
channels: AC power, DC input current and voltage, error code. Power channels are in hectowatts until 4 August
2018 and in watts afterwards; the switch is detected from the power-to-irradiance ratio.

There is no fault log, so precision cannot be measured. What can be measured: how many alarms each method
raises per device-year, and whether it finds the outages that are plain in the data (an inverter at zero for
hours in full sun while its neighbours produce). The thresholds come from the simulated plants; a sweep shows how
the alarm count falls with the threshold and with the shortest event kept. The fleet's KPIs use a site set-up
chosen without fault labels: the lowest peer threshold that still finds every plain outage at no more than one
other alarm per inverter-month, events of at least 2 h.

Usage:
    python -m pv_outage real
"""
from __future__ import annotations

import json
import logging
import pickle
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pv_outage import detectors, events as ev, features, kpi, pipeline, schema

log = logging.getLogger(__name__)

BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata"
YEARS = [2016, 2017, 2018]
TZ = "Etc/GMT+8"
META = {"lat": 36.1952, "lon": -115.1582, "tilt": 11.0, "azimuth": 180.0,
        "pdc_kw": {"A": 146.6, "B": 68.5, "D1": 85.7, "D2": 85.7, "G": 121.7}}
# unit -> (system, prefix of AC power, DC current, DC voltage, error code)
UNITS = {
    "A": (34, "ac_power_hw", "dc_current", "dc_voltage", "inverter_error_code"),
    "B": (1276, "ac_power_hw", "input_current", "input_voltage", "inverter_error_code"),
    "D1": (1278, "inv1_ac_power_hw", "inv1_input_current", "inv1_input_voltage", "inv1_error_code"),
    "D2": (1278, "inv2_ac_power_hw", "inv2_input_current", "inv2_input_voltage", "inv2_error_code"),
    "G": (35, "ac_power_hw", "dc_current", "dc_voltage", "inverter_error_code"),
}
MET_PRIORITY = [1276, 34, 1278, 35]
PLANT = "CAMPUS"


def day_url(system_id: int, day: pd.Timestamp) -> str:
    return (f"{BASE_URL}/system_id={system_id}/year={day.year}/month={day.month}/day={day.day}/"
            f"system_{system_id}__date_{day:%Y_%m_%d}.csv")


def fetch(raw_dir: Path, years=YEARS, workers: int = 8) -> int:
    """Download the missing daily files (about 60 MB for 2016-2018)."""
    jobs = []
    for sid in sorted({u[0] for u in UNITS.values()}):
        out = Path(raw_dir) / f"system_{sid}"
        out.mkdir(parents=True, exist_ok=True)
        for day in pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D"):
            path = out / f"system_{sid}__date_{day:%Y_%m_%d}.csv"
            if not path.exists():
                jobs.append((sid, day, path))

    def get(job):
        sid, day, path = job
        try:
            with urllib.request.urlopen(day_url(sid, day), timeout=60) as r:
                path.write_bytes(r.read())
            return 1
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return 0
            raise

    with ThreadPoolExecutor(workers) as pool:
        n = sum(pool.map(get, jobs))
    log.info("PVDAQ: %d of %d missing daily files downloaded", n, len(jobs))
    return n


def _column(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.rsplit("__", 1)[0] == prefix]
    return pd.to_numeric(df[cols[0]], errors="coerce") if cols else pd.Series(np.nan, index=df.index)


def read_system(raw_dir: Path, sid: int, years=YEARS) -> pd.DataFrame:
    files = sorted((Path(raw_dir) / f"system_{sid}").glob(f"system_{sid}__date_*.csv"))
    files = [p for p in files if int(p.stem.split("_date_")[1][:4]) in years and p.stat().st_size > 0]
    frames = [pd.read_csv(p) for p in files]
    frames = [f for f in frames if "measured_on" in f.columns]
    raw = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(raw["measured_on"]).dt.round("15min").dt.tz_localize(TZ).dt.tz_convert("UTC")
    raw = raw.set_index(t).drop(columns="measured_on")
    return raw[~raw.index.duplicated()].sort_index()


def kw_factor(raw: pd.Series, poa: pd.Series, jump: float = 10.0) -> np.ndarray:
    """0.1 while the channel reports hectowatts, 0.001 once it reports watts (switch found in the data)."""
    ratio = (raw.where(raw > 0) / poa.where(poa > 50)).to_numpy()
    day = np.isfinite(ratio)
    looks_watt = ratio[day] > jump * np.nanmedian(ratio)
    factor = np.full(len(raw), 0.1)
    if looks_watt.any():
        tail = np.cumsum(looks_watt[::-1])[::-1] / np.arange(len(looks_watt), 0, -1)
        k = np.flatnonzero(looks_watt & (tail >= 0.95))
        if len(k):
            factor[np.flatnonzero(day)[k[0]]:] = 0.001
    return factor


def to_long(raw_dir: Path, years=YEARS) -> pd.DataFrame:
    """The PVDAQ campus in the tool's long format: one plant row (met station) and five inverters."""
    systems = {sid: read_system(raw_dir, sid, years) for sid in sorted({u[0] for u in UNITS.values()})}
    start = pd.Timestamp(f"{min(years)}-01-01", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(f"{max(years) + 1}-01-01", tz=TZ).tz_convert("UTC")
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    poa = pd.Series(np.nan, index=grid)
    tmod = pd.Series(np.nan, index=grid)
    for sid in MET_PRIORITY:
        df = systems[sid].reindex(grid)
        poa = poa.fillna(_column(df, "poa_irradiance"))
        tf = _column(df, "module_temp_f")
        tmod = tmod.fillna(((tf - 32) * 5 / 9).where((tf != 0) & (tf > -40)))
    poa = poa.clip(lower=0)
    blocks = [pd.DataFrame({"timestamp": grid, "device_id": PLANT, "parent_id": "", "level": "plant",
                            "p_ac": np.nan, "i_dc": np.nan, "v_dc": np.nan, "poa": poa.to_numpy(),
                            "t_mod": tmod.to_numpy(), "status": np.nan})]
    for unit, (sid, p_col, i_col, v_col, e_col) in UNITS.items():
        df = systems[sid].reindex(grid)
        raw = _column(df, p_col)
        p = (raw * kw_factor(raw, poa)).clip(lower=0)
        blocks.append(pd.DataFrame({"timestamp": grid, "device_id": unit, "parent_id": PLANT, "level": "inverter",
                                    "p_ac": p.to_numpy(), "i_dc": _column(df, i_col).clip(lower=0).to_numpy(),
                                    "v_dc": _column(df, v_col).clip(lower=0).to_numpy(), "poa": np.nan,
                                    "t_mod": np.nan, "status": _column(df, e_col).to_numpy()}))
    out = pd.concat(blocks, ignore_index=True)
    for c in ("p_ac", "i_dc", "v_dc", "poa", "t_mod", "status"):
        out[c] = out[c].astype("float32")
    return out


def run(data_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(data_dir) / "pvdaq"
    if not any(raw_dir.glob("system_*/*.csv")):
        fetch(raw_dir)
    long_path = Path(data_dir) / "pvdaq_long.parquet"
    if not long_path.exists():
        to_long(raw_dir).to_parquet(long_path, index=False)
    panel = schema.to_panel(pd.read_parquet(long_path), meta=META)
    th = pipeline.load_thresholds(Path(out_dir) / "thresholds.json")
    theta = th["theta"]
    f = features.compute(panel, calibration="hour_month")
    models = None
    mp = Path(data_dir) / "models.pkl"
    if mp.exists():  # written locally by `benchmark`
        with open(mp, "rb") as fh:
            models = pickle.load(fh)
    lv = panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    years_obs = {panel.devices.index[j]: float(f.light[:, j].sum() * panel.step_h / 4380.0) for j in inv}

    # the plain outages: inverter at zero for at least 2 daylight hours in sun while a neighbour produces
    plain = _plain_outages(f)
    dev_years = sum(years_obs.values())

    def summarise(det, method, theta_, min_h):
        det = det.copy()
        det["plain"] = [_overlaps(r, plain) for r in det.itertuples()]
        found = sum(_overlaps(p, det) for p in plain.itertuples())
        inv_ev = det[det["level"] == "inverter"]
        return det, {"method": method, "theta": theta_, "min_hours": min_h, "alarms": len(det),
                     "alarms_inverter": len(inv_ev), "alarms_plant": int((det["level"] == "plant").sum()),
                     "alarms_per_device_year": len(inv_ev) / dev_years,
                     "other_per_device_year": int((~inv_ev["plain"]).sum()) / dev_years,
                     "plain_outages": len(plain), "plain_found": found,
                     "alarm_hours_per_device_year": float(inv_ev["hours"].sum()) / dev_years}

    rows, sweep, scores = [], [], {}
    for m in ["threshold", "expected", "peer", "hampel", "cusum", "iforest", "gbm", "hybrid"]:
        if m == "gbm" and models is None:
            continue
        if m == "hybrid":
            fl = pipeline.hybrid_flags(f, theta)
        else:
            scores[m] = detectors.score(m, f, model=models["step_gbm"] if models else None)
            fl = detectors.flags(m, scores[m], f, theta.get(m, {}))
        det, cf = ev.detect_events(fl, f)
        rows.append(summarise(ev.estimate_loss(det, f), m, None, 0.5)[1])
    # how the alarm count falls with the threshold and the shortest event kept (inverter level)
    inv_only = (lv == "inverter")[None, :]
    for m in ("expected", "peer", "hampel", "cusum", "iforest"):
        for th_ in detectors.GRIDS[m]:
            fl = detectors.flags(m, scores[m], f, {"inverter": th_}) & inv_only
            for min_h in (0.5, 2.0):
                det, _ = ev.detect_events(fl, f, min_steps=round(min_h / panel.step_h))
                sweep.append(summarise(det, m, th_, min_h)[1])
    res = pd.DataFrame(rows)
    sw = pd.DataFrame(sweep)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "real_alarms.csv", index=False)
    sw.to_csv(out_dir / "real_sweep.csv", index=False)
    plain.drop(columns=["j"]).to_csv(out_dir / "real_plain_outages.csv", index=False)

    # site set-up: the lowest peer threshold that finds every plain outage at <= 1 other alarm per inverter-month
    # with events of at least 2 h (no fault labels are used; this is how a site's alarm rule would be tuned)
    ok = sw[(sw["method"] == "peer") & (sw["min_hours"] == 2.0) & (sw["other_per_device_year"] <= 12)
            & (sw["plain_found"] == len(plain))]
    site_theta = float(ok["theta"].min()) if len(ok) else 0.25
    site = {"theta": {**theta, "peer": {**theta.get("peer", {}), "inverter": site_theta}},
            "rules_precision": th.get("rules_precision", {})}
    (out_dir / "real_site_thresholds.json").write_text(json.dumps({"peer_inverter": site_theta, "min_hours": 2.0}))

    # the tool end to end on the real fleet, with the site set-up
    full = pipeline.analyse(panel, site, calibration="hour_month", min_hours=2.0, f=f)
    evs = full["events"]
    cols = [c for c in pipeline.EVENT_COLUMNS if c in evs.columns]
    evs[cols].to_csv(out_dir / "real_events.csv", index=False)
    per_year, calc = [], {}
    for y in YEARS:
        sub = _year_slice(panel, y)
        r = pipeline.analyse(sub, site, calibration="hour_month", min_hours=2.0)
        k = r["kpi"]
        per_year.append({"year": y, "availability_time": k["availability_time"],
                         "availability_energy": k["availability_energy"], "energy_mwh": k["energy_kwh"] / 1000,
                         **{f"lost_{c}_mwh": v / 1000 for c, v in k["lost_by_category_kwh"].items()},
                         "events": len(r["events"])})
        r["per_inverter"].assign(year=y).to_csv(out_dir / f"real_kpi_inverters_{y}.csv", index=False)
        calc[str(y)] = {"tool": kpi.calculator_data(r["events"], r["features"])}
    (out_dir / "real_calculator.json").write_text(json.dumps(calc, indent=1))
    pd.DataFrame(per_year).to_csv(out_dir / "real_kpi_by_year.csv", index=False)
    _daily(f, out_dir)
    log.info("PVDAQ: %d plain outages, alarms per device-year %s, site peer threshold %.2f", len(plain),
             dict(zip(res["method"], res["alarms_per_device_year"].round(1))), site_theta)
    return {"alarms": res, "plain": plain, "events": evs, "sweep": sw, "site_theta": site_theta}


def _year_slice(panel: schema.Panel, year: int) -> schema.Panel:
    m = panel.time.tz_convert(TZ).year == year
    return schema.Panel(time=panel.time[m], devices=panel.devices, data={k: v[m] for k, v in panel.data.items()},
                        step_h=panel.step_h, meta=panel.meta)


def _plain_outages(f: features.Features, min_hours: float = 2.0) -> pd.DataFrame:
    """Inverter at zero with POA > 300 W/m2 while at least one other inverter produces, >= ``min_hours``."""
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    sun = np.nan_to_num(f.poa[:, inv]) > 300
    y = f.y[:, inv]
    producing = np.nan_to_num(y) > 0.05 * np.nanmax(y, axis=0)
    zero = sun & np.isfinite(y) & (np.nan_to_num(y) <= 0.01 * np.nanmax(y, axis=0))
    other = np.array([(producing & ~np.eye(len(inv), dtype=bool)[k][None, :]).any(axis=1) for k in range(len(inv))]).T
    fl = np.zeros(f.light.shape, bool)
    fl[:, inv] = zero & other
    out = ev.extract(fl, f, max_gap=40, min_steps=int(min_hours / f.panel.step_h))
    out["status_codes"] = [
        ",".join(sorted({str(int(c)) for c in f.panel.data["status"][r.i0:r.i1 + 1, r.j] if np.isfinite(c) and c}))
        for r in out.itertuples()]
    return out


def _overlaps(r, df: pd.DataFrame) -> bool:
    if not len(df):
        return False
    same = df[df["j"] == r.j] if "j" in df else df
    return bool(((same["i0"] <= r.i1) & (same["i1"] >= r.i0)).any())


def _daily(f: features.Features, out_dir: Path):
    """Daily energy per inverter and its peer reference (for the dashboard)."""
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    day = f.panel.time.tz_convert(TZ).normalize().tz_localize(None)
    act = pd.DataFrame(np.nan_to_num(f.y[:, inv]) * f.panel.step_h, index=day).groupby(level=0).sum()
    ref = pd.DataFrame(np.where(f.light[:, inv], np.nan_to_num(f.ref_peer[:, inv]), 0) * f.panel.step_h,
                       index=day).groupby(level=0).sum()
    act.columns = ref.columns = list(f.panel.devices.index[inv])
    act.to_csv(out_dir / "real_daily_actual.csv")
    ref.to_csv(out_dir / "real_daily_peer.csv")
    (out_dir / "real_meta.json").write_text(json.dumps({"units": list(UNITS), "years": YEARS}))
