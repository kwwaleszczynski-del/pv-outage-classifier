"""Cause of an event: signature rules against a gradient boosting classifier on the same signature.

The signature of an event is computed over its daylight steps on its device:

| feature | what it captures |
|---|---|
| level, span_days, hours | where and how long |
| zero_share | output at zero (trip, open string, grid, set-point 0) |
| depth, depth_exp, trend | how deep against the siblings / the expected model; slow onset (soiling) |
| rv, rvself | voltage vs siblings (stuck MPPT, bypass diodes), vs its own normal (shed power, open circuit) |
| twin_share | the other string on the same MPPT input is low too |
| recur, low_sun_share, day_cover | the loss comes back with the sun in the same place and covers only part of
the day (obstruction), or covers the whole day (a defect) |
| frac_low, plant_low | siblings / the whole plant low at the same time |
| tmod | median module temperature during the event (snow keeps modules near 0 degC) |
| sensor_dev, sensor_shape | irradiance sensor vs clear sky on sunny steps: day level, dip within the day |
| status class, missing_share, frozen_share | status code meaning (if a map is given), data quality |
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_outage.features import Features
from pv_outage.taxonomy import CATEGORY, CLASSES

EVENT_FEATURES = ["lvl", "span_days", "hours", "zero_share", "depth", "depth_exp", "trend", "rv", "rvself",
                  "twin_share", "recur", "low_sun_share", "day_cover", "frac_low", "plant_low", "tmod", "sensor_dev",
                  "sensor_shape",
                  "status_share", "status_limited", "status_grid", "status_fault", "missing_share", "frozen_share",
                  "near_pac_share"]
LEVEL_CODE = {"plant": 0, "inverter": 1, "string": 2}


def _plant_col(f: Features) -> int | None:
    lv = f.panel.devices["level"].to_numpy()
    p = np.flatnonzero(lv == "plant")
    return int(p[0]) if len(p) else None


def _day_id(f: Features) -> np.ndarray:
    """Day number of every step (cached on the features)."""
    if "day_id" not in f.info:
        t = f.panel.time
        f.info["day_id"] = np.asarray((t.normalize() - t[0].normalize()).days)
    return f.info["day_id"]


def signature(f: Features, j: int, steps: np.ndarray, i0: int, i1: int, status_map: dict | None = None) -> dict:
    """Signature of one event on device j; ``steps`` are its flagged step indices, [i0, i1] its span.

    For a plant-wide event the voltage, zero, status and data-quality features are taken over the plant's
    inverters (the meter row has no voltage or status).
    """
    lv = f.panel.devices["level"].iloc[j]
    kids = np.flatnonzero((f.panel.devices["parent"].to_numpy() == j)
                          & (f.panel.devices["level"].to_numpy() == "inverter")) if lv == "plant" else np.array([j])
    if not len(kids):
        kids = np.array([j])
    span = np.arange(i0, i1 + 1)
    light_span = span[f.light[span, j]]
    st = steps[f.light[steps, j]] if len(steps) else steps
    if not len(st):
        st = light_span
    ok = st[~f.missing[st, j] & ~f.frozen[st, j]]
    base = np.where(np.isfinite(f.r_peer[:, j]), f.r_peer[:, j], f.r_exp[:, j])
    if lv == "plant" and not np.isfinite(f.y[:, j]).any():  # no meter: the median of the inverters
        with np.errstate(all="ignore"):
            base = np.nanmedian(f.r_exp[:, kids].astype(float), axis=1)
    plant = _plant_col(f)

    def med(x):
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else 0.0

    depth_series = -base[ok] if len(ok) else np.array([])
    if len(ok) >= 6:
        third = max(1, len(ok) // 3)
        trend = med(-base[ok[-third:]]) - med(-base[ok[:third]])
    else:
        trend = 0.0
    twin = 0.0
    g = f.mppt_group[j]
    if g >= 0 and len(ok):
        mates = [k for k in np.flatnonzero(f.mppt_group == g) if k != j]
        if mates:
            twin = float(np.mean(np.nanmean(f.r_peer[np.ix_(ok, mates)], axis=1) < -0.03))
    plant_low = med(f.frac_low[ok, plant]) if plant is not None and len(ok) else 0.0
    sd_win = np.arange(max(0, i0 - 96), min(len(f.sensor_dev), i1 + 97))
    sd = f.sensor_dev[sd_win]
    ss = f.sensor_shape[sd_win]
    status = f.panel.data["status"][np.ix_(st, kids)].ravel() if len(st) else np.array([])
    smap = {int(k): v for k, v in (status_map or {}).items()}
    s_cls = [smap.get(int(c)) for c in status[np.isfinite(status) & (status != 0)]]
    n_st = max(len(status), 1)
    if len(st):  # share of the sunny daylight steps of the affected days that the event covers
        day_id = _day_id(f)
        on_days = np.isin(day_id, np.unique(day_id[st])) & f.light[:, j]
        sunny = on_days & (np.nan_to_num(f.kc, nan=1.0) > 0.6)
        day_cover = float(np.isin(st, np.flatnonzero(sunny)).sum() / max(sunny.sum(), 1))
    else:
        day_cover = 0.0

    def kid_med(x):
        return med(x[np.ix_(st, kids)].astype(float).ravel()) if len(st) else 0.0
    span_light = max(len(light_span), 1)
    return {
        "lvl": LEVEL_CODE[lv], "span_days": (i1 - i0 + 1) * f.panel.step_h / 24, "hours": len(st) * f.panel.step_h,
        "zero_share": float(f.zero[np.ix_(st, kids)].mean()) if len(st) else 0.0,
        "depth": med(depth_series), "depth_exp": med(-f.r_exp[ok, j].astype(float)) if len(ok) else 0.0,
        "trend": trend, "rv": kid_med(f.r_v), "rvself": kid_med(f.r_vself), "twin_share": twin,
        "recur": med(f.recur[ok, j].astype(float)) if len(ok) else 0.0,
        "low_sun_share": float(np.mean(np.nan_to_num(f.sun_el[st], nan=45) < 20)) if len(st) else 0.0,
        "day_cover": day_cover,
        "frac_low": med(f.frac_low[ok, j].astype(float)) if len(ok) else 0.0, "plant_low": plant_low,
        "tmod": med(f.tmod[st, j].astype(float)) if len(st) else 20.0,
        "sensor_dev": med(sd) if np.isfinite(sd).sum() >= 4 else 0.0,
        "sensor_shape": float(np.nanmin(ss)) if np.isfinite(ss).sum() >= 4 else 0.0,
        "status_share": float(f.status_bad[np.ix_(st, kids)].mean()) if len(st) else 0.0,
        "status_limited": s_cls.count("curtailment") / n_st, "status_grid": s_cls.count("grid_outage") / n_st,
        "status_fault": s_cls.count("inverter_trip") / n_st,
        "missing_share": float(f.missing[np.ix_(light_span, kids)].mean()) if len(light_span) else 0.0,
        "frozen_share": float(f.frozen[np.ix_(light_span, kids)].sum() / (span_light * len(kids))),
        "near_pac_share": float(f.near_pac[st, j].mean()) if len(st) else 0.0,
    }


def detected_table(events: pd.DataFrame, f: Features, flags: np.ndarray, status_map=None) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        steps = np.flatnonzero(flags[r.i0:r.i1 + 1, r.j]) + r.i0
        rows.append(signature(f, r.j, steps, r.i0, r.i1, status_map))
    return pd.DataFrame(rows, columns=EVENT_FEATURES, index=events.index)


def oracle_table(truth: pd.DataFrame, f: Features, labels: np.ndarray, status_map=None) -> pd.DataFrame:
    """Signatures over the true effect steps of each truth event (classification with perfect segmentation)."""
    pos = {d: i for i, d in enumerate(f.panel.devices.index)}
    time = f.panel.time
    rows, idx = [], []
    for tid, t in truth.iterrows():
        j = pos[t["device_id"]]
        i0 = time.get_indexer([t["start"]], method="bfill")[0]
        i1 = time.get_indexer([t["end"] - pd.Timedelta(hours=f.panel.step_h)], method="ffill")[0]
        ci = CLASSES.index(t["cls"])
        steps = np.flatnonzero(labels[i0:i1 + 1, j] == ci) + i0
        rows.append(signature(f, j, steps, i0, i1, status_map))
        idx.append(tid)
    return pd.DataFrame(rows, columns=EVENT_FEATURES, index=idx)


# ----------------------------------------------------------------------------------------------- rules
def rule(e: dict | pd.Series) -> str:
    """Signature rules, in order (thresholds set on the training plants only)."""
    if e["missing_share"] >= 0.5:
        return "comm_loss"
    if e["frozen_share"] >= 0.5:
        return "frozen_data"
    # a status code with a known meaning settles it
    for key, cls in (("status_grid", "grid_outage"), ("status_limited", "curtailment"),
                     ("status_fault", "inverter_trip")):
        if e[key] >= 0.5 and not (cls == "inverter_trip" and e["lvl"] == 0):
            return cls
    snowy = e["tmod"] <= 3.0
    lvl = e["lvl"]
    if lvl == 0:  # plant-wide
        if (abs(e["sensor_dev"]) >= 0.15 or e["sensor_shape"] <= -0.2) and e["zero_share"] < 0.3 and not snowy:
            return "sensor_fault"
        if e["zero_share"] >= 0.6:
            return "snow" if snowy and e["rvself"] < 0.08 else "grid_outage"
        if e["rvself"] >= 0.03:  # inverters pushed above their MPP voltage: they shed power on purpose
            return "curtailment"
        if e["tmod"] <= 5.0 or (e["depth"] >= 0.1 and e["tmod"] <= 10.0):
            return "snow"
        return "soiling"
    if lvl == 1:  # inverter
        if snowy and e["plant_low"] >= 0.3:
            return "snow"
        if e["zero_share"] >= 0.6:
            return "inverter_trip"
        if e["plant_low"] >= 0.6 and e["rvself"] >= 0.03:
            return "curtailment"
        if e["day_cover"] < 0.4 and e["span_days"] < 1.5 and e["depth"] >= 0.15:
            return "partial_shading"  # deep but only part of the day (an inverter without string monitoring)
        return "soiling" if e["depth"] < 0.3 else "inverter_trip"
    # string
    if snowy and e["frac_low"] >= 0.3 and e["plant_low"] >= 0.3:
        return "snow"
    if e["zero_share"] >= 0.6:
        return "string_open"
    if e["day_cover"] < 0.4 and (e["low_sun_share"] >= 0.5 or e["recur"] >= 0.2):
        return "partial_shading"  # part of the day only, with the sun low or in the same place
    if abs(e["rv"]) >= 0.1:  # the MPPT input runs far from the other inputs' voltage
        return "mppt_failure"
    return "bypass_short"


def classify_rules(table: pd.DataFrame) -> pd.Series:
    return table.apply(rule, axis=1)


def train_event_gbm(table: pd.DataFrame, y: pd.Series, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    cls, cnt = np.unique(y, return_counts=True)
    w = y.map({c: float(np.clip(np.median(cnt) / n, 0.5, 5.0)) for c, n in zip(cls, cnt)}).to_numpy()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                           min_samples_leaf=5, l2_regularization=1.0, random_state=seed)
    model.fit(table[EVENT_FEATURES].to_numpy(dtype=float), y.to_numpy(), sample_weight=w)
    return model


def classify_gbm(model, table: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(table[EVENT_FEATURES].to_numpy(dtype=float)), index=table.index)


def confusion(true: pd.Series, pred: pd.Series, classes=CLASSES) -> pd.DataFrame:
    cm = pd.crosstab(pd.Categorical(true, classes), pd.Categorical(pred, classes), dropna=False)
    cm.index.name, cm.columns.name = "true", "predicted"
    return cm


def category(cls: str) -> str:
    return CATEGORY.get(cls, "unknown")
