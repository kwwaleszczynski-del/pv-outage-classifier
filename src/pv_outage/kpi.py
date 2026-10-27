"""Availability, lost energy by cause and performance ratio, in the spirit of IEC 61724-1 and IEC TS 63019.

Both standards leave room for contracts: which events count against the plant, which are excluded (outside the
operator's responsibility), how data gaps are treated. The treatment of each class is therefore a parameter:

- ``unavailable``: the affected capacity counts as down (time) and its lost energy counts against the plant;
- ``excluded``: the period is taken out of the denominator (external causes such as grid outages and
  curtailment, unknown state during data gaps);
- ``available``: the plant counts as available; the loss shows up in the performance ratio, not in availability.

Time-based availability is weighted by capacity: an inverter outage takes 1/N of the plant, an open string
1/(N x strings per inverter), and only daylight steps (POA above 50 W/m2) count. Energy-based availability is
1 - lost energy of unavailable classes / potential energy outside excluded periods.

The exact definitions of the standards (information categories of IEC TS 63019, the irradiance threshold, the
handling of partial capability) are summarised here, not quoted, and need checking against the current editions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_outage.features import Features
from pv_outage.taxonomy import CATEGORY, CLASSES

DEFAULT_TREATMENT = {
    "inverter_trip": "unavailable", "string_open": "unavailable",
    "bypass_short": "available", "mppt_failure": "available",
    "partial_shading": "available", "soiling": "available", "snow": "available",
    "curtailment": "excluded", "grid_outage": "excluded",
    "comm_loss": "excluded", "frozen_data": "excluded", "sensor_fault": "available",
}
OUTAGE_CLASSES = {"inverter_trip", "string_open", "grid_outage", "curtailment", "snow", "comm_loss", "frozen_data"}


def unit_masks(events: pd.DataFrame, f: Features) -> dict:
    """Per class: (time, inverter) share of each inverter's capacity affected, over each event's span.

    ``events`` needs device_id, i0, i1, cls. Plant events cover every inverter, string events 1/n of theirs.
    """
    lv = f.panel.devices["level"].to_numpy()
    parent = f.panel.devices["parent"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    pos_inv = {j: k for k, j in enumerate(inv)}
    n_str = {j: max(1, int(((parent == j) & (lv == "string")).sum())) for j in inv}
    T = len(f.panel.time)
    masks = {}
    for r in events.itertuples():
        m = masks.setdefault(r.cls, np.zeros((T, len(inv))))
        sl = slice(int(r.i0), int(r.i1) + 1)
        if lv[r.j] == "plant":
            m[sl, :] = 1.0
        elif lv[r.j] == "inverter":
            m[sl, pos_inv[r.j]] = 1.0
        else:
            k = pos_inv.get(parent[r.j])
            if k is not None:
                m[sl, k] = np.minimum(1.0, m[sl, k] + 1.0 / n_str[parent[r.j]])
    return masks


def availability(events: pd.DataFrame, f: Features, treatment: dict | None = None,
                 loss_col: str = "loss_kwh") -> dict:
    """Plant KPIs from classified events (columns device_id, j, i0, i1, cls and a lost-energy column)."""
    treatment = {**DEFAULT_TREATMENT, **(treatment or {})}
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    light = f.light[:, inv] if len(inv) else f.light
    w = np.nanmedian(np.where(np.isfinite(f.k[:, inv]), f.k[:, inv], np.nan), axis=0) if len(inv) else np.ones(1)
    w = np.nan_to_num(w / np.nansum(w), nan=1.0 / max(len(inv), 1))
    masks = unit_masks(events, f) if len(events) else {}
    T = len(f.panel.time)
    down = np.zeros((T, len(inv)))
    excl = np.zeros((T, len(inv)))
    for cls, m in masks.items():
        tr = treatment.get(cls, "available")
        if tr == "unavailable" and cls in OUTAGE_CLASSES:
            down = np.maximum(down, m)
        elif tr == "excluded":
            excl = np.maximum(excl, m)
    down = np.where(excl > 0, 0.0, down)
    lw = light * w[None, :]
    denom = float((lw * (1 - excl)).sum())
    a_time = 1 - float((lw * down).sum()) / denom if denom > 0 else np.nan

    e_act = _energy(f)
    lost = events.groupby("cls")[loss_col].sum() if len(events) else pd.Series(dtype=float)
    lost = lost.reindex(CLASSES).fillna(0.0)
    lost_unav = sum(lost[c] for c in CLASSES if treatment.get(c) == "unavailable")
    lost_excl = sum(lost[c] for c in CLASSES if treatment.get(c) == "excluded")
    potential = e_act + float(lost.sum())
    a_energy = 1 - lost_unav / (potential - lost_excl) if potential - lost_excl > 0 else np.nan
    by_cat = {cat: float(sum(lost[c] for c in CLASSES if CATEGORY[c] == cat)) for cat in set(CATEGORY.values())}
    pr = performance_ratio(f, e_act, float(lost.sum()),
                           float(sum(lost[c] for c in CLASSES if CATEGORY[c] == "external")))
    return {"availability_time": a_time, "availability_energy": a_energy, "energy_kwh": e_act,
            "potential_kwh": potential, "lost_by_class_kwh": lost.to_dict(), "lost_by_category_kwh": by_cat, **pr}


def _energy(f: Features) -> float:
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    if len(inv):
        return float(np.nansum(f.panel.data["p_ac"][:, inv]) * f.panel.step_h)
    plant = np.flatnonzero(lv == "plant")
    return float(np.nansum(f.panel.data["p_ac"][:, plant]) * f.panel.step_h)


def performance_ratio(f: Features, e_ac: float, lost_all: float, lost_ext: float) -> dict:
    """PR = E_AC / (P_STC x H_POA / 1 kW/m2); also with external losses and with all identified losses added back."""
    pdc = (f.panel.meta.get("pdc_kw") or {})
    lv = f.panel.devices["level"].to_numpy()
    ids = f.panel.devices.index
    total = sum(float(pdc[d]) for d, l in zip(ids, lv) if l == "inverter" and d in pdc)
    if not total:
        return {"pr": np.nan, "pr_ext_corrected": np.nan, "pr_potential": np.nan}
    plant = np.flatnonzero(lv == "plant")
    poa = f.panel.data["poa"][:, plant[0]] if len(plant) else np.nanmedian(f.poa, axis=1)
    h = float(np.nansum(np.clip(poa, 0, None)) * f.panel.step_h / 1000.0)
    ref = total * h
    return {"pr": e_ac / ref, "pr_ext_corrected": (e_ac + lost_ext) / ref, "pr_potential": (e_ac + lost_all) / ref}


def truth_events(truth: pd.DataFrame, f: Features) -> pd.DataFrame:
    """Truth table in the event format used here (span from the first effect to the end)."""
    pos = {d: i for i, d in enumerate(f.panel.devices.index)}
    time = f.panel.time
    rows = []
    for t in truth.itertuples():
        start = t.first_effect if pd.notna(t.first_effect) else t.start
        i0 = time.get_indexer([start], method="bfill")[0]
        i1 = time.get_indexer([t.end - pd.Timedelta(hours=f.panel.step_h)], method="ffill")[0]
        for d in str(t.devices).split(";"):
            rows.append({"device_id": d, "j": pos[d], "i0": i0, "i1": i1, "cls": t.cls,
                         "loss_kwh": t.true_loss_kwh / len(str(t.devices).split(";"))})
    return pd.DataFrame(rows)


def calculator_data(events: pd.DataFrame, f: Features, loss_col: str = "loss_kwh") -> dict:
    """Per-class totals for the dashboard's availability calculator (classes are added up, overlaps ignored).

    ``base_h``: capacity-weighted daylight hours of the plant; per class ``down_h`` (capacity-weighted daylight
    hours affected) and ``lost_kwh``; ``energy_kwh`` the measured energy.
    """
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    light = f.light[:, inv]
    w = np.nanmedian(np.where(np.isfinite(f.k[:, inv]), f.k[:, inv], np.nan), axis=0)
    w = np.nan_to_num(w / np.nansum(w), nan=1.0 / max(len(inv), 1))
    masks = unit_masks(events, f) if len(events) else {}
    step_h = f.panel.step_h
    lost = events.groupby("cls")[loss_col].sum() if len(events) else pd.Series(dtype=float)
    classes = {}
    for c in CLASSES:
        m = masks.get(c)
        down = float((m * light * w[None, :]).sum() * step_h) if m is not None else 0.0
        classes[c] = {"down_h": round(down, 3), "lost_kwh": round(float(lost.get(c, 0.0)), 1),
                      "outage": c in OUTAGE_CLASSES}
    return {"base_h": round(float((light * w[None, :]).sum() * step_h), 2), "energy_kwh": round(_energy(f), 1),
            "classes": classes}
