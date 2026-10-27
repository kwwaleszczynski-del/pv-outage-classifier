"""The tool on any plant in the long format: data quality, detection, consolidation, cause, lost energy, KPIs.

Usage:
    panel = schema.to_panel(schema.read_long(Path("plant.parquet")), meta=json.loads(Path("meta.json").read_text()))
    result = analyse(panel)
    result["events"].to_csv("events.csv"); print(result["kpi"])
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv_outage import classify, detectors, events as ev, features, kpi
from pv_outage.schema import Panel
from pv_outage.taxonomy import CATEGORY

# thresholds calibrated on the synthetic training plants; results/thresholds.json (written by the benchmark)
# overrides them, and also carries how often each rule was right on the training events (the confidence)
THETA = {"peer": {"inverter": 0.06, "string": 0.06}, "expected": {"plant": 0.1}}
THRESHOLDS_FILE = Path(__file__).resolve().parents[2] / "results" / "thresholds.json"
EVENT_COLUMNS = ["start", "end", "device_id", "level", "cls", "category", "loss_kwh", "loss_peer_kwh",
                 "loss_exp_kwh", "hours", "confidence", "source"]


def load_thresholds(path: Path | None = None) -> dict:
    """{"theta": {...}, "rules_precision": {...}} from the benchmark, or the built-in defaults."""
    path = Path(path) if path else THRESHOLDS_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {"theta": THETA, "rules_precision": {}}


def hybrid_flags(f: features.Features, theta: dict) -> np.ndarray:
    """Peer comparison for inverters and strings, the expected model for the plant, zero / status everywhere."""
    lv = f.panel.devices["level"].to_numpy()
    fl = np.zeros(f.light.shape, bool)
    for m, levels in (("peer", ("inverter", "string")), ("expected", ("plant",)),
                      ("threshold", ("plant", "inverter", "string"))):
        s = detectors.score(m, f)
        fl |= detectors.flags(m, s, f, theta.get(m, {})) & np.isin(lv, levels)[None, :]
    return fl


def analyse(panel: Panel, thresholds: dict | None = None, calibration: str = "poa", event_model=None,
            treatment: dict | None = None, min_hours: float = 0.5, f: features.Features | None = None) -> dict:
    """``min_hours``: shortest event kept (flagged daylight time); ``f``: features already computed."""
    thresholds = thresholds or load_thresholds()
    theta = thresholds.get("theta", THETA)
    f = f if f is not None else features.compute(panel, calibration=calibration)
    det, cflags = ev.detect_events(hybrid_flags(f, theta), f, min_steps=max(1, round(min_hours / panel.step_h)))
    det = ev.estimate_loss(det, f)
    status_map = panel.meta.get("status_map")
    table = classify.detected_table(det, f, cflags, status_map)
    if len(det):
        det["cls"] = (classify.classify_gbm(event_model, table) if event_model is not None
                      else classify.classify_rules(table)).to_numpy()
        det["confidence"] = _confidence(table, det["cls"], event_model, thresholds.get("rules_precision", {}))
    det["source"] = "detector"
    dq, _ = ev.dq_events(f)
    if len(dq):
        dq["loss_kwh"] = dq["loss_peer_kwh"] = dq["loss_exp_kwh"] = 0.0
        dq["confidence"] = 1.0
        dq["source"] = "data_quality"
    events = pd.concat([det, dq], ignore_index=True) if len(dq) else det
    if not len(events):
        events = pd.DataFrame(columns=EVENT_COLUMNS + ["j", "i0", "i1"])
    events["category"] = events["cls"].map(CATEGORY)
    events = events.sort_values(["start", "device_id"]).reset_index(drop=True)
    k = kpi.availability(events, f, treatment)
    return {"events": events, "kpi": k, "per_inverter": per_inverter(events, f, treatment), "features": f}


def _confidence(table: pd.DataFrame, cls: pd.Series, event_model, rules_precision: dict) -> np.ndarray:
    """Class probability of the event model; for the rules, how often that rule was right on training events."""
    if event_model is not None:
        p = event_model.predict_proba(table[classify.EVENT_FEATURES].to_numpy(dtype=float))
        return np.round(p.max(axis=1), 2)
    return np.array([round(float(rules_precision.get(c, np.nan)), 2) for c in cls])


def per_inverter(events: pd.DataFrame, f: features.Features, treatment: dict | None = None) -> pd.DataFrame:
    """Time availability, lost energy by category and energy per inverter."""
    treatment = {**kpi.DEFAULT_TREATMENT, **(treatment or {})}
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    masks = kpi.unit_masks(events, f) if len(events) else {}
    T = len(f.panel.time)
    down = np.zeros((T, len(inv)))
    excl = np.zeros((T, len(inv)))
    for cls, m in masks.items():
        tr = treatment.get(cls, "available")
        if tr == "unavailable" and cls in kpi.OUTAGE_CLASSES:
            down = np.maximum(down, m)
        elif tr == "excluded":
            excl = np.maximum(excl, m)
    down = np.where(excl > 0, 0, down)
    light = f.light[:, inv]
    rows = []
    parent = f.panel.devices["parent"].to_numpy()
    for k, j in enumerate(inv):
        denom = float((light[:, k] * (1 - excl[:, k])).sum())
        mine = events[(events["j"] == j) | events["j"].map(lambda x: parent[int(x)] == j)] if len(events) else events
        lost = mine.groupby("category")["loss_kwh"].sum() if len(mine) else pd.Series(dtype=float)
        rows.append({"device_id": f.panel.devices.index[j],
                     "availability_time": 1 - float((light[:, k] * down[:, k]).sum()) / denom if denom else np.nan,
                     "energy_kwh": float(np.nansum(f.panel.data["p_ac"][:, j]) * f.panel.step_h),
                     **{f"lost_{c}_kwh": float(lost.get(c, 0.0)) for c in ("fault", "environment", "external")},
                     "events": int(len(mine))})
    return pd.DataFrame(rows)
