"""Real, labelled shading: a 4 kW rooftop with two twin inverters on La Réunion (one shaded on purpose, one not).

Source: the "PV Array Fault Dataset" on Zenodo (record 23114394, CC BY 4.0), which aligns the measurements of
Graillet, A. et al. (2022), "Electrical production data of a domestic grid-connected rooftop PV plant in normal and
shading faults conditions associated with solar and meteorological data in a tropical climate", Zenodo,
https://doi.org/10.5281/zenodo.7828879 (CC BY 4.0), into 60-second windows with irradiance >= 120 W/m2. Only the
real records are used here (classes ``normal`` and ``shading_real``); the record's injected soiling and hotspot
classes are left out.

The minutes are averaged to 15-minute steps (at least 10 minutes present), so the thresholds tuned on the
simulated plants apply unchanged. A step is shaded when at least half of its minutes are. Each shaded day is one
truth event on inverter 1. Only inverter-level flags are scored (the question is whether a method points at the
shaded inverter and stays quiet otherwise); a sweep over each method's thresholds shows the trade-off between
events found and false alarms on this site.

Usage:
    python -m pv_outage shading
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pv_outage import classify, detectors, events as ev, features, pipeline, schema

log = logging.getLogger(__name__)

FILE = "final_pv_fault_dataset.csv"
RECORD_URL = "https://zenodo.org/records/23114394/files/final_pv_fault_dataset.csv?download=1"
META = {"lat": -20.90126, "lon": 55.48422}
METHODS = ["threshold", "expected", "peer", "hampel", "cusum", "iforest", "gbm", "hybrid"]


def load(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """15-minute long table (plant + two inverters) and the shaded share of each step on inverter 1."""
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["Provenance"] == "real"]
    t = pd.to_datetime(raw["time"], utc=True)
    df = pd.DataFrame({"poa": raw["GTI"].to_numpy(), "t_mod": raw["TPV"].to_numpy(),
                       "p1": raw["Pg_1"].to_numpy() / 1000, "i1": raw["Ia_1"].to_numpy(), "v1": raw["Va_1"].to_numpy(),
                       "p2": raw["Pg_2"].to_numpy() / 1000, "i2": raw["Ia_2"].to_numpy(), "v2": raw["Va_2"].to_numpy(),
                       "shaded": (raw["Class"] == "shading_real").to_numpy().astype(float)},
                      index=pd.DatetimeIndex(t))
    g = df.resample("15min")
    q = g.mean()
    q = q[g["poa"].count() >= 10]
    grid = pd.date_range(q.index[0].floor("D"), q.index[-1].ceil("D"), freq="15min", inclusive="left")
    q = q.reindex(grid)
    blocks = [pd.DataFrame({"timestamp": grid, "device_id": "PLANT", "parent_id": "", "level": "plant",
                            "p_ac": q["p1"].to_numpy() + q["p2"].to_numpy(), "poa": q["poa"].to_numpy(),
                            "t_mod": q["t_mod"].to_numpy()})]
    for k in (1, 2):
        blocks.append(pd.DataFrame({"timestamp": grid, "device_id": f"INV{k}", "parent_id": "PLANT",
                                    "level": "inverter", "p_ac": q[f"p{k}"].to_numpy(), "i_dc": q[f"i{k}"].to_numpy(),
                                    "v_dc": q[f"v{k}"].to_numpy()}))
    long = pd.concat(blocks, ignore_index=True)
    for c in ("p_ac", "i_dc", "v_dc", "poa", "t_mod"):
        if c not in long:
            long[c] = np.nan
    long["status"] = np.nan
    return long, q["shaded"]


def run(data_dir: Path, out_dir: Path, models: dict | None = None) -> dict | None:
    path = Path(data_dir) / "zenodo" / FILE
    if not path.exists():
        log.warning("%s missing (69 MB, CC BY 4.0, %s); skipping the real shading test", path, RECORD_URL)
        return None
    long, shaded = load(path)
    panel = schema.to_panel(long, meta=META)
    f = features.compute(panel)
    th = pipeline.load_thresholds(Path(out_dir) / "thresholds.json")
    theta = th["theta"]
    j1 = list(panel.devices.index).index("INV1")
    lab = np.full(f.light.shape, -1)
    sh = (shaded.reindex(panel.time).to_numpy() >= 0.5)
    day = panel.time.normalize()
    days = pd.Index(day[sh]).unique()
    for k, d in enumerate(days):
        lab[sh & (day == d), j1] = k
    truth = pd.DataFrame({"cls": "partial_shading", "true_loss_kwh": np.nan}, index=pd.Index(range(len(days)),
                                                                                          name="event_id"))
    effects = ev.effect_steps(lab)
    sunny_days = int(pd.Index(day[f.light[:, j1]]).nunique())
    inv_only = (panel.devices["level"].to_numpy() == "inverter")[None, :]
    inv_years = 2 * sunny_days / 365.25

    def evaluate(fl):
        det, cf = ev.detect_events(fl & inv_only, f)
        det = ev.match(ev.estimate_loss(det, f), cf, lab, f)
        return det, cf, ev.score(det, cf, truth, f, effects)

    # threshold sweep: events found against false alarms per inverter-year
    sweep, scores = [], {}
    for m in ("expected", "peer", "hampel", "cusum", "iforest"):
        scores[m] = detectors.score(m, f)
        for th_ in detectors.GRIDS[m]:
            det, cf, sc = evaluate(detectors.flags(m, scores[m], f, {"inverter": th_}))
            sweep.append({"method": m, "theta": th_, "found": sc["found"],
                          "fa_per_inverter_year": int((det["truth_id"] < 0).sum()) / inv_years,
                          "calibrated": bool(np.isclose(th_, theta.get(m, {}).get("inverter", np.nan)))})
    pd.DataFrame(sweep).to_csv(Path(out_dir) / "shading_real_sweep.csv", index=False)

    rows, hits_cls = [], {}
    for m in METHODS:
        if m == "gbm" and not models:
            continue
        if m == "hybrid":
            fl = pipeline.hybrid_flags(f, theta)
        else:
            s = scores[m] if m in scores else detectors.score(m, f, model=(models or {}).get("step_gbm"))
            fl = detectors.flags(m, s, f, theta.get(m, {}))
        det, cf, sc = evaluate(fl)
        fa = det[det["truth_id"] < 0]
        # step level on inverter 1: share of shaded steps flagged, share of unshaded steps flagged
        on = cf[:, j1] & f.light[:, j1]
        valid = f.light[:, j1] & np.isfinite(f.y[:, j1])
        rows.append({"method": m, "events_found": sc["found"], "events": sc["n_truth"],
                     "precision": sc["precision"], "detections": sc["n_det"],
                     "false_alarms": len(fa), "false_alarms_per_inverter_year": len(fa) / inv_years,
                     "step_recall": float(on[sh & valid].mean()) if (sh & valid).any() else np.nan,
                     "step_false_rate": float(on[~sh & valid].mean()),
                     "median_delay_h": float(sc["per_truth"]["delay_h"].median()) if sc["found"] else np.nan})
        if m == "hybrid" and len(det):
            table = classify.detected_table(det, f, cf)
            det["cls"] = classify.classify_rules(table).to_numpy()
            det["recur"] = table["recur"].to_numpy()
            hits_cls = {"labelled": det[det["truth_id"] >= 0]["cls"].value_counts().to_dict(),
                        "unlabelled": det[det["truth_id"] < 0].groupby(["device_id", "cls"]).size()
                        .rename("n").reset_index().to_dict(orient="records")}
            det.assign(start=det["start"].astype(str), end=det["end"].astype(str))[
                ["device_id", "start", "end", "hours", "loss_kwh", "truth_id", "cls", "recur"]].to_csv(
                out_dir / "shading_real_events.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(Path(out_dir) / "shading_real.csv", index=False)
    (Path(out_dir) / "shading_real_classes.json").write_text(json.dumps(hits_cls, indent=1))
    # daily ratio inverter 1 / inverter 2 for the dashboard
    y = pd.DataFrame(f.y[:, 1:3], index=panel.time, columns=["INV1", "INV2"])
    dd = y.groupby(y.index.normalize()).sum(min_count=8)
    ratio = (dd["INV1"] / dd["INV2"]).dropna()
    shaded_day = pd.Series(sh, index=panel.time).groupby(panel.time.normalize()).any().reindex(ratio.index)
    pd.DataFrame({"ratio": ratio.round(4), "shaded": shaded_day.astype(int)}).to_csv(
        Path(out_dir) / "shading_real_daily.csv")
    log.info("real shading: %d shaded days; %s", len(days), dict(zip(res["method"], res["events_found"])))
    return {"results": res, "classes": hits_cls, "days": len(days)}
