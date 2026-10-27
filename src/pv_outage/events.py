"""From flagged steps to events, and from events to scores against the truth.

1. Consolidation (alarm flood control): when at least 80 % of an inverter's strings are flagged at the same step,
   the inverter gets the flag and the strings lose it; the same for inverters under a plant. A parent flag that is
   already explained by some of its children (a partial loss with one string down) is dropped in favour of the
   more specific children.
2. Events: runs of flagged daylight steps per device. Gaps of up to ``max_gap`` daylight steps (2 h at 15 min)
   are bridged and nights never split an event, so a three-day outage is one event.
3. Matching (simulation only): a detected event is a hit when most of its flagged steps fall on steps where the
   truth says that device was affected; it belongs to the truth event behind most of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_outage.features import Features


def consolidate(flags: np.ndarray, f: Features, share: float = 0.8, data_flags: bool = False) -> np.ndarray:
    """Roll children's simultaneous flags up to the parent (``data_flags``: flags of missing/frozen data)."""
    flags = flags.copy()
    panel = f.panel
    lv = panel.devices["level"].to_numpy()
    parent = panel.devices["parent"].to_numpy()
    has_data = f.light if data_flags else f.light & ~f.missing & ~f.frozen
    for parent_level, child_level in (("inverter", "string"), ("plant", "inverter")):
        for p in np.flatnonzero(lv == parent_level):
            kids = np.flatnonzero((parent == p) & (lv == child_level))
            if not len(kids):
                continue
            n = has_data[:, kids].sum(axis=1)
            k = (flags[:, kids] & has_data[:, kids]).sum(axis=1)
            full = (n > 0) & (k >= share * n)
            some = (k > 0) & ~full
            flags[full, p] = True
            flags[np.ix_(full, kids)] = False
            if not data_flags:
                # a parent at zero explains whatever its children show
                flags[np.ix_(flags[:, p] & f.zero[:, p], kids)] = False
                # partial loss already explained by flagged children (not when the parent itself is at zero)
                flags[some & ~f.zero[:, p], p] = False
    return flags


def extract(flags: np.ndarray, f: Features, max_gap: int = 8, min_steps: int = 2) -> pd.DataFrame:
    """Events (one row each): device, level, first/last flagged step, flagged hours."""
    panel = f.panel
    light_any = f.light.any(axis=1)
    cl = np.cumsum(light_any)
    rows = []
    ids = panel.devices.index.to_numpy()
    lv = panel.devices["level"].to_numpy()
    step = pd.Timedelta(hours=panel.step_h)
    for j in range(panel.n):
        idx = np.flatnonzero(flags[:, j] & f.light[:, j])
        if not len(idx):
            continue
        gaps = cl[idx[1:]] - cl[idx[:-1]] - 1
        cut = np.flatnonzero(gaps > max_gap)
        starts = np.concatenate([[0], cut + 1])
        ends = np.concatenate([cut, [len(idx) - 1]])
        for a, b in zip(starts, ends):
            n = b - a + 1
            if n < min_steps:
                continue
            rows.append((ids[j], lv[j], j, idx[a], idx[b], n))
    df = pd.DataFrame(rows, columns=["device_id", "level", "j", "i0", "i1", "n_flag"])
    df["start"] = panel.time[df["i0"].to_numpy()] if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
    df["end"] = (panel.time[df["i1"].to_numpy()] + step) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
    df["hours"] = df["n_flag"] * panel.step_h
    return df.sort_values(["start", "device_id"]).reset_index(drop=True)


def detect_events(flags: np.ndarray, f: Features, min_steps: int = 2):
    """Consolidated flags and the events extracted from them."""
    c = consolidate(flags, f)
    return extract(c, f, min_steps=min_steps), c


# ----------------------------------------------------------------------------------------------- energy
def estimate_loss(events: pd.DataFrame, f: Features, flags: np.ndarray | None = None) -> pd.DataFrame:
    """Lost energy of each event against the peer reference and the expected model (kWh)."""
    out = events.copy()
    step_h = f.panel.step_h
    lv = f.panel.devices["level"].to_numpy()
    parent = f.panel.devices["parent"].to_numpy()
    no_signal = ~np.isfinite(f.y).any(axis=0)
    est_peer, est_exp, share_peer = [], [], []
    for r in events.itertuples():
        j, a, b = r.j, r.i0, r.i1 + 1
        if no_signal[j]:  # a plant without a meter: add up its inverters against the expected model
            kids = np.flatnonzero((parent == j) & (lv == "inverter"))
            ok_k = f.light[a:b][:, kids] & ~f.missing[a:b][:, kids] & ~f.frozen[a:b][:, kids]
            le = np.where(ok_k, np.clip(f.y_exp[a:b][:, kids] - f.y[a:b][:, kids], 0, None), 0.0)
            est_exp.append(float(le.sum() * step_h))
            est_peer.append(np.nan)
            share_peer.append(0.0)
            continue
        ok = f.light[a:b, j] & ~f.missing[a:b, j] & ~f.frozen[a:b, j]
        y = f.y[a:b, j]
        if f.is_amp[j]:
            v = np.where(np.isfinite(f.v_ref[a:b, j]), f.v_ref[a:b, j], np.nanmedian(f.v_ref[:, j]))
            conv = v / 1000.0 * 0.98
        else:
            conv = np.full(b - a, 0.98 if lv[j] == "string" else 1.0)
        lp = np.where(ok, np.clip(f.ref_peer[a:b, j] - y, 0, None) * conv, np.nan)
        le = np.where(ok, np.clip(f.y_exp[a:b, j] - y, 0, None) * conv, np.nan)
        est_peer.append(np.nansum(lp) * step_h)
        est_exp.append(np.nansum(le) * step_h)
        share_peer.append(float(np.isfinite(lp).sum() / max(ok.sum(), 1)))
    out["loss_peer_kwh"] = est_peer
    out["loss_exp_kwh"] = est_exp
    out["loss_kwh"] = np.where(np.array(share_peer) >= 0.5, est_peer, est_exp)
    return out


# ----------------------------------------------------------------------------------------------- matching
def match(det: pd.DataFrame, flags: np.ndarray, label_event: np.ndarray, f: Features, min_share: float = 0.5,
          allowed: set | None = None) -> pd.DataFrame:
    """Adds ``truth_id`` (-1 = false alarm) and ``truth_share`` to the detections.

    A detection is a hit when at least ``min_share`` of its flagged daylight steps fall on steps where the truth
    says its device was affected; it is assigned to the event behind most of them. ``allowed`` restricts the
    truth events that count (e.g. production losses only, so that alarms caused by a faulty irradiance sensor
    stay false alarms).
    """
    det = det.copy()
    tid = np.full(len(det), -1)
    share = np.zeros(len(det))
    for k, r in enumerate(det.itertuples()):
        st = np.flatnonzero(flags[r.i0:r.i1 + 1, r.j] & f.light[r.i0:r.i1 + 1, r.j]) + r.i0
        if not len(st):
            continue
        ids = label_event[st, r.j].astype(int)
        if allowed is not None:
            ids = np.where(np.isin(ids, list(allowed)), ids, -1)
        hit = ids[ids >= 0]
        share[k] = len(hit) / len(st)
        if share[k] >= min_share:
            vals, cnt = np.unique(hit, return_counts=True)
            tid[k] = vals[np.argmax(cnt)]
    det["truth_id"] = tid
    det["truth_share"] = share
    return det


def effect_steps(label_event: np.ndarray) -> dict:
    """event id -> sorted unique time indices at which any device is affected."""
    t, _ = np.nonzero(label_event >= 0)
    ids = label_event[label_event >= 0]
    df = pd.DataFrame({"id": ids, "t": t}).drop_duplicates().sort_values("t")
    return {int(k): g["t"].to_numpy() for k, g in df.groupby("id")}


def score(det: pd.DataFrame, flags: np.ndarray, truth: pd.DataFrame, f: Features, effects: dict) -> dict:
    """Event-level precision / recall plus delay, duration and energy of the matched truth events.

    ``truth`` is indexed by event id; ``effects`` maps event id -> its affected time steps.
    """
    cl = np.cumsum(f.light.any(axis=1))
    step_h = f.panel.step_h
    n_det = len(det)
    tp_det = int((det["truth_id"] >= 0).sum()) if n_det else 0
    rows = []
    for tid, t in truth.iterrows():
        m = det[det["truth_id"] == tid] if n_det else det
        eff = effects.get(int(tid), np.array([], int))
        row = {"truth_id": tid, "cls": t["cls"], "found": len(m) > 0, "true_hours": len(eff) * step_h,
               "true_kwh": t["true_loss_kwh"]}
        if len(m) and len(eff):
            steps = np.unique(np.concatenate([np.flatnonzero(flags[r.i0:r.i1 + 1, r.j]) + r.i0
                                              for r in m.itertuples()]))
            row["delay_h"] = float(max(0, cl[steps[0]] - cl[eff[0]])) * step_h
            row["det_hours"] = len(steps) * step_h
            row["est_kwh"] = float(m["loss_kwh"].sum()) if "loss_kwh" in m else np.nan
        rows.append(row)
    pt = pd.DataFrame(rows)
    n_truth = len(truth)
    found = int(pt["found"].sum()) if n_truth else 0
    prec = tp_det / n_det if n_det else np.nan
    rec = found / n_truth if n_truth else np.nan
    f1 = 2 * prec * rec / (prec + rec) if n_det and n_truth and (prec + rec) > 0 else 0.0
    return {"n_det": n_det, "tp_det": tp_det, "n_truth": n_truth, "found": found, "precision": prec,
            "recall": rec, "f1": f1, "per_truth": pt}


# ----------------------------------------------------------------------------------------------- data quality
def dq_events(f: Features, day_threshold: float = 0.15, shape_threshold: float = 0.2):
    """Data-quality events, the same for every detector: missing data, frozen values, irradiance sensor.

    Returns the events and the (time, device) flags they were extracted from."""
    out = []
    all_flags = np.zeros(f.light.shape, bool)
    for cls, fl in (("comm_loss", f.missing), ("frozen_data", f.frozen)):
        c = consolidate(fl, f, data_flags=True)
        all_flags |= c
        ev = extract(c, f, min_steps=2)
        ev["cls"] = cls
        out.append(ev)
    plant = np.flatnonzero(f.panel.devices["level"].to_numpy() == "plant")
    if len(plant) and np.isfinite(f.sensor_dev).any():
        fl = np.zeros(f.light.shape, bool)
        fl[:, plant[0]] = ((np.abs(np.nan_to_num(f.sensor_dev)) > day_threshold)
                           | (np.nan_to_num(f.sensor_shape) < -shape_threshold))
        # clear days are the only evidence; bridge up to about five days of cloud
        ev = extract(fl, f, max_gap=int(5 * 24 / f.panel.step_h / 2), min_steps=4)
        all_flags |= fl
        ev["cls"] = "sensor_fault"
        out.append(ev)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return (df.sort_values("start").reset_index(drop=True) if len(df) else df), all_flags
