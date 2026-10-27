"""Features, detectors, consolidation, events and matching on a small hand-made plant."""
import numpy as np
import pandas as pd
import pytest

from pv_outage import detectors, events as ev, features, schema

DAYS = 12
N_STR = 6


def hand_plant(seed: int = 0):
    """Plant P, inverters I1/I2 with six strings each (pairs share an MPPT voltage), 12 days of 15-min data.

    Injected: I1-S3 open on days 3-5, I2 at zero on day 7 from 10:00 to 14:00, I1 (and its strings) frozen on
    day 9 from 11:00 to 15:00, I2-S5 at 90 % from day 8 on.
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-05-01", periods=DAYS * 96, freq="15min", tz="UTC")
    hour = t.hour + t.minute / 60
    poa = np.clip(1000 * np.sin(np.pi * (hour - 5) / 14), 0, None) * (1 + 0.02 * rng.standard_normal(len(t)))
    poa = np.clip(poa, 0, None)
    day = np.asarray((t - t[0]).days)
    rows = []
    p_inv = {}
    for i in (1, 2):
        strings = {}
        for s in range(1, N_STR + 1):
            cur = 9.0 * poa / 1000 * (1 + 0.005 * rng.standard_normal(len(t)))
            if i == 1 and s == 3:
                cur = np.where((day >= 3) & (day <= 5), 0.0, cur)
            if i == 2 and s == 5:
                cur = np.where(day >= 8, 0.9 * cur, cur)
            strings[s] = cur
        v_mppt = {m: np.where(poa > 5, 700 + 0.002 * rng.standard_normal(len(t)), 0.0) for m in range(3)}
        off = (i == 2) & (day == 7) & (hour >= 10) & (hour < 14)
        for s, cur in strings.items():
            cur = np.where(off, 0.0, cur)
            v = np.where(off, 820.0, v_mppt[(s - 1) // 2])
            strings[s] = (cur, v)
        p = sum(c * v for c, v in strings.values()) / 1000 * 0.98
        p_inv[i] = p
        for s, (cur, v) in strings.items():
            rows.append(pd.DataFrame({"timestamp": t, "device_id": f"I{i}-S{s}", "parent_id": f"I{i}",
                                      "level": "string", "i_dc": cur, "v_dc": v}))
        rows.append(pd.DataFrame({"timestamp": t, "device_id": f"I{i}", "parent_id": "P", "level": "inverter",
                                  "p_ac": p, "i_dc": sum(c for c, _ in strings.values()),
                                  "v_dc": np.mean([v for _, v in strings.values()], axis=0), "status": 0.0}))
    rows.append(pd.DataFrame({"timestamp": t, "device_id": "P", "parent_id": "", "level": "plant",
                              "p_ac": (p_inv[1] + p_inv[2]) * 0.99, "poa": poa, "t_mod": 25 + poa / 40}))
    df = pd.concat(rows, ignore_index=True)
    frz = ((df["timestamp"].dt.day == 10) & (df["timestamp"].dt.hour.between(11, 14))
           & df["device_id"].str.startswith("I1"))
    for dev, g in df[frz].groupby("device_id"):
        first = g.index[0]
        for c in ("p_ac", "i_dc", "v_dc"):
            df.loc[g.index, c] = df.loc[first, c]
    return schema.to_panel(df)


@pytest.fixture(scope="module")
def feats():
    return features.compute(hand_plant(), calibration="scalar")


def col(f, name):
    return list(f.panel.devices.index).index(name)


def test_peer_residual_and_groups(feats):
    f = feats
    j = col(f, "I2-S5")
    day = np.asarray((f.panel.time - f.panel.time[0]).days)
    later = f.light[:, j] & (day >= 8) & (np.nan_to_num(f.poa[:, j]) > 300)
    # -10 % for a third of the period: the median calibration absorbs part of it, the loss stays visible
    assert -0.11 < np.nanmedian(f.r_peer[later, j]) < -0.06
    # pairs of strings with identical voltage readings form MPPT groups
    g = f.mppt_group
    assert g[col(f, "I1-S1")] == g[col(f, "I1-S2")] != g[col(f, "I1-S3")]


def test_frozen_and_zero(feats):
    f = feats
    j = col(f, "I1")
    assert f.frozen[:, j].sum() >= 8
    k = col(f, "I1-S3")
    assert f.zero[:, k].sum() > 50


def test_threshold_and_consolidation(feats):
    f = feats
    s = detectors.score("threshold", f)
    fl = detectors.flags("threshold", s, f, {})
    det, cf = ev.detect_events(fl, f)
    # the inverter outage: strings roll up into one inverter event
    inv_events = det[det["device_id"] == "I2"]
    assert len(inv_events) == 1
    assert inv_events["hours"].iloc[0] == pytest.approx(4.0, abs=0.5)
    strings = det[det["device_id"].str.startswith("I2-S")]
    assert not len(strings) or strings["start"].min() > inv_events["end"].iloc[0]
    # the open string: one event across nights
    s3 = det[det["device_id"] == "I1-S3"]
    assert len(s3) == 1 and s3["hours"].iloc[0] > 20


def test_peer_and_cusum_find_the_small_loss(feats):
    f = feats
    j = col(f, "I2-S5")
    fl = detectors.flags("peer", detectors.score("peer", f), f, {"string": 0.06, "inverter": 0.06})
    assert fl[:, j].sum() > 100
    u = detectors.score("cusum", f)
    cf = detectors.flags("cusum", u, f, {"string": 20.0, "inverter": 20.0, "plant": 20.0})
    first = np.flatnonzero(cf[:, j])
    assert len(first) and (f.panel.time[first[0]] - f.panel.time[0]).days >= 7  # not before the change


def test_cusum_restarts_and_stops():
    u = np.zeros((300, 1))
    u[100:150] = 3.0
    fl = detectors.cusum_flags(u, h=10.0)
    hit = np.flatnonzero(fl[:, 0])
    assert hit.min() >= 100 and hit.max() < 155


def test_dq_events_and_loss(feats):
    f = feats
    dq, _ = ev.dq_events(f)
    assert "frozen_data" in set(dq["cls"])
    det, cf = ev.detect_events(detectors.flags("threshold", detectors.score("threshold", f), f, {}), f)
    det = ev.estimate_loss(det, f)
    row = det[det["device_id"] == "I2"].iloc[0]
    # 4 h of a ~50 kW inverter around noon: roughly 150-200 kWh
    assert 100 < row["loss_kwh"] < 260


def test_match_and_score(feats):
    f = feats
    det, cf = ev.detect_events(detectors.flags("threshold", detectors.score("threshold", f), f, {}), f)
    label = np.full(f.light.shape, -1)
    j = col(f, "I2")
    t7 = np.flatnonzero((np.asarray((f.panel.time - f.panel.time[0]).days) == 7) & (f.panel.time.hour >= 10)
                        & (f.panel.time.hour < 14))
    label[t7, j] = 0
    m = ev.match(det, cf, label, f)
    assert (m.loc[m["device_id"] == "I2", "truth_id"] == 0).all()
    truth = pd.DataFrame({"cls": ["inverter_trip"], "true_loss_kwh": [180.0]}, index=pd.Index([0], name="event_id"))
    sc = ev.score(ev.estimate_loss(m, f), cf, truth, f, ev.effect_steps(label))
    assert sc["recall"] == 1.0
    assert sc["per_truth"]["delay_h"].iloc[0] == 0.0
