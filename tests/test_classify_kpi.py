import numpy as np
import pandas as pd
import pytest

from pv_outage import classify, kpi
from pv_outage.taxonomy import CLASSES

BASE = {c: 0.0 for c in classify.EVENT_FEATURES}
BASE.update(tmod=20.0, span_days=0.5)


def sig(**kw):
    return {**BASE, **kw}


@pytest.mark.parametrize("e, expected", [
    (sig(missing_share=1.0), "comm_loss"),
    (sig(frozen_share=0.9), "frozen_data"),
    (sig(lvl=0, zero_share=1.0, rvself=0.2), "grid_outage"),
    (sig(lvl=0, zero_share=1.0, status_limited=1.0), "curtailment"),
    (sig(lvl=0, depth=0.4, rvself=0.08), "curtailment"),
    (sig(lvl=0, depth=0.5, tmod=0.5), "snow"),
    (sig(lvl=0, depth=0.06, span_days=12), "soiling"),
    (sig(lvl=0, depth=0.05, sensor_shape=-0.5), "sensor_fault"),
    (sig(lvl=1, zero_share=0.95), "inverter_trip"),
    (sig(lvl=1, depth=0.08, span_days=6), "soiling"),
    (sig(lvl=2, zero_share=0.9), "string_open"),
    (sig(lvl=2, depth=0.2, rv=-0.25), "mppt_failure"),
    (sig(lvl=2, depth=0.08, recur=0.5), "partial_shading"),
    (sig(lvl=2, depth=0.05, span_days=20), "bypass_short"),
    (sig(lvl=2, depth=0.9, tmod=0.0, frac_low=0.5, plant_low=0.5), "snow"),
])
def test_rules(e, expected):
    assert classify.rule(e) == expected


def test_confusion_has_all_classes():
    cm = classify.confusion(pd.Series(["snow", "soiling"]), pd.Series(["snow", "snow"]))
    assert list(cm.index) == CLASSES and cm.loc["soiling", "snow"] == 1


class _Panel:
    def __init__(self, T, levels, parents, ids):
        self.time = pd.date_range("2024-06-01", periods=T, freq="15min", tz="UTC")
        self.devices = pd.DataFrame({"level": levels, "parent": parents, "parent_id": ""}, index=ids)
        self.step_h = 0.25
        self.meta = {"pdc_kw": {"I1": 100.0, "I2": 100.0}}
        self.data = {"p_ac": np.zeros((T, len(ids))), "poa": np.zeros((T, len(ids)))}


class _F:
    pass


def fake_features(T=96):
    f = _F()
    ids = ["P", "I1", "I2", "I1-S1", "I1-S2", "I2-S1", "I2-S2"]
    f.panel = _Panel(T, ["plant", "inverter", "inverter", "string", "string", "string", "string"],
                     [-1, 0, 0, 1, 1, 2, 2], ids)
    f.light = np.zeros((T, len(ids)), bool)
    f.light[24:72] = True  # 12 daylight hours
    f.k = np.ones((T, len(ids)))
    f.poa = np.where(f.light, 800.0, 0.0)
    f.panel.data["p_ac"][24:72, 1:3] = 50.0
    f.panel.data["poa"][24:72, 0] = 800.0
    return f


def test_availability_inverter_and_string_outages():
    f = fake_features()
    events = pd.DataFrame([
        {"device_id": "I1", "j": 1, "i0": 24, "i1": 47, "cls": "inverter_trip", "loss_kwh": 300.0},  # 6 h
        {"device_id": "I2-S1", "j": 5, "i0": 24, "i1": 71, "cls": "string_open", "loss_kwh": 150.0},  # 12 h, 1/2
        {"device_id": "P", "j": 0, "i0": 60, "i1": 71, "cls": "grid_outage", "loss_kwh": 100.0},  # 3 h excluded
    ])
    k = kpi.availability(events, f)
    # weights 1/2 each; denominator 2 x 9 h; down: I1 6 h (minus 0 inside the grid outage) + I2 0.5 x 9 h
    expected = 1 - (0.5 * 6 + 0.5 * 0.5 * 9) / (0.5 * 9 + 0.5 * 9)
    assert k["availability_time"] == pytest.approx(expected)
    e_act = 2 * 50.0 * 12
    assert k["availability_energy"] == pytest.approx(1 - 450 / (e_act + 550 - 100))
    assert k["lost_by_category_kwh"]["external"] == pytest.approx(100.0)
    # the grid outage counted against the plant instead
    k2 = kpi.availability(events, f, treatment={"grid_outage": "unavailable"})
    assert k2["availability_time"] < k["availability_time"]
