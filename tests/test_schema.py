import numpy as np
import pandas as pd
import pytest

from pv_outage import schema


def tiny_long(unit: str = "ns") -> pd.DataFrame:
    t = pd.date_range("2024-06-01 08:00", periods=4, freq="15min", tz="UTC").as_unit(unit)
    rows = []
    for ts in t:
        rows.append({"timestamp": ts, "device_id": "P", "parent_id": None, "level": "plant", "p_ac": 50.0,
                     "poa": 800.0, "t_mod": 40.0})
        for i in (1, 2):
            rows.append({"timestamp": ts, "device_id": f"I{i}", "parent_id": "P", "level": "inverter",
                         "p_ac": 25.0, "i_dc": 40.0, "v_dc": 650.0, "status": 0})
            for s in (1, 2):
                rows.append({"timestamp": ts, "device_id": f"I{i}-S{s}", "parent_id": f"I{i}", "level": "string",
                             "i_dc": 20.0, "v_dc": 650.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_roundtrip_independent_of_datetime_resolution(unit):
    df = tiny_long(unit)
    panel = schema.to_panel(df)
    assert panel.n == 7 and len(panel.time) == 4
    assert panel.step_h == pytest.approx(0.25)
    assert list(panel.devices["level"]) == ["plant", "inverter", "inverter", "string", "string", "string", "string"]
    j = list(panel.devices.index).index("I2-S1")
    assert panel.devices.index[panel.devices["parent"].iloc[j]] == "I2"
    back = schema.from_panel(panel)
    assert len(back) == 28
    assert np.nansum(back["p_ac"]) == pytest.approx(4 * (50 + 2 * 25))


def test_missing_timestamps_become_gaps_and_inherit():
    df = tiny_long()
    df = df[df["timestamp"] != df["timestamp"].unique()[2]]
    panel = schema.to_panel(df)
    assert len(panel.time) == 4
    assert np.isnan(panel.data["p_ac"][2]).all()
    poa = panel.inherited("poa")
    assert np.allclose(poa[0], 800.0)  # strings and inverters get the plant's irradiance


def test_validation_messages():
    df = tiny_long()
    assert schema.validate(df) == []
    assert "missing column level" in schema.validate(df.drop(columns="level"))[0]
    bad = df.copy()
    bad.loc[0, "level"] = "combiner"
    assert any("unknown level" in p for p in schema.validate(bad))
    bad = df.copy()
    bad.loc[bad["device_id"] == "I1-S1", "parent_id"] = "nope"
    assert any("parent_id not found" in p for p in schema.validate(bad))
    with pytest.raises(ValueError):
        schema.to_panel(df.drop(columns="device_id"))
