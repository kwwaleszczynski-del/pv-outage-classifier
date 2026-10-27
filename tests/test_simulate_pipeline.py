import json

import numpy as np
import pandas as pd

from pv_outage import cli, real, schema, weather
from pv_outage.taxonomy import CATEGORY, CLASSES

from conftest import fake_era5


def test_downscale_keeps_hourly_energy():
    raw = fake_era5(days=10)
    wx = weather.downscale(raw, 51.9, 19.5, 30, 180, seed=1, year=2023)
    hourly = wx["ghi"].resample("1h", label="right", closed="left").mean()
    common = raw.index.intersection(hourly.index)
    assert np.allclose(hourly.loc[common].to_numpy(), raw.loc[common, "shortwave_radiation"].to_numpy(), atol=1.0)
    assert (wx["poa"] >= 0).all() and wx.index.freq is not None or len(wx) == 10 * 96


def test_simulation_truth_and_format(small_sim):
    sim = small_sim
    assert schema.validate(sim.long) == []
    assert set(sim.truth["cls"]) <= set(CLASSES)
    assert (sim.truth["category"] == sim.truth["cls"].map(CATEGORY)).all()
    # layers only remove energy
    e = [p.sum() for p in sim.layers_p]
    assert all(a >= b - 1e-6 for a, b in zip(e, e[1:]))
    loss = sim.truth[sim.truth["category"] != "data"]
    assert (loss["true_loss_kwh"] >= 0).all() and (loss["effect_hours"] >= 0.25).all()
    # labels point at events that exist and only inside their windows
    ids = set(np.unique(sim.label_event[sim.label_event >= 0]))
    assert ids <= set(sim.truth["event_id"])
    assert sim.labels.shape == (len(sim.wx), len(sim.device_ids))


def test_inverter_trip_shows_open_circuit(small_sim):
    sim = small_sim
    trips = sim.truth[sim.truth["cls"] == "inverter_trip"]
    assert len(trips)
    t = trips.iloc[0]
    inv = sim.long[(sim.long["device_id"] == t["device_id"]) & (sim.long["timestamp"] >= t["start"])
                   & (sim.long["timestamp"] < t["end"])]
    sunny = inv["timestamp"].isin(sim.wx.index[sim.wx["poa"] > 300])
    if sunny.any():
        assert (inv.loc[sunny, "p_ac"] == 0).all()


def test_pipeline_on_simulated_plant(small_sim, tmp_path):
    sim = small_sim
    path = tmp_path / "plant.parquet"
    sim.long.to_parquet(path, index=False)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({**sim.meta, "status_map": {"1": "curtailment", "2": "grid_outage",
                                                           "3": "inverter_trip"}}))
    out = tmp_path / "out"
    cli.main(["run", "--input", str(path), "--meta", str(meta), "--out", str(out),
              "--thresholds", str(tmp_path / "none.json")])
    ev = pd.read_csv(out / "events.csv")
    assert len(ev) > 0 and set(ev["cls"]) <= set(CLASSES)
    k = json.loads((out / "kpi.json").read_text())
    assert 0.5 < k["availability_time"] <= 1.0 and 0.5 < k["availability_energy"] <= 1.0
    per = pd.read_csv(out / "kpi_per_inverter.csv")
    assert len(per) == 4
    # every long outage of the truth is covered by some event on its inverter
    trips = sim.truth[(sim.truth["cls"] == "inverter_trip") & (sim.truth["effect_hours"] >= 3)]
    for t in trips.itertuples():
        hit = ev[(ev["device_id"] == t.device_id) & (pd.to_datetime(ev["start"]) < t.end)
                 & (pd.to_datetime(ev["end"]) > t.start)]
        assert len(hit), t


def test_pvdaq_unit_switch():
    poa = pd.Series(np.tile(np.r_[np.zeros(40), np.full(56, 800.0)], 20))
    raw = pd.Series(np.where(poa > 0, 1000.0, 0.0))  # 100 kW in hectowatts
    raw.iloc[1536:] = np.where(poa.iloc[1536:] > 0, 100000.0, 0.0)  # the same 100 kW in watts (last 20 %)
    f = real.kw_factor(raw, poa)
    kw = raw * f
    assert np.allclose(kw[poa > 0], 100.0)
