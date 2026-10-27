"""Shared fixtures: synthetic hourly weather in the ERA5 layout (no network) and small simulated plants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_outage import simulate


def fake_era5(start: str = "2023-03-01", days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Hourly weather with clear, broken and overcast days, a snowfall and some rain (UTC, ERA5 column names)."""
    import pvlib

    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta("1h"), periods=days * 24, freq="1h")
    loc = pvlib.location.Location(51.9, 19.5)
    cs = loc.get_clearsky(idx - pd.Timedelta("30min"))
    regime = rng.choice([1.0, 0.6, 0.25], size=days, p=[0.5, 0.3, 0.2])
    k = np.repeat(regime, 24) * np.clip(1 + 0.15 * rng.standard_normal(len(idx)), 0.5, 1.2)
    k = np.clip(k, 0.05, 1.05)
    ghi = cs["ghi"].to_numpy() * k
    hours = np.arange(len(idx))
    t2m = 5 + 8 * np.sin(2 * np.pi * (hours % 24 - 9) / 24) + 0.05 * hours / 24
    snow = np.zeros(len(idx))
    snow[24 * 3 + 2: 24 * 3 + 8] = 1.0  # 6 cm overnight on day 4
    t2m[24 * 3: 24 * 5] = -3.0
    rain = np.where(rng.random(len(idx)) < 0.02, rng.uniform(1, 6, len(idx)), 0.0)
    return pd.DataFrame({
        "shortwave_radiation": ghi, "direct_normal_irradiance": np.nan, "diffuse_radiation": np.nan,
        "temperature_2m": t2m, "wind_speed_10m": 3.0, "precipitation": rain, "snowfall": snow, "snow_depth": np.nan,
    }, index=pd.DatetimeIndex(idx, name="time"))


def small_config(**kw) -> simulate.PlantConfig:
    base = dict(n_inverters=4, strings_per_inverter=6, n_trip_short=2, n_trip_long=1, n_string_open=2, n_bypass=1,
                n_mppt=1, n_shading=0, n_curtail=2, n_grid=1, n_comm=3, n_frozen=1, n_sensor=0)
    base.update(kw)
    return simulate.PlantConfig(**base)


@pytest.fixture(scope="session")
def small_sim():
    raw = fake_era5()
    return simulate.simulate(raw, year=2023, seed=5, cfg=small_config())
