"""Real weather for the simulated plant: hourly ERA5 from the Open-Meteo archive, downscaled to 15 minutes.

ERA5 gives hourly means of global, direct and diffuse irradiance (over the preceding hour), 2 m temperature, 10 m
wind, precipitation, snowfall and snow depth on a ~31 km grid. Hourly area means are smoother than what a plant
sees, so the global irradiance is downscaled to 15 minutes with a stochastic clear-sky index that keeps every
hour's energy exactly:

1. clear-sky index of the hour, k_h = GHI_h / GHI_cs,h (Ineichen clear sky, Linke turbidity climatology);
2. linear interpolation of k_h between hour midpoints plus an AR(1) fluctuation whose amplitude peaks at
   intermediate k (broken clouds) and vanishes for clear or overcast hours;
3. the four 15-minute values of each hour are rescaled to the hour's ERA5 mean.

Erbs decomposition and the Perez transposition then give plane-of-array irradiance. Temperature and wind are
interpolated linearly, snowfall and precipitation are spread evenly over the hour.

Usage:
    raw = fetch_era5(51.9, 19.5, 2021, 2024, Path("data/era5"))
    wx = downscale(raw, 51.9, 19.5, tilt=30, azimuth=180, seed=1)
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m",
             "wind_speed_10m", "precipitation", "snowfall", "snow_depth"]
STEP = "15min"


def fetch_era5(lat: float, lon: float, start_year: int, end_year: int, cache_dir: Path) -> pd.DataFrame:
    """Hourly ERA5 for one point (UTC, cached as Parquet). Radiation is the mean over the preceding hour."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"era5_{lat:.2f}_{lon:.2f}_{start_year}_{end_year}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31",
        "hourly": ",".join(VARIABLES), "models": "era5", "timezone": "UTC", "wind_speed_unit": "ms"})
    log.info("Open-Meteo ERA5 archive: %d variables, %d-%d, one point", len(VARIABLES), start_year, end_year)
    with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=300) as r:
        payload = json.loads(r.read())
    hourly = payload["hourly"]
    df = pd.DataFrame({v: hourly[v] for v in VARIABLES},
                      index=pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="time").tz_localize("UTC"))
    df = df.astype(float)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def downscale(raw: pd.DataFrame, lat: float, lon: float, tilt: float, azimuth: float, seed: int,
              year: int | None = None, sigma_max: float = 0.15, phi: float = 0.6) -> pd.DataFrame:
    """15-minute weather (interval start labels) with POA irradiance on the given plane.

    Columns: ghi, dni, dhi, poa, poa_beam, poa_diffuse, ghi_cs, poa_cs, t_air, wind, precip_mm, snow_cm,
    sun_el, sun_az, kt (clear-sky index of the global irradiance, 15 min).
    """
    import pvlib

    rng = np.random.default_rng(seed)
    hourly = raw.copy()
    if year is not None:
        hourly = hourly[(hourly.index >= pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta("1h"))
                        & (hourly.index <= pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))]
    # an hourly value stamped t covers (t - 1 h, t]; its 15-minute slots start at t-60, t-45, t-30, t-15 min
    start, end = hourly.index[0] - pd.Timedelta("1h"), hourly.index[-1]
    grid = pd.date_range(start, end, freq=STEP, inclusive="left")
    owner = grid.ceil("1h")
    owner = owner.where(owner != grid, grid + pd.Timedelta("1h"))  # slot starting on the hour belongs to hour + 1

    loc = pvlib.location.Location(lat, lon, tz="UTC")
    mid = grid + pd.Timedelta("7.5min")
    solpos = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen", solar_position=solpos)
    ghi_cs = cs["ghi"].to_numpy()

    ghi_h = hourly["shortwave_radiation"].reindex(owner).to_numpy()
    ghi_cs_h = pd.Series(ghi_cs, index=grid).groupby(owner).transform("mean").to_numpy()
    k_h = np.where(ghi_cs_h > 10, np.clip(ghi_h / np.maximum(ghi_cs_h, 1e-9), 0, 1.3), np.nan)

    # smooth k between hour midpoints (hour stamped t has its midpoint at t - 30 min)
    k_hour = pd.Series(k_h, index=owner - pd.Timedelta("30min")).groupby(level=0).first()
    k_smooth = (k_hour.reindex(k_hour.index.union(mid)).interpolate(method="time", limit_area="inside")
                .reindex(mid).to_numpy())
    k_smooth = np.where(np.isfinite(k_smooth), k_smooth, np.nan_to_num(k_h))
    eps = rng.standard_normal(len(grid))
    ar = np.empty(len(grid))
    ar[0] = eps[0]
    for i in range(1, len(grid)):
        ar[i] = phi * ar[i - 1] + np.sqrt(1 - phi ** 2) * eps[i]
    kk = np.clip(k_smooth, 0, 1.2)
    k15 = np.clip(kk + sigma_max * 4 * kk * (1 - np.minimum(kk, 1)) * ar, 0, 1.3)
    ghi = k15 * ghi_cs
    hour_sum = pd.Series(ghi, index=grid).groupby(owner).transform("mean").to_numpy()
    ghi = np.where(hour_sum > 0, ghi * np.nan_to_num(ghi_h) / np.maximum(hour_sum, 1e-9), 0.0)
    ghi = np.where(ghi_cs > 0, ghi, 0.0)

    zen = solpos["apparent_zenith"].to_numpy()
    dni_extra = pvlib.irradiance.get_extra_radiation(mid).to_numpy()
    erbs = pvlib.irradiance.erbs(ghi, zen, mid)
    dni, dhi = np.nan_to_num(erbs["dni"]), np.nan_to_num(erbs["dhi"])
    airmass = pvlib.atmosphere.get_relative_airmass(zen)
    poa = pvlib.irradiance.get_total_irradiance(tilt, azimuth, zen, solpos["azimuth"].to_numpy(), dni, ghi, dhi,
                                                dni_extra=dni_extra, airmass=airmass, model="perez")
    poa_cs = pvlib.irradiance.get_total_irradiance(tilt, azimuth, zen, solpos["azimuth"].to_numpy(),
                                                   cs["dni"].to_numpy(), ghi_cs, cs["dhi"].to_numpy(),
                                                   dni_extra=dni_extra, airmass=airmass, model="perez")

    def interp(col: str) -> np.ndarray:
        s = hourly[col]
        return s.reindex(s.index.union(mid)).interpolate(method="time").reindex(mid).bfill().ffill().to_numpy()

    per_slot = lambda col: hourly[col].reindex(owner).fillna(0).to_numpy() / 4.0  # noqa: E731
    out = pd.DataFrame({
        "ghi": ghi, "dni": dni, "dhi": dhi,
        "poa": np.nan_to_num(np.asarray(poa["poa_global"], dtype=float)).clip(0),
        "poa_beam": np.nan_to_num(np.asarray(poa["poa_direct"], dtype=float)).clip(0),
        "poa_diffuse": np.nan_to_num(np.asarray(poa["poa_diffuse"], dtype=float)).clip(0),
        "ghi_cs": ghi_cs, "poa_cs": np.nan_to_num(np.asarray(poa_cs["poa_global"], dtype=float)).clip(0),
        "t_air": interp("temperature_2m"), "wind": interp("wind_speed_10m"),
        "precip_mm": per_slot("precipitation"), "snow_cm": per_slot("snowfall"),
        "sun_el": 90 - zen, "sun_az": solpos["azimuth"].to_numpy(),
        "kt": np.where(ghi_cs > 20, ghi / np.maximum(ghi_cs, 1e-9), np.nan),
    }, index=pd.DatetimeIndex(grid, name="time"))
    return out
