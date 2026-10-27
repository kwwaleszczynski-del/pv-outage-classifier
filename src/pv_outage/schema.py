"""The input format of the tool: one long table of measurements, and its wide in-memory form.

Long format (Parquet or CSV), one row per device and timestamp:

| column | type | meaning |
|---|---|---|
| timestamp | datetime, UTC | start of the averaging interval (regular step, e.g. 5, 10 or 15 min) |
| device_id | string | unique device name |
| parent_id | string or empty | device one level up (string -> inverter -> plant) |
| level | string | ``plant``, ``inverter`` or ``string`` |
| p_ac | float, kW | AC power (inverters, plant meter); empty for strings |
| i_dc | float, A | DC current (string current, inverter input current) |
| v_dc | float, V | DC voltage (MPPT input voltage for strings, input voltage for inverters) |
| poa | float, W/m2 | plane-of-array irradiance (usually only on the plant row) |
| t_mod | float, degC | module temperature (usually only on the plant row) |
| status | integer, optional | inverter status code, 0 = normal |

Missing values are allowed everywhere. A device without its own irradiance or module temperature uses the nearest
ancestor's. Strings of one MPPT input usually report the same voltage; the tool groups them by that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["plant", "inverter", "string"]
CHANNELS = ["p_ac", "i_dc", "v_dc", "poa", "t_mod", "status"]
REQUIRED = ["timestamp", "device_id", "parent_id", "level"]


@dataclass
class Panel:
    """Wide form: arrays of shape (time, device), devices ordered plant -> inverter -> string."""

    time: pd.DatetimeIndex
    devices: pd.DataFrame  # index device_id; columns parent_id, level, parent (int position or -1)
    data: dict[str, np.ndarray]
    step_h: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.devices)

    def level_idx(self, level: str) -> np.ndarray:
        return np.flatnonzero(self.devices["level"].to_numpy() == level)

    def children(self, j: int) -> np.ndarray:
        return np.flatnonzero(self.devices["parent"].to_numpy() == j)

    def siblings(self, j: int) -> np.ndarray:
        p = self.devices["parent"].to_numpy()
        lv = self.devices["level"].to_numpy()
        return np.flatnonzero((p == p[j]) & (lv == lv[j]) & (np.arange(self.n) != j))

    def ancestors(self, j: int) -> list[int]:
        out, p = [], self.devices["parent"].to_numpy()
        while p[j] >= 0:
            j = int(p[j])
            out.append(j)
        return out

    def inherited(self, channel: str) -> np.ndarray:
        """Channel with gaps filled from the nearest ancestor (irradiance, module temperature)."""
        x = self.data[channel].copy()
        p = self.devices["parent"].to_numpy()
        order = np.argsort([len(self.ancestors(j)) for j in range(self.n)], kind="stable")
        for j in order:
            if p[j] >= 0:
                miss = ~np.isfinite(x[:, j])
                x[miss, j] = x[miss, p[j]]
        return x


def read_long(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Problems that stop the analysis (empty list = fine)."""
    problems = [f"missing column {c}" for c in REQUIRED if c not in df.columns]
    if problems:
        return problems
    bad = sorted(set(df["level"].dropna().unique()) - set(LEVELS))
    if bad:
        problems.append(f"unknown level(s) {bad}; use {LEVELS}")
    if not any(c in df.columns for c in ("p_ac", "i_dc")):
        problems.append("need at least one of p_ac, i_dc")
    lv = df.drop_duplicates("device_id").set_index("device_id")
    if lv.index.has_duplicates:
        problems.append("device_id not unique")
    parents = set(lv["parent_id"].dropna().astype(str)) - {""}
    unknown = parents - set(lv.index.astype(str))
    if unknown:
        problems.append(f"parent_id not found as device_id: {sorted(unknown)[:5]}")
    return problems


def to_panel(df: pd.DataFrame, meta: dict | None = None) -> Panel:
    """Long table -> Panel on a regular time grid (missing timestamps become NaN)."""
    problems = validate(df)
    if problems:
        raise ValueError("; ".join(problems))
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["device_id"] = df["device_id"].astype(str)
    df["parent_id"] = df["parent_id"].where(df["parent_id"].notna(), "").astype(str).replace({"nan": "", "None": ""})
    dev = df.drop_duplicates("device_id")[["device_id", "parent_id", "level"]].set_index("device_id")
    dev["rank"] = dev["level"].map({lv: i for i, lv in enumerate(LEVELS)})
    dev = dev.sort_values(["rank"], kind="stable")
    # within a level keep parents' order, then device name
    pos_parent = {d: i for i, d in enumerate(dev.index)}
    dev["porder"] = dev["parent_id"].map(lambda p: pos_parent.get(p, -1))
    dev = dev.reset_index().sort_values(["rank", "porder", "device_id"], kind="stable").set_index("device_id")
    pos = {d: i for i, d in enumerate(dev.index)}
    dev["parent"] = dev["parent_id"].map(lambda p: pos.get(p, -1)).astype(int)
    dev = dev[["parent_id", "level", "parent"]]

    times = pd.DatetimeIndex(df["timestamp"].unique()).sort_values()
    # the most common spacing (ties: the shortest), so gaps in the data do not stretch the grid
    diffs = pd.Series(times).diff().dropna()
    step = diffs.value_counts().sort_index().idxmax() if len(diffs) else pd.Timedelta("15min")
    grid = pd.date_range(times[0], times[-1], freq=step)
    ti = grid.get_indexer(df["timestamp"])
    if (ti < 0).any():  # irregular stamps: snap to the grid
        ti = grid.get_indexer(df["timestamp"], method="nearest")
    di = dev.index.get_indexer(df["device_id"])
    data = {}
    for ch in CHANNELS:
        arr = np.full((len(grid), len(dev)), np.nan)
        if ch in df.columns:
            arr[ti, di] = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
        data[ch] = arr
    return Panel(time=grid, devices=dev, data=data, step_h=step / pd.Timedelta("1h"), meta=dict(meta or {}))


def from_panel(panel: Panel) -> pd.DataFrame:
    """Panel -> long table (rows where all channels are missing are kept, so gaps stay visible)."""
    t, n = len(panel.time), panel.n
    out = pd.DataFrame({
        "timestamp": np.repeat(panel.time.to_numpy(), n),
        "device_id": np.tile(panel.devices.index.to_numpy(), t),
        "parent_id": np.tile(panel.devices["parent_id"].to_numpy(), t),
        "level": np.tile(panel.devices["level"].to_numpy(), t),
    })
    for ch in CHANNELS:
        out[ch] = panel.data[ch].reshape(-1).astype("float32")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out
