"""Detectors: one anomaly score per device and step (higher = worse, NaN = not assessable), plus the threshold.

| method | score | needs |
|---|---|---|
| threshold | zero output in daylight, or a non-zero status code | nothing |
| expected | shortfall against the PVWatts expectation from POA and module temperature | irradiance sensor |
| peer | shortfall against the leave-one-out median of the siblings | siblings |
| hampel | robust z-score of the residual against its own trailing 7-day median and MAD | siblings or sensor |
| cusum | one-sided CUSUM of the standardised residual (slow, persistent losses) | siblings or sensor |
| iforest | isolation forest on the step features, fitted on the plant's own data | - |
| gbm | gradient boosting classifier trained on labelled synthetic plants, 1 - P(normal) | training plants |

Hampel and CUSUM use the peer residual where a device has siblings and the expected-model residual otherwise
(the plant meter). Steps with missing or frozen data are never scored: the data-quality stage owns them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_outage.features import Features

METHODS = ["threshold", "expected", "peer", "hampel", "cusum", "iforest", "gbm"]
LEVELS = ["plant", "inverter", "string"]
GRIDS = {
    "expected": [0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6],
    "peer": [0.02, 0.03, 0.04, 0.06, 0.1, 0.15, 0.25, 0.4],
    "hampel": [2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0],
    "cusum": [3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
    "iforest": [0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999],  # quantile of the level's daylight scores
}
DEFAULTS = {"threshold": 0.5, "expected": 0.15, "peer": 0.1, "hampel": 5.0, "cusum": 20.0, "iforest": 0.99,
            "gbm": 0.5}

STEP_FEATURES = ["r_exp", "r_peer", "r_v", "r_vself", "zero", "near_pac", "parent_limited", "status_bad", "recur",
                 "frac_low", "sensor_dev", "sensor_shape", "cold", "sun_el", "kc", "lvl_plant", "lvl_inverter",
                 "lvl_string",
                 "r_peer_day", "r_exp_day", "twin_r_peer", "plant_r_exp"]


def base_residual(f: Features) -> np.ndarray:
    return np.where(np.isfinite(f.r_peer), f.r_peer, f.r_exp).astype(float)


def assessable(f: Features) -> np.ndarray:
    return f.light & ~f.missing & ~f.frozen


def score(method: str, f: Features, model=None) -> np.ndarray:
    ok = assessable(f)
    if method == "threshold":
        return np.where(ok, (f.zero | f.status_bad).astype(float), np.nan)
    if method == "expected":
        return np.where(ok, -f.r_exp, np.nan)
    if method == "peer":
        return np.where(ok, -f.r_peer, np.nan)
    if method == "hampel":
        return np.where(ok, hampel_z(base_residual(f), ok, f.panel.time), np.nan)
    if method == "cusum":  # standardised shortfall; the CUSUM itself runs in ``flags`` (it needs h)
        return np.where(ok, standardise(base_residual(f), ok), np.nan)
    if method == "iforest":
        return np.where(ok, iforest_score(f, ok), np.nan)
    if method == "gbm":
        if model is None:
            raise ValueError("gbm needs a trained model")
        p = model.predict_proba(step_matrix(f))
        normal = list(model.classes_).index(-1)
        return np.where(ok, (1 - p[:, normal]).reshape(ok.shape), np.nan)
    raise ValueError(method)


def flags(method: str, s: np.ndarray, f: Features, theta: dict) -> np.ndarray:
    """Flags from scores with one threshold per level (iforest: a quantile of that level's scores)."""
    lv = f.panel.devices["level"].to_numpy()
    out = np.zeros(s.shape, bool)
    for level in LEVELS:
        cols = np.flatnonzero(lv == level)
        if not len(cols):
            continue
        th = theta.get(level, DEFAULTS[method]) if isinstance(theta, dict) else theta
        sub = s[:, cols]
        if method == "cusum":
            out[:, cols] = cusum_flags(sub, th)
            continue
        if method == "iforest":
            fin = sub[np.isfinite(sub)]
            th = np.quantile(fin, th) if fin.size else np.inf
        with np.errstate(invalid="ignore"):
            out[:, cols] = np.nan_to_num(sub, nan=-np.inf) > th
    return out


# ----------------------------------------------------------------------------------------------- hampel
def hampel_z(r: np.ndarray, ok: np.ndarray, time: pd.DatetimeIndex, window: str = "7D") -> np.ndarray:
    """-(r - median) / (1.4826 MAD) over a trailing window of the device's own assessable steps."""
    rows = ok.any(axis=1)
    df = pd.DataFrame(np.where(ok, r, np.nan)[rows], index=time[rows])
    med = df.rolling(window, min_periods=20).median()
    mad = (df - med).abs().rolling(window, min_periods=20).median()
    z = -(df - med) / (1.4826 * mad + 0.005)
    out = np.full(r.shape, np.nan)
    out[rows] = z.to_numpy()
    return out


# ----------------------------------------------------------------------------------------------- cusum
def standardise(r: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Shortfall in units of each device's robust spread: -(r - median) / (1.4826 MAD)."""
    rr = np.where(ok, r, np.nan)
    med = np.nanmedian(rr, axis=0)
    mad = np.nanmedian(np.abs(rr - med), axis=0)
    return -(rr - med) / (1.4826 * np.nan_to_num(mad, nan=1.0) + 0.003)


def cusum_flags(u: np.ndarray, h: float, kappa: float = 0.5, cap: float = 50.0) -> np.ndarray:
    """Lower CUSUM S = max(0, S + u - kappa) with a restart after each alarm.

    Every step from the start of the excursion (the last S = 0, the change-point estimate) up to an alarm is
    flagged, so a persistent loss is covered end to end and the flags stop as soon as alarms stop recurring.
    """
    T, D = u.shape
    out = np.zeros((T, D), bool)
    rows = np.flatnonzero(np.isfinite(u).any(axis=1))
    s = np.zeros(D)
    since = np.full(D, -1)  # position in ``rows`` where the current excursion started
    for k, t in enumerate(rows):
        ut = u[t]
        fin = np.isfinite(ut)
        s_new = np.maximum(0.0, s + np.clip(np.nan_to_num(ut), -cap, cap) - kappa)
        s = np.where(fin, s_new, s)
        since[fin & (s > 0) & (since < 0)] = k
        since[fin & (s <= 0)] = -1
        alarm = fin & (s > h)
        for j in np.flatnonzero(alarm):
            out[rows[since[j]:k + 1], j] = True
        s[alarm] = 0.0
        since[alarm] = -1
    return out


# ----------------------------------------------------------------------------------------------- isolation forest
def iforest_score(f: Features, ok: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    X = step_matrix(f)
    out = np.full(ok.shape, np.nan)
    lv = f.panel.devices["level"].to_numpy()
    flat_ok = ok.reshape(-1)
    for level in LEVELS:
        colmask = np.broadcast_to(lv == level, ok.shape).reshape(-1)
        rows = np.flatnonzero(flat_ok & colmask)
        if len(rows) < 100:
            continue
        rng = np.random.default_rng(seed)
        fit_rows = rng.choice(rows, size=min(len(rows), 50000), replace=False)
        cols = [STEP_FEATURES.index(c) for c in ("r_exp", "r_peer", "r_v", "r_vself", "zero", "frac_low", "sun_el",
                                                   "kc", "r_peer_day", "r_exp_day")]
        model = IsolationForest(n_estimators=200, random_state=seed).fit(X[fit_rows][:, cols])
        flat = out.reshape(-1)
        flat[rows] = -model.score_samples(X[rows][:, cols])
        out = flat.reshape(ok.shape)
    return out


# ----------------------------------------------------------------------------------------------- step features
def step_matrix(f: Features) -> np.ndarray:
    """(T*D, n_features) matrix of STEP_FEATURES, NaN replaced by 0 (row-major: time, then device)."""
    T, D = f.y.shape
    lv = f.panel.devices["level"].to_numpy()
    parent = f.panel.devices["parent"].to_numpy()
    rp = np.nan_to_num(f.r_peer.astype(float))
    re = np.nan_to_num(f.r_exp.astype(float))
    day = np.asarray(f.panel.time.normalize().asi8)
    ok = assessable(f)
    r_peer_day = _day_median(np.where(ok, f.r_peer, np.nan), day)
    r_exp_day = _day_median(np.where(ok, f.r_exp, np.nan), day)
    twin = np.zeros((T, D))
    for g in np.unique(f.mppt_group[f.mppt_group >= 0]):
        idx = np.flatnonzero(f.mppt_group == g)
        for j in idx:
            others = [k for k in idx if k != j]
            twin[:, j] = np.nanmean(rp[:, others], axis=1)
    plant_r = np.zeros(T)
    plants = np.flatnonzero(lv == "plant")
    if len(plants):
        plant_r = np.nan_to_num(f.r_exp[:, plants[0]].astype(float))
    cols = {
        "r_exp": re, "r_peer": rp, "r_v": np.nan_to_num(f.r_v), "r_vself": np.nan_to_num(f.r_vself),
        "zero": f.zero, "near_pac": f.near_pac, "parent_limited": f.parent_limited, "status_bad": f.status_bad,
        "recur": np.nan_to_num(f.recur), "frac_low": np.nan_to_num(f.frac_low),
        "sensor_dev": np.broadcast_to(np.nan_to_num(f.sensor_dev)[:, None], (T, D)),
        "sensor_shape": np.broadcast_to(np.nan_to_num(f.sensor_shape)[:, None], (T, D)),
        "cold": np.broadcast_to(f.cold[:, None], (T, D)),
        "sun_el": np.broadcast_to(np.nan_to_num(f.sun_el)[:, None], (T, D)),
        "kc": np.broadcast_to(np.nan_to_num(f.kc, nan=0.8)[:, None], (T, D)),
        "lvl_plant": np.broadcast_to(lv == "plant", (T, D)), "lvl_inverter": np.broadcast_to(lv == "inverter", (T, D)),
        "lvl_string": np.broadcast_to(lv == "string", (T, D)),
        "r_peer_day": np.nan_to_num(r_peer_day), "r_exp_day": np.nan_to_num(r_exp_day), "twin_r_peer": twin,
        "plant_r_exp": np.broadcast_to(plant_r[:, None], (T, D)),
    }
    _ = parent
    return np.stack([np.asarray(cols[c], dtype=np.float32).reshape(-1) for c in STEP_FEATURES], axis=1)


def _day_median(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(x)
    return df.groupby(day).transform("median").to_numpy()


def train_gbm(mats: list[np.ndarray], labels: list[np.ndarray], oks: list[np.ndarray], seed: int = 0,
              normal_share: float = 0.15):
    """Multi-class gradient boosting on daylight steps of the training plants (normal steps subsampled)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    Xs, ys, ws = [], [], []
    for X, lab, ok in zip(mats, labels, oks):
        lab = lab.reshape(-1)
        okf = ok.reshape(-1)
        rows = np.flatnonzero(okf)
        normal = rows[lab[rows] == -1]
        fault = rows[lab[rows] != -1]
        keep_n = rng.choice(normal, size=int(len(normal) * normal_share), replace=False) if len(normal) else normal
        r = np.concatenate([keep_n, fault])
        Xs.append(X[r])
        ys.append(lab[r])
        ws.append(np.where(lab[r] == -1, 1.0 / normal_share, 1.0))
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    # normal steps keep their prevalence; rare fault classes (bypass, MPPT) get up to 5x weight
    cls, cnt = np.unique(y[y != -1], return_counts=True)
    med = np.median(cnt) if len(cnt) else 1.0
    cw = {c: float(np.clip(med / n, 1.0, 5.0)) for c, n in zip(cls, cnt)}
    w = np.where(y == -1, w, np.array([cw.get(c, 1.0) for c in y]))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(X, y, sample_weight=w)
    return model
