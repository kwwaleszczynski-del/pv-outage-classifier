"""The synthetic experiment: train on four simulated plant-years, test on four others.

Training plants use 2021-2022 weather, test plants 2023-2024; every plant has its own events, layout and noise.
Everything that is tuned (detector thresholds per level, the gradient boosting models, the rule thresholds) is
tuned on the training plants only; the test plants are scored once.

Usage:
    python -m pv_outage benchmark --jobs 4
"""
from __future__ import annotations

import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pv_outage import classify, detectors, events as ev, features, kpi, schema
from pv_outage.taxonomy import CATEGORY, CLASSES, DATA_CLASSES, LOSS_CLASSES

log = logging.getLogger(__name__)

TRAIN = [(2021, 1), (2021, 2), (2022, 3), (2022, 4)]
TEST = [(2023, 11), (2023, 12), (2024, 13), (2024, 14)]
METHODS = ["threshold", "expected", "peer", "hampel", "cusum", "iforest", "gbm", "hybrid"]
LAT, LON = 51.9, 19.5
STATUS_MAP = {1: "curtailment", 2: "grid_outage", 3: "inverter_trip"}


@dataclass
class Plant:
    year: int
    seed: int
    f: features.Features
    truth: pd.DataFrame  # indexed by event_id
    labels: np.ndarray
    label_event: np.ndarray
    clip: np.ndarray  # (T, inverter) clipping of the healthy plant
    effects: dict

    @property
    def name(self) -> str:
        return f"{self.year}-s{self.seed}"


# ----------------------------------------------------------------------------------------------- data
def sim_paths(data_dir: Path, year: int, seed: int) -> dict:
    d = Path(data_dir) / "sim"
    stem = f"plant_{year}_s{seed}"
    return {"long": d / f"{stem}.parquet", "truth": d / f"{stem}_truth.csv", "arrays": d / f"{stem}_labels.npz",
            "meta": d / f"{stem}_meta.json"}


def make_plant(data_dir: Path, year: int, seed: int) -> Path:
    """Simulate one plant-year into the cache (long Parquet, truth, step labels, metadata)."""
    from pv_outage import simulate, weather

    p = sim_paths(data_dir, year, seed)
    if all(x.exists() for x in p.values()):
        return p["long"]
    raw = weather.fetch_era5(LAT, LON, 2021, 2024, Path(data_dir) / "era5")
    sim = simulate.simulate(raw, year=year, seed=seed)
    p["long"].parent.mkdir(parents=True, exist_ok=True)
    sim.long.to_parquet(p["long"], index=False)
    sim.truth.to_csv(p["truth"], index=False)
    np.savez_compressed(p["arrays"], labels=sim.labels, label_event=sim.label_event, clip=sim.clip,
                        p_healthy=sim.p_healthy.astype(np.float32), p_actual=sim.p_actual.astype(np.float32))
    meta = {**sim.meta, "status_map": {str(k): v for k, v in STATUS_MAP.items()}}
    p["meta"].write_text(json.dumps(meta, indent=1))
    log.info("simulated plant %d seed %d: %d events", year, seed, len(sim.truth))
    return p["long"]


def load_plant(data_dir: Path, year: int, seed: int) -> Plant:
    p = sim_paths(data_dir, year, seed)
    meta = json.loads(p["meta"].read_text())
    panel = schema.to_panel(pd.read_parquet(p["long"]), meta=meta)
    f = features.compute(panel)
    truth = pd.read_csv(p["truth"], parse_dates=["start", "end", "first_effect"])
    for c in ("start", "end", "first_effect"):
        truth[c] = pd.to_datetime(truth[c], utc=True)
    truth = truth.set_index("event_id")
    arr = np.load(p["arrays"])
    return Plant(year, seed, f, truth, arr["labels"], arr["label_event"], arr["clip"],
                 ev.effect_steps(arr["label_event"]))


# ----------------------------------------------------------------------------------------------- detection
def method_flags(method: str, plant: Plant, theta: dict, gbm=None, scores: dict | None = None) -> np.ndarray:
    f = plant.f
    if method == "hybrid":
        fl = np.zeros(f.light.shape, bool)
        lv = f.panel.devices["level"].to_numpy()
        for m, levels in (("peer", ("inverter", "string")), ("expected", ("plant",)), ("threshold", LEVELS_ALL)):
            s = scores[m] if scores and m in scores else detectors.score(m, f)
            sub = detectors.flags(m, s, f, theta.get(m, {}))
            fl |= sub & np.isin(lv, levels)[None, :]
        return fl
    s = scores[method] if scores and method in scores else detectors.score(method, f, model=gbm)
    return detectors.flags(method, s, f, theta.get(method, {}))


LEVELS_ALL = ("plant", "inverter", "string")


def evaluate(det: pd.DataFrame, cflags: np.ndarray, plant: Plant, ids: set) -> dict:
    det = ev.estimate_loss(det, plant.f)
    det = ev.match(det, cflags, plant.label_event, plant.f, allowed=ids)
    truth = plant.truth.loc[sorted(ids)]
    sc = ev.score(det, cflags, truth, plant.f, plant.effects)
    sc["det"] = det
    return sc


def false_alarm_causes(det: pd.DataFrame, cflags: np.ndarray, plant: Plant, method: str) -> dict:
    """Unmatched detections whose flagged steps are mostly healthy clipping, or inside a sensor-fault window.

    The sensor counts only for methods that read the irradiance sensor at that level (the expected model,
    isolation forest, gradient boosting, and Hampel / CUSUM on the plant meter)."""
    f = plant.f
    lv = f.panel.devices["level"].to_numpy()
    parent = f.panel.devices["parent"].to_numpy()
    inv = list(np.flatnonzero(lv == "inverter"))
    sensor_ids = set(plant.truth.index[plant.truth["cls"] == "sensor_fault"])
    plant_col = int(np.flatnonzero(lv == "plant")[0])
    out = {"clipping": 0, "sensor": 0, "other": 0}
    for r in det[det["truth_id"] < 0].itertuples():
        st = np.flatnonzero(cflags[r.i0:r.i1 + 1, r.j]) + r.i0
        k = inv.index(r.j) if r.j in inv else (inv.index(parent[r.j]) if parent[r.j] in inv else None)
        clip = plant.clip[st, k].mean() if k is not None and len(st) else (plant.clip[st].any(axis=1).mean()
                                                                           if len(st) else 0)
        reads_sensor = method in ("expected", "iforest", "gbm") or (r.level == "plant" and method != "peer")
        sens = (np.isin(plant.label_event[st, plant_col], list(sensor_ids)).mean()
                if len(st) and reads_sensor else 0)
        if clip >= 0.5:
            out["clipping"] += 1
        elif sens >= 0.5:
            out["sensor"] += 1
        else:
            out["other"] += 1
    return out


def calibrate(method: str, plants: list[Plant], score_cache: dict) -> dict:
    """Per level, the threshold from the method's grid with the best event F1 on the training plants."""
    if method in ("threshold", "gbm"):
        return {}
    theta = {}
    for level in LEVELS_ALL:
        best = None
        for th in detectors.GRIDS[method]:
            f1s = []
            for pl in plants:
                f = pl.f
                lv = f.panel.devices["level"].to_numpy()
                s = score_cache[(pl.name, method)]
                fl = detectors.flags(method, s, f, {lv_: (th if lv_ == level else np.inf) for lv_ in LEVELS_ALL}
                                     if method != "iforest" else {lv_: (th if lv_ == level else 1.0)
                                                                  for lv_ in LEVELS_ALL})
                fl &= (lv == level)[None, :]
                det = ev.extract(fl, f)
                ids = set(pl.truth.index[pl.truth["cls"].isin(LOSS_CLASSES)])
                ids_level = {i for i in ids if _visible_at(pl, i, level)}
                det = ev.match(det, fl, pl.label_event, f, allowed=ids)
                n_det = len(det)
                prec = (det["truth_id"] >= 0).mean() if n_det else 0.0
                rec = len(set(det["truth_id"]) & ids_level) / max(len(ids_level), 1)
                f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
            m = float(np.mean(f1s))
            if best is None or m > best[0]:
                best = (m, th)
        theta[level] = best[1]
        log.info("calibrated %s %s: theta %s (train F1 %.2f)", method, level, best[1], best[0])
    return theta


def _visible_at(pl: Plant, eid: int, level: str) -> bool:
    lv = pl.f.panel.devices["level"].to_numpy()
    cols = np.flatnonzero(lv == level)
    return bool((pl.label_event[:, cols] == eid).any())


# ----------------------------------------------------------------------------------------------- gbm
def gbm_training_rows(plant: Plant, normal_share: float = 0.1, seed: int = 0):
    """Subsampled step matrix of one plant (all affected steps + a share of normal ones)."""
    X = detectors.step_matrix(plant.f)
    ok = detectors.assessable(plant.f).reshape(-1)
    lab = plant.labels.reshape(-1).astype(int)
    data_cls = [CLASSES.index(c) for c in DATA_CLASSES]
    lab = np.where(np.isin(lab, data_cls), -1, lab)  # data problems are the data-quality stage's job
    rng = np.random.default_rng(seed + plant.seed)
    rows = np.flatnonzero(ok)
    normal = rows[lab[rows] == -1]
    keep = np.concatenate([rng.choice(normal, int(len(normal) * normal_share), replace=False),
                           rows[lab[rows] != -1]])
    w = np.where(lab[keep] == -1, 1.0 / normal_share, 1.0)
    return X[keep], lab[keep], w


def train_step_gbm(rows: list, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    w = np.concatenate([r[2] for r in rows])
    cls, cnt = np.unique(y[y != -1], return_counts=True)
    med = np.median(cnt)
    cw = {c: float(np.clip(med / n, 1.0, 5.0)) for c, n in zip(cls, cnt)}
    w = np.where(y == -1, w, np.array([cw.get(c, 1.0) for c in y]))
    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_leaf_nodes=63,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(X, y, sample_weight=w)
    return model


# ----------------------------------------------------------------------------------------------- pipeline
def pipeline_events(plant: Plant, flags: np.ndarray, event_model=None) -> pd.DataFrame:
    """The tool end to end: data-quality events + detected loss events, classified, with lost energy."""
    f = plant.f
    det, cflags = ev.detect_events(flags, f)
    det = ev.estimate_loss(det, f)
    table = classify.detected_table(det, f, cflags, STATUS_MAP)
    det["cls_rules"] = classify.classify_rules(table).to_numpy() if len(det) else np.array([], dtype=object)
    det["cls_gbm"] = (classify.classify_gbm(event_model, table).to_numpy() if event_model is not None and len(det)
                      else det["cls_rules"])
    dq, dqf = ev.dq_events(f)
    dq["loss_kwh"] = 0.0
    dq["cls_rules"] = dq["cls"]
    dq["cls_gbm"] = dq["cls"]
    out = pd.concat([det.assign(source="detector"), dq.assign(source="data_quality")], ignore_index=True)
    allf = cflags | dqf
    return out, allf, table


def plant_kpis(plant: Plant, events: pd.DataFrame, cls_col: str) -> dict:
    e = events.rename(columns={cls_col: "cls_k"}).drop(columns=["cls"], errors="ignore").rename(
        columns={"cls_k": "cls"})
    return kpi.availability(e, plant.f)


def truth_kpis(plant: Plant) -> dict:
    return kpi.availability(kpi.truth_events(plant.truth.reset_index(), plant.f), plant.f)


# ----------------------------------------------------------------------------------------------- main
def _prepare(args):
    data_dir, year, seed = args
    make_plant(Path(data_dir), year, seed)
    return year, seed


def run(data_dir: Path, out_dir: Path, jobs: int = 1) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(str(data_dir), y, s) for y, s in TRAIN + TEST]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            list(pool.map(_prepare, todo))
    else:
        for a in todo:
            _prepare(a)

    # ---- training plants: scores, GBM rows, oracle event tables
    score_cache, gbm_rows, train_tables, train_y = {}, [], [], []
    train_plants = []
    for y, s in TRAIN:
        pl = load_plant(data_dir, y, s)
        for m in ("expected", "peer", "hampel", "cusum", "iforest"):
            score_cache[(pl.name, m)] = detectors.score(m, pl.f).astype(np.float32)
        gbm_rows.append(gbm_training_rows(pl))
        tab = classify.oracle_table(pl.truth, pl.f, pl.labels, STATUS_MAP)
        train_tables.append(tab)
        train_y.append(pl.truth.loc[tab.index, "cls"])
        train_plants.append(pl)
        log.info("training plant %s ready", pl.name)

    theta = {m: calibrate(m, train_plants, score_cache) for m in ("expected", "peer", "hampel", "cusum", "iforest")}
    step_gbm = train_step_gbm(gbm_rows)
    del gbm_rows

    # event classifier: oracle windows + detected events of the training plants (hybrid detector)
    for pl in train_plants:
        sc = {m: score_cache[(pl.name, m)] for m in ("expected", "peer")}
        fl = method_flags("hybrid", pl, theta, scores=sc)
        evs, allf, table = pipeline_events(pl, fl)
        evs = ev.match(evs, allf, pl.label_event, pl.f)
        hit = evs[(evs["truth_id"] >= 0) & (evs["source"] == "detector")]
        train_tables.append(table.loc[hit.index])
        train_y.append(pl.truth.loc[hit["truth_id"], "cls"].set_axis(hit.index))
    X_ev = pd.concat(train_tables, ignore_index=True)
    y_ev = pd.concat(train_y, ignore_index=True)
    event_gbm = classify.train_event_gbm(X_ev, y_ev)
    rules_train = classify.classify_rules(X_ev)
    log.info("rules on training events: accuracy %.2f", float((rules_train.to_numpy() == y_ev.to_numpy()).mean()))
    rules_precision = (rules_train.to_numpy() == y_ev.to_numpy())
    rules_precision = pd.Series(rules_precision).groupby(rules_train.to_numpy()).mean().round(3).to_dict()
    (out_dir / "thresholds.json").write_text(json.dumps({"theta": theta, "rules_precision": rules_precision},
                                                        indent=1, default=_json_default))
    del train_plants, score_cache
    with open(data_dir / "models.pkl", "wb") as fh:
        pickle.dump({"step_gbm": step_gbm, "event_gbm": event_gbm, "theta": theta}, fh)

    # ---- test plants
    parts = [score_test_plant(load_plant(data_dir, y, s), theta, step_gbm, event_gbm, out_dir,
                              example=(i == 0)) for i, (y, s) in enumerate(TEST)]
    results = [r for p in parts for r in p["results"]]
    per_truth = [r for p in parts for r in p["per_truth"]]
    fa_rows = [r for p in parts for r in p["dq"]]
    oracle_rows = [p["oracle"] for p in parts]
    pipe_rows = [p["pipeline"] for p in parts]
    kpi_rows = [r for p in parts for r in p["kpi"]]
    calc = {p["name"]: p["calc"] for p in parts}

    res = pd.DataFrame(results)
    res.to_csv(out_dir / "detection_by_plant.csv", index=False)
    pt = pd.concat(per_truth, ignore_index=True)
    pt.to_csv(out_dir / "detection_per_truth_event.csv", index=False)
    pd.DataFrame(fa_rows).to_csv(out_dir / "data_quality.csv", index=False)
    orc = pd.concat(oracle_rows)
    orc.to_csv(out_dir / "classification_oracle.csv")
    pipe = pd.concat(pipe_rows, ignore_index=True)
    pipe.to_csv(out_dir / "classification_pipeline.csv", index=False)
    kp = pd.DataFrame(kpi_rows)
    kp.to_csv(out_dir / "kpi_by_plant.csv", index=False)
    (out_dir / "calculator.json").write_text(json.dumps(calc, indent=1, default=_json_default))
    summary = summarise(res, pt, orc, pipe, kp, pd.DataFrame(fa_rows), theta)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    return summary


def score_test_plant(pl: Plant, theta: dict, step_gbm, event_gbm, out_dir: Path, example: bool = False) -> dict:
    """Every method on one test plant: detection scores, the pipeline with both classifiers, KPIs, DQ."""
    loss_ids = set(pl.truth.index[pl.truth["cls"].isin(LOSS_CLASSES)])
    out = {"name": pl.name, "results": [], "per_truth": [], "kpi": [], "dq": [], "calc": {}}
    scores = {}
    n_dev = {lv_: int((pl.f.panel.devices["level"] == lv_).sum()) for lv_ in LEVELS_ALL}
    for m in METHODS:
        if m == "hybrid":
            fl = method_flags(m, pl, theta, scores=scores)
        else:
            scores[m] = detectors.score(m, pl.f, model=step_gbm)
            fl = detectors.flags(m, scores[m], pl.f, theta.get(m, {}))
        det, cf = ev.detect_events(fl, pl.f)
        sc = evaluate(det, cf, pl, loss_ids)
        fa = false_alarm_causes(sc["det"], cf, pl, m)
        fa_lv = sc["det"][sc["det"]["truth_id"] < 0].groupby("level").size()
        out["results"].append({"plant": pl.name, "method": m, **{k: sc[k] for k in
                               ("n_det", "tp_det", "n_truth", "found", "precision", "recall", "f1")},
                               "fa_clipping": fa["clipping"], "fa_sensor": fa["sensor"], "fa_other": fa["other"],
                               **{f"fa_per_dev_{lv_}": float(fa_lv.get(lv_, 0)) / max(n_dev[lv_], 1)
                                  for lv_ in LEVELS_ALL}})
        out["per_truth"].append(sc["per_truth"].assign(plant=pl.name, method=m))
        # end to end with classification and KPIs
        evs, allf, table = pipeline_events(pl, fl, event_gbm)
        for col in ("cls_rules", "cls_gbm"):
            out["kpi"].append({"plant": pl.name, "method": m, "classifier": col[4:],
                               **_flat_kpi(plant_kpis(pl, evs, col))})
        if m == "hybrid":
            out["calc"]["tool"] = kpi.calculator_data(evs.drop(columns=["cls"]).rename(columns={"cls_rules": "cls"}),
                                                      pl.f)
            evm = ev.match(evs, allf, pl.label_event, pl.f)
            evm["true_cls"] = evm["truth_id"].map(pl.truth["cls"]).fillna("false_alarm")
            out["pipeline"] = evm.assign(plant=pl.name)[["plant", "device_id", "level", "start", "end", "hours",
                                                         "source", "cls_rules", "cls_gbm", "true_cls", "loss_kwh",
                                                         "truth_id"]]
            if example:
                _save_example(out_dir, pl, evm)
    out["kpi"].append({"plant": pl.name, "method": "truth", "classifier": "truth", **_flat_kpi(truth_kpis(pl))})
    out["calc"]["truth"] = kpi.calculator_data(kpi.truth_events(pl.truth.reset_index(), pl.f), pl.f)
    # classification with perfect segmentation
    tab = classify.oracle_table(pl.truth, pl.f, pl.labels, STATUS_MAP)
    out["oracle"] = pd.DataFrame({"plant": pl.name, "true_cls": pl.truth.loc[tab.index, "cls"].to_numpy(),
                                  "rules": classify.classify_rules(tab).to_numpy(),
                                  "gbm": classify.classify_gbm(event_gbm, tab).to_numpy()}, index=tab.index)
    # data-quality stage
    dq, dqf = ev.dq_events(pl.f)
    data_ids = set(pl.truth.index[pl.truth["cls"].isin(DATA_CLASSES)])
    dqm = ev.match(dq, dqf, pl.label_event, pl.f, allowed=data_ids)
    dq_truth = pl.truth.loc[sorted(data_ids)]
    for c in DATA_CLASSES:
        tid = set(dq_truth.index[dq_truth["cls"] == c])
        d_c = dqm[dqm["cls"] == c]
        out["dq"].append({"plant": pl.name, "cls": c, "n_truth": len(tid), "found": len(tid & set(d_c["truth_id"])),
                          "n_det": len(d_c), "tp_det": int(d_c["truth_id"].isin(tid).sum())})
    log.info("test plant %s scored", pl.name)
    return out


def _flat_kpi(k: dict) -> dict:
    out = {kk: v for kk, v in k.items() if not isinstance(v, dict)}
    for c, v in k["lost_by_class_kwh"].items():
        out[f"lost_{c}"] = v
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _save_example(out_dir: Path, pl: Plant, events: pd.DataFrame | None):
    """Truth and the hybrid pipeline's events of one test plant, for the dashboard's event overview."""
    t = pl.truth.reset_index()[["event_id", "cls", "category", "device_id", "devices", "start", "end",
                                "effect_hours", "true_loss_kwh", "note"]]
    t.to_csv(out_dir / "example_truth.csv", index=False)
    if events is not None:
        events.to_csv(out_dir / "example_events.csv", index=False)
    # daily energy per inverter: actual and expected (for the overview chart)
    f = pl.f
    lv = f.panel.devices["level"].to_numpy()
    inv = np.flatnonzero(lv == "inverter")
    day = f.panel.time.normalize()
    act = pd.DataFrame(np.nan_to_num(f.y[:, inv]) * f.panel.step_h, index=day).groupby(level=0).sum()
    exp = pd.DataFrame(np.nan_to_num(np.where(f.light[:, inv], f.ref_peer[:, inv], 0)) * f.panel.step_h,
                       index=day).groupby(level=0).sum()
    act.columns = exp.columns = f.panel.devices.index[inv]
    act.to_csv(out_dir / "example_daily_actual.csv")
    exp.to_csv(out_dir / "example_daily_peer.csv")


def summarise(res, pt, orc, pipe, kp, dq, theta) -> dict:
    agg = res.groupby("method")[["n_det", "tp_det", "n_truth", "found", "fa_clipping", "fa_sensor",
                                 "fa_other"]].sum()
    agg["precision"] = agg["tp_det"] / agg["n_det"]
    agg["recall"] = agg["found"] / agg["n_truth"]
    agg["f1"] = 2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])
    for lv_ in LEVELS_ALL:
        agg[f"fa_per_dev_year_{lv_}"] = res.groupby("method")[f"fa_per_dev_{lv_}"].mean()
    f1_sd = res.groupby("method")["f1"].std()
    agg["f1_sd_plants"] = f1_sd
    found = pt[pt["found"]]
    agg["delay_h_median"] = found.groupby("method")["delay_h"].median()
    rel_dur = (found["det_hours"] - found["true_hours"]) / found["true_hours"].clip(lower=0.25)
    agg["duration_rel_err_median"] = rel_dur.abs().groupby(found["method"]).median()
    e = found[found["true_kwh"] > 1]
    rel_e = (e["est_kwh"] - e["true_kwh"]) / e["true_kwh"]
    agg["energy_rel_err_median"] = rel_e.abs().groupby(e["method"]).median()
    agg["energy_bias_total"] = (e.groupby("method")["est_kwh"].sum() / e.groupby("method")["true_kwh"].sum() - 1)
    recall_cls = pt.groupby(["method", "cls"])["found"].mean().unstack()
    delay_cls = found.groupby(["method", "cls"])["delay_h"].median().unstack()

    def acc(a, b):
        return float((a == b).mean())

    orc_cm = {c: classify.confusion(orc["true_cls"], orc[c]).to_dict() for c in ("rules", "gbm")}
    hits = pipe[pipe["true_cls"] != "false_alarm"]
    pipe_cm = {c: classify.confusion(hits["true_cls"], hits[f"cls_{c}"]).to_dict() for c in ("rules", "gbm")}
    truth_k = kp[kp["method"] == "truth"].set_index("plant")
    tool = kp[kp["method"] != "truth"].copy()
    tool["err_time_pp"] = (tool["availability_time"] - tool["plant"].map(truth_k["availability_time"])) * 100
    tool["err_energy_pp"] = (tool["availability_energy"] - tool["plant"].map(truth_k["availability_energy"])) * 100
    kerr = tool.groupby(["method", "classifier"])[["err_time_pp", "err_energy_pp"]].agg(
        lambda x: float(np.mean(np.abs(x))))
    kbias = tool.groupby(["method", "classifier"])[["err_time_pp", "err_energy_pp"]].mean()
    cat_cols = {cat: [f"lost_{c}" for c in CLASSES if CATEGORY[c] == cat] for cat in set(CATEGORY.values())}
    lost_cat = {}
    for name, g in kp.groupby(["method", "classifier"]):
        lost_cat["|".join(name)] = {cat: float(g[cols].sum(axis=1).sum()) for cat, cols in cat_cols.items()}
    dq_s = dq.groupby("cls")[["n_truth", "found", "n_det", "tp_det"]].sum()
    return {
        "theta": theta,
        "detection": agg.reset_index().to_dict(orient="records"),
        "recall_by_class": recall_cls.reset_index().to_dict(orient="records"),
        "delay_by_class": delay_cls.reset_index().to_dict(orient="records"),
        "oracle_accuracy": {c: acc(orc["true_cls"], orc[c]) for c in ("rules", "gbm")},
        "oracle_confusion": orc_cm,
        "pipeline_accuracy": {c: acc(hits["true_cls"], hits[f"cls_{c}"]) for c in ("rules", "gbm")},
        "pipeline_confusion": pipe_cm,
        "pipeline_false_alarms": int((pipe["true_cls"] == "false_alarm").sum()),
        "pipeline_events": int(len(pipe)),
        "kpi_abs_err": kerr.reset_index().to_dict(orient="records"),
        "kpi_bias": kbias.reset_index().to_dict(orient="records"),
        "kpi_truth": truth_k.reset_index().to_dict(orient="records"),
        "lost_by_category": lost_cat,
        "data_quality": dq_s.reset_index().to_dict(orient="records"),
        "n_test_events": int(pt[pt["method"] == "peer"].shape[0]),
    }


def sensitivity(data_dir: Path, out_dir: Path, min_hours=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Test plants re-scored with longer minimum event durations (a check of one setting, not a re-tuning)."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    with open(data_dir / "models.pkl", "rb") as fh:  # written by `run` on this machine
        models = pickle.load(fh)
    theta = models["theta"]
    rows = []
    for y, s in TEST:
        pl = load_plant(data_dir, y, s)
        loss_ids = set(pl.truth.index[pl.truth["cls"].isin(LOSS_CLASSES)])
        n_dev = {lv_: int((pl.f.panel.devices["level"] == lv_).sum()) for lv_ in LEVELS_ALL}
        scores = {}
        for m in METHODS:
            if m == "hybrid":
                fl = method_flags(m, pl, theta, scores=scores)
            else:
                scores[m] = detectors.score(m, pl.f, model=models["step_gbm"])
                fl = detectors.flags(m, scores[m], pl.f, theta.get(m, {}))
            for h in min_hours:
                det, cf = ev.detect_events(fl, pl.f, min_steps=max(1, round(h / pl.f.panel.step_h)))
                sc = evaluate(det, cf, pl, loss_ids)
                unmatched = sc["det"][sc["det"]["truth_id"] < 0]
                fa = unmatched.groupby("level").size()
                # what an operator would open: one ticket per found event, one per device and day of false alarms
                fa_episodes = unmatched.groupby([unmatched["device_id"],
                                                 pd.to_datetime(unmatched["start"]).dt.date]).ngroups
                rows.append({"plant": pl.name, "method": m, "min_hours": h, "n_det": sc["n_det"],
                             "tp_det": sc["tp_det"], "n_truth": sc["n_truth"], "found": sc["found"],
                             "fa_episodes": fa_episodes,
                             "fa_per_dev_year": float(sum(fa.get(lv_, 0) for lv_ in LEVELS_ALL))
                             / sum(n_dev.values())})
        log.info("sensitivity: %s done", pl.name)
    df = pd.DataFrame(rows)
    g = df.groupby(["method", "min_hours"])[["n_det", "tp_det", "n_truth", "found", "fa_episodes"]].sum()
    g["precision"] = g["tp_det"] / g["n_det"]
    g["recall"] = g["found"] / g["n_truth"]
    g["f1"] = 2 * g["precision"] * g["recall"] / (g["precision"] + g["recall"])
    g["fa_per_dev_year"] = df.groupby(["method", "min_hours"])["fa_per_dev_year"].mean()
    g["fragments_per_event"] = g["tp_det"] / g["found"]
    g["event_precision"] = g["found"] / (g["found"] + g["fa_episodes"])
    g = g.reset_index()
    g.to_csv(out_dir / "sensitivity_min_duration.csv", index=False)
    return g
