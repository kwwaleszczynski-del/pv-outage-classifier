"""Figures for the README (docs/img) and the data behind the dashboard (docs/data.js).

Usage:
    python -m pv_outage report
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pv_outage.taxonomy import CATEGORY, CLASSES, LOSS_CLASSES

log = logging.getLogger(__name__)

METHOD_NAMES = {
    "threshold": "Threshold (zero / status)", "expected": "Expected model (PR)", "peer": "Peer median",
    "hampel": "Hampel z-score", "cusum": "CUSUM", "iforest": "Isolation forest", "gbm": "Gradient boosting",
    "hybrid": "Hybrid (tool)",
}
CLASS_NAMES = {
    "inverter_trip": "Inverter trip", "string_open": "Open string", "bypass_short": "Shorted bypass diode",
    "mppt_failure": "MPPT failure", "partial_shading": "Partial shading", "soiling": "Soiling", "snow": "Snow",
    "curtailment": "Curtailment", "grid_outage": "Grid outage", "comm_loss": "Communication loss",
    "frozen_data": "Frozen values", "sensor_fault": "Irradiance sensor fault",
}
# categorical slots of the reference palette, fixed order: fault, environment, external, data
CAT_COLOR = {"fault": "#2a78d6", "environment": "#eb6834", "external": "#1baf7a", "data": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": INK,
        "figure.dpi": 110, "savefig.bbox": "tight", "savefig.dpi": 130, "legend.frameon": False,
    })
    return plt


def run(results: Path, docs: Path) -> None:
    results, docs = Path(results), Path(docs)
    (docs / "img").mkdir(parents=True, exist_ok=True)
    plt = _style()
    s = json.loads((results / "summary.json").read_text())
    det = pd.DataFrame(s["detection"]).set_index("method").reindex(list(METHOD_NAMES))
    rec = pd.DataFrame(s["recall_by_class"]).set_index("method").reindex(list(METHOD_NAMES))
    rec = rec[[c for c in LOSS_CLASSES if c in rec.columns]]
    fig_methods(plt, det, docs / "img" / "method_comparison.png")
    fig_heat(plt, rec, docs / "img" / "recall_by_class.png", "Share of truth events found, test plants", "{:.0%}")
    orc = pd.read_csv(results / "classification_oracle.csv", index_col=0)
    fig_confusion(plt, orc, docs / "img" / "confusion.png")
    kp = pd.read_csv(results / "kpi_by_plant.csv")
    fig_kpi(plt, kp, docs / "img" / "availability_error.png")
    ex_t = pd.read_csv(results / "example_truth.csv", parse_dates=["start", "end"])
    ex_e = pd.read_csv(results / "example_events.csv", parse_dates=["start", "end"])
    fig_timeline(plt, ex_t, ex_e, docs / "img" / "example_timeline.png")
    real = None
    if (results / "real_alarms.csv").exists():
        real = pd.read_csv(results / "real_alarms.csv").set_index("method")
        fig_real(plt, real, results, docs / "img" / "real_pvdaq.png")
    if (results / "shading_real.csv").exists():
        fig_shading(plt, results, docs / "img" / "real_shading.png")
    write_data(results, docs / "data.js", s, det, rec, orc, kp, ex_t, ex_e, real)
    log.info("report written to %s", docs)


# ----------------------------------------------------------------------------------------------- figures
def fig_methods(plt, det, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    names = [METHOD_NAMES[m] for m in det.index]
    y = np.arange(len(det))[::-1]
    for ax, col, title in zip(axes, ("precision", "recall", "f1"), ("Precision", "Recall", "F1")):
        colors = [BLUE[5] if m == "hybrid" else BLUE[2] for m in det.index]
        ax.barh(y, det[col], color=colors, height=0.62)
        for yy, v in zip(y, det[col]):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    fig.suptitle("Event-level detection of production losses, 4 test plant-years", x=0.01, ha="left",
                 fontweight="bold", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_heat(plt, df, path, title, fmt):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + BLUE)
    vals = df.to_numpy(dtype=float)
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(df.shape[1]), [CLASS_NAMES[c] for c in df.columns], rotation=30, ha="right")
    ax.set_yticks(range(df.shape[0]), [METHOD_NAMES.get(m, m) for m in df.index])
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else INK)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(plt, orc, path):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff"] + BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    present = [c for c in CLASSES if c in set(orc["true_cls"])]
    for ax, col, title in zip(axes, ("rules", "gbm"), ("Signature rules", "Gradient boosting")):
        cm = pd.crosstab(pd.Categorical(orc["true_cls"], present), pd.Categorical(orc[col], present), dropna=False)
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        ax.imshow(norm.to_numpy(), cmap=cmap, vmin=0, vmax=1)
        for i in range(len(present)):
            for j in range(len(present)):
                n = cm.iloc[i, j]
                if n:
                    ax.text(j, i, str(n), ha="center", va="center", fontsize=7,
                            color="white" if norm.iloc[i, j] > 0.6 else INK)
        lab = [CLASS_NAMES[c] for c in present]
        ax.set_xticks(range(len(present)), lab, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(present)), lab if col == "rules" else [""] * len(present), fontsize=7)
        acc = float((orc["true_cls"] == orc[col]).mean())
        ax.set_title(f"{title}: accuracy {acc:.0%}", loc="left")
        ax.set_xlabel("predicted")
        ax.grid(False)
    axes[0].set_ylabel("true cause")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def kpi_errors(kp: pd.DataFrame) -> pd.DataFrame:
    truth = kp[kp["method"] == "truth"].set_index("plant")
    tool = kp[(kp["method"] != "truth") & (kp["classifier"] == "rules")].copy()
    tool["err_time_pp"] = (tool["availability_time"] - tool["plant"].map(truth["availability_time"])) * 100
    tool["err_energy_pp"] = (tool["availability_energy"] - tool["plant"].map(truth["availability_energy"])) * 100
    return tool


def fig_kpi(plt, kp, path):
    tool = kpi_errors(kp)
    g = tool.groupby("method")[["err_time_pp", "err_energy_pp"]].agg(lambda x: float(np.mean(np.abs(x))))
    g = g.reindex(list(METHOD_NAMES)).dropna(how="all")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    y = np.arange(len(g))[::-1]
    for ax, col, title in zip(axes, ("err_time_pp", "err_energy_pp"),
                              ("Time-based availability", "Energy-based availability")):
        colors = [BLUE[5] if m == "hybrid" else BLUE[2] for m in g.index]
        ax.barh(y, g[col], color=colors, height=0.62)
        for yy, v in zip(y, g[col]):
            ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_title(f"{title}: mean |error| [pp]", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [METHOD_NAMES[m] for m in g.index])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _rows(df: pd.DataFrame) -> pd.Series:
    """Timeline row of an event: the plant, or the inverter (string events go on their inverter's row)."""
    return df["device_id"].astype(str).str.split("-").str[0]


def fig_timeline(plt, truth, events, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    rows = sorted(set(_rows(truth)) | set(_rows(events)), key=lambda r: (r != "PLANT", r))
    pos = {r: i for i, r in enumerate(rows)}
    events = events.copy()
    events["cat"] = events["cls_rules"].map(CATEGORY)
    for ax, df, cat_col, title in ((axes[0], truth, "category", "Truth"),
                                   (axes[1], events[events["true_cls"] != "false_alarm"], "cat",
                                    "Tool (hybrid detector + rules), hits"),):
        for cat, color in CAT_COLOR.items():
            sub = df[df[cat_col] == cat]
            for r in sub.itertuples():
                dur = max((r.end - r.start) / pd.Timedelta("1D"), 0.15)
                ax.barh(pos[_rows(pd.DataFrame({"device_id": [r.device_id]})).iloc[0]], dur,
                        left=r.start, color=color, height=0.7)
        ax.set_yticks(range(len(rows)), rows)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fa = events[events["true_cls"] == "false_alarm"]
    for r in fa.itertuples():
        axes[1].plot([r.start], [pos.get(_rows(pd.DataFrame({"device_id": [r.device_id]})).iloc[0], 0)], "x",
                     color=INK2, ms=4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
    handles.append(plt.Line2D([], [], marker="x", ls="", color=INK2))
    axes[0].legend(handles, list(CAT_COLOR) + ["false alarm"], ncol=5, loc="lower left", bbox_to_anchor=(0, 1.08),
                   fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_real(plt, real, results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
    r = real.reindex([m for m in METHOD_NAMES if m in real.index])
    y = np.arange(len(r))[::-1]
    axes[0].barh(y, r["other_per_device_year"], color=[BLUE[5] if m == "hybrid" else BLUE[2] for m in r.index],
                 height=0.62)
    for yy, v, fnd, n in zip(y, r["other_per_device_year"], r["plain_found"], r["plain_outages"]):
        axes[0].text(v + 0.3, yy, f"{v:.1f}  ({fnd}/{n} outages)", va="center", fontsize=8, color=INK2)
    axes[0].set_yticks(y, [METHOD_NAMES[m] for m in r.index])
    axes[0].set_title("Other alarms per inverter-year (PVDAQ)", loc="left")
    axes[0].grid(axis="y", visible=False)
    act = pd.read_csv(results / "real_daily_actual.csv", index_col=0, parse_dates=True)
    ref = pd.read_csv(results / "real_daily_peer.csv", index_col=0, parse_dates=True)
    ratio = (act / ref.where(ref > 0)).clip(0, 1.3)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ratio", ["#e34948", "#f6d4d4", "#f4f8fd", "#cde2fb"], N=256)
    axes[1].imshow(ratio.T.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1.2, interpolation="nearest")
    axes[1].set_yticks(range(ratio.shape[1]), ratio.columns)
    ticks = [i for i, d in enumerate(ratio.index) if d.day == 1 and d.month in (1, 7)]
    axes[1].set_xticks(ticks, [ratio.index[i].strftime("%Y-%m") for i in ticks])
    axes[1].grid(False)
    axes[1].set_title("Daily energy / peer reference per inverter (red = shortfall)", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_shading(plt, results, path):
    sw = pd.read_csv(results / "shading_real_sweep.csv")
    res = pd.read_csv(results / "shading_real.csv").set_index("method")
    daily = pd.read_csv(results / "shading_real_daily.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
    colors = {"expected": "#2a78d6", "peer": "#eb6834", "hampel": "#1baf7a", "cusum": "#eda100", "iforest": "#6250d6"}
    for m, g in sw.groupby("method"):
        g = g.sort_values("fa_per_inverter_year")
        axes[0].plot(g["fa_per_inverter_year"].clip(lower=0.1), g["found"], "-o", ms=4, lw=2, color=colors[m],
                     label=METHOD_NAMES[m])
        cal = g[g["calibrated"]]
        if len(cal):
            axes[0].plot(cal["fa_per_inverter_year"].clip(lower=0.1), cal["found"], "o", ms=10, mfc="none",
                         color=colors[m])
    n = int(res["events"].iloc[0])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alarms outside the labels per inverter-year")
    axes[0].set_ylabel(f"shaded days found (of {n})")
    axes[0].set_title("Threshold sweep (ring: threshold from the simulation)", loc="left")
    axes[0].legend(fontsize=7, loc="lower right")
    d = daily["ratio"].clip(0.3, 1.2)
    axes[1].plot(d.index, d, ".", ms=3, color=MUTED, label="normal day")
    sh = daily["shaded"] == 1
    axes[1].plot(d.index[sh], d[sh], "o", ms=5, color="#eb6834", label="labelled shading on inverter 1")
    axes[1].set_ylabel("daily energy, inverter 1 / inverter 2")
    axes[1].set_title("Twin inverters on La Reunion", loc="left")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- dashboard data
def _records(df: pd.DataFrame, cols=None) -> list:
    d = df if cols is None else df[cols]
    return json.loads(d.to_json(orient="records", date_format="iso", double_precision=4))


def write_data(results, path, s, det, rec, orc, kp, ex_t, ex_e, real):
    pt = pd.read_csv(results / "detection_per_truth_event.csv")
    delay = pt[pt["found"]].groupby(["method", "cls"])["delay_h"].median().unstack().reindex(list(METHOD_NAMES))
    pipe = pd.read_csv(results / "classification_pipeline.csv")
    hits = pipe[pipe["true_cls"] != "false_alarm"]
    conf = {}
    for name, df, tcol, pcols in (("oracle", orc, "true_cls", {"rules": "rules", "gbm": "gbm"}),
                                  ("pipeline", hits, "true_cls", {"rules": "cls_rules", "gbm": "cls_gbm"})):
        conf[name] = {}
        for k, c in pcols.items():
            cm = pd.crosstab(pd.Categorical(df[tcol], CLASSES), pd.Categorical(df[c], CLASSES), dropna=False)
            conf[name][k] = {"matrix": cm.to_numpy().tolist(),
                             "accuracy": float((df[tcol] == df[c]).mean()) if len(df) else None}
    tool = kpi_errors(kp)
    kpi_err = tool.groupby("method")[["err_time_pp", "err_energy_pp"]].agg(lambda x: float(np.mean(np.abs(x))))
    daily_a = pd.read_csv(results / "example_daily_actual.csv", index_col=0)
    daily_p = pd.read_csv(results / "example_daily_peer.csv", index_col=0)
    data = {
        "methods": [{"id": m, "name": METHOD_NAMES[m], **{k: _num(det.loc[m, k]) for k in (
            "precision", "recall", "f1", "n_det", "tp_det", "n_truth", "found", "delay_h_median",
            "duration_rel_err_median", "energy_rel_err_median", "energy_bias_total", "fa_per_dev_year_plant",
            "fa_per_dev_year_inverter", "fa_per_dev_year_string", "fa_clipping", "fa_sensor", "fa_other")},
                     "kpi_time_pp": _num(kpi_err["err_time_pp"].get(m)),
                     "kpi_energy_pp": _num(kpi_err["err_energy_pp"].get(m))} for m in METHOD_NAMES if m in det.index],
        "loss_classes": LOSS_CLASSES, "classes": CLASSES, "class_names": CLASS_NAMES, "category": CATEGORY,
        "recall": {m: {c: _num(rec.loc[m, c]) for c in rec.columns} for m in rec.index},
        "delay": {m: {c: _num(delay.loc[m, c]) if c in delay.columns else None for c in LOSS_CLASSES}
                  for m in delay.index},
        "confusion": conf,
        "data_quality": s.get("data_quality"),
        "theta": s.get("theta"),
        "example": {
            "truth": _records(ex_t.assign(start=ex_t["start"].astype(str), end=ex_t["end"].astype(str)),
                              ["cls", "category", "device_id", "start", "end", "effect_hours", "true_loss_kwh",
                               "note"]),
            "events": _records(ex_e.assign(start=ex_e["start"].astype(str), end=ex_e["end"].astype(str)),
                               ["device_id", "level", "start", "end", "hours", "source", "cls_rules", "cls_gbm",
                                "true_cls", "loss_kwh"]),
            "daily": {"dates": list(daily_a.index.astype(str).str[:10]),
                      "actual": {c: np.round(daily_a[c].to_numpy(), 1).tolist() for c in daily_a.columns},
                      "peer": {c: np.round(daily_p[c].to_numpy(), 1).tolist() for c in daily_p.columns}},
        },
        "calculator": json.loads((results / "calculator.json").read_text()) if (results / "calculator.json").exists()
        else {},
    }
    if real is not None:
        rc = results / "real_calculator.json"
        data["real"] = {
            "alarms": _records(real.reset_index()),
            "plain": _records(pd.read_csv(results / "real_plain_outages.csv")),
            "events": _records(pd.read_csv(results / "real_events.csv")),
            "kpi_by_year": _records(pd.read_csv(results / "real_kpi_by_year.csv")),
            "calculator": json.loads(rc.read_text()) if rc.exists() else {},
        }
        if (results / "real_sweep.csv").exists():
            data["real"]["sweep"] = _records(pd.read_csv(results / "real_sweep.csv"))
        site = results / "real_site_thresholds.json"
        data["real"]["site"] = json.loads(site.read_text()) if site.exists() else None
        act = pd.read_csv(results / "real_daily_actual.csv", index_col=0)
        ref = pd.read_csv(results / "real_daily_peer.csv", index_col=0)
        data["real"]["daily"] = {"dates": list(act.index.astype(str).str[:10]),
                                 "ratio": {c: [None if not np.isfinite(v) else round(float(v), 3)
                                               for v in (act[c] / ref[c].where(ref[c] > 0)).clip(0, 1.5)]
                                           for c in act.columns}}
    if (results / "shading_real.csv").exists():
        daily = pd.read_csv(results / "shading_real_daily.csv")
        data["shading"] = {
            "results": _records(pd.read_csv(results / "shading_real.csv")),
            "sweep": _records(pd.read_csv(results / "shading_real_sweep.csv")),
            "classes": json.loads((results / "shading_real_classes.json").read_text()),
            "daily": {"dates": daily.iloc[:, 0].astype(str).str[:10].tolist(),
                      "ratio": [round(float(v), 4) for v in daily["ratio"]], "shaded": daily["shaded"].tolist()},
        }
    if (results / "sensitivity_min_duration.csv").exists():
        data["sensitivity"] = _records(pd.read_csv(results / "sensitivity_min_duration.csv"))
    path.write_text("window.DATA = " + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n")


def _num(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if np.isfinite(v) else None
