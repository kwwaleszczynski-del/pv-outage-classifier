"""Command line.

Usage:
    python -m pv_outage run --input plant.parquet [--meta meta.json] --out results/my_plant
    python -m pv_outage simulate --year 2023 --seed 11          # a synthetic plant with its truth table
    python -m pv_outage benchmark --jobs 4                      # synthetic method comparison (results/)
    python -m pv_outage sensitivity                             # the same test plants, longer minimum events
    python -m pv_outage real                                    # NREL PVDAQ fleet (downloads ~60 MB once)
    python -m pv_outage site-facts                              # PVDAQ outage chronology, channel clocks
    python -m pv_outage shading                                 # labelled shading, Zenodo 23114394 (69 MB)
    python -m pv_outage report                                  # figures and docs/data.js
    python -m pv_outage all --jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("pv_outage")


def cmd_run(a):
    from pv_outage import pipeline, schema

    meta = json.loads(Path(a.meta).read_text()) if a.meta else {}
    df = schema.read_long(Path(a.input))
    problems = schema.validate(df)
    if problems:
        raise SystemExit("input problems: " + "; ".join(problems))
    panel = schema.to_panel(df, meta=meta)
    model = None
    if a.classifier == "gbm":
        # models.pkl is written locally by `benchmark`; never load one from elsewhere (pickle runs code)
        with open(Path(a.data) / "models.pkl", "rb") as fh:
            model = pickle.load(fh)["event_gbm"]
    treatment = json.loads(Path(a.treatment).read_text()) if a.treatment else None
    res = pipeline.analyse(panel, pipeline.load_thresholds(a.thresholds), calibration=a.calibration,
                           event_model=model, treatment=treatment, min_hours=a.min_hours)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = res["events"]
    cols = [c for c in pipeline.EVENT_COLUMNS if c in ev.columns]
    ev[cols].to_csv(out / "events.csv", index=False)
    res["per_inverter"].to_csv(out / "kpi_per_inverter.csv", index=False)
    (out / "kpi.json").write_text(json.dumps(res["kpi"], indent=1, default=float))
    k = res["kpi"]
    log.info("%d events; availability time %.2f %%, energy %.2f %%; lost energy by category %s", len(ev),
             100 * k["availability_time"], 100 * k["availability_energy"],
             {c: round(v) for c, v in k["lost_by_category_kwh"].items()})


def cmd_simulate(a):
    from pv_outage import benchmark

    path = benchmark.make_plant(Path(a.data), a.year, a.seed)
    log.info("long table %s (truth, labels and metadata next to it)", path)


def cmd_fetch_weather(a):
    from pv_outage import benchmark, weather

    weather.fetch_era5(benchmark.LAT, benchmark.LON, 2021, 2024, Path(a.data) / "era5")


def cmd_benchmark(a):
    from pv_outage import benchmark

    s = benchmark.run(Path(a.data), Path(a.results), jobs=a.jobs)
    for row in s["detection"]:
        log.info("%-10s P %.2f R %.2f F1 %.2f", row["method"], row["precision"], row["recall"], row["f1"])


def cmd_sensitivity(a):
    from pv_outage import benchmark

    g = benchmark.sensitivity(Path(a.data), Path(a.results))
    log.info("min-duration sensitivity:%s%s", chr(10), g.round(3).to_string())


def cmd_real(a):
    from pv_outage import real

    real.run(Path(a.data), Path(a.results))


def cmd_site_facts(a):
    from pv_outage import site_facts

    site_facts.run(Path(a.data), Path(a.results))


def cmd_shading(a):
    from pv_outage import shading_real

    models = None
    mp = Path(a.data) / "models.pkl"
    if mp.exists():  # written locally by `benchmark`; never load one from elsewhere (pickle runs code)
        with open(mp, "rb") as fh:
            models = pickle.load(fh)
    shading_real.run(Path(a.data), Path(a.results), models)


def cmd_report(a):
    from pv_outage import report

    report.run(Path(a.results), ROOT / "docs")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pv_outage", description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(ROOT / "data"), help="data and cache directory")
    p.add_argument("--results", default=str(ROOT / "results"), help="results directory")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse a plant in the long format")
    r.add_argument("--input", required=True, help="long table (.parquet or .csv), see README")
    r.add_argument("--meta", help="optional JSON: lat, lon, tilt, azimuth, pdc_kw, pac_kw, gamma_pdc, status_map")
    r.add_argument("--out", default="results/run")
    r.add_argument("--calibration", default="poa", choices=["poa", "scalar", "hour_month"],
                   help="hour_month when devices face other ways than the irradiance sensor")
    r.add_argument("--classifier", default="rules", choices=["rules", "gbm"])
    r.add_argument("--thresholds", help="thresholds JSON (default results/thresholds.json)")
    r.add_argument("--treatment", help="JSON {class: unavailable|excluded|available} for the KPIs")
    r.add_argument("--min-hours", type=float, default=0.5, help="shortest event kept (flagged daylight hours)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("simulate", help="simulate one plant-year with injected events")
    s.add_argument("--year", type=int, default=2023)
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=cmd_simulate)
    sub.add_parser("fetch-weather", help="ERA5 from Open-Meteo (one point, cached)").set_defaults(fn=cmd_fetch_weather)
    b = sub.add_parser("benchmark", help="synthetic method comparison")
    b.add_argument("--jobs", type=int, default=1)
    b.set_defaults(fn=cmd_benchmark)
    sub.add_parser("sensitivity", help="test plants with longer minimum event durations").set_defaults(
        fn=cmd_sensitivity)
    sub.add_parser("real", help="NREL PVDAQ Las Vegas fleet").set_defaults(fn=cmd_real)
    sub.add_parser("site-facts", help="PVDAQ campus: outage chronology and channel clock offsets").set_defaults(
        fn=cmd_site_facts)
    sub.add_parser("shading", help="real labelled shading, Zenodo 23114394 (file in data/zenodo)").set_defaults(
        fn=cmd_shading)
    sub.add_parser("report", help="figures and dashboard data").set_defaults(fn=cmd_report)
    al = sub.add_parser("all", help="benchmark, real, shading, report")
    al.add_argument("--jobs", type=int, default=1)
    al.set_defaults(fn=lambda a: (cmd_benchmark(a), cmd_sensitivity(a), cmd_real(a), cmd_shading(a), cmd_report(a)))
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    a.fn(a)


if __name__ == "__main__":
    main()
