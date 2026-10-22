"""Write ``reports/site/weather.json``, the export the portfolio site reads.

Run from anywhere after ``make pipeline`` and ``make analysis``::

    make export

The numbers come from ``reports/quality.json`` and ``reports/results.json`` (copied with their rounding) and from re-running the
five-fold, held-out-places cross-validation, so the file holds each fold's scores. The file is canonical JSON with no generation
time, so two runs give the same bytes, whatever the machine's time zone.

The five quarantined readings are read only to compute the count and the range of their wind speed. No reading, place or
time is written: the dataset's licence is not stated (RULING-facts item 6), so the export carries the rule, the count and the range.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "site" / "weather.json"
CSV = ROOT / "data" / "GlobalWeatherRepository.csv"
WAREHOUSE = ROOT / "data" / "warehouse.duckdb"
EXPECTED_CSV_SHA256 = "238e4958dee28485f749518f5a89f31862db9909df2bedd8e0a7251e82fb324d"
LICENCE = "Not stated (Kaggle: 'Other (specified in description)'; none given)"
THRESHOLD_KMH = 150  # sql/02_model.sql:23-26
SPARSE_CHECK = "locations observed fewer than 20 times"
FOLD_KEYS = ("accuracy", "f1_wet", "roc_auc", "avg_precision")


def dump(obj: dict) -> str:
    """Canonical JSON: sorted keys, one-space indent, no NaN, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def quarantine_rows(warehouse: Path = WAREHOUSE) -> list[dict]:
    """The wind speed of each quarantined reading, weakest first. Used for the count and the range only; never written to the export."""
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        con.execute("SET TimeZone='UTC'")
        rows = con.execute("SELECT wind_kph FROM quarantine_observation ORDER BY wind_kph").fetchall()
    finally:
        con.close()
    return [{"wind_kph": float(w)} for (w,) in rows]


def fold_rows(df: pd.DataFrame) -> list[dict]:
    """Per model and fold: the scores and the size of the held-out set. Same data, features and splits as analysis.rain_benchmark."""
    import analysis  # noqa: PLC0415  (it pulls in matplotlib and statsmodels, which the tests of build_export do not need)

    d = df[df.is_wet.notna()]
    x, y, groups = d[analysis.FEATURES].values, d.is_wet.astype(int).values, d.location_id.values
    splits = list(analysis.GroupKFold(5).split(x, y, groups=groups))
    rows = []
    for model, make in analysis.model_factories().items():
        scores, _ = analysis.fold_scores(make, x, y, splits)
        for i, (score, (_, te)) in enumerate(zip(scores, splits, strict=True), start=1):
            rows.append({"model": model, "fold": i, "test_places": int(len(set(groups[te]))), "test_rows": int(len(te)), **score})
    return rows


def _fact(value, unit: str, decimals: int, label: str, source: str) -> dict:
    return {"value": value, "unit": unit, "decimals": decimals, "label": label, "from": source}


def build_facts(quality: dict, results: dict, quarantine: list[dict]) -> dict:
    bench = results["benchmark_wet_vs_dry"]
    xgb, forest = bench["by_location_folds"]["XGBoost"], bench["by_location_folds"]["Random Forest"]
    sparse_at = next(i for i, c in enumerate(quality["checks"]) if c["check_name"] == SPARSE_CHECK)
    winds = [r["wind_kph"] for r in quarantine]
    q, r = "quality.json#/", "results.json#/"
    return {
        "data.observations": _fact(quality["tables"]["stg_weather"], "count", 0, "Weather observations loaded", q + "tables/stg_weather"),
        "data.locations": _fact(quality["tables"]["dim_location"], "count", 0, "Places observed", q + "tables/dim_location"),
        "data.countries": _fact(results["countries"], "count", 0, "Countries observed", r + "countries"),
        "data.wet_share": _fact(bench["wet_share"], "share", 0, "Share of labelled observations where it is raining now", r + "benchmark_wet_vs_dry/wet_share"),
        "checks.count": _fact(len(quality["checks"]), "count", 0, "Automated data-quality checks", q + "checks"),
        "checks.error_count": _fact(sum(c["severity"] == "error" for c in quality["checks"]), "count", 0,
                                    "Checks whose failure stops the pipeline (error level)", q + "checks"),
        "checks.sparse_locations": _fact(quality["checks"][sparse_at]["failing_rows"], "count", 0, "Places observed fewer than 20 times", q + f"checks/{sparse_at}/failing_rows"),
        "quarantine.rows": _fact(len(winds), "count", 0, "Readings held back as sensor errors", q + "tables/quarantine_observation"),
        "quarantine.max_wind_kmh": _fact(max(winds), "kmh", 0, "Highest held-back wind speed", "warehouse.duckdb#quarantine_observation/max(wind_kph)"),
        "quarantine.min_wind_kmh": _fact(min(winds), "kmh", 0, "Lowest held-back wind speed", "warehouse.duckdb#quarantine_observation/min(wind_kph)"),
        "quarantine.threshold_kmh": _fact(THRESHOLD_KMH, "kmh", 0, "Rule: wind above 150 km/h with ordinary conditions is held back", "sql/02_model.sql:23-26"),
        "model.xgb_f1": _fact(xgb["f1_wet"][0], "decimal", 2, "F1 on rain, XGBoost, tested on places it had not seen (mean of 5 folds)",
                              r + "benchmark_wet_vs_dry/by_location_folds/XGBoost/f1_wet/0"),
        "model.xgb_f1_sd": _fact(xgb["f1_wet"][1], "decimal", 2, "Standard deviation of that F1 across the 5 folds",
                                 r + "benchmark_wet_vs_dry/by_location_folds/XGBoost/f1_wet/1"),
        "model.xgb_auc": _fact(xgb["roc_auc"][0], "decimal", 2, "ROC AUC, XGBoost, tested on places it had not seen (mean of 5 folds)",
                               r + "benchmark_wet_vs_dry/by_location_folds/XGBoost/roc_auc/0"),
        "model.rf_f1": _fact(forest["f1_wet"][0], "decimal", 2, "F1 on rain, random forest, tested on places it had not seen (mean of 5 folds)",
                             r + "benchmark_wet_vs_dry/by_location_folds/Random Forest/f1_wet/0"),
        "baseline.dry_accuracy": _fact(bench["majority_class_accuracy"], "share", 0, "Accuracy of always predicting dry",
                                       r + "benchmark_wet_vs_dry/majority_class_accuracy"),
    }


def build_tables(quality: dict, folds: list[dict]) -> dict:
    return {
        "checks": [{"name": c["check_name"], "severity": c["severity"], "failing_rows": c["failing_rows"]} for c in quality["checks"]],
        "table_rows": dict(quality["tables"]),
        "cv_folds": [{"model": f["model"], "fold": f["fold"], "f1_rain": round(f["f1_wet"], 4), "roc_auc": round(f["roc_auc"], 4),
                      "avg_precision": round(f["avg_precision"], 4), "accuracy": round(f["accuracy"], 4),
                      "test_places": f["test_places"], "test_rows": f["test_rows"]} for f in folds],
    }


def build_export(quality: dict, results: dict, quarantine: list[dict], folds: list[dict], csv_sha256: str) -> dict:
    """``quarantine`` is read only to compute the count and the range; no row of it is written."""
    return {
        "_meta": {
            "schema": "weather/1", "source": "weather", "generator": "src/export_site.py",
            "inputs": [{"path": "data/GlobalWeatherRepository.csv", "sha256": csv_sha256, "rows": quality["tables"]["stg_weather"]}],
            "engine": {"duckdb": duckdb.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": ".".join(platform.python_version_tuple()[:2]),
                       "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
            "dataset": {"name": "Global Weather Repository", "url": "https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository",
                        "licence": LICENCE},
            "provenance": "public-data",
        },
        "facts": build_facts(quality, results, quarantine),
        "tables": build_tables(quality, folds),
    }


def main(out: Path = OUT) -> None:
    import analysis  # noqa: PLC0415

    digest = hashlib.sha256(CSV.read_bytes()).hexdigest()
    if digest != EXPECTED_CSV_SHA256:
        print(f"warning: {CSV.name} has sha256 {digest}, expected {EXPECTED_CSV_SHA256}; the export records the hash it saw", file=sys.stderr)
    quality = json.loads((ROOT / "reports" / "quality.json").read_text())
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump(build_export(quality, results, quarantine_rows(), fold_rows(analysis.load()), digest)))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
