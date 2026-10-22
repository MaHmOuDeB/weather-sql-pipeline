"""Tests for the site export (src/export_site.py).

Most tests read only committed files (reports/quality.json, reports/results.json and reports/site/weather.json), so they run in CI.
The last ones need the local warehouse (``make pipeline``) and are skipped without it.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from export_site import build_export, dump, quarantine_rows  # noqa: E402

QUALITY = json.loads((ROOT / "reports" / "quality.json").read_text())
RESULTS = json.loads((ROOT / "reports" / "results.json").read_text())
EXPORT_PATH = ROOT / "reports" / "site" / "weather.json"
WAREHOUSE = ROOT / "data" / "warehouse.duckdb"
CSV = ROOT / "data" / "GlobalWeatherRepository.csv"
LICENCE = "Not stated (Kaggle: 'Other (specified in description)'; none given)"
needs_warehouse = pytest.mark.skipif(not WAREHOUSE.exists(), reason="needs the local warehouse (make pipeline)")


@pytest.fixture(scope="module")
def export() -> dict:
    return json.loads(EXPORT_PATH.read_text())


def test_committed_export_round_trips():
    text = EXPORT_PATH.read_text()
    assert dump(json.loads(text)) == text


def test_dump_is_canonical():
    assert dump({"b": [1, 2], "a": "é"}) == '{\n "a": "é",\n "b": [\n  1,\n  2\n ]\n}\n'
    with pytest.raises(ValueError):
        dump({"x": float("nan")})


def test_meta(export):
    meta = export["_meta"]
    assert meta["schema"] == "weather/1" and meta["source"] == "weather" and meta["provenance"] == "public-data"
    assert meta["dataset"]["name"] == "Global Weather Repository"
    assert meta["dataset"]["url"].startswith("https://www.kaggle.com/datasets/")
    assert meta["dataset"]["licence"] == LICENCE
    assert meta["inputs"] == [{"path": "data/GlobalWeatherRepository.csv", "rows": QUALITY["tables"]["stg_weather"],
                               "sha256": "238e4958dee28485f749518f5a89f31862db9909df2bedd8e0a7251e82fb324d"}]
    assert "generated" not in json.dumps(meta).lower()


def test_checks_equal_quality_json_with_renamed_keys(export):
    checks = export["tables"]["checks"]
    assert len(checks) == 18
    assert checks == [{"name": c["check_name"], "severity": c["severity"], "failing_rows": c["failing_rows"]} for c in QUALITY["checks"]]


def test_table_rows_equal_quality_json_tables(export):
    assert export["tables"]["table_rows"] == QUALITY["tables"]


def test_no_quarantined_row_is_exported(export):
    """Licence ruling: only the rule, the count and the range are published, never the readings or where they were taken."""
    assert set(export["tables"]) == {"checks", "table_rows", "cv_folds"}
    assert "quarantine" not in export["tables"]
    text = json.dumps(export)
    for name in ("Suva", "Fiji", "2024-05-19", "observed_at", "location_name"):
        assert name not in text, name


def test_data_and_check_facts(export):
    f = export["facts"]
    assert (f["data.observations"]["value"], f["data.locations"]["value"], f["data.countries"]["value"]) == (8725, 236, 191)
    assert (f["checks.count"]["value"], f["checks.error_count"]["value"], f["checks.sparse_locations"]["value"]) == (18, 14, 43)
    for key in ("data.observations", "data.locations", "data.countries", "checks.count", "checks.error_count", "checks.sparse_locations"):
        assert f[key]["unit"] == "count" and f[key]["decimals"] == 0
    assert f["data.locations"]["value"] == QUALITY["tables"]["dim_location"] == RESULTS["locations"]
    assert f["data.countries"]["value"] == RESULTS["countries"]
    assert f["data.wet_share"]["value"] == RESULTS["benchmark_wet_vs_dry"]["wet_share"] == 0.101
    assert f["baseline.dry_accuracy"]["value"] == RESULTS["benchmark_wet_vs_dry"]["majority_class_accuracy"] == 0.899
    assert f["data.wet_share"]["unit"] == f["baseline.dry_accuracy"]["unit"] == "share"
    assert f["data.wet_share"]["decimals"] == f["baseline.dry_accuracy"]["decimals"] == 0


def test_quarantine_facts_are_the_rule_the_count_and_the_range(export):
    f = export["facts"]
    assert f["quarantine.rows"]["value"] == 5 == QUALITY["tables"]["quarantine_observation"]
    assert f["quarantine.min_wind_kmh"]["value"] == 172.1
    assert f["quarantine.max_wind_kmh"]["value"] == 2963.2
    assert f["quarantine.threshold_kmh"]["value"] == 150
    assert f["quarantine.threshold_kmh"]["from"] == "sql/02_model.sql:23-26"
    for key in ("min_wind_kmh", "max_wind_kmh", "threshold_kmh"):
        assert f[f"quarantine.{key}"]["unit"] == "kmh" and f[f"quarantine.{key}"]["decimals"] == 0
    assert "wind above 150 km/h with ordinary conditions" in f["quarantine.threshold_kmh"]["label"]


def test_model_facts_come_from_results(export):
    f, m = export["facts"], RESULTS["benchmark_wet_vs_dry"]["by_location_folds"]
    assert f["model.xgb_f1"]["value"] == m["XGBoost"]["f1_wet"][0] == 0.52
    assert f["model.xgb_f1_sd"]["value"] == m["XGBoost"]["f1_wet"][1] == 0.06
    assert f["model.xgb_auc"]["value"] == m["XGBoost"]["roc_auc"][0]
    assert f"{f['model.xgb_auc']['value']:.2f}" == "0.91"
    assert f["model.rf_f1"]["value"] == m["Random Forest"]["f1_wet"][0] == 0.46
    for key in ("model.xgb_f1", "model.xgb_f1_sd", "model.xgb_auc", "model.rf_f1"):
        assert f[key]["unit"] == "decimal" and f[key]["decimals"] == 2


def test_cv_folds_summarise_to_the_committed_results(export):
    folds = export["tables"]["cv_folds"]
    assert len(folds) == 20 and {r["fold"] for r in folds} == {1, 2, 3, 4, 5}
    by_model = RESULTS["benchmark_wet_vs_dry"]["by_location_folds"]
    assert [m for m in dict.fromkeys(r["model"] for r in folds)] == list(by_model)
    for model, expected in by_model.items():
        rows = [r for r in folds if r["model"] == model]
        assert len(rows) == 5
        for ours, theirs in (("accuracy", "accuracy"), ("f1_rain", "f1_wet"), ("roc_auc", "roc_auc"), ("avg_precision", "avg_precision")):
            values = [r[ours] for r in rows]
            assert [round(float(np.mean(values)), 3), round(float(np.std(values)), 3)] == expected[theirs], (model, ours)
    # the five held-out groups partition the 8,158 labelled rows and the 236 places
    for model in by_model:
        rows = [r for r in folds if r["model"] == model]
        assert sum(r["test_rows"] for r in rows) == RESULTS["benchmark_wet_vs_dry"]["rows"]
        assert sum(r["test_places"] for r in rows) <= RESULTS["locations"]


def test_no_word_leak_anywhere(export):
    assert "leak" not in EXPORT_PATH.read_text().lower()
    assert "leakage_demo" not in json.dumps(export)


def test_build_export_does_not_read_the_leakage_section():
    results = {**RESULTS, "leakage_demo": "LEAKAGE_SENTINEL"}
    folds = [{"model": m, "fold": i, "accuracy": .9, "f1_wet": .5, "roc_auc": .9, "avg_precision": .5, "test_places": 47, "test_rows": 1600}
             for m in RESULTS["benchmark_wet_vs_dry"]["by_location_folds"] for i in range(1, 6)]
    quarantine = [{"wind_kph": w} for w in (172.1, 300.0, 400.0, 500.0, 2963.2)]
    text = dump(build_export(QUALITY, results, quarantine, folds, "0" * 64))
    assert "SENTINEL" not in text and "leak" not in text.lower()


@needs_warehouse
def test_quarantine_query_returns_five_rows_sorted_with_suva_first():
    rows = quarantine_rows(WAREHOUSE)
    assert len(rows) == 5
    assert [r["wind_kph"] for r in rows] == sorted(r["wind_kph"] for r in rows)
    first = rows[0]
    assert (first["location_name"], first["country"], first["observed_at_utc"], first["wind_kph"]) == ("Suva", "Fiji Islands", "2024-05-19T14:00:00Z", 172.1)
    assert rows[-1]["wind_kph"] == 2963.2


@needs_warehouse
@pytest.mark.skipif(not CSV.exists(), reason="needs the local CSV")
def test_export_bytes_do_not_depend_on_the_timezone(tmp_path):
    hashes = {}
    for tz in ("UTC", "America/Los_Angeles"):
        out = tmp_path / f"{tz.replace('/', '_')}.json"
        subprocess.run([sys.executable, str(ROOT / "src" / "export_site.py"), str(out)], check=True, env={**os.environ, "TZ": tz}, cwd=tmp_path)
        hashes[tz] = hashlib.sha256(out.read_bytes()).hexdigest()
    assert len(set(hashes.values())) == 1, hashes
    assert (tmp_path / "UTC.json").read_text() == EXPORT_PATH.read_text()


@needs_warehouse
def test_fold_scores_summarise_to_cross_validate():
    import analysis

    df = analysis.load()
    d = df[df.is_wet.notna()]
    x, y, groups = d[analysis.FEATURES].values, d.is_wet.astype(int).values, d.location_id.values
    splits = list(analysis.GroupKFold(5).split(x, y, groups=groups))
    make = analysis.model_factories()["Random Forest"]
    folds, oof = analysis.fold_scores(make, x, y, splits)
    assert len(folds) == 5 and set(folds[0]) == {"accuracy", "f1_wet", "roc_auc", "avg_precision"}
    summary, oof2 = analysis.cross_validate(make, x, y, splits)
    for key, values in summary.items():
        scores = [f[key] for f in folds]
        assert values == [round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)]
    assert np.allclose(oof, oof2, atol=1e-9)  # the forest sums its trees on several threads, so the last bits can differ
    assert summary == RESULTS["benchmark_wet_vs_dry"]["by_location_folds"]["Random Forest"]
