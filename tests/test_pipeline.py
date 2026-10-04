"""The pipeline on a small synthetic file: cleaning, the wet/dry label, and that the checks really catch bad data."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline import build, run_checks  # noqa: E402

COLS = ["country", "location_name", "latitude", "longitude", "timezone", "last_updated_epoch", "last_updated", "temperature_celsius",
        "feels_like_celsius", "condition_text", "wind_kph", "wind_degree", "gust_kph", "pressure_mb", "precip_mm", "humidity", "cloud",
        "visibility_km", "uv_index", "air_quality_PM2.5", "temperature_fahrenheit", "wind_mph"]


def row(loc="Berlin", epoch=1716000000, cond="Sunny", temp=20.0, hum=50, **kw):
    r = dict(country="Germany", location_name=loc, latitude=52.5, longitude=13.4, timezone="Europe/Berlin", last_updated_epoch=epoch,
             last_updated="2024-05-18 04:00", temperature_celsius=temp, feels_like_celsius=temp, condition_text=cond, wind_kph=10.0,
             wind_degree=90, gust_kph=15.0, pressure_mb=1010, precip_mm=0.0, humidity=hum, cloud=20, visibility_km=10.0, uv_index=3,
             **{"air_quality_PM2.5": 5.0}, temperature_fahrenheit=round(temp * 9 / 5 + 32, 1), wind_mph=round(10.0 / 1.609344, 1))
    r.update(kw)
    return r


def make(tmp_path, rows):
    p = tmp_path / "weather.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    return p


def failing(tmp_path, rows):
    p = make(tmp_path, rows)
    return {c["check_name"]: c["failing_rows"] for c in run_checks(build(p), p)}


def test_clean_data_passes_every_error_check(tmp_path):
    res = failing(tmp_path, [row(epoch=1716000000), row(epoch=1716086400, cond="Partly Cloudy"), row(epoch=1716172800, cond="Partly cloudy")])
    assert all(v == 0 for k, v in res.items() if not k.startswith(("condition spellings", "locations observed", "wind above")))


def test_condition_spellings_are_merged_and_wet_label_is_correct(tmp_path):
    p = make(tmp_path, [row(cond="Partly Cloudy"), row(epoch=1716086400, cond="Partly cloudy"), row(epoch=1716172800, cond="Light rain"),
                        row(epoch=1716259200, cond="Patchy rain nearby")])
    con = build(p)
    assert con.execute("SELECT COUNT(*) FROM dim_condition WHERE condition_text = 'partly cloudy'").fetchone()[0] == 1
    wet = dict(con.execute("SELECT condition_text, is_wet FROM dim_condition").fetchall())
    assert wet["light rain"] is True and wet["partly cloudy"] is False and wet["patchy rain nearby"] is None


def test_checks_catch_impossible_humidity_duplicates_and_unit_mismatch(tmp_path):
    res = failing(tmp_path, [row(hum=140), row(), row(temperature_fahrenheit=0.0, epoch=1716086400)])
    assert res["humidity within 0-100"] == 1
    assert res["location + observation time is unique"] == 1          # first two rows share location and time
    assert res["Celsius and Fahrenheit columns agree (within 0.5 F)"] == 1
