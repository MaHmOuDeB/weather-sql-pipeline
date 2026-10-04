"""Load the raw CSV, build the star schema in DuckDB, run the data-quality checks.

    python src/pipeline.py [path/to/GlobalWeatherRepository.csv]

Writes data/warehouse.duckdb (only if every error-level check passes) and reports/quality.json.
"""
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "sql"


def sql_path(csv_path) -> str:
    """The CSV path as a SQL string literal body (single quotes doubled)."""
    return str(csv_path).replace("'", "''")


def build(csv_path, db_path=":memory:"):
    con = duckdb.connect(str(db_path))
    for name in ("01_staging.sql", "02_model.sql"):
        con.execute((SQL / name).read_text().replace("{csv_path}", sql_path(csv_path)))
    return con


def run_checks(con, csv_path):
    sql = (SQL / "03_checks.sql").read_text().replace("{csv_path}", sql_path(csv_path)).rstrip().rstrip(";")
    return con.execute(sql).df().to_dict("records")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "GlobalWeatherRepository.csv"
    db_path = ROOT / "data" / "warehouse.duckdb"
    building = ROOT / "data" / "warehouse.building.duckdb"
    building.unlink(missing_ok=True)
    con = build(csv_path, building)
    checks = run_checks(con, csv_path)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("stg_weather", "dim_location", "dim_condition", "fact_observation", "quarantine_observation")}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "quality.json").write_text(json.dumps({"tables": counts, "checks": checks}, indent=2, default=int))
    for c in checks:
        flag = "OK  " if c["failing_rows"] == 0 else c["severity"].upper().ljust(4)
        print(f"{flag} {c['check_name']}: {c['failing_rows']}")
    print(counts)
    con.close()
    if any(c["severity"] == "error" and c["failing_rows"] > 0 for c in checks):
        building.unlink(missing_ok=True)
        sys.exit("pipeline stopped: an error-level check failed, no warehouse written")
    db_path.unlink(missing_ok=True)
    building.rename(db_path)


if __name__ == "__main__":
    main()
