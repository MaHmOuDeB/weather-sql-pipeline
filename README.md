# Global weather: SQL pipeline, data-quality checks and a fair model benchmark

A small, tested data pipeline over 8,725 weather observations from 236 locations in 191 countries (16 May to 28 Jun
2024): raw CSV, staging, star schema, automated data-quality checks, then five statistical questions and a rain
classifier that is tested on places it has never seen.

Rebuilt in 2026 from my Master's project *Data Engineering and Business Analytics* (2024). Rebuilding it turned up a
real problem in the first version: it predicted each target from all the other columns, including that target's own unit
twin (temperature in Fahrenheit for Celsius, visibility in miles for kilometres), and it scaled the data before splitting.
That is why it reported errors close to zero. This version removes the leak and shows its effect (see below).

## Data

Kaggle, *Global Weather Repository* (daily snapshots collected from WeatherAPI.com). The file is not in this repo;
download it from Kaggle (check the dataset page for the licence) and place it at `data/GlobalWeatherRepository.csv`.

## Pipeline (`sql/`, `src/pipeline.py`)

| Step | What it does |
|---|---|
| `01_staging.sql` | Typed, snake_case table; trims text and lower-cases conditions; drops the redundant Fahrenheit, mph, inch and mile columns (kept only to prove they are exact conversions) |
| `02_model.sql` | `dim_location` (236), `dim_condition` (27, with a condition group and a wet/dry flag), `fact_observation` (8,725) |
| `03_checks.sql` | 16 checks: row counts, key uniqueness, missing keys, value ranges, unit consistency, plausibility. `error` checks stop the pipeline; `warn` and `info` are reported |

Result on this file: all error-level checks pass. Findings worth knowing: 5 observations with wind above 150 km/h (kept,
flagged for review), 43 of the 236 locations have fewer than 20 observations, and one condition appears under two
spellings ("Partly Cloudy" and "Partly cloudy"), which the cleaning merges. The wet/dry flag leaves "rain nearby" and
"rain possible" as unknown, since they do not say it is raining at the location.

`tests/` runs the same SQL on a small synthetic file and proves that the checks catch impossible humidity, duplicate
observations and mismatched units, and that the label logic and spelling merge work.

## Five questions, with effect sizes

| Question | Answer |
|---|---|
| Does visibility differ by condition? | Yes. Fog and mist have a median of 4 km versus 10 km for clear and cloudy (Kruskal-Wallis, epsilon-squared 0.18) |
| Does precipitation differ by condition? | Yes, with the largest effect (epsilon-squared 0.36). Median 0.13 mm in rain and thunder, 0 elsewhere |
| Does temperature differ by condition? | Barely. The effect is small (epsilon-squared 0.04, medians 27.1 C clear versus 24.8 C rain). Adding latitude lifts R-squared from 0.04 to 0.20, so *where* a place is matters more than the weather it is having |
| UV index versus cloud cover | Moderate negative relation: Spearman rho -0.41 (95% CI -0.43 to -0.39). Using only 10:00 to 15:59 local time gives -0.40, so the result is not a day/night artefact |
| Temperature versus humidity | Pooled rho -0.35, but -0.62 within a place over time. Comparing places hides how strongly the two move together at one place |

![trends](reports/figures/trends.png)

## Rain classifier, tested fairly

Target: is it raining, drizzling, showering or thundering right now (10% of 8,163 labelled rows). Features exclude
precipitation and the condition text, since they define the target. Folds are grouped by location, so every model is
scored on places it has never seen. "Always dry" already reaches 90% accuracy, so accuracy alone says nothing; F1 on the
rain class and ROC AUC do.

| Model (held-out places, 5 folds) | Accuracy | F1 (rain) | ROC AUC |
|---|---|---|---|
| Always predict dry | 0.900 | 0.000 | 0.500 |
| k-NN | 0.905 | 0.30 | 0.84 |
| Random Forest | 0.921 | 0.47 | 0.90 |
| XGBoost | 0.924 | 0.52 | 0.91 |

Splitting rows at random instead of by place flatters the models only a little here (XGBoost F1 0.54 versus 0.52), because
each place has few observations. XGBoost is best, but an F1 of about 0.5 still means many rain cases are missed or
false alarms, so I would not use it as a forecast.

![benchmark](reports/figures/benchmark.png)

## The leakage trap

Predicting temperature in Celsius with a random forest (held-out places, R-squared):

| Features | R-squared |
|---|---|
| Humidity, cloud, pressure, wind, visibility, UV, latitude, longitude, local hour | 0.77 |
| The same plus temperature in Fahrenheit | 0.999 |

The jump to 0.999 is not skill: Fahrenheit is the same number in other units. Any near-perfect score should trigger a check
of what the model can see.

## Limits

- Six weeks of data and mostly one observation a day per place, so this says nothing about seasons or forecasting.
- Conditions are the weather service's text labels, not ground truth, and the wet label is a judgement call (documented above).
- The first version of this project used a standardised copy of the data; this one works on the original units.

## Run it

```bash
pip install -r requirements.txt
python src/pipeline.py     # builds data/warehouse.duckdb, runs the checks, writes reports/quality.json
python src/analysis.py     # statistics and the benchmark -> reports/results.json, reports/figures/
pytest
```
