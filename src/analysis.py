"""Five weather questions with effect sizes, plus a rain classifier tested on places it has never seen.

Run from the repo root after ``python src/pipeline.py``::

    python src/analysis.py

Outputs ``reports/results.json`` and ``reports/figures/*.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
SEED = 42
GROUPS = ["clear / sunny", "cloudy", "fog / mist", "rain / thunder"]
FEATURES = ["humidity", "cloud", "pressure_mb", "temperature_celsius", "wind_kph", "gust_kph", "visibility_km", "uv_index", "wind_degree", "abs_lat", "longitude"]
HONEST_TEMPERATURE_FEATURES = ["humidity", "cloud", "pressure_mb", "wind_kph", "visibility_km", "uv_index", "abs_lat", "longitude", "local_hour"]
BLUE, LIGHT = "#2f5fb3", "#9db8e0"


def load() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    df = con.execute("""
        SELECT f.*, l.country, l.location_name, l.latitude, l.longitude, c.condition_group, c.is_wet,
               EXTRACT(hour FROM f.observed_at_local) AS local_hour
        FROM fact_observation f
        JOIN dim_location l USING (location_id) JOIN dim_condition c USING (condition_id)
        ORDER BY f.location_id, f.observed_at_utc""").df()
    df["abs_lat"] = df.latitude.abs()
    return df


# ---------- statistics ----------
def kruskal_effect(groups: list[pd.Series]) -> dict:
    """Kruskal-Wallis test with epsilon-squared as the effect size."""
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return {"H": round(float(h), 1), "p_value": float(p), "epsilon_squared": round(float((h - len(groups) + 1) / (n - len(groups))), 3)}


def spearman_ci(x: pd.Series, y: pd.Series, n_boot: int = 1000) -> dict:
    """Spearman correlation with a bootstrap 95% interval."""
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(x))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx))
        boots.append(stats.spearmanr(x.iloc[s], y.iloc[s])[0])
    return {"rho": round(float(stats.spearmanr(x, y)[0]), 3), "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)], "n": len(x)}


def trend_statistics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    g = df[df.condition_group.isin(GROUPS)]
    out: dict = {}
    for key, col in [("visibility_by_condition", "visibility_km"), ("precip_by_condition", "precip_mm"), ("temperature_by_condition", "temperature_celsius")]:
        out[key] = {**kruskal_effect([g[g.condition_group == c][col] for c in GROUPS]),
                    "medians": {c: round(float(g[g.condition_group == c][col].median()), 2) for c in GROUPS}}
    # Temperature differs by condition mostly because of where a place is, so add latitude as a control
    out["temperature_by_condition"]["ols_with_latitude_control"] = {
        "r2_condition_only": round(float(smf.ols("temperature_celsius ~ C(condition_group)", data=g).fit().rsquared), 3),
        "r2_with_latitude": round(float(smf.ols("temperature_celsius ~ C(condition_group) + abs_lat", data=g).fit().rsquared), 3)}
    # UV depends on the hour (it sits at its floor of 1 overnight), so compare all hours with midday only
    midday = df[df.local_hour.between(10, 15)]
    out["uv_vs_cloud"] = {"all_hours": spearman_ci(df.cloud, df.uv_index), "midday_10_to_15h_local": spearman_ci(midday.cloud, midday.uv_index),
                          "midday_rows": len(midday), "share_of_rows_with_uv_at_floor_of_1": round(float((df.uv_index == 1).mean()), 3)}
    within = df.assign(t=df.temperature_celsius - df.groupby("location_id").temperature_celsius.transform("mean"),
                       h=df.humidity - df.groupby("location_id").humidity.transform("mean"))
    out["temperature_vs_humidity"] = {"pooled": spearman_ci(df.temperature_celsius, df.humidity), "within_location": spearman_ci(within.t, within.h)}
    return out, g, midday, within


def plot_trends(g: pd.DataFrame, midday: pd.DataFrame, within: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].boxplot([g[g.condition_group == c].visibility_km for c in GROUPS], tick_labels=[c.replace(" / ", "/\n") for c in GROUPS], showfliers=False)
    axes[0].set_title("Visibility (km) by condition")
    axes[1].scatter(midday.cloud, midday.uv_index, s=4, alpha=.35, color=BLUE)
    axes[1].set(xlabel="Cloud cover (%)", ylabel="UV index", title="UV vs cloud, 10:00 to 15:59 local")
    axes[2].scatter(within.t, within.h, s=4, alpha=.25, color=BLUE)
    axes[2].set(xlabel="Temperature vs location mean (C)", ylabel="Humidity vs location mean (%)", title="Humidity vs temperature, within a place")
    fig.tight_layout()
    fig.savefig(FIG / "trends.png")
    plt.close(fig)


# ---------- rain classifier ----------
def model_factories() -> dict:
    return {
        "k-NN (k=15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "Random Forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=.9, random_state=SEED, eval_metric="logloss", n_jobs=1),
    }


def cross_validate(make, x: np.ndarray, y: np.ndarray, splits) -> tuple[dict, np.ndarray]:
    """Mean and standard deviation of the metrics across folds, plus out-of-fold probabilities."""
    acc, f1, auc, ap = [], [], [], []
    oof = np.zeros(len(y))
    for tr, te in splits:
        model = make().fit(x[tr], y[tr])
        pred, prob = model.predict(x[te]), model.predict_proba(x[te])[:, 1]
        oof[te] = prob
        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, zero_division=0))
        auc.append(roc_auc_score(y[te], prob))
        ap.append(average_precision_score(y[te], prob))
    pack = lambda v: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]  # noqa: E731
    return {"accuracy": pack(acc), "f1_wet": pack(f1), "roc_auc": pack(auc), "avg_precision": pack(ap)}, oof


def rain_benchmark(df: pd.DataFrame) -> tuple[dict, dict, np.ndarray]:
    d = df[df.is_wet.notna()].copy()
    x, y, groups = d[FEATURES].values, d.is_wet.astype(int).values, d.location_id.values
    by_place = list(GroupKFold(5).split(x, y, groups=groups))
    by_row = list(KFold(5, shuffle=True, random_state=SEED).split(x))
    out = {"label": "rain, drizzle, showers or thunder happening now (ambiguous 'nearby/possible' rows left out)", "rows": len(d), "wet_share": round(float(y.mean()), 3),
           "majority_class_accuracy": round(float(1 - y.mean()), 3), "features": FEATURES, "by_location_folds": {}, "random_row_folds": {}}
    oof = {}
    for name, make in model_factories().items():
        out["by_location_folds"][name], oof[name] = cross_validate(make, x, y, by_place)
        out["random_row_folds"][name], _ = cross_validate(make, x, y, by_row)
    return out, oof, y


def plot_benchmark(bench: dict, oof: dict, y: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    names = list(oof)
    x, w = np.arange(len(names)), .38
    axes[0].bar(x - w / 2, [bench["random_row_folds"][n]["f1_wet"][0] for n in names], w, label="random row folds", color=LIGHT)
    axes[0].bar(x + w / 2, [bench["by_location_folds"][n]["f1_wet"][0] for n in names], w, label="held-out places", color=BLUE)
    axes[0].set_xticks(x, names)
    axes[0].set(ylim=(0, .7), ylabel="F1 on the rain class", title="'Always dry' scores 90% accuracy but F1 = 0")
    axes[0].legend(fontsize=7, loc="upper left")
    for name in names:
        p, r, _ = precision_recall_curve(y, oof[name])
        axes[1].plot(r, p, label=f"{name} (AP {average_precision_score(y, oof[name]):.2f})")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=.8, label=f"No skill ({y.mean():.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, held-out places", ylim=(0, 1))
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "benchmark.png")
    plt.close(fig)


# ---------- the leakage trap ----------
def leakage_demo(df: pd.DataFrame) -> dict:
    t = df.copy()
    t["temperature_fahrenheit"] = t.temperature_celsius * 9 / 5 + 32   # the raw file's twin column, rebuilt exactly
    target, groups = t.temperature_celsius.values, t.location_id.values

    def r2(cols: list[str]) -> float:
        x, scores = t[cols].values, []
        for tr, te in GroupKFold(5).split(x, target, groups=groups):
            model = RandomForestRegressor(200, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(x[tr], target[tr])
            scores.append(r2_score(target[te], model.predict(x[te])))
        return round(float(np.mean(scores)), 3)

    return {"target": "temperature_celsius",
            "r2_with_fahrenheit_and_feels_like": r2(HONEST_TEMPERATURE_FEATURES + ["temperature_fahrenheit", "feels_like_celsius"]),
            "r2_with_fahrenheit_only": r2(HONEST_TEMPERATURE_FEATURES + ["temperature_fahrenheit"]),
            "r2_honest_features_only": r2(HONEST_TEMPERATURE_FEATURES)}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    df = load()
    results: dict = {"rows": len(df), "locations": int(df.location_id.nunique()), "countries": int(df.country.nunique()),
                     "period": [str(df.observed_at_utc.min())[:10], str(df.observed_at_utc.max())[:10]]}
    trends, g, midday, within = trend_statistics(df)
    results["trends"] = trends
    plot_trends(g, midday, within)
    bench, oof, y = rain_benchmark(df)
    results["benchmark_wet_vs_dry"] = bench
    plot_benchmark(bench, oof, y)
    results["leakage_demo"] = leakage_demo(df)
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
