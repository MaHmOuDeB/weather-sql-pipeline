"""Five weather questions with proper statistics, plus a leakage-free model benchmark.

    python src/pipeline.py && python src/analysis.py        ->  reports/results.json, reports/figures/*.png
"""
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
df = con.execute("""
    SELECT f.*, l.country, l.location_name, l.latitude, l.longitude, c.condition_group, c.is_wet,
           EXTRACT(hour FROM f.observed_at_local) AS local_hour
    FROM fact_observation f JOIN dim_location l USING (location_id) JOIN dim_condition c USING (condition_id)""").df()
df["abs_lat"] = df.latitude.abs()
R = {"rows": len(df), "locations": int(df.location_id.nunique()), "countries": int(df.country.nunique()),
     "period": [str(df.observed_at_utc.min())[:10], str(df.observed_at_utc.max())[:10]]}

def kruskal_eps2(groups):
    h, p = stats.kruskal(*groups); n = sum(len(g) for g in groups)
    return {"H": round(float(h), 1), "p_value": float(p), "epsilon_squared": round(float((h - len(groups) + 1) / (n - len(groups))), 3)}

def spearman_ci(x, y, n_boot=1000):
    rho = stats.spearmanr(x, y)[0]; idx = np.arange(len(x)); boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx)); boots.append(stats.spearmanr(x.iloc[s], y.iloc[s])[0])
    return {"rho": round(float(rho), 3), "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)], "n": len(x)}

order = ["clear / sunny", "cloudy", "fog / mist", "rain / thunder"]
g = df[df.condition_group.isin(order)]

# ---- Trends ----
T = {}
for key, col in [("visibility_by_condition", "visibility_km"), ("precip_by_condition", "precip_mm"), ("temperature_by_condition", "temperature_celsius")]:
    T[key] = {**kruskal_eps2([g[g.condition_group == c][col] for c in order]),
              "medians": {c: round(float(g[g.condition_group == c][col].median()), 2) for c in order}}
# temperature differs by condition mostly because of where it is: add latitude as a control
import statsmodels.formula.api as smf
m = smf.ols("temperature_celsius ~ C(condition_group) + abs_lat", data=g).fit()
T["temperature_by_condition"]["ols_with_latitude_control"] = {"r2_condition_only": round(float(smf.ols("temperature_celsius ~ C(condition_group)", data=g).fit().rsquared), 3),
                                                              "r2_with_latitude": round(float(m.rsquared), 3)}
# UV vs cloud: UV depends on the hour (it sits at its floor of 1 overnight), so compare all hours with midday only
day = df[df.local_hour.between(10, 15)]
T["uv_vs_cloud"] = {"all_hours": spearman_ci(df.cloud, df.uv_index), "midday_10_to_15h_local": spearman_ci(day.cloud, day.uv_index),
                    "midday_rows": len(day), "share_of_rows_with_uv_at_floor_of_1": round(float((df.uv_index == 1).mean()), 3)}
# Temperature vs humidity: across places, and within a place over time
T["temperature_vs_humidity"] = {"pooled": spearman_ci(df.temperature_celsius, df.humidity)}
dm = df.assign(t=df.temperature_celsius - df.groupby("location_id").temperature_celsius.transform("mean"),
               h=df.humidity - df.groupby("location_id").humidity.transform("mean"))
T["temperature_vs_humidity"]["within_location"] = spearman_ci(dm.t, dm.h)
R["trends"] = T

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
axes[0].boxplot([g[g.condition_group == c].visibility_km for c in order], labels=[c.replace(" / ", "/\n") for c in order], showfliers=False)
axes[0].set_title("Visibility (km) by condition")
axes[1].scatter(day.cloud, day.uv_index, s=4, alpha=.35); axes[1].set_xlabel("Cloud cover (%)"); axes[1].set_ylabel("UV index"); axes[1].set_title("UV vs cloud, 10:00 to 15:59 local")
axes[2].scatter(dm.t, dm.h, s=4, alpha=.25); axes[2].set_xlabel("Temperature vs location mean (C)"); axes[2].set_ylabel("Humidity vs location mean (%)")
axes[2].set_title("Humidity vs temperature, within a place")
fig.tight_layout(); fig.savefig(FIG / "trends.png"); plt.close(fig)

# ---- Benchmark A: will it be raining? (classification, tested on places the model has never seen) ----
d = df[df.is_wet.notna()].copy(); d["wet"] = d.is_wet.astype(int)
FEATS = ["humidity", "cloud", "pressure_mb", "temperature_celsius", "wind_kph", "gust_kph", "visibility_km", "uv_index", "wind_degree", "abs_lat", "longitude"]
X, y, groups = d[FEATS].values, d.wet.values, d.location_id.values
models = {
    "k-NN (k=15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "Random Forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=.9, random_state=SEED, eval_metric="logloss", n_jobs=1),
}
def cv(make, splitter, **kw):
    acc, f1, auc = [], [], []
    for tr, te in splitter.split(X, y, **kw):
        m = make().fit(X[tr], y[tr]); p = m.predict(X[te]); pr = m.predict_proba(X[te])[:, 1]
        acc.append(accuracy_score(y[te], p)); f1.append(f1_score(y[te], p, zero_division=0)); auc.append(roc_auc_score(y[te], pr))
    return {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in (("accuracy", acc), ("f1_wet", f1), ("roc_auc", auc))}
bench = {"label": "rain, drizzle, showers or thunder happening now (ambiguous 'nearby/possible' rows left out)", "rows": len(d), "wet_share": round(float(y.mean()), 3),
         "majority_class_accuracy": round(float(1 - y.mean()), 3), "features": FEATS}
gkf, kf = GroupKFold(5), KFold(5, shuffle=True, random_state=SEED)
bench["by_location_folds"] = {n: cv(mk, gkf, groups=groups) for n, mk in models.items()}
bench["random_row_folds"] = {n: cv(mk, kf) for n, mk in models.items()}
R["benchmark_wet_vs_dry"] = bench

# ---- Benchmark B: the leakage trap. Predict temperature with and without the Fahrenheit twin and feels-like ----
t = df.copy()
tgt = t.temperature_celsius.values
def reg_r2(cols):
    X2 = t[cols].values; scores = []
    for tr, te in GroupKFold(5).split(X2, tgt, groups=t.location_id.values):
        scores.append(r2_score(tgt[te], RandomForestRegressor(200, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(X2[tr], tgt[tr]).predict(X2[te])))
    return round(float(np.mean(scores)), 3)
t["temperature_fahrenheit"] = t.temperature_celsius * 9 / 5 + 32       # the raw file's twin column, rebuilt exactly
honest = ["humidity", "cloud", "pressure_mb", "wind_kph", "visibility_km", "uv_index", "abs_lat", "longitude", "local_hour"]
R["leakage_demo"] = {"target": "temperature_celsius",
                     "r2_with_fahrenheit_and_feels_like": reg_r2(honest + ["temperature_fahrenheit", "feels_like_celsius"]),
                     "r2_with_fahrenheit_only": reg_r2(honest + ["temperature_fahrenheit"]),
                     "r2_honest_features_only": reg_r2(honest)}

# figure: F1 on the rain class (accuracy is useless here: 'always dry' already scores 90%)
fig, ax = plt.subplots(figsize=(6, 3.2)); names = list(models); w = .38; x = np.arange(len(names))
ax.bar(x - w/2, [bench["random_row_folds"][n]["f1_wet"][0] for n in names], w, label="random row folds", color="#9db8e0")
ax.bar(x + w/2, [bench["by_location_folds"][n]["f1_wet"][0] for n in names], w, label="held-out places", color="#2f5fb3")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, .7); ax.set_ylabel("F1 on the rain class"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Is it raining? 'Always dry' scores 90% accuracy and F1 = 0", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "benchmark.png"); plt.close(fig)

(ROOT / "reports" / "results.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=2, default=str))
