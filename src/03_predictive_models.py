"""
DriveWise AI+ — Phase 2: Predictive Models
=============================================
Trains on data/drivewise_market_history.csv (real) and
data/drivewise_acquisition_pool.csv (real + simulated dealer layer).

Produces 4 models:
  1. Fair value regression   -- REAL data, REAL target. The core model.
  2. Depreciation curves     -- REAL data, fitted per brand.
  3. Purchase-risk score     -- REAL data, unsupervised outlier detection
                                 (deliberately independent of the simulated
                                 condition_grade, so it's a genuine model,
                                 not one that just re-learns a rule).
  4. Demand / days-on-market -- trained on SIMULATED days_on_market.
                                 Flagged clearly: this demonstrates the
                                 pipeline shape, not a validated real-world
                                 demand signal, since no real
                                 sell-through/time-on-market data exists.

Saves trained models to models/ and prints a metrics summary for the report.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from pathlib import Path

MARKET_PATH = "../data/drivewise_market_history.csv"
POOL_PATH = "../data/drivewise_acquisition_pool.csv"
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. FAIR VALUE MODEL
# ----------------------------------------------------------------------
def build_fair_value_model(market: pd.DataFrame):
    print("\n" + "=" * 60)
    print("1. FAIR VALUE MODEL")
    print("=" * 60)

    # High-cardinality categoricals (model, city) get frequency-encoded
    # rather than one-hot -- one-hot on 295 cities / 700+ models would blow
    # up dimensionality for little gain. Low-cardinality ones (brand,
    # transmission, fuel_type) are one-hot encoded normally.
    df = market.copy()
    model_freq = df["model"].value_counts(normalize=True)
    city_freq = df["city"].value_counts(normalize=True)
    df["model_freq"] = df["model"].map(model_freq)
    df["city_freq"] = df["city"].map(city_freq)

    features_num = ["year", "mileage_km", "car_age", "mileage_per_year",
                     "model_freq", "city_freq"]
    features_cat = ["brand", "transmission", "fuel_type"]
    target = "price_inr"

    X = df[features_num + features_cat]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ], remainder="passthrough")

    models = {
        "LinearRegression": Pipeline([("prep", preprocess), ("model", LinearRegression())]),
        "RandomForest": Pipeline([("prep", preprocess),
                                   ("model", RandomForestRegressor(
                                       n_estimators=300, max_depth=18,
                                       min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1))]),
        "XGBoost": Pipeline([("prep", preprocess),
                              ("model", xgb.XGBRegressor(
                                  n_estimators=400, max_depth=7, learning_rate=0.06,
                                  subsample=0.85, colsample_bytree=0.85,
                                  random_state=RANDOM_STATE, n_jobs=-1))]),
    }

    results = []
    best_name, best_model, best_r2 = None, None, -np.inf
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
        results.append({"model": name, "RMSE": round(rmse), "MAE": round(mae),
                         "R2": round(r2, 4), "MAPE_%": round(mape, 2)})
        print(f"  {name:16s}  RMSE=Rs.{rmse:>10,.0f}  MAE=Rs.{mae:>10,.0f}  "
              f"R2={r2:.4f}  MAPE={mape:.1f}%")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, pipe, r2

    print(f"\n  Best model: {best_name} (R2={best_r2:.4f})")
    joblib.dump(best_model, MODELS_DIR / "fair_value_model.joblib")
    joblib.dump({"model_freq": model_freq, "city_freq": city_freq},
                 MODELS_DIR / "fair_value_freq_maps.joblib")

    pd.DataFrame(results).to_csv(MODELS_DIR / "fair_value_model_comparison.csv", index=False)
    return best_model, pd.DataFrame(results)


# ----------------------------------------------------------------------
# 2. DEPRECIATION CURVES (per brand)
# ----------------------------------------------------------------------
def build_depreciation_curves(market: pd.DataFrame, top_n_brands: int = 8):
    print("\n" + "=" * 60)
    print("2. DEPRECIATION CURVES")
    print("=" * 60)

    top_brands = market["brand"].value_counts().head(top_n_brands).index.tolist()
    curves = {}
    for brand in top_brands:
        sub = market[market["brand"] == brand]
        curve = sub.groupby("car_age")["price_inr"].median()
        curve = curve[curve.index <= 15]  # keep it to a readable age range
        if len(curve) < 4:
            continue
        # fit an exponential decay: price = a * exp(-b * age)
        ages = curve.index.values.astype(float)
        prices = curve.values.astype(float)
        log_prices = np.log(prices.clip(min=1))
        b, log_a = np.polyfit(ages, log_prices, 1)
        a = np.exp(log_a)
        annual_decay_pct = (1 - np.exp(b)) * 100
        curves[brand] = {"a": a, "b": b, "annual_decay_pct": round(annual_decay_pct, 1),
                          "n_listings": len(sub)}
        print(f"  {brand:15s}  ~{annual_decay_pct:5.1f}%/yr value loss  (n={len(sub)})")

    pd.DataFrame(curves).T.to_csv(MODELS_DIR / "depreciation_curves.csv")
    return curves


# ----------------------------------------------------------------------
# 3. PURCHASE-RISK SCORE (unsupervised, real data only)
# ----------------------------------------------------------------------
def build_risk_model(pool: pd.DataFrame):
    print("\n" + "=" * 60)
    print("3. PURCHASE-RISK SCORE (Isolation Forest, real features only)")
    print("=" * 60)
    print("  Deliberately built from REAL columns only (price deviation, mileage")
    print("  pattern, age) -- not from the simulated condition_grade, so this is")
    print("  a genuine statistical model, not one re-deriving a known rule.")

    features = ["price_vs_model_median_pct", "mileage_per_year", "car_age", "price_per_km"]
    X = pool[features].replace([np.inf, -np.inf], np.nan).dropna()

    iso = IsolationForest(n_estimators=300, contamination=0.12, random_state=RANDOM_STATE)
    iso.fit(X)
    raw_score = -iso.score_samples(X)  # higher = more anomalous = riskier
    risk_pct = pd.Series(raw_score, index=X.index).rank(pct=True) * 100

    pool = pool.copy()
    pool.loc[X.index, "risk_score_0_100"] = risk_pct.round(1)
    pool["risk_score_0_100"] = pool["risk_score_0_100"].fillna(pool["risk_score_0_100"].median())

    print(f"  scored {len(X)} listings")
    print("  risk score distribution:")
    print(pool["risk_score_0_100"].describe().round(1))

    joblib.dump(iso, MODELS_DIR / "risk_model_isolationforest.joblib")
    return pool, iso


# ----------------------------------------------------------------------
# 4. DEMAND / DAYS-ON-MARKET MODEL (trained on SIMULATED target -- flagged)
# ----------------------------------------------------------------------
def build_demand_model(pool: pd.DataFrame):
    print("\n" + "=" * 60)
    print("4. DEMAND / DAYS-ON-MARKET MODEL")
    print("  *** trained on SIMULATED days_on_market -- see caveat in report ***")
    print("=" * 60)

    df = pool.copy()
    model_freq = df["model"].value_counts(normalize=True)
    df["model_freq"] = df["model"].map(model_freq)

    features_num = ["price_vs_model_median_pct", "mileage_km", "car_age", "model_freq"]
    features_cat = ["brand", "fuel_type"]
    target = "days_on_market"

    X = df[features_num + features_cat]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ], remainder="passthrough")
    pipe = Pipeline([("prep", preprocess),
                      ("model", RandomForestRegressor(n_estimators=200, max_depth=10,
                                                        random_state=RANDOM_STATE, n_jobs=-1))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"  RMSE={rmse:.1f} days   R2={r2:.4f}")
    print("  (high R2 here mostly reflects that days_on_market was itself generated")
    print("   from these same features -- expected, not evidence of real predictive power)")

    joblib.dump(pipe, MODELS_DIR / "demand_dom_model.joblib")
    return pipe


def main():
    market = pd.read_csv(MARKET_PATH)
    pool = pd.read_csv(POOL_PATH)

    fair_value_model, comparison = build_fair_value_model(market)
    curves = build_depreciation_curves(market)
    pool_scored, risk_model = build_risk_model(pool)
    demand_model = build_demand_model(pool)

    pool_scored.to_csv("../data/drivewise_acquisition_pool_scored.csv", index=False)
    print(f"\nSaved scored pool -> ../data/drivewise_acquisition_pool_scored.csv")
    print(f"All models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
