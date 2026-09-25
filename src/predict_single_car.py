"""
DriveWise AI+ -- Interactive Single Vehicle Prediction & Appraisal Tool
=======================================================================
Allows you or an evaluator to test the trained machine learning pipeline
on ANY vehicle (either pre-configured samples or custom inputs).

Outputs:
  - Fair Market Value Appraisal (Rs.)
  - Asking Price Value Gap (% discount or premium)
  - Unsupervised Purchase Risk Score (0-100)
  - Parametric Brand Annual Depreciation Rate (%/year)
  - Predicted Days on Market (DOM)
  - Composite Dealer Intelligence Index (DII Score 0-100)
  - Action Recommendation (STRONG_BUY / ACCUMULATE / HOLD / REJECT)
  - Negotiation Strategy: Opening Offer & Walk-Away Ceiling (Rs.)
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# File paths
FAIR_VALUE_MODEL = BASE_DIR / "fair_value_model.joblib"
FREQ_MAPS        = BASE_DIR / "fair_value_freq_maps.joblib"
RISK_MODEL       = BASE_DIR / "risk_model_isolationforest.joblib"
DEMAND_MODEL     = BASE_DIR / "demand_dom_model.joblib"
DEPRECIATION_CSV = BASE_DIR / "depreciation_curves.csv"
MARKET_CSV       = BASE_DIR / "drivewise_market_history (1).csv"

CURRENT_YEAR = 2026


def load_artifacts():
    """Load all pre-trained model binaries and metadata."""
    fv_pipe = joblib.load(FAIR_VALUE_MODEL)
    freqs = joblib.load(FREQ_MAPS)
    risk_iso = joblib.load(RISK_MODEL)
    dom_pipe = joblib.load(DEMAND_MODEL)
    dep_df = pd.read_csv(DEPRECIATION_CSV, index_col=0)
    market_df = pd.read_csv(MARKET_CSV)
    return fv_pipe, freqs, risk_iso, dom_pipe, dep_df, market_df


def predict_car(brand: str, model: str, year: int, mileage_km: float,
                transmission: str, fuel_type: str, city: str, asking_price: float):
    """Run end-to-end DriveWise AI+ prediction on a single vehicle."""
    fv_pipe, freqs, risk_iso, dom_pipe, dep_df, market_df = load_artifacts()

    # 1. Feature Engineering
    car_age = max(1, CURRENT_YEAR - year + 1)
    mileage_per_year = mileage_km / car_age
    price_per_km = asking_price / max(1.0, mileage_km)

    # Model and City frequency encoding
    model_freq_map = freqs["model_freq"]
    city_freq_map = freqs["city_freq"]

    min_m_freq = min(model_freq_map.values()) if isinstance(model_freq_map, dict) else 0.0001
    min_c_freq = min(city_freq_map.values()) if isinstance(city_freq_map, dict) else 0.0001

    m_freq = model_freq_map.get(model, min_m_freq) if isinstance(model_freq_map, dict) else min_m_freq
    c_freq = city_freq_map.get(city, min_c_freq) if isinstance(city_freq_map, dict) else min_c_freq

    # Price vs Model Median %
    model_sub = market_df[market_df["model"] == model]
    model_median = model_sub["price_inr"].median() if len(model_sub) > 0 else market_df["price_inr"].median()
    price_vs_model_median_pct = ((asking_price - model_median) / model_median) * 100

    # 2. Fair Market Value Prediction (XGBoost)
    X_val = pd.DataFrame([{
        "year": year,
        "mileage_km": mileage_km,
        "car_age": car_age,
        "mileage_per_year": mileage_per_year,
        "model_freq": m_freq,
        "city_freq": c_freq,
        "brand": brand,
        "transmission": transmission,
        "fuel_type": fuel_type,
    }])
    predicted_fair_val = float(fv_pipe.predict(X_val)[0])
    value_gap_pct = ((asking_price - predicted_fair_val) / predicted_fair_val) * 100

    # 3. Unsupervised Purchase Risk Scoring (Isolation Forest)
    X_risk = pd.DataFrame([{
        "price_vs_model_median_pct": price_vs_model_median_pct,
        "mileage_per_year": mileage_per_year,
        "car_age": car_age,
        "price_per_km": price_per_km,
    }])
    raw_anomaly = -float(risk_iso.score_samples(X_risk)[0])
    # Calibrate to 0-100 percentile score (typical raw anomaly is between 0.35 and 0.65)
    risk_score = float(np.clip((raw_anomaly - 0.38) / (0.62 - 0.38) * 100, 0, 100))

    # 4. Brand Depreciation Rate
    if brand in dep_df.index:
        annual_deprec = float(dep_df.loc[brand, "annual_decay_pct"])
    else:
        annual_deprec = float(dep_df["annual_decay_pct"].median())

    # 5. Days on Market Prediction (DOM)
    X_dom = pd.DataFrame([{
        "price_vs_model_median_pct": price_vs_model_median_pct,
        "mileage_km": mileage_km,
        "car_age": car_age,
        "model_freq": m_freq,
        "brand": brand,
        "fuel_type": fuel_type,
    }])
    predicted_dom = max(5, int(round(float(dom_pipe.predict(X_dom)[0]))))

    # 6. Dealer Operational Simulation
    # Condition grade
    if mileage_per_year < 9000 and car_age <= 4:
        condition = "Excellent"
        recon_cost = 3000
    elif mileage_per_year < 15000 and car_age <= 8:
        condition = "Good"
        recon_cost = 9000
    elif mileage_per_year < 22000:
        condition = "Fair"
        recon_cost = 22000
    else:
        condition = "Poor"
        recon_cost = 45000

    acq_target_cost = round(asking_price * 0.78 - recon_cost)
    holding_cost_per_day = round(acq_target_cost * 0.0022, 2)
    expected_holding_cost = holding_cost_per_day * predicted_dom
    expected_profit = round(predicted_fair_val - acq_target_cost - recon_cost - expected_holding_cost)
    roi_pct = (expected_profit / acq_target_cost) * 100 if acq_target_cost > 0 else 0

    # 7. Composite DII Score (0-100)
    # Component percentiles
    s_profit = np.clip((expected_profit / 300_000) * 100, 5, 95)
    s_demand = np.clip(100 - (predicted_dom / 75) * 100, 5, 95)
    s_holding = np.clip(100 - (holding_cost_per_day / 500) * 100, 5, 95)
    s_roi = np.clip((roi_pct / 45) * 100, 5, 95)
    s_deprec = np.clip(100 - (annual_deprec / 15) * 100, 5, 95)
    s_certainty = np.clip(100 - risk_score, 5, 95)

    dii_score = round(
        0.30 * s_profit +
        0.15 * s_demand +
        0.10 * s_holding +
        0.20 * s_roi +
        0.10 * s_deprec +
        0.15 * s_certainty,
        1
    )

    # 8. Dynamic Resale Pricing & Negotiation
    target_dom = 25
    dom_gap = predicted_dom - target_dom
    price_adj = np.clip(-dom_gap / target_dom * 0.5, -0.08, 0.08)
    recommended_resale = round(predicted_fair_val * (1 + price_adj), -2)

    opening_offer = round(acq_target_cost * 0.93, -2)
    walk_away_ceiling = round(acq_target_cost, -2)

    # 9. Recommendation Action
    if dii_score >= 68 and value_gap_pct <= 2:
        action = "STRONG_BUY (High profit potential, low capital risk)"
    elif dii_score >= 56:
        action = "ACCUMULATE (Attractive return, verify mechanical scope)"
    elif dii_score >= 42:
        action = "HOLD (Moderate margin, negotiate aggressive discount)"
    else:
        action = "REJECT (Unfavorable risk-adjusted return)"

    # Print Clean Dashboard Output
    print("\n" + "=" * 75)
    print("  DRIVEWISE AI+ -- VEHICLE VALUATION & ACQUISITION INTELLIGENCE")
    print("=" * 75)
    print(f"Vehicle: {year} {brand} {model} | {transmission} | {fuel_type} | {city}")
    print(f"Odometer: {mileage_km:,.0f} km ({car_age} years old | {mileage_per_year:,.0f} km/year)")
    print(f"Condition Grade: {condition} (Est. Reconditioning Cost: Rs. {recon_cost:,})")
    print("-" * 75)
    print("1. VALUATION & FAIR MARKET APPRAISAL (XGBoost Regressor):")
    print(f"   Asking Price:             Rs. {asking_price:>12,.0f}")
    print(f"   Predicted Fair Value:     Rs. {predicted_fair_val:>12,.0f}")
    if value_gap_pct < 0:
        print(f"   Valuation Gap:            {value_gap_pct:>12.1f}% (Priced BELOW fair value -> Bargain)")
    else:
        print(f"   Valuation Gap:            +{value_gap_pct:>11.1f}% (Priced ABOVE fair value -> Premium)")
    print("-" * 75)
    print("2. RISK & VELOCITY ASSESSMENT:")
    print(f"   Purchase Risk Score:      {risk_score:>12.1f} / 100 (Unsupervised Isolation Forest)")
    print(f"   Predicted Days on Market: {predicted_dom:>12d} days (Target: {target_dom} days)")
    print(f"   Brand Value Decay:        {annual_deprec:>12.1f}% / year ({brand})")
    print(f"   Daily Carrying Cost:      Rs. {holding_cost_per_day:>12,.2f} / day (0.22% of capital)")
    print("-" * 75)
    print("3. DEALER INTELLIGENCE INDEX (DII COMPOSITE METRIC):")
    print(f"   DII Score (0-100):        {dii_score:>12.1f} / 100")
    print(f"   Expected Net Profit:      Rs. {expected_profit:>12,.0f}")
    print(f"   Expected Return on Inv:   {roi_pct:>12.1f}%")
    print(f"   SYSTEM RECOMMENDATION:    >>> {action} <<<")
    print("-" * 75)
    print("4. EXECUTION, PRICING & NEGOTIATION BOUNDARIES:")
    print(f"   Recommended Resale Price: Rs. {recommended_resale:>12,.0f} ({price_adj*100:+.1f}% velocity adjustment)")
    print(f"   Negotiation Opening Offer:Rs. {opening_offer:>12,.0f} (Target opening bid)")
    print(f"   Walk-Away Ceiling:        Rs. {walk_away_ceiling:>12,.0f} (Strict break-even limit)")
    print("=" * 75)


def run_demo():
    print("Running DriveWise AI+ Prediction Pipeline on 3 Diverse Test Vehicles...")

    # Case A: High DII Budget Hatchback (Maruti Baleno)
    predict_car(
        brand="Maruti Suzuki",
        model="Baleno",
        year=2021,
        mileage_km=32000,
        transmission="Manual",
        fuel_type="Petrol",
        city="Delhi",
        asking_price=520000
    )

    # Case B: High-Value Mid-Market SUV (Kia Seltos)
    predict_car(
        brand="Kia",
        model="Seltos",
        year=2022,
        mileage_km=28000,
        transmission="Automatic",
        fuel_type="Diesel",
        city="Bangalore",
        asking_price=1250000
    )

    # Case C: High-Risk Overpriced Luxury Asset (Mercedes-Benz)
    predict_car(
        brand="Mercedes-Benz",
        model="GLC",
        year=2017,
        mileage_km=98000,
        transmission="Automatic",
        fuel_type="Diesel",
        city="Mumbai",
        asking_price=2450000
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_demo()
