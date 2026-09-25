"""
DriveWise AI+ -- Phase 6: Explainable AI (XAI)
================================================
Generates feature-importance and SHAP-based explanations for the fair
value model and risk model, producing human-readable recommendation
explanations for individual vehicles.

Outputs:
  - drivewise_feature_importance.csv     (global feature importance)
  - drivewise_shap_explanations.csv      (per-vehicle explanations)
  - drivewise_xai_summary.json           (summary statistics)
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MARKET_PATH      = BASE_DIR / "drivewise_market_history (1).csv"
POOL_PATH        = BASE_DIR / "drivewise_pool_with_dii.csv"
FAIR_VALUE_MODEL = BASE_DIR / "fair_value_model.joblib"
FREQ_MAPS        = BASE_DIR / "fair_value_freq_maps.joblib"
RISK_MODEL       = BASE_DIR / "risk_model_isolationforest.joblib"

IMPORTANCE_OUT   = BASE_DIR / "drivewise_feature_importance.csv"
EXPLANATIONS_OUT = BASE_DIR / "drivewise_shap_explanations.csv"
XAI_SUMMARY_OUT  = BASE_DIR / "drivewise_xai_summary.json"

try:
    import shap
    SHAP_AVAILABLE = True
    print("[XAI] SHAP library available -- using TreeExplainer")
except (ImportError, OSError, Exception) as e:
    SHAP_AVAILABLE = False
    print(f"[XAI] SHAP/numba unavailable on this OS ({type(e).__name__}) -- using tree feature importance & permutation attribution")


def get_min_freq(freq_obj):
    if isinstance(freq_obj, dict):
        return min(freq_obj.values()) if freq_obj else 0.0
    return freq_obj.min() if hasattr(freq_obj, "min") else 0.0


def load_model_and_data():
    """Load the fair value pipeline, frequency maps, and market data."""
    pipeline = joblib.load(FAIR_VALUE_MODEL)
    freq_maps = joblib.load(FREQ_MAPS)
    market = pd.read_csv(MARKET_PATH)
    pool = pd.read_csv(POOL_PATH)

    model_freq = freq_maps["model_freq"]
    city_freq  = freq_maps["city_freq"]

    min_m = get_min_freq(model_freq)
    min_c = get_min_freq(city_freq)

    market["model_freq"] = market["model"].map(model_freq).fillna(min_m)
    market["city_freq"]  = market["city"].map(city_freq).fillna(min_c)

    features_num = ["year", "mileage_km", "car_age", "mileage_per_year",
                    "model_freq", "city_freq"]
    features_cat = ["brand", "transmission", "fuel_type"]
    X = market[features_num + features_cat]

    return pipeline, X, market, pool, features_num, features_cat


def get_feature_names(pipeline, features_num, features_cat):
    """Extract feature names from the fitted pipeline's preprocessor."""
    preprocessor = pipeline.named_steps["prep"]
    try:
        cat_names = list(preprocessor.named_transformers_["cat"]
                         .get_feature_names_out(features_cat))
    except Exception:
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = [f"{features_cat[i]}_{c}"
                     for i, cats in enumerate(ohe.categories_)
                     for c in cats]
    all_names = cat_names + features_num
    return all_names


def compute_model_importance(pipeline, feature_names):
    """Extract feature importance directly from the tree model."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        n_feats = min(len(feature_names), len(importances))
        df = pd.DataFrame({
            "feature": feature_names[:n_feats],
            "importance": importances[:n_feats],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
        df["rank"] = range(1, len(df) + 1)
        return df
    return None


def compute_shap_values(pipeline, X, feature_names, n_background=200, n_explain=500):
    """Compute SHAP values using TreeExplainer on the underlying model."""
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray()
    else:
        X_dense = np.array(X_transformed)

    bg_size = min(n_background, len(X_dense))
    bg_idx = np.random.RandomState(42).choice(len(X_dense), bg_size, replace=False)

    explainer = shap.TreeExplainer(model, data=X_dense[bg_idx])

    exp_size = min(n_explain, len(X_dense))
    explain_idx = np.random.RandomState(42).choice(len(X_dense), exp_size, replace=False)
    X_explain = X_dense[explain_idx]
    
    raw_shap = explainer.shap_values(X_explain)
    if isinstance(raw_shap, list):
        shap_array = np.array(raw_shap[0])
    else:
        shap_array = np.array(raw_shap)

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    n_feats = min(len(feature_names), len(mean_abs_shap))
    imp_df = pd.DataFrame({
        "feature": feature_names[:n_feats],
        "mean_abs_shap": mean_abs_shap[:n_feats],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["importance_pct"] = (imp_df["mean_abs_shap"] / imp_df["mean_abs_shap"].sum() * 100).round(2)
    imp_df["rank"] = range(1, len(imp_df) + 1)

    return imp_df, shap_array, explain_idx


def generate_vehicle_explanations(pool, pipeline, freq_maps_path, feature_names, n_explain=50):
    """Generate human-readable explanations for individual vehicles."""
    freq_data = joblib.load(freq_maps_path)
    model_freq = freq_data["model_freq"]
    city_freq  = freq_data["city_freq"]

    min_m = get_min_freq(model_freq)
    min_c = get_min_freq(city_freq)

    df = pool.copy()
    df["model_freq"] = df["model"].map(model_freq).fillna(min_m)
    df["city_freq"]  = df["city"].map(city_freq).fillna(min_c)

    features_num = ["year", "mileage_km", "car_age", "mileage_per_year",
                    "model_freq", "city_freq"]
    features_cat = ["brand", "transmission", "fuel_type"]

    # Sample top candidates (high DII) and lower candidates (low DII)
    half = n_explain // 2
    top_df = df.head(half)
    bottom_df = df.tail(half)
    subset = pd.concat([top_df, bottom_df]).drop_duplicates(subset=["listing_id"])

    X_subset = subset[features_num + features_cat]
    predicted_values = pipeline.predict(X_subset)

    explanations = []
    for idx_pos, (_, row) in enumerate(subset.iterrows()):
        predicted = float(predicted_values[idx_pos])
        actual = float(row["price_inr"])
        gap_pct = (actual - predicted) / predicted * 100 if predicted > 0 else 0.0

        factors = []
        if row["car_age"] <= 3:
            factors.append(f"relatively new ({int(row['car_age'])} yrs old)")
        elif row["car_age"] >= 10:
            factors.append(f"aging vehicle ({int(row['car_age'])} yrs old, higher depreciation)")

        if row["mileage_per_year"] > 18000:
            factors.append(f"high annual usage ({int(row['mileage_per_year']):,} km/yr)")
        elif row["mileage_per_year"] < 8000:
            factors.append(f"low annual usage ({int(row['mileage_per_year']):,} km/yr)")

        if gap_pct < -10:
            factors.append(f"priced {abs(gap_pct):.0f}% below fair value (acquisition opportunity)")
        elif gap_pct > 10:
            factors.append(f"priced {gap_pct:.0f}% above fair value (overpriced asking)")

        risk_val = float(row.get("risk_score_0_100", 50))
        if risk_val > 75:
            factors.append("elevated purchase risk profile")
        elif risk_val < 25:
            factors.append("low operational risk profile")

        dii_val = float(row.get("DII", 50))
        if dii_val > 70:
            factors.append(f"high composite DII score ({dii_val:.1f}/100)")
        elif dii_val < 35:
            factors.append(f"low composite DII score ({dii_val:.1f}/100)")

        explanation = "; ".join(factors) if factors else "Standard balanced profile"

        if dii_val >= 68 and gap_pct <= 0:
            recommendation = "STRONG_BUY - High profit potential, low capital risk"
        elif dii_val >= 58:
            recommendation = "ACCUMULATE - Attractive margin, verify reconditioning scope"
        elif dii_val >= 42:
            recommendation = "HOLD - Moderate return profile, negotiate on price"
        else:
            recommendation = "REJECT - Unfavorable risk-adjusted return"

        explanations.append({
            "listing_id": row.get("listing_id", f"DW-{idx_pos}"),
            "brand": row["brand"],
            "model": row["model"],
            "year": int(row["year"]),
            "price_inr": int(actual),
            "predicted_fair_value": int(predicted),
            "value_gap_pct": round(gap_pct, 1),
            "risk_score": round(risk_val, 1),
            "DII": round(dii_val, 1),
            "explanation": explanation,
            "recommendation": recommendation,
        })

    return pd.DataFrame(explanations)


def compute_risk_model_importance():
    """Analyze which features the risk model (Isolation Forest) relies on."""
    iso = joblib.load(RISK_MODEL)
    features = ["price_vs_model_median_pct", "mileage_per_year", "car_age", "price_per_km"]

    pool = pd.read_csv(POOL_PATH)
    X = pool[features].replace([np.inf, -np.inf], np.nan).dropna()
    base_scores = -iso.score_samples(X.values)

    importances = {}
    rng = np.random.RandomState(42)
    for i, feat in enumerate(features):
        X_perm = X.values.copy()
        rng.shuffle(X_perm[:, i])
        perm_scores = -iso.score_samples(X_perm)
        change = float(np.mean(np.abs(perm_scores - base_scores)))
        importances[feat] = change

    total = sum(importances.values()) if sum(importances.values()) > 0 else 1.0
    result = pd.DataFrame([
        {"feature": k, "importance": v, "importance_pct": round(v / total * 100, 2)}
        for k, v in sorted(importances.items(), key=lambda x: -x[1])
    ])
    result["rank"] = range(1, len(result) + 1)
    return result


def main():
    print("=" * 70)
    print("  DriveWise AI+ -- Phase 6: Explainable AI (XAI)")
    print("=" * 70)

    pipeline, X, market, pool, features_num, features_cat = load_model_and_data()
    feature_names = get_feature_names(pipeline, features_num, features_cat)

    print("\n[>>] Computing global feature importance...")
    imp_df = None
    method_used = "tree_feature_importances"

    if SHAP_AVAILABLE:
        try:
            imp_df, shap_vals, explain_idx = compute_shap_values(
                pipeline, X, feature_names, n_background=150, n_explain=300)
            method_used = "SHAP_TreeExplainer"
            print(f"  [OK] SHAP computed on {len(explain_idx)} background-calibrated samples")
        except Exception as e:
            print(f"  [WARN] SHAP TreeExplainer encountered exception: {e}")
            print("  Falling back to tree feature importances...")
            imp_df = compute_model_importance(pipeline, feature_names)
            method_used = "tree_feature_importances"
    else:
        imp_df = compute_model_importance(pipeline, feature_names)

    if imp_df is None:
        imp_df = compute_model_importance(pipeline, feature_names)

    if imp_df is not None:
        imp_df.to_csv(IMPORTANCE_OUT, index=False)
        print(f"  [OK] Feature importance saved -> {IMPORTANCE_OUT}")
        print(f"\n  Top 10 features ({method_used}):")
        for _, r in imp_df.head(10).iterrows():
            print(f"    {int(r['rank']):2d}. {r['feature']:30s}  {r['importance_pct']:6.2f}%")

    print("\n[>>] Computing risk model feature sensitivity (permutation)...")
    risk_imp = compute_risk_model_importance()
    print("  Risk model feature sensitivity:")
    for _, r in risk_imp.iterrows():
        print(f"    {int(r['rank']):2d}. {r['feature']:30s}  {r['importance_pct']:6.2f}%")

    print("\n[>>] Generating per-vehicle explainable decision cards...")
    explanations_df = generate_vehicle_explanations(
        pool, pipeline, FREQ_MAPS, feature_names, n_explain=50)
    explanations_df.to_csv(EXPLANATIONS_OUT, index=False)
    print(f"  [OK] {len(explanations_df)} vehicle explanations saved -> {EXPLANATIONS_OUT}")

    print("\n  Sample decision explanations:")
    for _, r in explanations_df.head(4).iterrows():
        print(f"\n  [{r['listing_id']}] {r['brand']} {r['model']} {r['year']}")
        print(f"    Asking: Rs.{r['price_inr']:,} | Fair Value: Rs.{r['predicted_fair_value']:,} | Gap: {r['value_gap_pct']:+.1f}%")
        print(f"    Risk: {r['risk_score']:.1f}/100 | DII: {r['DII']:.1f}/100")
        print(f"    Key Drivers: {r['explanation']}")
        print(f"    Action: {r['recommendation']}")

    summary = {
        "method": method_used,
        "n_features_evaluated": len(imp_df) if imp_df is not None else 0,
        "top5_value_drivers": imp_df.head(5)["feature"].tolist() if imp_df is not None else [],
        "n_explained_cards": len(explanations_df),
        "strong_buy_count": int((explanations_df["recommendation"].str.startswith("STRONG_BUY")).sum()),
        "accumulate_count": int((explanations_df["recommendation"].str.startswith("ACCUMULATE")).sum()),
        "hold_count": int((explanations_df["recommendation"].str.startswith("HOLD")).sum()),
        "reject_count": int((explanations_df["recommendation"].str.startswith("REJECT")).sum()),
        "risk_model_primary_driver": risk_imp.iloc[0]["feature"] if len(risk_imp) > 0 else None,
    }
    with open(XAI_SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] XAI summary report saved -> {XAI_SUMMARY_OUT}")


if __name__ == "__main__":
    main()
