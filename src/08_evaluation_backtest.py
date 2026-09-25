"""
DriveWise AI+ -- Phase 7: Evaluation & Backtesting
====================================================
Comprehensive quantitative evaluation of the predictive models, DII metric,
and portfolio optimization framework.

Key Components:
  1. Out-of-sample Model Generalization & Subgroup Evaluation:
     - Metrics across price tiers (Budget, Mid-Market, Luxury)
     - Metrics across vehicle age bins and top brands
  2. DII Decile Ranking & Deal Identification Efficacy:
     - Monotonicity test of DII vs. Profit, ROI, Risk, and DOM
  3. Multi-Strategy Portfolio Backtest (5 Strategies x 4 Budgets):
     - Random Selection (100-run Monte Carlo)
     - Lowest-Price-First (Volume heuristic)
     - Greedy by Profit
     - Greedy by DII
     - DriveWise AI+ MILP Optimizer
  4. Dealership Downside Stress-Testing:
     - Capital resilience under 4 economic distress scenarios

Outputs:
  - evaluation_metrics_by_segment.csv
  - evaluation_dii_deciles.csv
  - evaluation_strategy_comparison.csv
  - evaluation_stress_testing.csv
  - drivewise_evaluation_summary.json
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import pulp
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MARKET_PATH      = BASE_DIR / "drivewise_market_history (1).csv"
POOL_PATH        = BASE_DIR / "drivewise_pool_with_dii.csv"
FAIR_VALUE_MODEL = BASE_DIR / "fair_value_model.joblib"
FREQ_MAPS        = BASE_DIR / "fair_value_freq_maps.joblib"

SEGMENT_OUT      = BASE_DIR / "evaluation_metrics_by_segment.csv"
DECILES_OUT      = BASE_DIR / "evaluation_dii_deciles.csv"
STRATEGY_OUT     = BASE_DIR / "evaluation_strategy_comparison.csv"
STRESS_OUT       = BASE_DIR / "evaluation_stress_testing.csv"
SUMMARY_OUT      = BASE_DIR / "drivewise_evaluation_summary.json"


def get_min_freq(freq_obj):
    if isinstance(freq_obj, dict):
        return min(freq_obj.values()) if freq_obj else 0.0
    return freq_obj.min() if hasattr(freq_obj, "min") else 0.0


# ═════════════════════════════════════════════════════════════════════
#  1. MODEL GENERALIZATION & SUBGROUP BACKTEST
# ═════════════════════════════════════════════════════════════════════
def evaluate_model_segments():
    print("\n" + "=" * 70)
    print("1. OUT-OF-SAMPLE MODEL SEGMENT EVALUATION")
    print("=" * 70)

    market = pd.read_csv(MARKET_PATH)
    pipeline = joblib.load(FAIR_VALUE_MODEL)
    freq_maps = joblib.load(FREQ_MAPS)

    model_freq = freq_maps["model_freq"]
    city_freq  = freq_maps["city_freq"]
    min_m = get_min_freq(model_freq)
    min_c = get_min_freq(city_freq)

    market["model_freq"] = market["model"].map(model_freq).fillna(min_m)
    market["city_freq"]  = market["city"].map(city_freq).fillna(min_c)

    features_num = ["year", "mileage_km", "car_age", "mileage_per_year",
                    "model_freq", "city_freq"]
    features_cat = ["brand", "transmission", "fuel_type"]
    target = "price_inr"

    X = market[features_num + features_cat]
    y = market[target]

    # Recreate identical 80/20 train-test split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    test_df = market.loc[X_test.index].copy()
    test_df["predicted_price"] = pipeline.predict(X_test)
    test_df["error"] = test_df["predicted_price"] - test_df["price_inr"]
    test_df["abs_pct_error"] = (test_df["error"].abs() / test_df["price_inr"]) * 100

    results = []

    # Overall Test Set
    rmse = mean_squared_error(test_df["price_inr"], test_df["predicted_price"]) ** 0.5
    mae = mean_absolute_error(test_df["price_inr"], test_df["predicted_price"])
    r2 = r2_score(test_df["price_inr"], test_df["predicted_price"])
    mape = test_df["abs_pct_error"].mean()
    results.append({
        "segment_type": "Overall",
        "segment_name": "Full Test Set",
        "n_samples": len(test_df),
        "mean_actual_price": round(test_df["price_inr"].mean()),
        "RMSE": round(rmse),
        "MAE": round(mae),
        "R2": round(r2, 4),
        "MAPE_pct": round(mape, 2),
    })

    # Segment by Price Tier
    price_bins = [
        ("Budget (< Rs.6L)", test_df["price_inr"] < 600_000),
        ("Mid-Market (Rs.6L - Rs.15L)", (test_df["price_inr"] >= 600_000) & (test_df["price_inr"] <= 1_500_000)),
        ("Premium/Luxury (> Rs.15L)", test_df["price_inr"] > 1_500_000),
    ]
    for name, mask in price_bins:
        sub = test_df[mask]
        if len(sub) > 10:
            results.append({
                "segment_type": "Price Tier",
                "segment_name": name,
                "n_samples": len(sub),
                "mean_actual_price": round(sub["price_inr"].mean()),
                "RMSE": round(mean_squared_error(sub["price_inr"], sub["predicted_price"]) ** 0.5),
                "MAE": round(mean_absolute_error(sub["price_inr"], sub["predicted_price"])),
                "R2": round(r2_score(sub["price_inr"], sub["predicted_price"]), 4),
                "MAPE_pct": round(sub["abs_pct_error"].mean(), 2),
            })

    # Segment by Age
    age_bins = [
        ("Newer (<= 3 yrs)", test_df["car_age"] <= 3),
        ("Mid-Age (4 - 7 yrs)", (test_df["car_age"] >= 4) & (test_df["car_age"] <= 7)),
        ("Older (>= 8 yrs)", test_df["car_age"] >= 8),
    ]
    for name, mask in age_bins:
        sub = test_df[mask]
        if len(sub) > 10:
            results.append({
                "segment_type": "Vehicle Age",
                "segment_name": name,
                "n_samples": len(sub),
                "mean_actual_price": round(sub["price_inr"].mean()),
                "RMSE": round(mean_squared_error(sub["price_inr"], sub["predicted_price"]) ** 0.5),
                "MAE": round(mean_absolute_error(sub["price_inr"], sub["predicted_price"])),
                "R2": round(r2_score(sub["price_inr"], sub["predicted_price"]), 4),
                "MAPE_pct": round(sub["abs_pct_error"].mean(), 2),
            })

    # Segment by Top Brands
    top_brands = test_df["brand"].value_counts().head(6).index
    for brand in top_brands:
        sub = test_df[test_df["brand"] == brand]
        results.append({
            "segment_type": "Top Brand",
            "segment_name": brand,
            "n_samples": len(sub),
            "mean_actual_price": round(sub["price_inr"].mean()),
            "RMSE": round(mean_squared_error(sub["price_inr"], sub["predicted_price"]) ** 0.5),
            "MAE": round(mean_absolute_error(sub["price_inr"], sub["predicted_price"])),
            "R2": round(r2_score(sub["price_inr"], sub["predicted_price"]), 4),
            "MAPE_pct": round(sub["abs_pct_error"].mean(), 2),
        })

    seg_df = pd.DataFrame(results)
    seg_df.to_csv(SEGMENT_OUT, index=False)
    print(seg_df.to_string(index=False))
    return seg_df


# ═════════════════════════════════════════════════════════════════════
#  2. DII DECILE RANKING EFFICACY
# ═════════════════════════════════════════════════════════════════════
def evaluate_dii_deciles():
    print("\n" + "=" * 70)
    print("2. DII DECILE RANKING & DEAL IDENTIFICATION EFFICACY")
    print("=" * 70)

    pool = pd.read_csv(POOL_PATH)
    pool["dii_decile"] = pd.qcut(pool["DII"], q=10, labels=[f"D{i}" for i in range(10, 0, -1)])

    summary = pool.groupby("dii_decile", observed=False).agg(
        n_vehicles=("listing_id", "count"),
        avg_dii=("DII", "mean"),
        avg_profit=("expected_profit", "mean"),
        avg_roi_pct=("roi", lambda x: round(x.mean() * 100, 1)),
        avg_dom=("days_on_market", "mean"),
        avg_risk=("risk_score_0_100", "mean"),
        avg_acquisition_cost=("acquisition_cost", "mean"),
        avg_resale_price=("expected_resale_price", "mean"),
    ).reset_index()

    summary["avg_dii"] = summary["avg_dii"].round(1)
    summary["avg_profit"] = summary["avg_profit"].round(0)
    summary["avg_dom"] = summary["avg_dom"].round(1)
    summary["avg_risk"] = summary["avg_risk"].round(1)
    summary["avg_acquisition_cost"] = summary["avg_acquisition_cost"].round(0)
    summary["avg_resale_price"] = summary["avg_resale_price"].round(0)

    summary.to_csv(DECILES_OUT, index=False)
    print(summary.to_string(index=False))
    return summary


# ═════════════════════════════════════════════════════════════════════
#  3. MULTI-STRATEGY PORTFOLIO BACKTEST
# ═════════════════════════════════════════════════════════════════════
def evaluate_strategies():
    print("\n" + "=" * 70)
    print("3. MULTI-STRATEGY PORTFOLIO BACKTEST ACROSS 4 BUDGET SCENARIOS")
    print("=" * 70)

    pool = pd.read_csv(POOL_PATH)
    budgets = [2_500_000, 5_000_000, 10_000_000, 20_000_000]
    max_vehicles = 40

    all_results = []

    for budget in budgets:
        budget_label = f"Rs.{budget/100_000:.0f} Lakh" if budget < 10_000_000 else f"Rs.{budget/10_000_000:.1f} Crore"

        # --- Strategy 1: Random Selection (100 Monte Carlo runs) ---
        mc_profits, mc_spends, mc_rois, mc_risks, mc_doms, mc_counts = [], [], [], [], [], []
        rng = np.random.RandomState(42)
        for _ in range(100):
            shuffled = pool.sample(frac=1, random_state=rng.randint(0, 1_000_000)).reset_index(drop=True)
            chosen_cost = 0
            sel = []
            for _, r in shuffled.iterrows():
                if len(sel) >= max_vehicles:
                    break
                if chosen_cost + r["acquisition_cost"] <= budget:
                    sel.append(r)
                    chosen_cost += r["acquisition_cost"]
            if sel:
                sdf = pd.DataFrame(sel)
                mc_profits.append(sdf["expected_profit"].sum())
                mc_spends.append(sdf["acquisition_cost"].sum())
                mc_rois.append((sdf["expected_profit"].sum() / sdf["acquisition_cost"].sum()) * 100)
                mc_risks.append(sdf["risk_score_0_100"].mean())
                mc_doms.append(sdf["days_on_market"].mean())
                mc_counts.append(len(sdf))

        all_results.append({
            "budget_inr": budget,
            "budget_label": budget_label,
            "strategy": "Random Selection (Monte Carlo avg)",
            "n_vehicles": round(np.mean(mc_counts)),
            "capital_deployed": round(np.mean(mc_spends)),
            "capital_utilization_pct": round(np.mean(mc_spends) / budget * 100, 1),
            "total_profit": round(np.mean(mc_profits)),
            "portfolio_roi_pct": round(np.mean(mc_rois), 2),
            "avg_risk_score": round(np.mean(mc_risks), 1),
            "avg_days_on_market": round(np.mean(mc_doms), 1),
        })

        # --- Strategy 2: Cheapest First (Volume Heuristic) ---
        cheap_sorted = pool.sort_values("acquisition_cost", ascending=True)
        cheap_sel = []
        cheap_cost = 0
        for _, r in cheap_sorted.iterrows():
            if len(cheap_sel) >= max_vehicles:
                break
            if cheap_cost + r["acquisition_cost"] <= budget:
                cheap_sel.append(r)
                cheap_cost += r["acquisition_cost"]
        cd_df = pd.DataFrame(cheap_sel)
        all_results.append({
            "budget_inr": budget,
            "budget_label": budget_label,
            "strategy": "Cheapest First (Volume Heuristic)",
            "n_vehicles": len(cd_df),
            "capital_deployed": round(cd_df["acquisition_cost"].sum()),
            "capital_utilization_pct": round(cd_df["acquisition_cost"].sum() / budget * 100, 1),
            "total_profit": round(cd_df["expected_profit"].sum()),
            "portfolio_roi_pct": round((cd_df["expected_profit"].sum() / cd_df["acquisition_cost"].sum()) * 100, 2),
            "avg_risk_score": round(cd_df["risk_score_0_100"].mean(), 1),
            "avg_days_on_market": round(cd_df["days_on_market"].mean(), 1),
        })

        # --- Strategy 3: Greedy by Expected Profit ---
        profit_sorted = pool.sort_values("expected_profit", ascending=False)
        profit_sel = []
        profit_cost = 0
        for _, r in profit_sorted.iterrows():
            if len(profit_sel) >= max_vehicles:
                break
            if profit_cost + r["acquisition_cost"] <= budget:
                profit_sel.append(r)
                profit_cost += r["acquisition_cost"]
        p_df = pd.DataFrame(profit_sel)
        all_results.append({
            "budget_inr": budget,
            "budget_label": budget_label,
            "strategy": "Greedy by Profit",
            "n_vehicles": len(p_df),
            "capital_deployed": round(p_df["acquisition_cost"].sum()),
            "capital_utilization_pct": round(p_df["acquisition_cost"].sum() / budget * 100, 1),
            "total_profit": round(p_df["expected_profit"].sum()),
            "portfolio_roi_pct": round((p_df["expected_profit"].sum() / p_df["acquisition_cost"].sum()) * 100, 2),
            "avg_risk_score": round(p_df["risk_score_0_100"].mean(), 1),
            "avg_days_on_market": round(p_df["days_on_market"].mean(), 1),
        })

        # --- Strategy 4: Greedy by DII ---
        dii_sorted = pool.sort_values("DII", ascending=False)
        dii_sel = []
        dii_cost = 0
        for _, r in dii_sorted.iterrows():
            if len(dii_sel) >= max_vehicles:
                break
            if dii_cost + r["acquisition_cost"] <= budget:
                dii_sel.append(r)
                dii_cost += r["acquisition_cost"]
        dii_df = pd.DataFrame(dii_sel)
        all_results.append({
            "budget_inr": budget,
            "budget_label": budget_label,
            "strategy": "Greedy by DII",
            "n_vehicles": len(dii_df),
            "capital_deployed": round(dii_df["acquisition_cost"].sum()),
            "capital_utilization_pct": round(dii_df["acquisition_cost"].sum() / budget * 100, 1),
            "total_profit": round(dii_df["expected_profit"].sum()),
            "portfolio_roi_pct": round((dii_df["expected_profit"].sum() / dii_df["acquisition_cost"].sum()) * 100, 2),
            "avg_risk_score": round(dii_df["risk_score_0_100"].mean(), 1),
            "avg_days_on_market": round(dii_df["days_on_market"].mean(), 1),
        })

        # --- Strategy 5: DriveWise AI+ MILP Optimizer ---
        prob = pulp.LpProblem(f"DriveWise_Evaluation_{budget}", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}
        prob += pulp.lpSum(x[i] * pool.loc[i, "expected_profit"] for i in pool.index)
        prob += pulp.lpSum(x[i] * pool.loc[i, "acquisition_cost"] for i in pool.index) <= budget
        prob += pulp.lpSum(x[i] for i in pool.index) <= max_vehicles
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        chosen_opt = [i for i in pool.index if x[i].value() == 1]
        opt_df = pool.loc[chosen_opt]
        all_results.append({
            "budget_inr": budget,
            "budget_label": budget_label,
            "strategy": "DriveWise AI+ MILP Optimizer",
            "n_vehicles": len(opt_df),
            "capital_deployed": round(opt_df["acquisition_cost"].sum()),
            "capital_utilization_pct": round(opt_df["acquisition_cost"].sum() / budget * 100, 1),
            "total_profit": round(opt_df["expected_profit"].sum()),
            "portfolio_roi_pct": round((opt_df["expected_profit"].sum() / opt_df["acquisition_cost"].sum()) * 100, 2),
            "avg_risk_score": round(opt_df["risk_score_0_100"].mean(), 1),
            "avg_days_on_market": round(opt_df["days_on_market"].mean(), 1),
        })

    strat_df = pd.DataFrame(all_results)
    strat_df.to_csv(STRATEGY_OUT, index=False)
    print(strat_df.to_string(index=False))
    return strat_df


# ═════════════════════════════════════════════════════════════════════
#  4. DOWNSIDE STRESS TESTING & RISK RESILIENCE
# ═════════════════════════════════════════════════════════════════════
def evaluate_stress_scenarios():
    print("\n" + "=" * 70)
    print("4. PORTFOLIO RESILIENCE UNDER DOWNSIDE MARKET STRESS SCENARIOS")
    print("=" * 70)

    pool = pd.read_csv(POOL_PATH)
    budget = 10_000_000  # 1 Crore portfolio

    # Get portfolios for Optimizer vs Greedy vs Random
    # Optimizer
    prob = pulp.LpProblem("Stress_Opt", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}
    prob += pulp.lpSum(x[i] * pool.loc[i, "expected_profit"] for i in pool.index)
    prob += pulp.lpSum(x[i] * pool.loc[i, "acquisition_cost"] for i in pool.index) <= budget
    prob += pulp.lpSum(x[i] for i in pool.index) <= 40
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    opt_portfolio = pool.loc[[i for i in pool.index if x[i].value() == 1]].copy()

    # Greedy by DII
    dii_sorted = pool.sort_values("DII", ascending=False)
    dii_sel, dii_cost = [], 0
    for _, r in dii_sorted.iterrows():
        if len(dii_sel) >= 40:
            break
        if dii_cost + r["acquisition_cost"] <= budget:
            dii_sel.append(r)
            dii_cost += r["acquisition_cost"]
    dii_portfolio = pd.DataFrame(dii_sel)

    # Volume/Cheapest first
    cheap_sorted = pool.sort_values("acquisition_cost", ascending=True)
    cheap_sel, cheap_cost = [], 0
    for _, r in cheap_sorted.iterrows():
        if len(cheap_sel) >= 40:
            break
        if cheap_cost + r["acquisition_cost"] <= budget:
            cheap_sel.append(r)
            cheap_cost += r["acquisition_cost"]
    cheap_portfolio = pd.DataFrame(cheap_sel)

    scenarios = [
        ("Base Case", 0.00, 1.00, 1.00),
        ("Mild Softening (-5% price, +10% DOM)", -0.05, 1.10, 1.05),
        ("Moderate Downturn (-10% price, +25% DOM, +15% recon)", -0.10, 1.25, 1.15),
        ("Severe Shock (-15% price, +50% DOM, +30% recon)", -0.15, 1.50, 1.30),
    ]

    stress_results = []
    strategies = [
        ("DriveWise AI+ Optimizer", opt_portfolio),
        ("Greedy by DII", dii_portfolio),
        ("Cheapest First (Volume)", cheap_portfolio),
    ]

    for strat_name, p_df in strategies:
        base_profit = p_df["expected_profit"].sum()
        for scen_name, price_shock, dom_mult, recon_mult in scenarios:
            stressed_resale = p_df["expected_resale_price"] * (1 + price_shock)
            stressed_dom = p_df["days_on_market"] * dom_mult
            stressed_recon = p_df["recon_cost_est"] * recon_mult
            stressed_holding = p_df["holding_cost_per_day"] * stressed_dom

            stressed_profit = (
                stressed_resale - p_df["acquisition_cost"] - stressed_recon - stressed_holding
            ).sum()
            drawdown_pct = ((stressed_profit - base_profit) / base_profit) * 100

            stress_results.append({
                "strategy": strat_name,
                "scenario": scen_name,
                "capital_invested": round(p_df["acquisition_cost"].sum()),
                "realized_profit": round(stressed_profit),
                "profit_margin_pct": round(stressed_profit / p_df["acquisition_cost"].sum() * 100, 1),
                "profit_retention_pct": round(stressed_profit / base_profit * 100, 1),
                "profit_drawdown_pct": round(drawdown_pct, 1),
            })

    stress_df = pd.DataFrame(stress_results)
    stress_df.to_csv(STRESS_OUT, index=False)
    print(stress_df.to_string(index=False))
    return stress_df


def main():
    print("=" * 70)
    print("  DriveWise AI+ -- Phase 7: Evaluation & Backtesting Engine")
    print("=" * 70)

    seg_df = evaluate_model_segments()
    dec_df = evaluate_dii_deciles()
    strat_df = evaluate_strategies()
    stress_df = evaluate_stress_scenarios()

    # Calculate key benchmark uplifts for report
    mid_strat = strat_df[strat_df["budget_inr"] == 10_000_000]
    opt_row = mid_strat[mid_strat["strategy"] == "DriveWise AI+ MILP Optimizer"].iloc[0]
    rnd_row = mid_strat[mid_strat["strategy"] == "Random Selection (Monte Carlo avg)"].iloc[0]
    vol_row = mid_strat[mid_strat["strategy"] == "Cheapest First (Volume Heuristic)"].iloc[0]
    dii_row = mid_strat[mid_strat["strategy"] == "Greedy by DII"].iloc[0]

    uplift_vs_random = ((opt_row["total_profit"] - rnd_row["total_profit"]) / rnd_row["total_profit"]) * 100
    uplift_vs_volume = ((opt_row["total_profit"] - vol_row["total_profit"]) / vol_row["total_profit"]) * 100
    uplift_vs_greedy = ((opt_row["total_profit"] - dii_row["total_profit"]) / dii_row["total_profit"]) * 100

    summary = {
        "dataset_total_records": 10468,
        "acquisition_pool_size": 1500,
        "test_set_size": 2094,
        "overall_r2": float(seg_df[seg_df["segment_type"] == "Overall"]["R2"].iloc[0]),
        "overall_mape_pct": float(seg_df[seg_df["segment_type"] == "Overall"]["MAPE_pct"].iloc[0]),
        "overall_rmse_inr": float(seg_df[seg_df["segment_type"] == "Overall"]["RMSE"].iloc[0]),
        "top_decile_dii_avg_profit": float(dec_df[dec_df["dii_decile"] == "D1"]["avg_profit"].iloc[0]),
        "bottom_decile_dii_avg_profit": float(dec_df[dec_df["dii_decile"] == "D10"]["avg_profit"].iloc[0]),
        "top_decile_dii_avg_roi_pct": float(dec_df[dec_df["dii_decile"] == "D1"]["avg_roi_pct"].iloc[0]),
        "bottom_decile_dii_avg_roi_pct": float(dec_df[dec_df["dii_decile"] == "D10"]["avg_roi_pct"].iloc[0]),
        "mid_budget_optimizer_profit_inr": float(opt_row["total_profit"]),
        "mid_budget_random_profit_inr": float(rnd_row["total_profit"]),
        "profit_uplift_vs_random_pct": round(uplift_vs_random, 1),
        "profit_uplift_vs_volume_pct": round(uplift_vs_volume, 1),
        "profit_uplift_vs_greedy_dii_pct": round(uplift_vs_greedy, 1),
        "severe_stress_opt_profit_retention_pct": float(
            stress_df[(stress_df["strategy"] == "DriveWise AI+ Optimizer") & 
                      (stress_df["scenario"].str.startswith("Severe"))]["profit_retention_pct"].iloc[0]
        ),
        "severe_stress_vol_profit_retention_pct": float(
            stress_df[(stress_df["strategy"] == "Cheapest First (Volume)") & 
                      (stress_df["scenario"].str.startswith("Severe"))]["profit_retention_pct"].iloc[0]
        ),
    }

    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Phase 7 evaluation artifacts successfully saved to:")
    print(f"  - {SEGMENT_OUT}")
    print(f"  - {DECILES_OUT}")
    print(f"  - {STRATEGY_OUT}")
    print(f"  - {STRESS_OUT}")
    print(f"  - {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
