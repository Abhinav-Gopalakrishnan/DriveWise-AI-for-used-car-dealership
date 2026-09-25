"""
DriveWise AI+ — Phase 4: Portfolio Optimization + Pricing + Negotiation
===========================================================================
Three things a dealer actually needs, built on top of Phase 3's DII scores:

1. PORTFOLIO OPTIMIZER
   Given a budget, pick the SET of vehicles to acquire that maximizes total
   expected profit -- not just "take the highest-DII ones until money runs
   out" (that's the greedy baseline, included for comparison), but an
   actual constrained optimization (0/1 knapsack via linear programming)
   that can trade off a few high-cost/high-profit cars against many
   low-cost/modest-profit ones.

2. DYNAMIC RESALE PRICING
   For each acquired vehicle, recommend a resale price: start from the
   fair-value model's own price estimate for that vehicle profile, then
   nudge it based on how the vehicle's predicted days-on-market compares
   to a target -- price down slightly to sell faster, price up slightly
   if it's a low-competition/high-demand profile.

3. NEGOTIATION OFFER
   A recommended opening acquisition offer: below the seller's asking
   price, anchored to the simulated acquisition_cost figure but expressed
   as a percentage-off-asking recommendation with a walk-away ceiling.
"""

import numpy as np
import pandas as pd
import pulp

POOL_PATH = "../data/drivewise_pool_with_dii.csv"
OUT_PATH = "../data/drivewise_portfolio_recommendations.csv"

BUDGET_SCENARIOS_INR = [5_000_000, 10_000_000, 20_000_000]  # 50L, 1Cr, 2Cr
MAX_VEHICLES_PER_SCENARIO = 40   # forecourt/lot capacity constraint
TARGET_DAYS_ON_MARKET = 25       # dealer's target sell-through time


# ----------------------------------------------------------------------
# 1. PORTFOLIO OPTIMIZER
# ----------------------------------------------------------------------
def optimize_portfolio(pool: pd.DataFrame, budget: float, max_vehicles: int) -> pd.DataFrame:
    """0/1 knapsack: maximize total expected_profit subject to a budget
    and a max-vehicle-count constraint. Solved as a linear program with
    binary decision variables via PuLP (CBC solver, bundled with PuLP)."""
    prob = pulp.LpProblem("DriveWise_Portfolio", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}

    prob += pulp.lpSum(x[i] * pool.loc[i, "expected_profit"] for i in pool.index)
    prob += pulp.lpSum(x[i] * pool.loc[i, "acquisition_cost"] for i in pool.index) <= budget
    prob += pulp.lpSum(x[i] for i in pool.index) <= max_vehicles

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen_idx = [i for i in pool.index if x[i].value() == 1]
    return pool.loc[chosen_idx].copy()


def greedy_baseline(pool: pd.DataFrame, budget: float, max_vehicles: int) -> pd.DataFrame:
    """Baseline to compare the optimizer against: just take vehicles in
    DII-descending order until the budget or vehicle cap runs out."""
    ranked = pool.sort_values("DII", ascending=False)
    chosen, spent = [], 0.0
    for i, row in ranked.iterrows():
        if len(chosen) >= max_vehicles:
            break
        if spent + row["acquisition_cost"] > budget:
            continue  # skip -- a car that doesn't fit isn't a stopping condition
        chosen.append(i)
        spent += row["acquisition_cost"]
    return pool.loc[chosen].copy()


def run_budget_scenarios(pool: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PORTFOLIO OPTIMIZATION -- optimizer vs. greedy-by-DII baseline")
    print("=" * 70)

    all_results = []
    for budget in BUDGET_SCENARIOS_INR:
        opt = optimize_portfolio(pool, budget, MAX_VEHICLES_PER_SCENARIO)
        greedy = greedy_baseline(pool, budget, MAX_VEHICLES_PER_SCENARIO)

        opt_profit = opt["expected_profit"].sum()
        greedy_profit = greedy["expected_profit"].sum()
        uplift = (opt_profit / greedy_profit - 1) * 100 if greedy_profit else float("nan")

        print(f"\nBudget: Rs.{budget:,.0f}  (max {MAX_VEHICLES_PER_SCENARIO} vehicles)")
        print(f"  Optimizer : {len(opt):3d} vehicles, spend Rs.{opt['acquisition_cost'].sum():>12,.0f}, "
              f"profit Rs.{opt_profit:>12,.0f}")
        print(f"  Greedy    : {len(greedy):3d} vehicles, spend Rs.{greedy['acquisition_cost'].sum():>12,.0f}, "
              f"profit Rs.{greedy_profit:>12,.0f}")
        print(f"  Optimizer profit uplift vs. greedy: {uplift:+.1f}%")

        opt = opt.copy()
        opt["budget_scenario"] = budget
        opt["selection_method"] = "optimizer"
        all_results.append(opt)

    return pd.concat(all_results, ignore_index=True)


# ----------------------------------------------------------------------
# 2. DYNAMIC RESALE PRICING
# ----------------------------------------------------------------------
def recommend_resale_price(df: pd.DataFrame, target_dom: int = TARGET_DAYS_ON_MARKET) -> pd.DataFrame:
    df = df.copy()
    # days_on_market here is Phase 2's predicted/simulated DOM for this
    # profile at its current (asking) price -- use the gap to target DOM
    # to nudge the resale price up or down.
    dom_gap = df["days_on_market"] - target_dom
    # +-8% max adjustment, scaled by how far off target the predicted DOM is
    adjustment_pct = np.clip(-dom_gap / target_dom * 0.5, -0.08, 0.08)
    df["recommended_resale_price"] = (df["expected_resale_price"] * (1 + adjustment_pct)).round(-2)
    df["resale_adjustment_pct"] = (adjustment_pct * 100).round(1)
    return df


# ----------------------------------------------------------------------
# 3. NEGOTIATION OFFER
# ----------------------------------------------------------------------
def recommend_negotiation_offer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Opening offer: a bit below the simulated acquisition_cost target to
    # leave room to negotiate up; ceiling: the acquisition_cost itself
    # (walk away above this -- it's the point where expected profit
    # starts getting squeezed below an acceptable margin).
    df["opening_offer"] = (df["acquisition_cost"] * 0.93).round(-2)
    df["walk_away_ceiling"] = df["acquisition_cost"].round(-2)
    df["discount_off_asking_pct"] = (
        (df["price_inr"] - df["walk_away_ceiling"]) / df["price_inr"] * 100
    ).round(1)
    return df


def main():
    pool = pd.read_csv(POOL_PATH)

    portfolios = run_budget_scenarios(pool)
    portfolios = recommend_resale_price(portfolios)
    portfolios = recommend_negotiation_offer(portfolios)

    portfolios.to_csv(OUT_PATH, index=False)
    print(f"\nSaved all scenario portfolios + pricing/offers -> {OUT_PATH}")

    mid_budget = BUDGET_SCENARIOS_INR[1]
    sample = portfolios[portfolios["budget_scenario"] == mid_budget].head(8)
    cols = ["listing_id", "brand", "model", "price_inr", "opening_offer",
            "walk_away_ceiling", "recommended_resale_price", "expected_profit", "DII"]
    print(f"\nSample recommendations (Rs.{mid_budget:,.0f} budget scenario):")
    print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    main()
