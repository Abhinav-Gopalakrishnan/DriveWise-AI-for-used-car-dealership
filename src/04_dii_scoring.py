"""
DriveWise AI+ — Phase 3: Dealer Intelligence Index (DII)
============================================================
Combines Phase 2's outputs into one composite acquisition-attractiveness
score per vehicle, 0-100.

DII components (each normalized to a 0-100 percentile rank before
combining, so no single raw-unit scale dominates):

  1. profit_score       -- expected profit margin, percentile-ranked
  2. demand_score        -- inverse of predicted days_on_market (faster
                             sale = better)
  3. holding_cost_score   -- inverse of holding_cost_per_day (cheaper to
                             hold = better)
  4. capital_efficiency_score -- profit per rupee of capital tied up
                             (acquisition_cost), i.e. ROI-style efficiency,
                             not just absolute profit
  5. depreciation_score  -- inverse of the vehicle's brand's annual
                             depreciation rate from Phase 2's fitted curves
                             (slower-depreciating brand = better)
  6. certainty_score     -- inverse of the Isolation Forest risk score
                             (lower risk = better; renamed from
                             "market uncertainty" in the proposal since
                             this is what the risk model actually captures)

DII = weighted sum of the six percentile scores. Weights are a modelling
choice -- defaults below are a reasoned starting point (profit and
certainty weighted highest, since a good deal that's also a real, safe
deal matters more than a technically-cheap one), NOT a claim of a uniquely
correct weighting. The sensitivity_analysis() function shows how rankings
move under alternative weightings -- use that to justify the choice in
the report rather than asserting it.
"""

import numpy as np
import pandas as pd

POOL_PATH = "../data/drivewise_acquisition_pool_scored.csv"
DEPRECIATION_PATH = "../models/depreciation_curves.csv"
OUT_PATH = "../data/drivewise_pool_with_dii.csv"

DEFAULT_WEIGHTS = {
    "profit_score": 0.30,
    "demand_score": 0.15,
    "holding_cost_score": 0.10,
    "capital_efficiency_score": 0.20,
    "depreciation_score": 0.10,
    "certainty_score": 0.15,
}
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"


def pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentile-rank a series to 0-100. Flip direction when lower raw
    values should score higher (e.g. cost, risk, depreciation rate)."""
    ranked = series.rank(pct=True) * 100
    return ranked if higher_is_better else (100 - ranked)


def compute_dii(pool: pd.DataFrame, depreciation: pd.DataFrame,
                 weights: dict = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    df = pool.copy()

    # -- 1. Expected profit --
    df["expected_holding_total"] = df["holding_cost_per_day"] * df["days_on_market"]
    df["expected_profit"] = (
        df["expected_resale_price"] - df["acquisition_cost"]
        - df["recon_cost_est"] - df["expected_holding_total"]
    )
    df["profit_score"] = pct_rank(df["expected_profit"])

    # -- 2. Demand (faster sale = better) --
    df["demand_score"] = pct_rank(df["days_on_market"], higher_is_better=False)

    # -- 3. Holding cost (cheaper = better) --
    df["holding_cost_score"] = pct_rank(df["holding_cost_per_day"], higher_is_better=False)

    # -- 4. Capital efficiency: profit per rupee tied up --
    df["roi"] = df["expected_profit"] / df["acquisition_cost"].replace(0, np.nan)
    df["capital_efficiency_score"] = pct_rank(df["roi"])

    # -- 5. Depreciation (slower-depreciating brand = better) --
    dep_map = depreciation["annual_decay_pct"].to_dict()
    overall_median_decay = depreciation["annual_decay_pct"].median()
    df["brand_annual_decay_pct"] = df["brand"].map(dep_map).fillna(overall_median_decay)
    df["depreciation_score"] = pct_rank(df["brand_annual_decay_pct"], higher_is_better=False)

    # -- 6. Certainty (inverse of risk) --
    df["certainty_score"] = pct_rank(df["risk_score_0_100"], higher_is_better=False)

    # -- Combine --
    df["DII"] = sum(df[col] * w for col, w in weights.items())
    df["DII"] = df["DII"].round(1)

    return df.sort_values("DII", ascending=False).reset_index(drop=True)


def sensitivity_analysis(pool: pd.DataFrame, depreciation: pd.DataFrame, top_k: int = 20):
    """Recompute DII under a few alternative weightings and check how much
    the top-K acquisition list changes -- evidence for the report that the
    ranking isn't fragile to the exact weight choice."""
    alt_weightings = {
        "default": DEFAULT_WEIGHTS,
        "profit_heavy": {**DEFAULT_WEIGHTS, "profit_score": 0.45, "certainty_score": 0.10,
                          "demand_score": 0.10, "holding_cost_score": 0.05,
                          "capital_efficiency_score": 0.20, "depreciation_score": 0.10},
        "risk_averse": {**DEFAULT_WEIGHTS, "certainty_score": 0.35, "profit_score": 0.20,
                         "demand_score": 0.15, "holding_cost_score": 0.10,
                         "capital_efficiency_score": 0.10, "depreciation_score": 0.10},
    }
    for name, w in alt_weightings.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{name} weights must sum to 1"

    top_sets = {}
    for name, w in alt_weightings.items():
        scored = compute_dii(pool, depreciation, weights=w)
        top_sets[name] = set(scored.head(top_k)["listing_id"])

    base = top_sets["default"]
    print(f"\nSensitivity check -- overlap with default top-{top_k} list:")
    for name, s in top_sets.items():
        overlap = len(base & s)
        print(f"  {name:15s}  {overlap}/{top_k} vehicles in common with default weighting")


def main():
    pool = pd.read_csv(POOL_PATH)
    depreciation = pd.read_csv(DEPRECIATION_PATH, index_col=0)

    scored = compute_dii(pool, depreciation)
    scored.to_csv(OUT_PATH, index=False)
    print(f"Scored {len(scored)} vehicles -> {OUT_PATH}")

    print("\nTop 10 acquisition candidates by DII:")
    cols = ["listing_id", "brand", "model", "year", "price_inr", "acquisition_cost",
            "expected_profit", "risk_score_0_100", "DII"]
    print(scored[cols].head(10).to_string(index=False))

    print("\nBottom 5 (worst DII):")
    print(scored[cols].tail(5).to_string(index=False))

    print("\nDII distribution:")
    print(scored["DII"].describe().round(1))

    sensitivity_analysis(pool, depreciation)


if __name__ == "__main__":
    main()
