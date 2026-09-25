"""
DriveWise AI+ — Phase 5: Multi-Agent Orchestration
=====================================================
Wraps the complete buy-to-sell pipeline into a multi-agent architecture
where each agent encapsulates a single analytical responsibility.

Agent architecture:
  ┌─────────────────────────────────────────────────────┐
  │                   ORCHESTRATOR                      │
  │  (sequences agents, manages shared state, logs)     │
  └──┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘
     │      │      │      │      │      │      │
  ┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐┌──▼──┐
  │Data ││Valu-││Risk ││Dema-││DII  ││Port-││Expla-│
  │Agent││ation││Agent││nd   ││Agent││folio││in    │
  │     ││Agent││     ││Agent││     ││Agent││Agent │
  └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└──────┘

Each agent has a standard interface:
  - name:       human-readable label
  - process():  takes shared state dict, mutates it, returns status

The orchestrator runs them in dependency order and logs each step.
"""

import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# ── Paths (all files live in the same directory) ─────────────────────
BASE_DIR = Path(__file__).resolve().parent
MARKET_PATH         = BASE_DIR / "drivewise_market_history (1).csv"
POOL_PATH           = BASE_DIR / "drivewise_acquisition_pool.csv"
SCORED_POOL_PATH    = BASE_DIR / "drivewise_acquisition_pool_scored.csv"
DII_POOL_PATH       = BASE_DIR / "drivewise_pool_with_dii.csv"
PORTFOLIO_PATH      = BASE_DIR / "drivewise_portfolio_recommendations.csv"
FAIR_VALUE_MODEL    = BASE_DIR / "fair_value_model.joblib"
FREQ_MAPS           = BASE_DIR / "fair_value_freq_maps.joblib"
RISK_MODEL          = BASE_DIR / "risk_model_isolationforest.joblib"
DEMAND_MODEL        = BASE_DIR / "demand_dom_model.joblib"
DEPRECIATION_CSV    = BASE_DIR / "depreciation_curves.csv"

ORCHESTRATOR_LOG    = BASE_DIR / "orchestrator_run_log.json"
ORCHESTRATOR_OUTPUT = BASE_DIR / "drivewise_full_pipeline_output.csv"


# ═════════════════════════════════════════════════════════════════════
#  AGENT BASE CLASS
# ═════════════════════════════════════════════════════════════════════
class BaseAgent:
    """Standard interface every agent implements."""
    name: str = "BaseAgent"

    def process(self, state: dict) -> dict:
        """Receive shared state, perform work, return updated state."""
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


# ═════════════════════════════════════════════════════════════════════
#  1. DATA AGENT — loads and validates all inputs
# ═════════════════════════════════════════════════════════════════════
class DataAgent(BaseAgent):
    name = "DataAgent"

    def process(self, state: dict) -> dict:
        market = pd.read_csv(MARKET_PATH)
        pool   = pd.read_csv(POOL_PATH)

        # Basic validation
        assert len(market) >= 10_000, f"Market data too small: {len(market)}"
        assert len(pool)   >= 1_000,  f"Pool data too small: {len(pool)}"
        required_cols = ["brand", "model", "year", "price_inr", "mileage_km",
                         "transmission", "fuel_type", "city"]
        for col in required_cols:
            assert col in market.columns, f"Missing column in market: {col}"

        state["market"] = market
        state["pool"]   = pool
        state["n_market"] = len(market)
        state["n_pool"]   = len(pool)
        state["n_brands"] = market["brand"].nunique()
        state["n_cities"] = market["city"].nunique()
        return state


# ═════════════════════════════════════════════════════════════════════
#  2. VALUATION AGENT — predicts fair market value
# ═════════════════════════════════════════════════════════════════════
class ValuationAgent(BaseAgent):
    name = "ValuationAgent"

    def process(self, state: dict) -> dict:
        model = joblib.load(FAIR_VALUE_MODEL)
        freq_maps = joblib.load(FREQ_MAPS)
        pool = state["pool"].copy()

        model_freq = freq_maps["model_freq"]
        city_freq  = freq_maps["city_freq"]
        min_m = min(model_freq.values()) if isinstance(model_freq, dict) else model_freq.min()
        min_c = min(city_freq.values()) if isinstance(city_freq, dict) else city_freq.min()
        pool["model_freq"] = pool["model"].map(model_freq).fillna(min_m)
        pool["city_freq"]  = pool["city"].map(city_freq).fillna(min_c)

        features_num = ["year", "mileage_km", "car_age", "mileage_per_year",
                        "model_freq", "city_freq"]
        features_cat = ["brand", "transmission", "fuel_type"]
        X = pool[features_num + features_cat]

        pool["predicted_fair_value"] = model.predict(X).round(0)
        pool["value_gap_pct"] = (
            (pool["price_inr"] - pool["predicted_fair_value"])
            / pool["predicted_fair_value"] * 100
        ).round(1)

        state["pool"] = pool
        state["fair_value_model"] = model
        state["valuation_summary"] = {
            "mean_predicted": float(pool["predicted_fair_value"].mean()),
            "mean_value_gap_pct": float(pool["value_gap_pct"].mean()),
            "underpriced_count": int((pool["value_gap_pct"] < -5).sum()),
            "overpriced_count":  int((pool["value_gap_pct"] > 5).sum()),
        }
        return state


# ═════════════════════════════════════════════════════════════════════
#  3. RISK AGENT — scores purchase risk
# ═════════════════════════════════════════════════════════════════════
class RiskAgent(BaseAgent):
    name = "RiskAgent"

    def process(self, state: dict) -> dict:
        iso = joblib.load(RISK_MODEL)
        pool = state["pool"].copy()

        features = ["price_vs_model_median_pct", "mileage_per_year",
                     "car_age", "price_per_km"]
        X = pool[features].replace([np.inf, -np.inf], np.nan).fillna(
            pool[features].median()
        )

        raw_score = -iso.score_samples(X)
        risk_pct  = pd.Series(raw_score, index=pool.index).rank(pct=True) * 100
        pool["risk_score_0_100"] = risk_pct.round(1)

        state["pool"] = pool
        state["risk_model"] = iso
        state["risk_summary"] = {
            "mean_risk": float(pool["risk_score_0_100"].mean()),
            "high_risk_count": int((pool["risk_score_0_100"] > 80).sum()),
            "low_risk_count":  int((pool["risk_score_0_100"] < 20).sum()),
        }
        return state


# ═════════════════════════════════════════════════════════════════════
#  4. DEMAND AGENT — predicts days on market
# ═════════════════════════════════════════════════════════════════════
class DemandAgent(BaseAgent):
    name = "DemandAgent"

    def process(self, state: dict) -> dict:
        demand_model = joblib.load(DEMAND_MODEL)
        pool = state["pool"].copy()

        model_freq = pool["model"].value_counts(normalize=True)
        pool["model_freq_demand"] = pool["model"].map(model_freq)

        features_num = ["price_vs_model_median_pct", "mileage_km",
                        "car_age", "model_freq_demand"]
        features_cat = ["brand", "fuel_type"]
        X = pool[features_num + features_cat].copy()
        X.columns = ["price_vs_model_median_pct", "mileage_km", "car_age",
                      "model_freq", "brand", "fuel_type"]

        pool["predicted_dom"] = demand_model.predict(X).round(0).astype(int)

        state["pool"] = pool
        state["demand_summary"] = {
            "mean_predicted_dom": float(pool["predicted_dom"].mean()),
            "fast_sellers": int((pool["predicted_dom"] < 20).sum()),
            "slow_sellers": int((pool["predicted_dom"] > 50).sum()),
        }
        return state


# ═════════════════════════════════════════════════════════════════════
#  5. DEPRECIATION AGENT — maps brand depreciation rates
# ═════════════════════════════════════════════════════════════════════
class DepreciationAgent(BaseAgent):
    name = "DepreciationAgent"

    def process(self, state: dict) -> dict:
        dep = pd.read_csv(DEPRECIATION_CSV, index_col=0)
        pool = state["pool"].copy()

        dep_map = dep["annual_decay_pct"].to_dict()
        median_decay = dep["annual_decay_pct"].median()
        pool["brand_annual_decay_pct"] = pool["brand"].map(dep_map).fillna(median_decay)

        state["pool"] = pool
        state["depreciation"] = dep
        state["depreciation_summary"] = {
            "brands_tracked": len(dep),
            "fastest_depreciating": dep["annual_decay_pct"].idxmax(),
            "slowest_depreciating": dep["annual_decay_pct"].idxmin(),
        }
        return state


# ═════════════════════════════════════════════════════════════════════
#  6. DII AGENT — computes the Dealer Intelligence Index
# ═════════════════════════════════════════════════════════════════════
class DIIAgent(BaseAgent):
    name = "DIIAgent"

    WEIGHTS = {
        "profit_score":             0.30,
        "demand_score":             0.15,
        "holding_cost_score":       0.10,
        "capital_efficiency_score": 0.20,
        "depreciation_score":       0.10,
        "certainty_score":          0.15,
    }

    def _pct_rank(self, s, higher_is_better=True):
        ranked = s.rank(pct=True) * 100
        return ranked if higher_is_better else (100 - ranked)

    def process(self, state: dict) -> dict:
        pool = state["pool"].copy()

        # Component scores
        pool["expected_holding_total"] = pool["holding_cost_per_day"] * pool["days_on_market"]
        pool["expected_profit"] = (
            pool["expected_resale_price"] - pool["acquisition_cost"]
            - pool["recon_cost_est"] - pool["expected_holding_total"]
        )

        pool["profit_score"]             = self._pct_rank(pool["expected_profit"])
        pool["demand_score"]             = self._pct_rank(pool["days_on_market"], False)
        pool["holding_cost_score"]       = self._pct_rank(pool["holding_cost_per_day"], False)
        pool["roi"] = pool["expected_profit"] / pool["acquisition_cost"].replace(0, np.nan)
        pool["capital_efficiency_score"] = self._pct_rank(pool["roi"])
        pool["depreciation_score"]       = self._pct_rank(pool["brand_annual_decay_pct"], False)
        pool["certainty_score"]          = self._pct_rank(pool["risk_score_0_100"], False)

        # Composite DII
        pool["DII"] = sum(pool[c] * w for c, w in self.WEIGHTS.items()).round(1)
        pool = pool.sort_values("DII", ascending=False).reset_index(drop=True)

        state["pool"] = pool
        state["dii_summary"] = {
            "mean_dii": float(pool["DII"].mean()),
            "max_dii":  float(pool["DII"].max()),
            "min_dii":  float(pool["DII"].min()),
            "top5_brands": pool.head(20)["brand"].value_counts().head(5).to_dict(),
        }
        return state


# ═════════════════════════════════════════════════════════════════════
#  7. PORTFOLIO AGENT — optimizes vehicle selection under budget
# ═════════════════════════════════════════════════════════════════════
class PortfolioAgent(BaseAgent):
    name = "PortfolioAgent"

    BUDGETS = [5_000_000, 10_000_000, 20_000_000]
    MAX_VEHICLES = 40

    def process(self, state: dict) -> dict:
        import pulp
        pool = state["pool"]
        all_results = []

        for budget in self.BUDGETS:
            prob = pulp.LpProblem(f"Portfolio_{budget}", pulp.LpMaximize)
            x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pool.index}
            prob += pulp.lpSum(x[i] * pool.loc[i, "expected_profit"] for i in pool.index)
            prob += pulp.lpSum(x[i] * pool.loc[i, "acquisition_cost"] for i in pool.index) <= budget
            prob += pulp.lpSum(x[i] for i in pool.index) <= self.MAX_VEHICLES
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            chosen = [i for i in pool.index if x[i].value() == 1]
            portfolio = pool.loc[chosen].copy()
            portfolio["budget_scenario"] = budget
            portfolio["selection_method"] = "optimizer"
            all_results.append(portfolio)

        portfolios = pd.concat(all_results, ignore_index=True)

        # Dynamic resale pricing
        target_dom = 25
        dom_gap = portfolios["days_on_market"] - target_dom
        adj_pct = np.clip(-dom_gap / target_dom * 0.5, -0.08, 0.08)
        portfolios["recommended_resale_price"] = (
            portfolios["expected_resale_price"] * (1 + adj_pct)
        ).round(-2)
        portfolios["resale_adjustment_pct"] = (adj_pct * 100).round(1)

        # Negotiation offers
        portfolios["opening_offer"]     = (portfolios["acquisition_cost"] * 0.93).round(-2)
        portfolios["walk_away_ceiling"] = portfolios["acquisition_cost"].round(-2)
        portfolios["discount_off_asking_pct"] = (
            (portfolios["price_inr"] - portfolios["walk_away_ceiling"])
            / portfolios["price_inr"] * 100
        ).round(1)

        state["portfolios"] = portfolios
        state["portfolio_summary"] = {}
        for budget in self.BUDGETS:
            sub = portfolios[portfolios["budget_scenario"] == budget]
            state["portfolio_summary"][str(budget)] = {
                "n_vehicles": len(sub),
                "total_spend": float(sub["acquisition_cost"].sum()),
                "total_profit": float(sub["expected_profit"].sum()),
                "avg_dii": float(sub["DII"].mean()),
            }
        return state


# ═════════════════════════════════════════════════════════════════════
#  8. NEGOTIATION AGENT — generates human-readable negotiation advice
# ═════════════════════════════════════════════════════════════════════
class NegotiationAgent(BaseAgent):
    name = "NegotiationAgent"

    def process(self, state: dict) -> dict:
        portfolios = state["portfolios"].copy()

        def negotiate_advice(row):
            gap = row["discount_off_asking_pct"]
            if gap > 25:
                return "Strong leverage — significant overpricing vs. fair value. Open aggressively."
            elif gap > 15:
                return "Good leverage — moderate overpricing. Target mid-range discount."
            elif gap > 8:
                return "Moderate leverage — near fair value. Negotiate on reconditioning/extras."
            else:
                return "Limited leverage — priced near or below fair value. Secure quickly."

        portfolios["negotiation_advice"] = portfolios.apply(negotiate_advice, axis=1)
        state["portfolios"] = portfolios
        return state


# ═════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — coordinates all agents in dependency order
# ═════════════════════════════════════════════════════════════════════
class Orchestrator:
    """Runs all agents in sequence, manages shared state, and logs execution."""

    def __init__(self):
        self.agents = [
            DataAgent(),
            ValuationAgent(),
            RiskAgent(),
            DemandAgent(),
            DepreciationAgent(),
            DIIAgent(),
            PortfolioAgent(),
            NegotiationAgent(),
        ]
        self.state = {}
        self.log = []

    def run(self) -> dict:
        print("=" * 70)
        print("  DriveWise AI+ -- Multi-Agent Orchestrator")
        print("=" * 70)

        for agent in self.agents:
            t0 = time.time()
            print(f"\n[>>] Running {agent.name} ...")
            try:
                self.state = agent.process(self.state)
                elapsed = time.time() - t0
                self.log.append({
                    "agent": agent.name,
                    "status": "SUCCESS",
                    "elapsed_sec": round(elapsed, 2),
                })
                print(f"  [OK] {agent.name} completed in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - t0
                self.log.append({
                    "agent": agent.name,
                    "status": "FAILED",
                    "error": str(e),
                    "elapsed_sec": round(elapsed, 2),
                })
                print(f"  [ERROR] {agent.name} FAILED: {e}")
                raise

        # Save outputs
        if "portfolios" in self.state:
            self.state["portfolios"].to_csv(ORCHESTRATOR_OUTPUT, index=False)
            print(f"\n[OK] Full pipeline output -> {ORCHESTRATOR_OUTPUT}")

        # Save run log
        log_data = {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agents_run": len(self.log),
            "total_elapsed_sec": round(sum(e["elapsed_sec"] for e in self.log), 2),
            "agent_log": self.log,
            "summaries": {
                k: v for k, v in self.state.items()
                if k.endswith("_summary")
            },
        }
        with open(ORCHESTRATOR_LOG, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"[OK] Run log -> {ORCHESTRATOR_LOG}")

        return self.state


def main():
    orchestrator = Orchestrator()
    state = orchestrator.run()

    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for key, val in state.items():
        if key.endswith("_summary"):
            print(f"\n{key}:")
            for k, v in val.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
