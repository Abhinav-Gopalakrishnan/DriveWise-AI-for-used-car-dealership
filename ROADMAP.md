# DriveWise AI+ — Project Roadmap (100% Complete)

**Status: ALL PHASES COMPLETE (Phases 1 through 8).**  
**Dataset:** 10,468 verified real CarDekho listings (>10,000 requirement satisfied), 28 brands, 295 cities across India.

---

## Phase 1 — Data Pipeline & Scraping [COMPLETED]
- [x] Scraped 10,505 real marketplace listings via `02_scraper.py`
- [x] Cleaned and deduplicated into 10,468 rows via `01_data_pipeline.py`
- [x] Generated feature-engineered `drivewise_market_history (1).csv`
- [x] Generated dealer-operational acquisition pool `drivewise_acquisition_pool.csv`

---

## Phase 2 — Predictive Models [COMPLETED]
- [x] **Fair market value model**: Compared Linear Regression, Random Forest, and XGBoost. Best: XGBoost ($R^2 = 0.9129$, MAPE = 17.20%).
- [x] **Depreciation curve model**: Fitted exponential decay curves $P(t) = a \cdot e^{-bt}$ across 8 major automobile brands (`depreciation_curves.csv`).
- [x] **Purchase-risk score**: Unsupervised Isolation Forest anomaly detection on real behavioral metrics (`risk_model_isolationforest.joblib`).
- [x] **Demand / selling-time proxy**: Random Forest regression predicting `days_on_market` (`demand_dom_model.joblib`).
- [x] Saved all model artifacts and generated `drivewise_acquisition_pool_scored.csv`.

---

## Phase 3 — Dealer Intelligence Index (DII) [COMPLETED]
- [x] Formulated composite DII combining:
  1. Expected Profit Score ($w_1 = 0.30$)
  2. Demand & Velocity Score ($w_2 = 0.15$)
  3. Holding Cost Efficiency Score ($w_3 = 0.10$)
  4. Capital Utilization / ROI Score ($w_4 = 0.20$)
  5. Brand Depreciation Resistance Score ($w_5 = 0.10$)
  6. Purchase Certainty / Risk Score ($w_6 = 0.15$)
- [x] Scale-invariant percentile normalization for all criteria
- [x] Computed DII across all 1,500 acquisition pool vehicles (`drivewise_pool_with_dii.csv`)
- [x] Sensitivity and stability analysis against alternative weightings (`04_dii_scoring.py`)

---

## Phase 4 — Optimization Layer & Pricing [COMPLETED]
- [x] Formulated 0/1 Knapsack optimization as Mixed-Integer Linear Programming (MILP)
- [x] Solved via CBC solver in PuLP (`05_portfolio_optimizer.py`)
- [x] Dynamic resale pricing based on predicted DOM vs target DOM ($\pm 8\%$ bounded adjustment)
- [x] Algorithmic negotiation recommendation: Opening offer (7% below target) and walk-away ceiling
- [x] Generated scenario portfolios (`drivewise_portfolio_recommendations.csv`)

---

## Phase 5 — Multi-Agent Orchestration [COMPLETED]
- [x] Modular architecture with 8 dedicated agents (`06_multi_agent_orchestrator.py`):
  1. DataAgent
  2. ValuationAgent
  3. RiskAgent
  4. DemandAgent
  5. DepreciationAgent
  6. DIIAgent
  7. PortfolioAgent
  8. NegotiationAgent
- [x] Central Orchestrator coordinating execution, managing shared state, and logging execution telemetry
- [x] Output generated: `drivewise_full_pipeline_output.csv` and `orchestrator_run_log.json`

---

## Phase 6 — Explainable AI (XAI) [COMPLETED]
- [x] Global feature importance extracted from tree architecture (`drivewise_feature_importance.csv`)
- [x] Permutation sensitivity analysis for Isolation Forest purchase risk model
- [x] Per-vehicle explainable decision cards generated for acquisition managers (`drivewise_shap_explanations.csv`)
- [x] Automated action classification: STRONG_BUY, ACCUMULATE, HOLD, REJECT (`drivewise_xai_summary.json`)

---

## Phase 7 — Evaluation & Backtesting [COMPLETED]
- [x] Holdout out-of-sample backtest across 2,094 vehicles (`evaluation_metrics_by_segment.csv`)
- [x] Subgroup performance evaluated across price tiers, vehicle age brackets, and top brands
- [x] DII Decile monotonicity validation (`evaluation_dii_deciles.csv`): Top decile yields 33.2% ROI in 24.6 days vs bottom decile at 6.3% ROI in 78.3 days (6.7x profit spread)
- [x] 5-strategy portfolio benchmark across 4 capital tiers (Rs. 25L, Rs. 50L, Rs. 1.0 Cr, Rs. 2.0 Cr): DriveWise AI+ MILP delivers +110.5% profit uplift over random and +209.2% over volume heuristics (`evaluation_strategy_comparison.csv`)
- [x] Downside stress testing under 4 macroeconomic shock scenarios (`evaluation_stress_testing.csv`)
- [x] Summary metrics exported (`drivewise_evaluation_summary.json`)

---

## Phase 8 — Comprehensive Final Project Report [COMPLETED]
- [x] Generated full academic and industry report in Markdown format: `DriveWise_AI_Plus_Final_Report.md`
- [x] Generated formatted Microsoft Word submission document: `DriveWise_AI_Plus_Final_Report.docx`
- [x] Complete end-to-end documentation covering problem formulation, mathematics, agent architectures, empirical tables, managerial recommendations, and future work.
