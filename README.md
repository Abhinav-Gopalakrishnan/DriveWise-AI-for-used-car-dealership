# DriveWise AI+: A Multi-Agent Multi-Objective Intelligent Decision Support System for Used Car Acquisition, Dynamic Pricing, Portfolio Optimization, and Profit Maximization Using Machine Learning and Explainable AI

**Course:** Business Analytics — Individual Case Study  
**Course Code:** CB.SC.U4CSE23201  
**Candidate Name:** Abhinav (Individual Submission)  
**Academic Requirement:** Minimum 10,000 empirical marketplace records (**Achieved: 10,468 verified listings**)  
**Submission Repository Structure:**
- `README.md` — Project summary, problem statement, objectives, data sourcing, methodology, key findings, and references.
- `data/` — Collected raw scraped dataset and final cleaned/anonymized datasets used for analysis.
- `analysis.ipynb` — Complete, fully executed Jupyter Notebook containing preprocessing, visualization, predictive modeling, optimization, XAI, and evaluation.
- `Case_Study_Report.pdf` — Formal case study report prepared according to the prescribed 7-section format (8-10 pages).

---

## 1. Problem Statement
In the rapidly growing pre-owned automotive retail industry in India, automobile dealerships face severe operational friction across the complete "buy-to-sell" inventory lifecycle:
1. **Valuation Asymmetry & Overpayment:** Sourcing vehicles based on intuition rather than predictive statistical fair market valuation leads to acquisition premiums that cannot be recovered.
2. **The Inventory Holding Cost Penalty:** Pre-owned vehicles require significant working capital. Floor-plan interest, showroom rent, security, and maintenance impose carrying costs of **0.18% to 0.25% daily** (approx. 6% to 7.5% monthly). Misjudging sales velocity erodes gross profit margins.
3. **Suboptimal Heuristic Sourcing:** Dealerships traditionally buy either the cheapest available vehicles to maximize volume or chase vehicles with large headline price discounts, oblivious to slow-depreciating luxury assets with prohibitive holding costs.
4. **Black-Box Skepticism:** Dealership acquisition managers systematically reject opaque algorithmic advice unless transparent, explainable rationales and actionable negotiation guidance are provided.

---

## 2. Research Objectives
1. **Predictive Appraisal & Depreciation Modeling:** Formulate and validate machine learning regression pipelines to estimate vehicle fair market value and parametric brand depreciation curves using large-scale live-scraped data (>10,000 records).
2. **Multi-Objective Composite Evaluation (DII):** Develop the **Dealer Intelligence Index (DII)**, a novel scale-invariant metric that jointly optimizes expected profit, sales velocity, carrying costs, capital efficiency (ROI), brand depreciation, and unsupervised purchase risk.
3. **Constrained Portfolio Optimization & Execution:** Formulate a Mixed-Integer Linear Programming (MILP) 0/1 knapsack optimizer to construct maximum-profit vehicle portfolios under dealership capital budgets and showroom lot capacities, integrated with velocity-adjusted dynamic resale pricing and negotiation boundary rules.
4. **Explainable AI (XAI) & Autonomous Multi-Agent Orchestration:** Implement an 8-agent modular architecture providing explainable decision cards with natural-language drivers to establish managerial trust.

---

## 3. Data Collection Source & Methodology
- **Target Portal:** CarDekho India (`https://www.cardekho.com`).
- **Data Collection Method:** Custom polite web scraper (`02_scraper (1).py`) utilizing Python `requests`, `BeautifulSoup4`, and `lxml`.
- **Compliance & Safeguards:** Automated runtime compliance with CarDekho's `robots.txt` via `urllib.robotparser`, randomized 2.0 to 4.0 second request delay, and custom academic User-Agent header.
- **Geographic Coverage:** 18 major metropolitan/tier-2 seed cities, expanding into **295 distinct Indian cities**.
- **Records Harvested:** **10,505 raw scraped listings** $\rightarrow$ **10,468 verified listings** after deduplication, entity resolution, and outlier hygiene (price: Rs. 40k to 2 Cr; mileage: 100 to 400k km; year: 1998 to 2026).
- **Core Attributes:** Brand, Model, Registration Year, Listed Price (INR), Mileage (km), Transmission, Fuel Type, City, Vehicle Age, Mileage/Year, Price/km, Price vs Model Median %, Condition Grade, Days on Market, Reconditioning Cost, Acquisition Cost, and Daily Holding Cost.

---

## 4. Analytics Methods Applied
1. **Predictive Regression Analytics:**
   - Multiple Linear Regression (Baseline)
   - Random Forest Regressor (Ensemble Benchmark)
   - Extreme Gradient Boosting (**XGBoost Regressor** - Production Selected)
2. **Parametric Depreciation Analytics:**
   - Non-linear exponential decay regression: $\text{Price}(t) = a \cdot e^{-b \cdot t}$ across 8 automobile manufacturers.
3. **Unsupervised Anomaly & Risk Analytics:**
   - **Isolation Forest** anomaly detection modeling purchase risk on behavioral ratios independent of condition labels.
4. **Multi-Objective Decision Analytics:**
   - **Dealer Intelligence Index (DII):** Scale-invariant percentile normalization synthesizing 6 conflicting criteria.
5. **Prescriptive Operations Research / Optimization:**
   - **0/1 Knapsack Mixed-Integer Linear Programming (MILP)** solved via the COIN-OR Branch and Cut (CBC) solver in PuLP.
6. **Dynamic Pricing & Algorithmic Bargaining:**
   - Bounded velocity adjustments ($\pm 8\%$) based on forecasted days-on-market vs target turnover.
   - Anchor pricing: Opening offer (7% below target) and walk-away break-even ceiling.
7. **Explainable AI (XAI):**
   - Global tree feature importance, permutation sensitivity analysis, and per-vehicle explainability decision cards.

---

## 5. Key Empirical Results

1. **Valuation Accuracy:** XGBoost achieved $\mathbf{R^2 = 0.9129}$, $\mathbf{\text{MAPE} = 17.20\%}$, and $\mathbf{\text{MAE} = \text{Rs. } 1,89,796}$, substantially outperforming Random Forest ($R^2 = 0.8697$) and Linear Regression ($R^2 = 0.5261$).
2. **DII Decile Monotonic Validation:**
   - **Decile 1 (Top 10% DII):** Average Profit = **Rs. 2,62,577**, **33.2% ROI**, **24.6 days on market**, **Risk Score = 29.5/100**.
   - **Decile 10 (Bottom 10% DII):** Average Profit = **Rs. 39,236**, **6.3% ROI**, **78.3 days on market**, **Risk Score = 74.0/100**.
   - Proves a **6.7x profit separation** and **3.2x velocity acceleration** between top and bottom deciles.
3. **Portfolio Optimization vs. Industry Baselines (Rs. 1.0 Crore Budget):**
   - *Random Selection (Monte Carlo avg):* Rs. 19.21 Lakh profit (19.24% ROI)
   - *Cheapest-First (Volume Heuristic):* Rs. 13.08 Lakh profit (30.50% ROI, lots capped, capital wasted)
   - *Greedy by DII:* Rs. 35.98 Lakh profit (36.07% ROI, 21.8 days DOM)
   - *DriveWise AI+ MILP Optimizer:* **Rs. 40.43 Lakh profit** (**40.44% ROI**, **19.3 days DOM**)
   - Profit Uplift: **+110.5% vs. Random**, **+209.2% vs. Volume**, and **+12.4% vs. Greedy DII**.
4. **Downside Macroeconomic Stress Resilience:**
   - Under severe stress (-15% retail prices, +50% DOM, +30% recon costs), the traditional volume strategy collapses into a net loss (**-Rs. 1.23 Lakh**, -109.4% drawdown).
   - DriveWise AI+ preserves **Rs. 14.24 Lakh net profit** (+14.2% margin, 35.2% retention), proving robust capital defensibility.

---

## 6. Project Directory Structure
```
.
├── README.md                                  # Submission summary and technical specifications
├── analysis.ipynb                             # Complete Jupyter Notebook with code, plots & outputs
├── Case_Study_Report.pdf                      # Formal Case Study Report in prescribed format (8-10 pp)
├── Case_Study_Report.docx                     # Editable Word version of formal report
├── data/                                      # Data directory containing all datasets
│   ├── scraped_listings_raw.csv               # Raw scraped marketplace listings (10,505 rows)
│   ├── drivewise_market_history.csv           # Cleaned & engineered dataset (10,468 rows, 28 brands, 295 cities)
│   ├── drivewise_acquisition_pool.csv         # 1,500 sampled vehicles with dealer operational economics
│   ├── drivewise_pool_with_dii.csv            # Fully evaluated pool with 6 component percentiles and DII
│   ├── drivewise_portfolio_recommendations.csv # Optimal recommendations across budget tiers
│   └── DATA_DICTIONARY.md                     # Schema documentation and operational formulas
├── 01_data_pipeline.py                        # Data cleaning, deduplication & operational simulation
├── 02_scraper (1).py                          # Polite live web crawler for CarDekho India
├── 03_predictive_models.py                    # Predictive ML training (XGBoost, RF, Isolation Forest)
├── 04_dii_scoring.py                          # DII multi-objective scoring & sensitivity analysis
├── 05_portfolio_optimizer.py                  # MILP 0/1 knapsack optimization & dynamic pricing
├── 06_multi_agent_orchestrator.py             # 8-agent autonomous sequential orchestration
├── 07_explainability.py                       # Explainable AI feature importance & decision cards
├── 08_evaluation_backtest.py                  # Model subgroup evaluation, deciles & stress testing
├── 09_generate_final_report.py                # Final project report generator
├── 10_generate_project_audit_documentation.py # Master documentation generator
└── models/ & artifacts                        # Serialized .joblib models, CSV evaluation tables & logs
```

---

## 7. Key References
1. Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J. (2019). Car price prediction using machine learning techniques. *TEM Journal*, 8(1), 113-118.
2. Kuiper, S. (2008). Introduction to multiple regression: How much is your car worth? *Journal of Statistics Education*, 16(3), 1-14.
3. Lessmann, S., Sung, M. C., Johnson, J. E., & Ma, T. (2017). Automated valuation models and inventory management for pre-owned vehicles. *Journal of Business Research*, 75, 136-146.
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765-4774.
5. Pudaruth, S. (2014). Predicting the price of used cars using machine learning techniques. *International Journal of Information & Computation Technology*, 4(7), 753-764.
6. Sun, N., Liu, H., & Chen, J. (2021). Used car price prediction and residual value forecasting using XGBoost and LightGBM. *IEEE Access*, 9, 89345-89356.
