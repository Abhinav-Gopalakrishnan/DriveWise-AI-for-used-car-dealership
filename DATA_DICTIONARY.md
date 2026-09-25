# DriveWise AI+ — Phase 1 Data Dictionary (v3: real scraped data)

## Provenance
**10,468 real, live-scraped used-car listings** from CarDekho (India),
scraped by the project owner via `02_scraper.py`, covering **28 brands**
across **295 cities**. This replaces every earlier stand-in dataset — all
downstream modelling trains on this.

## Cleaning applied (0.3% of rows removed — minimal, logged)
- Deduplicated on brand/model/year/price/mileage/city
- Fixed a brand/model mis-split ("Maruti" vs "Maruti Suzuki" merged; "Land
  Rover" rejoined where the scraper's naive first-word split broke it)
- Rounded price to the nearest rupee (source had float noise from the
  Lakh/Crore conversion, e.g. `465000.00000000006`)
- Dropped price outliers (<₹40,000 or >₹2 crore) and mileage outliers
  (<100km or >400,000km — a handful of mileage values were clear regex
  mismatches from the scraper, e.g. 82 million km, and were dropped rather
  than guessed at)
- Dropped listings outside 1998–2026 registration years

## 1. `drivewise_market_history.csv` (10,468 rows, 28 brands, 295 cities)
Trains the predictive models in Phase 2.

| column | type | source |
|---|---|---|
| listing_id | generated | id |
| brand, model, year, price_inr, transmission, mileage_km, fuel_type, city | **real** | scraped |
| car_age, mileage_per_year, price_per_km, price_vs_model_median_pct | **derived** | computed from real columns |
| source_url | **real** | the actual listing page scraped |

## 2. `drivewise_acquisition_pool.csv` (1,500 rows)
Sample used to demo the DII scoring, agents, and portfolio optimizer.

| column | type | how it's generated |
|---|---|---|
| all market_history columns (incl. **real city** — no longer simulated) | real/derived | inherited |
| condition_grade | **rule-based** | from mileage/year thresholds |
| days_on_market | **simulated** | rises with price-above-model-median and mileage, plus noise |
| recon_cost_est | **rule-based** | fixed reconditioning cost per condition grade (₹) |
| acquisition_cost | **simulated** | (market price × 70–85% acquisition discount) − recon cost |
| holding_cost_per_day | **simulated** | 0.22%/day of acquisition cost |
| expected_resale_price | real | = observed market price (Phase 2 regression target) |

**What's still simulated and why:** the marketplace data is sale-side only
(what a listing is priced at) — there's no dealer-side transaction record
(what a dealer actually pays to acquire a car, real days-on-market, formal
condition grading). That gap is what DII/portfolio needs, so it's generated
with documented rules. Region/city is now genuine — a real strength of
having live-scraped data over the earlier stand-in — so the "cross-city
arbitrage" angle from your original case study is now backed by real
geographic price variation, not a simulated approximation.
