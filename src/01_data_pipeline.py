"""
DriveWise AI+ — Phase 1 (v3): Clean the real scraped dataset
================================================================
Input: data/scraped_listings_raw.csv (10,504 rows scraped from CarDekho
by the user, via 02_scraper.py, run locally). This REPLACES the earlier
stand-in market dataset now that real scraped data exists -- per the
project requirement, all downstream modelling should train on this.

Known raw-data quirks fixed here (found on inspection, not assumed):
  - "Maruti" and "Maruti Suzuki" are the same brand, split into two labels
    by the scraper's naive "first word = brand" split
  - "Land Rover" is a two-word brand, sometimes mis-split into
    brand="Land", model="Rover ..."
  - price has floating-point noise from the Lakh/Crore multiplication
    (e.g. 465000.00000000006) -- rounded to the nearest rupee
  - a small number of mileage values are clearly corrupted
    (e.g. 82,000,275 km) -- these are regex mismatches from the scraper
    picking up an unrelated number; dropped rather than guessed at

Outputs (same two-file structure as v1/v2):
1. drivewise_market_history.csv -- cleaned, feature-engineered, real.
2. drivewise_acquisition_pool.csv -- sampled subset + simulated
   dealer-operational layer (condition, days-on-market, acquisition cost,
   holding cost). NOTE: region is no longer simulated -- the scraper
   captured each listing's real city, so that field is genuine data now.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

RAW_PATH = "../data/scraped_listings_raw.csv"
MARKET_OUT = "../data/drivewise_market_history.csv"
POOL_OUT = "../data/drivewise_acquisition_pool.csv"

CURRENT_YEAR = 2026  # scrape was run live, current listings

TWO_WORD_BRANDS = ["Land Rover", "Mercedes Benz"]  # brands the naive split can break

BRAND_FIXES = {
    "Maruti": "Maruti Suzuki",
    "Land": "Land Rover",          # mis-split rows where brand caught only "Land"
    "Mercedes-Benz": "Mercedes-Benz",
}


def fix_brand_model_split(row):
    """Re-join brand/model for known two-word brands the scraper mis-split."""
    brand, model = row["brand"], row["model"]
    if brand == "Land" and not model.startswith("Rover"):
        # "Land Rover Discovery" -> brand="Land", model="Rover Discovery"
        return "Land Rover", model.removeprefix("Rover").strip()
    return brand, model


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)

    df = df.drop_duplicates(subset=["brand", "model", "year", "price_inr",
                                     "mileage_km", "city"])

    fixed = df.apply(fix_brand_model_split, axis=1, result_type="expand")
    df["brand"], df["model"] = fixed[0], fixed[1]
    df["brand"] = df["brand"].replace(BRAND_FIXES)

    df["price_inr"] = df["price_inr"].round(0)

    df = df[(df["price_inr"] >= 40_000) & (df["price_inr"] <= 20_000_000)]
    df = df[(df["mileage_km"] >= 100) & (df["mileage_km"] <= 400_000)]
    df = df[(df["year"] >= 1998) & (df["year"] <= CURRENT_YEAR)]
    df = df.dropna(subset=["brand", "model", "transmission", "fuel_type", "city"])

    removed = before - len(df)
    print(f"[clean] removed {removed} rows ({removed/before:.1%}) as duplicates/junk/outliers")

    df["city"] = df["city"].str.strip().str.title()
    df["brand"] = df["brand"].str.strip()
    df["model"] = df["model"].str.strip()

    return df.reset_index(drop=True)


def engineer_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = (CURRENT_YEAR - df["year"] + 1).clip(lower=1)
    df["mileage_per_year"] = (df["mileage_km"] / df["car_age"]).round(0)
    df["price_per_km"] = (df["price_inr"] / df["mileage_km"].replace(0, np.nan)).round(2)

    model_key = df["brand"] + "_" + df["model"]
    model_median = df.groupby(model_key)["price_inr"].transform("median")
    df["price_vs_model_median_pct"] = ((df["price_inr"] - model_median) / model_median * 100).round(1)

    keep_cols = ["brand", "model", "year", "price_inr", "transmission", "mileage_km",
                 "fuel_type", "city", "car_age", "mileage_per_year", "price_per_km",
                 "price_vs_model_median_pct", "source_url"]
    df = df[keep_cols].reset_index(drop=True)
    df.insert(0, "listing_id", [f"DW-{i:06d}" for i in range(1, len(df) + 1)])
    return df


def add_dealer_operational_layer(df: pd.DataFrame, n_sample: int = 1500) -> pd.DataFrame:
    n_sample = min(n_sample, len(df))
    pool = df.sample(n=n_sample, random_state=42).reset_index(drop=True)

    def grade(row):
        mpy, age = row["mileage_per_year"], row["car_age"]
        if mpy < 9000 and age <= 4:
            return "Excellent"
        elif mpy < 15000 and age <= 8:
            return "Good"
        elif mpy < 22000:
            return "Fair"
        return "Poor"

    pool["condition_grade"] = pool.apply(grade, axis=1)

    noise = RNG.normal(0, 6, size=len(pool))
    dom = (
        15
        + 0.8 * pool["price_vs_model_median_pct"].clip(lower=-30, upper=60)
        + 0.0004 * pool["mileage_km"]
        + noise
    )
    pool["days_on_market"] = dom.clip(lower=3).round(0).astype(int)

    recon_cost = {"Excellent": 3000, "Good": 9000, "Fair": 22000, "Poor": 45000}  # INR
    pool["recon_cost_est"] = pool["condition_grade"].map(recon_cost)

    acq_discount = RNG.uniform(0.70, 0.85, size=len(pool))
    pool["acquisition_cost"] = (pool["price_inr"] * acq_discount - pool["recon_cost_est"]).round(0)
    pool["acquisition_cost"] = pool["acquisition_cost"].clip(lower=10_000)

    pool["holding_cost_per_day"] = (pool["acquisition_cost"] * 0.0022).round(2)
    pool["expected_resale_price"] = pool["price_inr"]

    return pool


def main():
    df = load_and_clean(RAW_PATH)
    market = engineer_market_features(df)
    market.to_csv(MARKET_OUT, index=False)
    print(f"[market] saved {len(market)} rows, {market['brand'].nunique()} brands, "
          f"{market['city'].nunique()} cities -> {MARKET_OUT}")

    pool = add_dealer_operational_layer(market, n_sample=1500)
    pool.to_csv(POOL_OUT, index=False)
    print(f"[pool]   saved {len(pool)} rows -> {POOL_OUT}")

    print("\ntop brands:")
    print(market["brand"].value_counts().head(10))
    print("\ntop cities:")
    print(market["city"].value_counts().head(10))
    print("\nmarket history summary:")
    print(market[["price_inr", "mileage_km", "car_age"]].describe().round(1))


if __name__ == "__main__":
    main()
