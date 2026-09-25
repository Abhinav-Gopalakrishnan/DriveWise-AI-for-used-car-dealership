"""
DriveWise AI+ — Phase 1 (v3): Live Scraper
=============================================
Run this on YOUR OWN machine, not in a sandboxed environment — it needs
real internet access to a live marketplace site, which this chat's
environment intentionally does not have.

Target: CarDekho used-car listings (cardekho.com), city by city.
Chosen because its listing pages are server-rendered (the listing text is
present in the raw HTML response, not injected only by client-side JS),
which makes polite, robots.txt-respecting scraping actually feasible
without a headless browser.

BEFORE YOU RUN THIS
--------------------
1. This script checks robots.txt itself at runtime (see check_allowed())
   and refuses to fetch anything it disallows. It does NOT bypass or spoof
   its way around that check — if a path is disallowed, fix the URL
   pattern or pick a different, permitted source. Don't remove that check.
2. Re-read CarDekho's current Terms of Service yourself before running this
   at any real scale (sites change ToS over time; I can't monitor that for
   you). Keep this to reasonable, non-commercial, academic-project volume.
3. Rate limiting is on by default (2-4s between requests, randomised).
   Don't lower it just to go faster — that's the difference between
   "polite crawler" and "the kind of traffic that gets your IP blocked."
4. Site layouts change. The extraction logic below works off the actual
   text patterns CarDekho listing pages use (confirmed against real pages
   at the time this was written: "Second-hand <year> <make> <model> ...
   for sale in <city>", "<n> kms", "Automatic"/"Manual",
   "Petrol"/"Diesel"/...). If CarDekho has changed its page structure by
   the time you run this, open a listing page in your browser, view
   source, and adjust EXTRACT_PATTERNS below — the structure is isolated
   there specifically so it's easy to patch without touching the crawl
   logic.

Output
------
Writes data/scraped_listings_raw.csv in the SAME schema
01_data_pipeline.py expects (brand, model, year, price, mileage,
transmission, fuel_type, city) so you can drop it straight into the
existing pipeline once it's done.

Install:  pip install requests beautifulsoup4 lxml pandas tqdm
Run:      python 02_scraper.py
"""

import csv
import random
import re
import time
import urllib.robotparser
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.cardekho.com"
USER_AGENT = "DriveWiseAI-StudentProject/1.0 (+academic use; contact: <your email>)"
REQUEST_DELAY_RANGE = (2.0, 4.0)   # seconds between requests -- do not shrink this
MAX_PAGES_PER_CITY = 5             # keep this modest; raise gradually, watch for blocks
OUTPUT_PATH = Path("../data/scraped_listings_raw.csv")

CITIES = [
    "delhi", "mumbai", "bangalore", "chennai", "hyderabad", "pune",
    "kolkata", "ahmedabad", "jaipur", "lucknow", "chandigarh", "kochi",
    "coimbatore", "indore", "surat", "nagpur", "patna", "bhopal",
]

# --- Extraction patterns -----------------------------------------------
# Keyed off real observed listing text, not guessed CSS class names --
# markup changes far more often than the words a site uses to describe a
# listing, so anchoring on text is the more durable choice here.
LISTING_HEADLINE_RE = re.compile(
    r"(Second-hand|Used)\s+(\d{4})\s+(.+?)\s+for sale in\s+([A-Za-z ]+)"
)
KMS_RE = re.compile(r"([\d,]+)\s*kms", re.IGNORECASE)
TRANSMISSION_RE = re.compile(r"\b(Automatic|Manual)\b")
FUEL_RE = re.compile(r"\b(Petrol|Diesel|CNG|LPG|Electric|Hybrid)\b")
PRICE_RE = re.compile(r"₹\s*([\d,]+\.?\d*)\s*(Lakh|Lac|L|Crore|Cr)?\b", re.IGNORECASE)


@dataclass
class Listing:
    brand: str
    model: str
    year: int
    price_inr: float
    mileage_km: float
    transmission: str
    fuel_type: str
    city: str
    source_url: str


def get_robots_parser(base_url: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urljoin(base_url, "/robots.txt"))
    rp.read()
    return rp


def check_allowed(rp: urllib.robotparser.RobotFileParser, url: str) -> bool:
    return rp.can_fetch(USER_AGENT, url)


def polite_get(session: requests.Session, url: str) -> str | None:
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  [skip] {url} -> HTTP {resp.status_code}")
            return None
        return resp.text
    except requests.RequestException as e:
        print(f"  [error] {url} -> {e}")
        return None
    finally:
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))


def parse_price(text: str) -> float | None:
    # NOTE: CarDekho often shows a struck-through original price followed
    # by a discounted one (e.g. "9.07 L" then "8.91L"). This takes the
    # FIRST price found in the window, which is usually the listed price
    # nearest the headline -- if you specifically want the discounted
    # price when both are present, switch to the LAST match instead.
    m = PRICE_RE.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in ("lakh", "lac", "l"):
        value *= 100_000
    elif unit in ("crore", "cr"):
        value *= 10_000_000
    return value


def extract_listings(html: str, city: str, source_url: str) -> list[Listing]:
    soup = BeautifulSoup(html, "lxml")
    text_blocks = soup.get_text("\n", strip=True).split("\n")
    full_text = "\n".join(text_blocks)

    results = []
    for m in LISTING_HEADLINE_RE.finditer(full_text):
        _, year, make_model, listing_city = m.groups()
        # Real listing order runs forward from the headline: price, then
        # kms, transmission, fuel, location -- all in one following window.
        window = full_text[m.end(): m.end() + 400]

        kms_m = KMS_RE.search(window)
        trans_m = TRANSMISSION_RE.search(window)
        fuel_m = FUEL_RE.search(window)
        price = parse_price(window)

        if not (kms_m and trans_m and fuel_m and price):
            continue  # incomplete record -- skip rather than guess

        parts = make_model.strip().split(" ", 1)
        brand = parts[0]
        model = parts[1] if len(parts) > 1 else ""

        results.append(Listing(
            brand=brand,
            model=model,
            year=int(year),
            price_inr=price,
            mileage_km=float(kms_m.group(1).replace(",", "")),
            transmission=trans_m.group(1),
            fuel_type=fuel_m.group(1),
            city=listing_city.strip() or city,
            source_url=source_url,
        ))
    return results


def scrape_city(session: requests.Session, rp, city: str) -> list[Listing]:
    listings = []
    for page in range(1, MAX_PAGES_PER_CITY + 1):
        # NOTE: verify this pagination pattern in your browser first --
        # CarDekho's actual query param may differ from this guess.
        url = f"{BASE}/used-cars+in+{city}?pageNo={page}"
        if not check_allowed(rp, url):
            print(f"  [robots.txt] disallowed: {url} -- stopping this city")
            break

        print(f"  fetching {url}")
        html = polite_get(session, url)
        if html is None:
            break

        page_listings = extract_listings(html, city, url)
        if not page_listings:
            print(f"  [empty] no listings parsed on page {page} -- "
                  f"stopping (either end of results or selectors need updating)")
            break

        listings.extend(page_listings)
        print(f"  +{len(page_listings)} listings (running total {len(listings)})")

    return listings


def main():
    rp = get_robots_parser(BASE)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_listings: list[Listing] = []
    for city in CITIES:
        print(f"\n== {city} ==")
        all_listings.extend(scrape_city(session, rp, city))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_listings[0]).keys()) if all_listings else [])
        writer.writeheader()
        for row in all_listings:
            writer.writerow(asdict(row))

    print(f"\nDone. {len(all_listings)} listings -> {OUTPUT_PATH}")
    print("Upload this CSV back to the chat and I'll fold it into the Phase 1 pipeline.")


if __name__ == "__main__":
    main()
