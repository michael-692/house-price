"""
Download publicly available housing data for the collateral valuation model.

Data sources:
1. FHFA House Price Index (already in repo) - market trends by metro/state
2. King County, WA residential sales (GitHub mirror) - property-level features
3. Zillow ZHVI (Zillow Home Value Index) - current market values by zip code
4. Arlington County VA open data - local assessment context
"""

import requests
import pandas as pd
import numpy as np
import os
import sys

DATA_RAW = "data/raw"
os.makedirs(DATA_RAW, exist_ok=True)

def download(url, dest, label):
    if os.path.exists(dest):
        print(f"  [cached] {label}")
        return True
    try:
        print(f"  Downloading {label} ...")
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"  Saved {dest} ({len(r.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ── 1. King County, WA house sales (2014–2015) ──────────────────────────────
#    21,613 residential transactions with rich property features
KC_URLS = [
    "https://raw.githubusercontent.com/dssg/public_data/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/shrikant-temburwar/King-County-House-Price-Prediction/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/pkmklong/house-price-prediction/master/data/kc_house_data.csv",
    "https://raw.githubusercontent.com/aggarwalraj/House-Price-Prediction/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/Gimanh/my-study/master/data-science/datasets/kc_house_data.csv",
]

kc_ok = False
for url in KC_URLS:
    if download(url, f"{DATA_RAW}/kc_house_data.csv", "King County house sales"):
        try:
            df = pd.read_csv(f"{DATA_RAW}/kc_house_data.csv")
            if len(df) > 1000 and "price" in df.columns:
                print(f"  Verified: {len(df):,} rows")
                kc_ok = True
                break
            else:
                os.remove(f"{DATA_RAW}/kc_house_data.csv")
        except Exception:
            if os.path.exists(f"{DATA_RAW}/kc_house_data.csv"):
                os.remove(f"{DATA_RAW}/kc_house_data.csv")

if not kc_ok:
    print("  All King County URLs failed – will generate representative dataset")


# ── 2. Zillow ZHVI – Single-Family Homes by Zip Code ────────────────────────
ZILLOW_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
)
download(ZILLOW_URL, f"{DATA_RAW}/zillow_zhvi_zip_sfr.csv", "Zillow ZHVI by zip (SFR)")


# ── 3. Arlington County open data – Real Estate Assessments ─────────────────
ARLINGTON_URL = (
    "https://catalog.data.gov/api/action/package_show"
    "?id=aaf9f7ee-d0e3-4ab1-b56d-dc38b4e1b5b4"
)
ARLINGTON_CSV_URLS = [
    "https://data.arlingtonva.us/api/views/fjva-9mmx/rows.csv?accessType=DOWNLOAD",
    "https://opendata.arcgis.com/datasets/fjva9mmx_0.csv",
]
arlington_ok = False
for url in ARLINGTON_CSV_URLS:
    if download(url, f"{DATA_RAW}/arlington_assessments.csv", "Arlington County assessments"):
        try:
            df = pd.read_csv(f"{DATA_RAW}/arlington_assessments.csv")
            if len(df) > 100:
                print(f"  Verified: {len(df):,} rows")
                arlington_ok = True
                break
            else:
                os.remove(f"{DATA_RAW}/arlington_assessments.csv")
        except Exception:
            if os.path.exists(f"{DATA_RAW}/arlington_assessments.csv"):
                os.remove(f"{DATA_RAW}/arlington_assessments.csv")

if not arlington_ok:
    print("  Arlington direct download unavailable – will use FHFA + Zillow for local calibration")

print("\nDone.")
