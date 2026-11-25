import pandas as pd
import numpy as np
import os
import re

# === STEP 1: Load the three source extracts ===

copa_df = pd.read_csv("data/raw/copa_actuals_2018.csv", encoding="latin1", sep=None, engine="python")
coois_df = pd.read_csv("data/raw/production_goods_movements.csv", encoding="latin1", sep=None, engine="python")
mbew_df = pd.read_csv("data/raw/material_standard_costs.csv", encoding="latin1", sep=None, engine="python")

# === STEP 2: The European Number Cleanser ===

def clean_sap_number(value):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s == "":
        return None
    is_negative = s.endswith("-")
    if is_negative:
        s = s[:-1].strip()
    # Drop thousands separators, swap decimal comma for period
    s = s.replace(".", "").replace(",", ".")
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if is_negative else num

def find_dirty_numeric_columns(df, sample_size=200):
    pattern = re.compile(r"^-?[\d.]*\d,\d{2}-?$")
    dirty_cols = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(sample_size)
        if len(sample) == 0:
            continue
        hits = sample.str.match(pattern).sum()
        if hits / len(sample) > 0.5:
            dirty_cols.append(col)
    return dirty_cols

def cleanse_dataframe(df, label=""):
    dirty_cols = find_dirty_numeric_columns(df)
    for col in dirty_cols:
        df[col] = df[col].apply(clean_sap_number)
    return df

copa_df = cleanse_dataframe(copa_df, label="CO-PA")
coois_df = cleanse_dataframe(coois_df, label="COOIS")
mbew_df = cleanse_dataframe(mbew_df, label="MBEW")

# === STEP 3: Header Sanitization (Stripping SAP Artifacts) ===

dataframes = [copa_df, coois_df, mbew_df]
labels = ["CO-PA", "COOIS", "MBEW"]
for df in dataframes:
    df.columns = df.columns.str.replace('ï»¿', '')
    df.columns = df.columns.str.replace('"', '')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# === STEP 4: Standard Cost Per Unit (MBEW Baseline) ===

mbew_df["standard_cost_per_unit"] = (
    mbew_df["standard_price"] / mbew_df["price_unit"].replace(0, np.nan)
)

# === STEP 5: Key Alignment & Mismatch Fixes ===

coois_df["material"] = coois_df["material"].astype(str).str.strip().str.upper()
mbew_df["material"] = mbew_df["material"].astype(str).str.strip().str.upper()

# Filter MBEW for Plant 1010 and drop Split-Valuation duplicates
mbew_1010 = mbew_df[mbew_df['valuation_area'] == 1010].copy()
mbew_1010 = mbew_1010.drop_duplicates(subset=["material"], keep="first")

# === STEP 6: Merge (The OT-to-FI Bridge) ===

merged_df = coois_df.merge(
    mbew_1010[["material", "standard_cost_per_unit"]],
    on="material",
    how="left",
    validate="many_to_one",
)

# === STEP 7: Variance Math (Using Signed Quantities) ===

merged_df["standard_expected_cost"] = (
    merged_df["qty_in_uoe_+_sign"] * merged_df["standard_cost_per_unit"]
)
merged_df["total_variance"] = (
    merged_df["amt.in_loc.cur."] - merged_df["standard_expected_cost"]
)

# === STEP 8: Export ===

os.makedirs("data/processed", exist_ok=True)
merged_df.to_csv("data/processed/plantiq_variance_output.csv", index=False)
