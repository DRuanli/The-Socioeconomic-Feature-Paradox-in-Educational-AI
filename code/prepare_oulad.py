"""
OULAD data preparation script.

Downloads the OULAD dataset from the official Open University source,
harmonises it to the common schema used by THCSMK and UCI-Por, restricting
the prediction window to mid-semester features only (clicks aggregated to
day 100, first TMA assessment).

USAGE
-----
    python prepare_oulad.py

This will:
  1. Download anonymisedData.zip from analyse.kmi.open.ac.uk
  2. Unzip studentInfo.csv, studentVle.csv, studentAssessment.csv,
     assessments.csv into data/oulad_raw/
  3. Build a harmonised CSV at data/oulad_harmonised.csv
"""
import os, sys, zipfile, urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "oulad_raw"
URL = "https://analyse.kmi.open.ac.uk/resources/documents/anonymisedData.zip"
MIDPOINT_DAY = 100  # mid-semester cutoff


def download_oulad():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    RAW_DIR.mkdir(exist_ok=True, parents=True)
    zip_path = DATA_DIR / "anonymisedData.zip"
    if zip_path.exists():
        print(f"  ZIP already at {zip_path}, skipping download")
    else:
        print(f"  Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"  Extracting to {RAW_DIR} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DIR)
    print("  Done")


def harmonise():
    """Build the harmonised dataframe matching THCSMK/UCI schema."""
    print("  Loading OULAD tables ...")
    si = pd.read_csv(RAW_DIR / "studentInfo.csv")
    sv = pd.read_csv(RAW_DIR / "studentVle.csv")
    sa = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    asm = pd.read_csv(RAW_DIR / "assessments.csv")

    print(f"  studentInfo  : {len(si):,} rows")
    print(f"  studentVle   : {len(sv):,} rows")
    print(f"  studentAssm  : {len(sa):,} rows")

    # Filter completed registrations only (drop withdrawals as in original paper)
    # final_result: Pass, Fail, Distinction, Withdrawn
    # Conservative: Pass + Distinction -> 1, Fail -> 0, drop Withdrawn
    si = si[si.final_result.isin(["Pass", "Fail", "Distinction"])].copy()
    si["Y"] = (si.final_result.isin(["Pass", "Distinction"])).astype(int)

    # Protected attributes
    si["A_gender"] = (si.gender == "F").astype(int)  # 1 = female
    # IMD band: more deprived = lower decile. We code HighSES = imd >= 50%
    imd_to_pct = {
        "0-10%": 5, "10-20": 15, "20-30%": 25, "30-40%": 35, "40-50%": 45,
        "50-60%": 55, "60-70%": 65, "70-80%": 75, "80-90%": 85, "90-100%": 95
    }
    si["imd_pct"] = si.imd_band.map(imd_to_pct)
    si = si.dropna(subset=["imd_pct"])
    si["A_ses_hi"] = (si.imd_pct >= 50).astype(int)
    si["A_intersect"] = si.A_gender * 2 + si.A_ses_hi

    # Confounders (pre-enrollment): num_of_prev_attempts, studied_credits, age_band, region indicator
    si["W_prev_attempts"] = si.num_of_prev_attempts
    si["W_studied_credits"] = si.studied_credits
    si["W_age_band_old"] = (si.age_band != "0-35").astype(int)
    si["W_region_scotland"] = (si.region.fillna("").str.contains("Scotland")).astype(int)

    # MID-SEMESTER MEDIATORS
    # M1: academic = mean score on first TMA (TMA01) before day MIDPOINT
    # Join assessments with student assessment scores; filter TMA, date <= MIDPOINT_DAY
    asm_early = asm[(asm.assessment_type == "TMA") & (asm.date.fillna(9999) <= MIDPOINT_DAY)]
    early_ids = asm_early.id_assessment.unique()
    sa_early = sa[sa.id_assessment.isin(early_ids)]
    tma1 = sa_early.groupby("id_student").agg(
        M1_tma_mean=("score", "mean"),
        M1_tma_count=("score", "count")
    ).reset_index()

    # M2: behavioral = total VLE clicks before day MIDPOINT
    sv_early = sv[sv.date <= MIDPOINT_DAY]
    vle = sv_early.groupby("id_student").agg(
        M2_total_clicks=("sum_click", "sum"),
        M2_active_days=("date", "nunique")
    ).reset_index()

    # Merge
    df = si.merge(tma1, on="id_student", how="left").merge(vle, on="id_student", how="left")
    # Fill missing mediators with 0 (no assessment / no VLE engagement)
    for c in ["M1_tma_mean", "M1_tma_count", "M2_total_clicks", "M2_active_days"]:
        df[c] = df[c].fillna(0)

    # Drop rows with critical missingness
    df = df.dropna(subset=["A_gender", "A_ses_hi", "Y"])

    # Keep only one record per (id_student, code_module, code_presentation) — first
    df = df.drop_duplicates(subset=["id_student", "code_module", "code_presentation"])

    print(f"  After harmonisation: n = {len(df):,}")
    print(f"  Pass rate: {df.Y.mean():.3f}")
    print(f"  Cell sizes:")
    print(df.groupby("A_intersect").size().rename("n"))

    # Standard schema columns
    out = pd.DataFrame({
        "A_gender": df.A_gender,
        "A_ses_hi": df.A_ses_hi,
        "A_intersect": df.A_intersect,
        "W_prev_attempts": df.W_prev_attempts,
        "W_studied_credits": df.W_studied_credits,
        "W_age_band_old": df.W_age_band_old,
        "W_region_scotland": df.W_region_scotland,
        "M1_tma_mean": df.M1_tma_mean,
        "M1_tma_count": df.M1_tma_count,
        "M2_total_clicks": df.M2_total_clicks,
        "M2_active_days": df.M2_active_days,
        "Y": df.Y,
    })

    out_path = DATA_DIR / "oulad_harmonised.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return out


if __name__ == "__main__":
    print("STEP 1: Download OULAD")
    download_oulad()
    print("\nSTEP 2: Harmonise OULAD to common schema")
    harmonise()
    print("\nDone. Add OULAD to data_loaders.py via load_oulad() to integrate.")
