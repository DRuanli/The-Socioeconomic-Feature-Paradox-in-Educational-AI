"""
Data loaders for the causal fairness audit paper.

Harmonizes THCSMK (Vietnam) and UCI-Por (Portugal) datasets to a common
schema with the following columns:

    A_gender    binary: 1 = female, 0 = male
    A_ses_hi    binary: 1 = high SES, 0 = low SES (median split within dataset)
    A_intersect categorical 0..3: 0=Male×Low, 1=Male×High, 2=Fem×Low, 3=Fem×High
    W_*         observed confounders (pre-enrollment, not affected by A)
    M1_*        academic mediators (mid-semester only)
    M2_*        behavioral mediators (mid-semester only)
    Y           binary outcome: 1 = pass, 0 = fail

Strict rule for mid-semester window:
    - THCSMK: include only HK1 features (avg_kttx_h1, diem_giua_ky, absences h1)
              EXCLUDE: kttx3, kttx4, diem_cuoi_ky, tb_hoc_ky, all h2 deltas
    - UCI-Por: include only G1 and pre-enrollment vars
              EXCLUDE: G2 (period 2, post-midpoint), G3 (target)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _build_intersect(female: pd.Series, ses_hi: pd.Series) -> pd.Series:
    """0=Male×Low, 1=Male×High, 2=Fem×Low, 3=Fem×High."""
    return (female.astype(int) * 2 + ses_hi.astype(int)).astype(int)


def load_thcsmk() -> dict:
    """Load and harmonize THCSMK (Vietnamese THCS, n=675)."""
    df = pd.read_csv(DATA_DIR / "THCSMK.csv")

    # SES: sum of father + mother SES (0..8); HighSES = sum >= 6 per proposal
    df["ses_combined"] = df["ses_father"] + df["ses_mother"]
    df["A_ses_hi"] = (df["ses_combined"] >= 6).astype(int)
    df["A_gender"] = df["female"].astype(int)
    df["A_intersect"] = _build_intersect(df["A_gender"], df["A_ses_hi"])

    # Handle the single missing kttx4 — irrelevant (not used in mid-semester window)
    # Impute missing in mid-semester features (none expected)
    mid_features_academic = ["avg_kttx_h1", "diem_giua_ky"]
    mid_features_behavioral = ["abs_phep_h1", "abs_nophep_h1"]
    confounders = ["distance_band", "immigrant"]

    # Verify no missingness in mid-semester window
    miss = df[mid_features_academic + mid_features_behavioral + confounders].isna().sum().sum()
    if miss > 0:
        raise ValueError(f"THCSMK: missing values in mid-semester features: {miss}")

    Y = df["y"].astype(int)

    out = pd.DataFrame({
        "A_gender": df["A_gender"],
        "A_ses_hi": df["A_ses_hi"],
        "A_intersect": df["A_intersect"],
        # Confounders (pre-enrollment, not caused by gender/SES in our DAG)
        "W_distance_band": df["distance_band"],
        "W_immigrant": df["immigrant"],
        # Academic mediator (mid-semester)
        "M1_avg_kttx_h1": df["avg_kttx_h1"],
        "M1_diem_giua_ky": df["diem_giua_ky"],
        # Behavioral mediator (mid-semester)
        "M2_abs_phep_h1": df["abs_phep_h1"],
        "M2_abs_nophep_h1": df["abs_nophep_h1"],
        "Y": Y,
    })

    return {
        "name": "THCSMK",
        "data": out,
        "n": len(out),
        "feature_groups": {
            "W": ["W_distance_band", "W_immigrant"],
            "M1": ["M1_avg_kttx_h1", "M1_diem_giua_ky"],
            "M2": ["M2_abs_phep_h1", "M2_abs_nophep_h1"],
        },
        "context": "Vietnamese lower secondary school, one academic year"
    }


def load_uci_por() -> dict:
    """Load and harmonize UCI Portuguese student dataset (n=649)."""
    df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")

    # SES proxy: Medu + Fedu (0..8). Use median split for consistency
    df["ses_combined"] = df["Medu"] + df["Fedu"]
    median_ses = df["ses_combined"].median()  # 5.0
    df["A_ses_hi"] = (df["ses_combined"] >= median_ses).astype(int)
    df["A_gender"] = (df["sex"] == "F").astype(int)
    df["A_intersect"] = _build_intersect(df["A_gender"], df["A_ses_hi"])

    # Outcome: pass G3 >= 10
    Y = (df["G3"] >= 10).astype(int)

    # Confounders (pre-enrollment): school, address, family size, parent status
    # Encode categoricals to numeric
    df["W_school"] = (df["school"] == "GP").astype(int)
    df["W_urban"] = (df["address"] == "U").astype(int)
    df["W_famsize_GT3"] = (df["famsize"] == "GT3").astype(int)
    df["W_Pstatus_together"] = (df["Pstatus"] == "T").astype(int)

    # Mid-semester academic: G1 (period 1 grade), studytime, failures (past)
    # NOTE: G2 is EXCLUDED — it is period 2, post-midpoint
    # M1 = academic; M2 = behavioral
    out = pd.DataFrame({
        "A_gender": df["A_gender"],
        "A_ses_hi": df["A_ses_hi"],
        "A_intersect": df["A_intersect"],
        # Confounders
        "W_school": df["W_school"],
        "W_urban": df["W_urban"],
        "W_famsize_GT3": df["W_famsize_GT3"],
        "W_Pstatus_together": df["W_Pstatus_together"],
        # Academic mediator (mid-semester proxy)
        "M1_G1": df["G1"],
        "M1_studytime": df["studytime"],
        "M1_failures": df["failures"],
        # Behavioral mediator
        "M2_absences": df["absences"],
        "M2_goout": df["goout"],
        "Y": Y,
    })

    return {
        "name": "UCI-Por",
        "data": out,
        "n": len(out),
        "feature_groups": {
            "W": ["W_school", "W_urban", "W_famsize_GT3", "W_Pstatus_together"],
            "M1": ["M1_G1", "M1_studytime", "M1_failures"],
            "M2": ["M2_absences", "M2_goout"],
        },
        "context": "Portuguese secondary school, two schools (Cortez & Silva 2008)"
    }


def load_oulad() -> dict:
    """Load OULAD harmonised data (built by prepare_oulad.py).

    Requires user to first run `python code/prepare_oulad.py` to download
    the OULAD raw files from the Open University and harmonise them.
    Raises FileNotFoundError with instructions otherwise.
    """
    p = DATA_DIR / "oulad_harmonised.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"OULAD harmonised file not found at {p}. Run:\n"
            f"    python code/prepare_oulad.py\n"
            f"This downloads ~85MB from analyse.kmi.open.ac.uk and produces\n"
            f"the harmonised CSV with mid-semester features (clicks/TMA by day 100)."
        )
    out = pd.read_csv(p)
    return {
        "name": "OULAD",
        "data": out,
        "n": len(out),
        "feature_groups": {
            "W": ["W_prev_attempts", "W_studied_credits",
                  "W_age_band_old", "W_region_scotland"],
            "M1": ["M1_tma_mean", "M1_tma_count"],
            "M2": ["M2_total_clicks", "M2_active_days"],
        },
        "context": "UK Open University, 22 module-presentations, 2013-2014"
    }


def get_feature_columns(dataset: dict, include_ses: bool = True) -> list:
    """Return the column list for model training.

    include_ses controls the SES Inclusion Paradox experiment:
      True  -> include A_ses_hi as a model feature (SES-aware)
      False -> exclude A_ses_hi (SES-unaware)

    A_gender is always excluded from model features (it is the protected attribute
    we want to audit, not predict from). W, M1, M2 are always included.
    """
    fg = dataset["feature_groups"]
    cols = fg["W"] + fg["M1"] + fg["M2"]
    if include_ses:
        cols = cols + ["A_ses_hi"]
    return cols


def subgroup_label(intersect_code: int) -> str:
    return {0: "Male×LowSES", 1: "Male×HighSES",
            2: "Fem×LowSES",  3: "Fem×HighSES"}[intersect_code]


if __name__ == "__main__":
    print("=" * 70)
    print("Loading THCSMK ...")
    thcsmk = load_thcsmk()
    print(f"  n = {thcsmk['n']}")
    print(f"  Pass rate = {thcsmk['data']['Y'].mean():.3f}")
    print("  Intersectional cell sizes:")
    print(thcsmk['data'].groupby('A_intersect').agg(
        n=('Y','size'), pass_rate=('Y','mean')).round(3))

    print("\n" + "=" * 70)
    print("Loading UCI-Por ...")
    uci = load_uci_por()
    print(f"  n = {uci['n']}")
    print(f"  Pass rate = {uci['data']['Y'].mean():.3f}")
    print("  Intersectional cell sizes:")
    print(uci['data'].groupby('A_intersect').agg(
        n=('Y','size'), pass_rate=('Y','mean')).round(3))

    print("\n" + "=" * 70)
    print("Sample features (THCSMK):", get_feature_columns(thcsmk))
    print("Sample features (UCI-Por):", get_feature_columns(uci))
