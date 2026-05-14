"""Block 5: E-value sensitivity analysis (VanderWeele & Ding 2017).

For each CtfDE/CtfIE estimate, compute the minimum strength of unmeasured
confounding (on the risk-ratio scale) that would be required to fully
explain away the estimated effect.

Conversion: additive effect -> approximate risk ratio using baseline rate p0.
    p1 = p0 + effect
    RR = p1 / p0
E-value(RR) = RR + sqrt(RR * (RR - 1))   for RR >= 1
            = (1/RR) + sqrt((1/RR) * ((1/RR) - 1))   for RR < 1
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evalue(rr: float) -> float:
    if np.isnan(rr) or rr <= 0:
        return np.nan
    if rr >= 1:
        return rr + np.sqrt(rr * (rr - 1))
    inv = 1.0 / rr
    return inv + np.sqrt(inv * (inv - 1))


def effect_to_evalue(effect: float, baseline_p: float) -> float:
    """Convert additive effect on probability scale -> E-value on RR scale."""
    if np.isnan(effect) or baseline_p <= 0:
        return np.nan
    p_new = baseline_p + effect
    p_new = np.clip(p_new, 0.001, 0.999)
    rr = p_new / baseline_p
    return evalue(rr)


def main():
    # Load decomposition with baseline FPR for each subgroup
    decomp = pd.read_csv(RESULTS_DIR / "block2_decomposition.csv")
    sub = pd.read_csv(RESULTS_DIR / "block1_subgroup.csv")

    # For each row, get baseline FPR of reference subgroup (a0)
    rows = []
    sub_rf_aware = sub[(sub.model == "RF") & (sub.config == "SES-aware")]
    for _, r in decomp.iterrows():
        ds = r["dataset"]
        a0 = r["a0"]
        # baseline FPR for a0 in that dataset
        b = sub_rf_aware[(sub_rf_aware.dataset == ds) & (sub_rf_aware.subgroup == a0)]
        if len(b) == 0:
            base_p = 0.5
        else:
            base_p = float(b["FPR"].iloc[0])

        rows.append({
            "dataset": ds,
            "a0": a0, "a1": r["a1"],
            "baseline_FPR_a0": round(base_p, 4),
            "CtfDE": round(r["CtfDE"], 4),
            "E-value_CtfDE": round(effect_to_evalue(r["CtfDE"], base_p), 3),
            "CtfDE_lo": round(r["CtfDE_lo"], 4),
            "E-value_CtfDE_lo": round(effect_to_evalue(r["CtfDE_lo"], base_p), 3),
            "CtfIE": round(r["CtfIE"], 4),
            "E-value_CtfIE": round(effect_to_evalue(r["CtfIE"], base_p), 3),
            "CtfIE_lo": round(r["CtfIE_lo"], 4),
            "E-value_CtfIE_lo": round(effect_to_evalue(r["CtfIE_lo"], base_p), 3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "block5_evalues.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
