"""Block 3: SES Inclusion Paradox via causal decomposition.

For each dataset, fit two models (SES-aware vs SES-unaware) and compute
Delta(CtfDE), Delta(CtfIE), Delta(CtfSE). The hypothesis: adding SES
features increases CtfDE specifically (direct shortcut), not CtfIE
(historical mediation).
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from data_loaders import load_thcsmk, load_uci_por, get_feature_columns, subgroup_label
from causal_estimator import bootstrap_decomposition

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42

def make_rf():
    return RandomForestClassifier(
        n_estimators=80, max_depth=8,
        class_weight='balanced', random_state=SEED, n_jobs=1
    )

def run(ds_loader, ds_name, n_boot=80, n_mc=15):
    rows = []
    ds = ds_loader()
    df = ds["data"].copy()
    M_cols = ds["feature_groups"]["M1"] + ds["feature_groups"]["M2"]
    W_cols = ds["feature_groups"]["W"]
    a0 = 2
    # Headline comparison only: Male×HighSES vs Fem×LowSES
    a1 = 1
    for config_name, include_ses in [("SES-aware", True), ("SES-unaware", False)]:
        feature_cols = get_feature_columns(ds, include_ses=include_ses)
        A_in_features = "A_ses_hi" if include_ses else None
        t0 = time.time()
        print(f"  {ds_name} [{config_name}]: {subgroup_label(a1)} vs {subgroup_label(a0)}...", flush=True)
        res = bootstrap_decomposition(
            clf_factory=make_rf, df=df, feature_cols=feature_cols,
            A_col="A_intersect", A_in_features=A_in_features,
            M_cols=M_cols, W_cols=W_cols,
            a0=a0, a1=a1, condition_Y0=True,
            n_mc=n_mc, n_boot=n_boot, seed=SEED, refit_clf=True
        )
        print(f"    {(time.time()-t0)/60:.1f}min  TV={res['TV']:+.4f} DE={res['CtfDE']:+.4f} IE={res['CtfIE']:+.4f}", flush=True)
        rows.append({
            "dataset": ds_name, "config": config_name,
            "a0": subgroup_label(a0), "a1": subgroup_label(a1),
            "n_a0": res["n_a0"], "n_a1": res["n_a1"],
            "TV": res["TV"], "TV_lo": res["TV_ci"][0], "TV_hi": res["TV_ci"][1],
            "CtfDE": res["CtfDE"], "CtfDE_lo": res["CtfDE_ci"][0], "CtfDE_hi": res["CtfDE_ci"][1],
            "CtfIE": res["CtfIE"], "CtfIE_lo": res["CtfIE_ci"][0], "CtfIE_hi": res["CtfIE_ci"][1],
            "CtfSE": res["CtfSE"], "CtfSE_lo": res["CtfSE_ci"][0], "CtfSE_hi": res["CtfSE_ci"][1],
        })
    return rows

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "thcsmk"
    if which == "thcsmk":
        rows = run(load_thcsmk, "THCSMK")
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "block3_thcsmk.csv", index=False)
    elif which == "uci":
        rows = run(load_uci_por, "UCI-Por")
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "block3_uci.csv", index=False)
    df = pd.DataFrame(rows)
    print(df[['dataset','config','TV','CtfDE','CtfIE','CtfSE']].round(4).to_string(index=False))
    print(f"\nDelta (SES-aware - SES-unaware):")
    if len(df) == 2:
        d = df.iloc[0] - df.iloc[1]
        print(f"  dTV={df.iloc[0].TV - df.iloc[1].TV:+.4f}")
        print(f"  dDE={df.iloc[0].CtfDE - df.iloc[1].CtfDE:+.4f}")
        print(f"  dIE={df.iloc[0].CtfIE - df.iloc[1].CtfIE:+.4f}")
