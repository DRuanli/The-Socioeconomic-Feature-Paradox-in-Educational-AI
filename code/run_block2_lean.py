"""Lean Block 2 — run one dataset at a time."""
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

def run_dataset(ds_loader, ds_name, n_boot=80, n_mc=15):
    rows = []
    ds = ds_loader()
    df = ds["data"].copy()
    feature_cols = get_feature_columns(ds, include_ses=True)
    M_cols = ds["feature_groups"]["M1"] + ds["feature_groups"]["M2"]
    W_cols = ds["feature_groups"]["W"]
    a0 = 2
    for a1 in [0, 1, 3]:
        t0 = time.time()
        print(f"  {ds_name}: {subgroup_label(a1)} vs {subgroup_label(a0)} ...", flush=True)
        res = bootstrap_decomposition(
            clf_factory=make_rf, df=df, feature_cols=feature_cols,
            A_col="A_intersect", A_in_features="A_ses_hi",
            M_cols=M_cols, W_cols=W_cols,
            a0=a0, a1=a1, condition_Y0=True,
            n_mc=n_mc, n_boot=n_boot, seed=SEED, refit_clf=True
        )
        print(f"    Done in {(time.time()-t0)/60:.1f} min. "
              f"TV={res['TV']:+.4f} DE={res['CtfDE']:+.4f} IE={res['CtfIE']:+.4f} SE={res['CtfSE']:+.4f}", flush=True)
        rows.append({
            "dataset": ds_name,
            "a0_code": a0, "a0": subgroup_label(a0),
            "a1_code": a1, "a1": subgroup_label(a1),
            "n_a0": res["n_a0"], "n_a1": res["n_a1"],
            "TV": res["TV"], "TV_lo": res["TV_ci"][0], "TV_hi": res["TV_ci"][1],
            "CtfDE": res["CtfDE"], "CtfDE_lo": res["CtfDE_ci"][0], "CtfDE_hi": res["CtfDE_ci"][1],
            "CtfIE": res["CtfIE"], "CtfIE_lo": res["CtfIE_ci"][0], "CtfIE_hi": res["CtfIE_ci"][1],
            "CtfSE": res["CtfSE"], "CtfSE_lo": res["CtfSE_ci"][0], "CtfSE_hi": res["CtfSE_ci"][1],
            "n_boot_valid": res["n_boot_valid"],
            "condition_Y0": True,
        })
    return rows

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out_path = RESULTS_DIR / f"block2_{which}.csv"
    all_rows = []
    if which in ("thcsmk", "all"):
        all_rows.extend(run_dataset(load_thcsmk, "THCSMK"))
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    if which in ("uci", "all"):
        all_rows.extend(run_dataset(load_uci_por, "UCI-Por"))
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(pd.DataFrame(all_rows).round(4).to_string(index=False))
