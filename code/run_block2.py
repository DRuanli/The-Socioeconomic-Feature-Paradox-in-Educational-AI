"""
Block 2: Causal decomposition of subgroup FPR disparity into
direct (CtfDE), indirect (CtfIE), and spurious (CtfSE) effects.

Procedure:
  - For each dataset and each pair (a0, a1) of intersectional subgroups,
    decompose the TV in FPR-conditional predictions.
  - 200 bootstrap replicates for 95% CI.
  - SES-aware Random Forest model (primary configuration).

Outputs:
  results/block2_decomposition.csv
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from data_loaders import load_thcsmk, load_uci_por, get_feature_columns, subgroup_label
from causal_estimator import bootstrap_decomposition

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=None,
        class_weight="balanced", random_state=SEED, n_jobs=1
    )


def make_rf_fast():
    """Faster RF for bootstrap iterations (n_estimators=100, depth limited)."""
    return RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=SEED, n_jobs=1
    )


def main(n_boot=200, n_mc=30):
    rows = []
    for ds_loader, ds_name in [(load_thcsmk, "THCSMK"), (load_uci_por, "UCI-Por")]:
        print(f"\n{'='*72}\nDECOMPOSITION: {ds_name}\n{'='*72}", flush=True)
        ds = ds_loader()
        df = ds["data"].copy()
        feature_cols = get_feature_columns(ds, include_ses=True)
        M_cols = ds["feature_groups"]["M1"] + ds["feature_groups"]["M2"]
        W_cols = ds["feature_groups"]["W"]
        print(f"  Features: {feature_cols}", flush=True)
        print(f"  M_cols  : {M_cols}", flush=True)
        print(f"  W_cols  : {W_cols}", flush=True)

        # Reference: Fem×LowSES (a=2). Comparison: all other subgroups.
        # We also compute Male×HighSES (a=1) vs Fem×LowSES (a=2) as headline.
        a0 = 2  # Fem×LowSES — historically disadvantaged baseline
        for a1 in [0, 1, 3]:
            print(f"\n  Decomposing {subgroup_label(a1)} (a1={a1}) vs {subgroup_label(a0)} (a0={a0})...", flush=True)
            import time as _t
            _t0 = _t.time()
            res = bootstrap_decomposition(
                clf_factory=make_rf_fast,
                df=df,
                feature_cols=feature_cols,
                A_col="A_intersect",
                A_in_features="A_ses_hi",
                M_cols=M_cols, W_cols=W_cols,
                a0=a0, a1=a1,
                condition_Y0=True,
                n_mc=n_mc, n_boot=n_boot, seed=SEED,
                refit_clf=True
            )
            print(f"    Time: {(_t.time()-_t0)/60:.1f} min", flush=True)
            print(f"    TV    = {res['TV']:+.4f}  [{res['TV_ci'][0]:+.4f}, {res['TV_ci'][1]:+.4f}]", flush=True)
            print(f"    CtfDE = {res['CtfDE']:+.4f}  [{res['CtfDE_ci'][0]:+.4f}, {res['CtfDE_ci'][1]:+.4f}]", flush=True)
            print(f"    CtfIE = {res['CtfIE']:+.4f}  [{res['CtfIE_ci'][0]:+.4f}, {res['CtfIE_ci'][1]:+.4f}]", flush=True)
            print(f"    CtfSE = {res['CtfSE']:+.4f}  [{res['CtfSE_ci'][0]:+.4f}, {res['CtfSE_ci'][1]:+.4f}]", flush=True)
            if abs(res["TV"]) > 1e-6:
                de_pct = 100 * res["CtfDE"] / res["TV"]
                ie_pct = 100 * res["CtfIE"] / res["TV"]
                se_pct = 100 * res["CtfSE"] / res["TV"]
                print(f"    %TV  : DE={de_pct:+.1f}%  IE={ie_pct:+.1f}%  SE={se_pct:+.1f}%", flush=True)
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
            # Save progress after each pair in case of crash
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "block2_decomposition.csv", index=False)

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "block2_decomposition.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n\nSaved to {out_path}")
    print("\n" + "=" * 72)
    print("BLOCK 2 SUMMARY")
    print("=" * 72)
    show = out_df[["dataset", "a0", "a1", "n_a0", "n_a1",
                   "TV", "CtfDE", "CtfIE", "CtfSE"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    import time
    t0 = time.time()
    main(n_boot=200, n_mc=30)
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")
