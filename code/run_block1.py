"""
Block 1: Aggregate vs subgroup performance across datasets.

Outputs:
  results/block1_aggregate.csv
  results/block1_subgroup.csv
  results/block1_delong.csv

Procedure:
  - 5-fold stratified CV by (A_intersect, Y) to ensure balanced cells
  - Models: LogReg, RF, XGBoost*
  - Two configurations: SES-aware (with A_ses_hi) vs SES-unaware
  - Per-fold OOF predictions accumulated, metrics computed on full OOF probs
  - DeLong test for AUC differences across subgroups
  - Bootstrap 95% CI for FPR per subgroup
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from data_loaders import load_thcsmk, load_uci_por, get_feature_columns, subgroup_label
from metrics import (subgroup_metrics, delong_test_independent, bootstrap_ci,
                     fpr_at_threshold, safe_auc, ece)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42


def make_model(name: str):
    if name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
        ])
    elif name == "RF":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        )
    else:
        raise ValueError(name)


def get_oof_probs(df: pd.DataFrame, feature_cols: list, model_name: str,
                  stratify_col: str = "strata") -> np.ndarray:
    """Run 5-fold stratified CV and return out-of-fold predicted probabilities."""
    X = df[feature_cols].values
    y = df["Y"].values

    # Strata = A_intersect * 2 + Y (ensures both class and subgroup balance)
    strata = (df["A_intersect"].values * 2 + y).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(df), np.nan)
    for tr, te in skf.split(X, strata):
        clf = make_model(model_name)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def aggregate_metrics_row(name, ds_name, model, ses_config, y, probs):
    return {
        "dataset": ds_name, "model": model, "config": ses_config,
        "AUC": safe_auc(y, probs),
        "FPR@0.5": fpr_at_threshold(y, probs, 0.5),
        "ECE": ece(y, probs),
        "n": len(y),
        "pass_rate": float(y.mean())
    }


def main():
    rows_agg, rows_sub, rows_delong = [], [], []
    datasets = [(load_thcsmk, "THCSMK"), (load_uci_por, "UCI-Por")]
    try:
        from data_loaders import load_oulad
        _ = load_oulad()  # check
        datasets.append((load_oulad, "OULAD"))
    except (ImportError, FileNotFoundError):
        pass
    for ds_loader, ds_name in datasets:
        ds = ds_loader()
        df = ds["data"].copy()
        for ses_config in ["SES-aware", "SES-unaware"]:
            include_ses = (ses_config == "SES-aware")
            feature_cols = get_feature_columns(ds, include_ses=include_ses)
            for model_name in ["LR", "RF"]:
                probs = get_oof_probs(df, feature_cols, model_name)
                y = df["Y"].values
                A_int = df["A_intersect"].values

                # Aggregate
                rows_agg.append(aggregate_metrics_row(None, ds_name, model_name, ses_config, y, probs))

                # Subgroup
                sub_df = subgroup_metrics(y, probs, A_int)
                # FPR bootstrap CI for each subgroup
                for _, r in sub_df.iterrows():
                    sg = int(r["subgroup"])
                    if sg == -1:
                        sg_lab = "Aggregate"
                        mask = np.ones(len(y), dtype=bool)
                    else:
                        sg_lab = subgroup_label(sg)
                        mask = (A_int == sg)
                    fpr_ci = bootstrap_ci(
                        lambda yt, yp: fpr_at_threshold(yt, yp, 0.5),
                        y[mask], probs[mask], n_boot=2000, seed=SEED
                    )
                    rows_sub.append({
                        "dataset": ds_name, "model": model_name, "config": ses_config,
                        "subgroup_code": sg, "subgroup": sg_lab,
                        "n": int(r["n"]),
                        "AUC": float(r["AUC"]) if not np.isnan(r["AUC"]) else None,
                        "TPR": float(r["TPR"]) if not np.isnan(r["TPR"]) else None,
                        "FPR": float(r["FPR"]) if not np.isnan(r["FPR"]) else None,
                        "FPR_lo": fpr_ci["lo"], "FPR_hi": fpr_ci["hi"],
                        "ECE": float(r["ECE"]) if not np.isnan(r["ECE"]) else None,
                        "pass_rate": float(r["pass_rate"]),
                    })

                # DeLong between each pair of intersectional subgroups (independent)
                for a in range(4):
                    for b in range(a + 1, 4):
                        m_a = A_int == a
                        m_b = A_int == b
                        res = delong_test_independent(y[m_a], probs[m_a], y[m_b], probs[m_b])
                        rows_delong.append({
                            "dataset": ds_name, "model": model_name, "config": ses_config,
                            "group_a": subgroup_label(a), "group_b": subgroup_label(b),
                            "AUC_a": res["auc_a"], "AUC_b": res["auc_b"],
                            "n_a": int(m_a.sum()), "n_b": int(m_b.sum()),
                            "z_stat": res.get("z_stat"), "p_value": res.get("p_value"),
                        })

    pd.DataFrame(rows_agg).to_csv(RESULTS_DIR / "block1_aggregate.csv", index=False)
    pd.DataFrame(rows_sub).to_csv(RESULTS_DIR / "block1_subgroup.csv", index=False)
    pd.DataFrame(rows_delong).to_csv(RESULTS_DIR / "block1_delong.csv", index=False)

    # Print headline numbers
    print("\n" + "=" * 72)
    print("BLOCK 1 RESULTS — AGGREGATE")
    print("=" * 72)
    print(pd.DataFrame(rows_agg).round(3).to_string(index=False))

    print("\n" + "=" * 72)
    print("BLOCK 1 RESULTS — SUBGROUP (RF, SES-aware only)")
    print("=" * 72)
    sub_df = pd.DataFrame(rows_sub)
    show = sub_df[(sub_df["model"] == "RF") & (sub_df["config"] == "SES-aware")]
    print(show[["dataset", "subgroup", "n", "AUC", "TPR", "FPR", "FPR_lo", "FPR_hi", "ECE"]].round(3).to_string(index=False))

    print("\n" + "=" * 72)
    print("BLOCK 1 — Significant subgroup AUC differences (RF, SES-aware)")
    print("=" * 72)
    dl_df = pd.DataFrame(rows_delong)
    show2 = dl_df[(dl_df["model"] == "RF") & (dl_df["config"] == "SES-aware") & (dl_df["p_value"] < 0.05)]
    if len(show2) > 0:
        print(show2[["dataset", "group_a", "group_b", "AUC_a", "AUC_b", "p_value"]].round(3).to_string(index=False))
    else:
        print("  (none — but examine effect sizes; small n_a/n_b reduces power)")

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
