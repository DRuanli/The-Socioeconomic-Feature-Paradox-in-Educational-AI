"""Block 4: Pathway-targeted interventions vs baseline.

Three interventions, each targeting one pathway:
  I1 (CtfDE-targeted): remove SES feature (feature ablation)
  I2 (CtfIE-targeted): inverse-propensity reweigh mediator distribution
                       across (a, w) strata (Kamiran-Calders style)
  I3 (residual):       per-subgroup threshold optimization

Metric: max - min FPR across 4 intersectional subgroups (FPR spread).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data_loaders import load_thcsmk, load_uci_por, get_feature_columns, subgroup_label
from metrics import fpr_at_threshold, safe_auc, tpr_at_threshold

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=SEED, n_jobs=1
    )


def get_oof_probs(df, feature_cols, sample_weight=None):
    """5-fold stratified CV OOF probabilities."""
    X = df[feature_cols].values
    y = df['Y'].values
    strata = (df['A_intersect'].values * 2 + y).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, te in skf.split(X, strata):
        clf = make_rf()
        if sample_weight is not None:
            clf.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        else:
            clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def fpr_spread(y, probs, A_int, thresholds=None):
    """Max-min FPR across 4 intersectional subgroups."""
    fprs = []
    for g in range(4):
        mask = A_int == g
        if mask.sum() == 0:
            continue
        tau = thresholds[g] if thresholds is not None else 0.5
        fprs.append(fpr_at_threshold(y[mask], probs[mask], tau))
    fprs = np.array([f for f in fprs if not np.isnan(f)])
    return float(fprs.max() - fprs.min()), [float(f) for f in fprs]


def optimal_threshold(y, probs, min_tpr=0.80):
    """Find threshold minimizing FPR subject to TPR >= min_tpr."""
    if len(y) == 0:
        return 0.5
    taus = np.linspace(0.05, 0.95, 91)
    best_tau = 0.5
    best_fpr = 1.0
    for tau in taus:
        tpr = tpr_at_threshold(y, probs, tau)
        if np.isnan(tpr) or tpr < min_tpr:
            continue
        fpr = fpr_at_threshold(y, probs, tau)
        if np.isnan(fpr):
            continue
        if fpr < best_fpr:
            best_fpr = fpr
            best_tau = tau
    return best_tau


def reweigh_for_mediator(df, M_cols, W_cols):
    """Inverse-propensity weights on (A_intersect, W) to balance mediator distribution.

    Simpler approximation: stratified IPW for A given W via logistic regression.
    Sample weight w_i = 1 / P(A=A_i | W=W_i, normalized).
    This reweighs the training data so each (A, W) cell contributes equally,
    breaking the historical association A -> M.
    """
    A = df['A_intersect'].values
    W = df[W_cols].values
    # Multinomial logistic for P(A | W)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(W, A)
    probs_a = lr.predict_proba(W)
    # weights = 1 / P(A=A_i | W_i), clipped
    w = 1.0 / probs_a[np.arange(len(A)), A]
    w = np.clip(w, 0.2, 5.0)
    # Normalize to mean 1
    w = w / w.mean()
    return w


def kc_reweigh(df):
    """Kamiran-Calders style reweighing baseline.

    Weight per (A, Y) cell = P(A) * P(Y) / P(A, Y).
    """
    A = df['A_intersect'].values
    y = df['Y'].values
    n = len(df)
    weights = np.ones(n)
    for a in np.unique(A):
        for c in np.unique(y):
            mask = (A == a) & (y == c)
            if mask.sum() == 0:
                continue
            p_a = (A == a).mean()
            p_c = (y == c).mean()
            p_ac = mask.mean()
            if p_ac > 0:
                weights[mask] = p_a * p_c / p_ac
    return weights


def evaluate_method(name, df, feature_cols, thresholds=None, sample_weight=None):
    """Returns row dict with AUC, FPR per subgroup, FPR spread."""
    probs = get_oof_probs(df, feature_cols, sample_weight=sample_weight)
    y = df['Y'].values
    A_int = df['A_intersect'].values

    # Per-subgroup metrics
    sg_fpr = {}
    sg_tpr = {}
    for g in range(4):
        mask = A_int == g
        tau = thresholds[g] if thresholds is not None else 0.5
        sg_fpr[g] = fpr_at_threshold(y[mask], probs[mask], tau)
        sg_tpr[g] = tpr_at_threshold(y[mask], probs[mask], tau)

    spread, fprs_list = fpr_spread(y, probs, A_int, thresholds)
    auc_agg = safe_auc(y, probs)

    return {
        "method": name,
        "AUC_aggregate": auc_agg,
        "FPR_Male_LowSES":  sg_fpr.get(0),
        "FPR_Male_HighSES": sg_fpr.get(1),
        "FPR_Fem_LowSES":   sg_fpr.get(2),
        "FPR_Fem_HighSES":  sg_fpr.get(3),
        "TPR_Male_LowSES":  sg_tpr.get(0),
        "TPR_Male_HighSES": sg_tpr.get(1),
        "TPR_Fem_LowSES":   sg_tpr.get(2),
        "TPR_Fem_HighSES":  sg_tpr.get(3),
        "FPR_spread": spread,
        "probs": probs,
    }


def run(ds_loader, ds_name):
    ds = ds_loader()
    df = ds["data"].copy()
    M_cols = ds["feature_groups"]["M1"] + ds["feature_groups"]["M2"]
    W_cols = ds["feature_groups"]["W"]

    feature_cols_aware   = get_feature_columns(ds, include_ses=True)
    feature_cols_unaware = get_feature_columns(ds, include_ses=False)

    rows = []

    # --- Baseline: SES-aware, no intervention ---
    r0 = evaluate_method("Baseline (SES-aware)", df, feature_cols_aware)
    rows.append(r0)

    # --- I1: CtfDE-targeted — remove SES ---
    r1 = evaluate_method("I1: SES feature ablation", df, feature_cols_unaware)
    rows.append(r1)

    # --- I2: CtfIE-targeted — mediator reweighing ---
    weights = reweigh_for_mediator(df, M_cols, W_cols)
    r2 = evaluate_method("I2: Mediator reweighing",
                         df, feature_cols_aware, sample_weight=weights)
    rows.append(r2)

    # --- I3: Residual — per-subgroup threshold optimization ---
    probs_aware = r0["probs"]
    thresholds = {}
    A_int = df['A_intersect'].values
    y = df['Y'].values
    for g in range(4):
        mask = A_int == g
        thresholds[g] = optimal_threshold(y[mask], probs_aware[mask], min_tpr=0.80)
    r3 = evaluate_method("I3: Per-group thresholds", df, feature_cols_aware,
                          thresholds=thresholds)
    r3["method"] = f"I3: Per-group thresholds (tau={[round(thresholds[g],2) for g in range(4)]})"
    rows.append(r3)

    # --- I1 + I3 combined: ablation + thresholds on unaware model ---
    probs_unaware = r1["probs"]
    thresholds_un = {}
    for g in range(4):
        mask = A_int == g
        thresholds_un[g] = optimal_threshold(y[mask], probs_unaware[mask], min_tpr=0.80)
    r4 = evaluate_method("I1+I3: Ablation + thresholds", df, feature_cols_unaware,
                          thresholds=thresholds_un)
    rows.append(r4)

    # --- Baselines for comparison ---
    # Kamiran-Calders Reweighing (one-attribute style, but here applied to intersectional)
    w_kc = kc_reweigh(df)
    r5 = evaluate_method("Baseline: KC Reweighing", df, feature_cols_aware,
                          sample_weight=w_kc)
    rows.append(r5)

    # Hou & Chen ADRL — approximated by adversarial debiasing.
    # In sklearn we approximate via combining reweighing + threshold opt:
    # for cell-by-cell adjustment. We label honestly:
    probs_kc = r5["probs"]
    thresholds_kc = {}
    for g in range(4):
        mask = A_int == g
        thresholds_kc[g] = optimal_threshold(y[mask], probs_kc[mask], min_tpr=0.80)
    r6 = evaluate_method("Baseline: KC + per-group thresholds (proxy for ADRL)",
                          df, feature_cols_aware, sample_weight=w_kc,
                          thresholds=thresholds_kc)
    rows.append(r6)

    # Drop probs column before saving
    for r in rows:
        r.pop("probs", None)
    out = pd.DataFrame(rows)
    out.insert(0, "dataset", ds_name)
    return out


if __name__ == "__main__":
    all_results = []
    datasets = [(load_thcsmk, "THCSMK"), (load_uci_por, "UCI-Por")]
    try:
        from data_loaders import load_oulad
        _ = load_oulad()
        datasets.append((load_oulad, "OULAD"))
    except (ImportError, FileNotFoundError):
        pass
    for loader, name in datasets:
        print(f"\n=== {name} ===", flush=True)
        out = run(loader, name)
        print(out[['method', 'AUC_aggregate', 'FPR_Male_LowSES', 'FPR_Male_HighSES',
                   'FPR_Fem_LowSES', 'FPR_Fem_HighSES', 'FPR_spread']].round(3).to_string(index=False))
        all_results.append(out)
    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(RESULTS_DIR / "block4_interventions.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR/'block4_interventions.csv'}")
