"""
Fairness and performance metrics for the EWS audit.

Includes:
- subgroup_metrics: AUC, TPR, FPR, ECE per intersectional subgroup
- delong_test: DeLong (1988) test for AUC difference between two correlated ROC curves
- bootstrap_ci: stratified bootstrap CI for any metric
- ece: Expected Calibration Error
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats


# ----------------------------------------------------------------------
# Basic subgroup metrics
# ----------------------------------------------------------------------

def fpr_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """False Positive Rate at given decision threshold.

    Note for EWS: positive class = pass (Y=1). 'False positive' means we
    predict pass for a student who actually fails. We interpret this as
    'missed at-risk' for EWS purposes — these are the students who would not
    be flagged by the system but should be.
    """
    y_pred = (y_prob >= threshold).astype(int)
    # FPR = FP / (FP + TN) where negative class is 'fail' (Y=0)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    if (tn + fp) == 0:
        return np.nan
    return fp / (tn + fp)


def tpr_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """True Positive Rate at given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    if (tp + fn) == 0:
        return np.nan
    return tp / (tp + fn)


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width binning).

    ECE = sum over bins of (|bin| / n) * |acc_bin - conf_bin|
    """
    n = len(y_true)
    if n == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece_val += (mask.sum() / n) * abs(acc - conf)
    return ece_val


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC with NaN return if only one class present."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def subgroup_metrics(y_true: np.ndarray, y_prob: np.ndarray, A_intersect: np.ndarray,
                     threshold: float = 0.5) -> pd.DataFrame:
    """Compute per-subgroup AUC, TPR, FPR, ECE, n."""
    rows = []
    for g in sorted(np.unique(A_intersect)):
        mask = A_intersect == g
        rows.append({
            "subgroup": int(g),
            "n": int(mask.sum()),
            "AUC": safe_auc(y_true[mask], y_prob[mask]),
            "TPR": tpr_at_threshold(y_true[mask], y_prob[mask], threshold),
            "FPR": fpr_at_threshold(y_true[mask], y_prob[mask], threshold),
            "ECE": ece(y_true[mask], y_prob[mask]),
            "pass_rate": float(y_true[mask].mean()),
        })
    # Aggregate
    rows.append({
        "subgroup": -1,
        "n": len(y_true),
        "AUC": safe_auc(y_true, y_prob),
        "TPR": tpr_at_threshold(y_true, y_prob, threshold),
        "FPR": fpr_at_threshold(y_true, y_prob, threshold),
        "ECE": ece(y_true, y_prob),
        "pass_rate": float(y_true.mean()),
    })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# DeLong test for paired AUC comparison
# ----------------------------------------------------------------------

def _compute_midrank(x):
    """Internal helper: compute mid-ranks (handles ties)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # average rank, 1-indexed
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong implementation from Sun & Xu (2014).

    predictions_sorted_transposed : ndarray of shape (k, n)
        k = number of classifiers, n = number of examples; positives FIRST
    label_1_count : int, number of positive samples
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> dict:
    """DeLong test for two correlated ROC AUCs on the same labels.

    Returns dict with auc_a, auc_b, z_stat, p_value.
    """
    order = np.argsort(-y_true)  # positives first (descending sort by label)
    label_1_count = int(np.sum(y_true == 1))
    preds_sorted = np.vstack([y_prob_a[order], y_prob_b[order]])
    aucs, cov = _fast_delong(preds_sorted, label_1_count)
    auc_a, auc_b = aucs[0], aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": np.nan, "p_value": np.nan}
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b,
            "z_stat": z, "p_value": p, "var_diff": var_diff}


def delong_test_independent(y_true_a, y_prob_a, y_true_b, y_prob_b) -> dict:
    """DeLong-style test for AUC difference between two INDEPENDENT samples
    (i.e., disjoint subgroups). Uses normal approximation of AUC variance.

    For two independent subgroups (different students), the covariance between
    AUCs is zero, so the variance of the difference is var(A) + var(B).
    """
    auc_a = safe_auc(y_true_a, y_prob_a)
    auc_b = safe_auc(y_true_b, y_prob_b)
    if np.isnan(auc_a) or np.isnan(auc_b):
        return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": np.nan, "p_value": np.nan}

    # Hanley-McNeil variance approximation
    def hanley_var(y_true, y_prob):
        auc = safe_auc(y_true, y_prob)
        n1 = (y_true == 1).sum()
        n0 = (y_true == 0).sum()
        if n1 == 0 or n0 == 0:
            return np.nan
        Q1 = auc / (2 - auc)
        Q2 = 2 * auc**2 / (1 + auc)
        var = (auc * (1 - auc) + (n1 - 1) * (Q1 - auc**2)
               + (n0 - 1) * (Q2 - auc**2)) / (n1 * n0)
        return var

    var_a = hanley_var(y_true_a, y_prob_a)
    var_b = hanley_var(y_true_b, y_prob_b)
    if np.isnan(var_a) or np.isnan(var_b) or (var_a + var_b) <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": np.nan, "p_value": np.nan}
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b,
            "z_stat": z, "p_value": p, "var_a": var_a, "var_b": var_b}


# ----------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------

def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    stratify: np.ndarray = None
) -> dict:
    """Generic bootstrap CI for a metric over (y_true, y_prob).

    If stratify is provided, sampling is stratified by that grouping.
    Returns dict with point estimate, lo, hi.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_vals = []
    if stratify is not None:
        groups = np.unique(stratify)
        idx_by_group = {g: np.where(stratify == g)[0] for g in groups}
    for _ in range(n_boot):
        if stratify is not None:
            idx = np.concatenate([
                rng.choice(idx_by_group[g], size=len(idx_by_group[g]), replace=True)
                for g in groups
            ])
        else:
            idx = rng.integers(0, n, size=n)
        v = metric_fn(y_true[idx], y_prob[idx])
        if not np.isnan(v):
            boot_vals.append(v)
    boot_vals = np.array(boot_vals)
    point = metric_fn(y_true, y_prob)
    lo = float(np.quantile(boot_vals, alpha / 2))
    hi = float(np.quantile(boot_vals, 1 - alpha / 2))
    return {"point": float(point), "lo": lo, "hi": hi,
            "boot_mean": float(boot_vals.mean()), "n_boot_valid": len(boot_vals)}


if __name__ == "__main__":
    # Quick self-test
    rng = np.random.default_rng(0)
    n = 500
    A = rng.integers(0, 4, size=n)
    Y = rng.binomial(1, 0.5 + 0.05 * A, size=n)
    # Simulate a slightly biased classifier
    probs = np.clip(0.3 + 0.1 * A + rng.normal(0, 0.1, n), 0, 1)
    df = subgroup_metrics(Y, probs, A)
    print(df.round(3))

    # DeLong test (paired)
    probs_b = probs + rng.normal(0, 0.02, n)
    res = delong_test(Y, probs, probs_b)
    print(f"\nDeLong (paired): z={res['z_stat']:.3f}, p={res['p_value']:.3f}")

    # Bootstrap CI for AUC
    ci = bootstrap_ci(safe_auc, Y, probs, n_boot=500)
    print(f"\nAUC = {ci['point']:.3f}  [95% CI: {ci['lo']:.3f}, {ci['hi']:.3f}]")
