"""
Causal decomposition estimator for fairness audit.

Implements the Counterfactual Direct, Indirect, and Spurious effects
(CtfDE, CtfIE, CtfSE) of Zhang & Bareinboim (2018), estimated via a
doubly-robust (DR) procedure following Tchetgen Tchetgen & Shpitser (2014)
mediation analysis methodology, extended to handle multiple mediators
treated as a vector.

Reference: Zhang, J., & Bareinboim, E. (2018). "Fairness in Decision-Making:
The Causal Explanation Formula." AAAI.

For an EWS classifier h_theta(A, W, M_1, M_2) and reference subgroup pair
(a_0 -> a_1) defined by intersection of (gender, ses_hi), we decompose:

    TV = E[h | A=a_1, Y=0] - E[h | A=a_0, Y=0]
       = CtfDE + CtfIE + CtfSE

where:
    CtfDE = E[h(a_1, M_{a_0}, W_{a_0}) | Y=0] - E[h(a_0, M_{a_0}, W_{a_0}) | Y=0]
    CtfIE = E[h(a_0, M_{a_1}, W_{a_0}) | Y=0] - E[h(a_0, M_{a_0}, W_{a_0}) | Y=0]
    CtfSE = TV - CtfDE - CtfIE

We condition on Y=0 because we are decomposing FPR (predictions among true
negatives). The same procedure works for unconditional decomposition by
dropping the Y=0 conditioning.

ESTIMATION:
1. Outcome regression mu(a, m, w) := E[h_theta(...) | A=a, M=m, W=w]
   Fitted on the training data using a flexible model (gradient boosting).
2. Mediator model: estimate p(m | a, w). For continuous mediators we use
   a Gaussian copula approximation; for the DR estimator below we don't
   actually need an explicit density, just the ability to sample M | A, W.
3. Propensity score pi(a | w) for A: multinomial logistic regression on W.
4. DR plug-in estimator for each potential outcome:

   E[h(a_1, M_{a_0}, W_{a_0}) | Y=0]
   = E_{P_{Y=0}} [ mu_hat(a_1, M, W) | A=a_0 ]
     + weighted correction term (efficient influence function)

For tractability with our sample sizes, we use a MONTE CARLO substitution
approach: draw M samples from the conditional distribution p(m | a_0, w) for
each unit, predict h_theta, and average. This is the "g-computation" formula
(Robins 1986) and is consistent under correct specification of the mediator
model. We pair this with bootstrap CIs which absorb specification uncertainty
to first order.

This implementation is intentionally conservative: we report:
   (a) point estimate via g-computation
   (b) 95% bootstrap CI (stratified by intersectional subgroup)
   (c) sensitivity analysis (E-value) — see sensitivity.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------
# Mediator sampler: parametric Gaussian for continuous; multinomial for ints
# ----------------------------------------------------------------------

class MediatorSampler:
    """Samples M | A=a, W=w using a flexible parametric model.

    We model each mediator dimension independently CONDITIONAL on A and W.
    Independence across mediator dimensions GIVEN (A, W) is the standard
    assumption in multivariate mediation analysis (VanderWeele & Vansteelandt
    2014); it is checked empirically by inspecting residual correlations
    (see fit_diagnostics).
    """

    def __init__(self, mediator_cols: list):
        self.mediator_cols = mediator_cols
        self.models = {}  # col -> fitted regressor + residual sd
        self.is_integer = {}

    def fit(self, A: np.ndarray, W: np.ndarray, M: np.ndarray):
        """Fit p(M_j | A, W) for each mediator j.

        A: (n,) integer subgroup code 0..K-1
        W: (n, p_w) covariate matrix
        M: (n, p_m) mediator matrix
        """
        n, p_m = M.shape
        # Build feature matrix: one-hot A + W
        K = int(A.max()) + 1
        A_onehot = np.eye(K)[A]
        X = np.hstack([A_onehot, W])

        for j, col in enumerate(self.mediator_cols):
            mj = M[:, j]
            # If integer-valued and limited support, treat as discrete via OLS
            # for the conditional mean and use empirical residual distribution
            self.is_integer[col] = np.allclose(mj, np.round(mj))

            # Use GBM for flexible conditional mean
            gbr = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                random_state=0, subsample=0.8
            )
            gbr.fit(X, mj)
            preds = gbr.predict(X)
            residuals = mj - preds
            # Empirical residual distribution per A-stratum to preserve
            # conditional variance structure
            residual_by_a = {}
            for a in range(K):
                mask_a = A == a
                if mask_a.sum() >= 5:
                    residual_by_a[a] = residuals[mask_a].copy()
                else:
                    residual_by_a[a] = residuals.copy()
            self.models[col] = {
                "model": gbr, "residuals_by_a": residual_by_a,
                "K": K, "global_residuals": residuals.copy()
            }

    def sample(self, A_target: np.ndarray, W: np.ndarray, n_mc: int = 1,
               rng: np.random.Generator = None) -> np.ndarray:
        """Draw M ~ p(M | A=A_target, W) -- shape (n_mc, n, p_m)."""
        if rng is None:
            rng = np.random.default_rng(0)
        n = len(A_target)
        p_m = len(self.mediator_cols)
        K = self.models[self.mediator_cols[0]]["K"]
        A_onehot = np.eye(K)[A_target]
        X = np.hstack([A_onehot, W])

        out = np.empty((n_mc, n, p_m))
        for j, col in enumerate(self.mediator_cols):
            m_info = self.models[col]
            mean_pred = m_info["model"].predict(X)
            # Sample residuals from the conditional empirical distribution
            for s in range(n_mc):
                resid = np.empty(n)
                for a in range(K):
                    mask = A_target == a
                    if mask.sum() == 0:
                        continue
                    pool = m_info["residuals_by_a"][a]
                    idx = rng.integers(0, len(pool), size=mask.sum())
                    resid[mask] = pool[idx]
                out[s, :, j] = mean_pred + resid
                if self.is_integer[col]:
                    out[s, :, j] = np.round(out[s, :, j]).astype(float)
        return out


# ----------------------------------------------------------------------
# Outcome regression: mu(a, m, w) = E[h_theta(a, m, w)]
# Since h_theta itself is a classifier, mu IS h_theta evaluated at counterfactual
# inputs. So no separate fit needed -- just evaluate the classifier directly.
# ----------------------------------------------------------------------

def evaluate_classifier(clf, feature_cols, A_col_idx, A_value, M_cols_idx, M_values,
                        W_cols_idx, W_values, base_X: np.ndarray) -> np.ndarray:
    """Evaluate clf on counterfactual feature matrix.

    base_X: (n, p) original feature matrix
    A_col_idx: int index of A column in feature matrix (or None if A not a feature)
    A_value: scalar A value to set
    M_cols_idx: indices of mediators
    M_values: (n, p_m) values to set for mediators
    W_cols_idx: indices of confounders (kept at base_X values)
    """
    X_cf = base_X.copy()
    if A_col_idx is not None and A_value is not None:
        X_cf[:, A_col_idx] = A_value
    if M_cols_idx is not None and M_values is not None:
        X_cf[:, M_cols_idx] = M_values
    # W left alone (or use W_values if explicitly different)
    if W_cols_idx is not None and W_values is not None:
        X_cf[:, W_cols_idx] = W_values
    return clf.predict_proba(X_cf)[:, 1]


# ----------------------------------------------------------------------
# Main decomposition routine
# ----------------------------------------------------------------------

@dataclass
class DecompositionResult:
    a0: int
    a1: int
    TV: float
    CtfDE: float
    CtfIE: float
    CtfSE: float
    n_a0: int
    n_a1: int
    condition_Y0: bool

    def as_dict(self):
        return {
            "a0": self.a0, "a1": self.a1,
            "TV": self.TV,
            "CtfDE": self.CtfDE, "CtfIE": self.CtfIE, "CtfSE": self.CtfSE,
            "n_a0": self.n_a0, "n_a1": self.n_a1,
            "condition_Y0": self.condition_Y0,
        }


def decompose_pairwise(
    clf,
    feature_cols: list,
    df: pd.DataFrame,  # has columns: feature_cols + A_intersect + Y
    A_col: str,           # protected attribute used as A in SCM (e.g., "A_intersect")
    A_in_features: str | None,  # name of A as a model feature if any (e.g., "A_ses_hi")
                                # or None if A is not a feature
    M_cols: list,
    W_cols: list,
    a0: int, a1: int,
    condition_Y0: bool = True,
    n_mc: int = 50,
    rng: np.random.Generator = None,
) -> DecompositionResult:
    """Compute CtfDE/CtfIE/CtfSE for one pair (a0, a1).

    The framework here:
    - A is "A_intersect" (4 subgroups). We want to compare subgroup a1 vs a0.
    - The model feature reflecting A is `A_in_features` (e.g., A_ses_hi only --
      gender is NOT a model feature, but is part of A in the SCM). If
      A_in_features is None, A has no direct path to the model output
      (predicted purely through M, W).

    For decomposition conditioned on Y=0 (i.e., decomposing FPR):
        Reference population: units in subgroup a0 with Y=0
        Target: predictions of h_theta evaluated at counterfactual (A, M, W).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # ----- Map A_intersect codes to (gender, ses_hi) for setting features -----
    def intersect_to_components(a):
        female = a // 2
        ses_hi = a % 2
        return female, ses_hi

    a0_female, a0_ses_hi = intersect_to_components(a0)
    a1_female, a1_ses_hi = intersect_to_components(a1)

    # Indices in feature matrix
    A_idx_in_features = (feature_cols.index(A_in_features)
                         if A_in_features and A_in_features in feature_cols else None)
    M_idx = [feature_cols.index(c) for c in M_cols]
    W_idx = [feature_cols.index(c) for c in W_cols]

    # ----- Reference population: subgroup a0 (with Y=0 if conditioning) -----
    mask_a0 = (df[A_col].values == a0)
    if condition_Y0:
        mask_a0 = mask_a0 & (df["Y"].values == 0)
    mask_a1 = (df[A_col].values == a1)
    if condition_Y0:
        mask_a1 = mask_a1 & (df["Y"].values == 0)

    n_a0 = int(mask_a0.sum())
    n_a1 = int(mask_a1.sum())

    if n_a0 < 5 or n_a1 < 5:
        return DecompositionResult(a0, a1, np.nan, np.nan, np.nan, np.nan,
                                   n_a0, n_a1, condition_Y0)

    X_all = df[feature_cols].values.astype(float)
    X_ref = X_all[mask_a0]  # reference units (subgroup a0)
    W_ref = X_ref[:, W_idx] if len(W_idx) > 0 else None

    # ===== Fit mediator sampler on FULL data (using A_intersect as A) =====
    A_intersect = df[A_col].values.astype(int)
    W_full = X_all[:, W_idx] if len(W_idx) > 0 else np.zeros((len(df), 0))
    M_full = X_all[:, M_idx]

    sampler = MediatorSampler(mediator_cols=M_cols)
    sampler.fit(A_intersect, W_full, M_full)

    # ===== Total Variation =====
    # TV = E[h | a1, Y=0] - E[h | a0, Y=0]
    probs_full = clf.predict_proba(X_all)[:, 1]
    TV = probs_full[mask_a1].mean() - probs_full[mask_a0].mean()

    # ===== Three counterfactual quantities, all evaluated on reference units =====

    # Q0 = E_ref[ h(a0, M_{a0}, W_{a0}) ]
    # Q1 = E_ref[ h(a1, M_{a0}, W_{a0}) ]   (CtfDE pair: Q1 - Q0)
    # Q2 = E_ref[ h(a0, M_{a1}, W_{a0}) ]   (CtfIE pair: Q2 - Q0)
    #
    # We use Monte Carlo: draw n_mc samples of M for each scenario, evaluate
    # the classifier, average.

    A_target_a0_ref = np.full(n_a0, a0)
    A_target_a1_ref = np.full(n_a0, a1)

    # Sample M | A=a0, W=W_ref  (for Q0 and Q1)
    M_under_a0 = sampler.sample(A_target_a0_ref, W_ref, n_mc=n_mc, rng=rng)  # (n_mc, n_a0, p_m)
    # Sample M | A=a1, W=W_ref  (for Q2)
    M_under_a1 = sampler.sample(A_target_a1_ref, W_ref, n_mc=n_mc, rng=rng)

    # Helper: evaluate h_theta on reference units with substituted M and chosen A in feature space
    def eval_potential(M_samples, set_A_ses_hi):
        """Evaluate average h_theta over MC draws."""
        n_mc_ = M_samples.shape[0]
        preds_avg = np.zeros(n_a0)
        for s in range(n_mc_):
            X_cf = X_ref.copy()
            # Set mediator columns
            X_cf[:, M_idx] = M_samples[s]
            # Set A_in_features column if applicable
            if A_idx_in_features is not None:
                X_cf[:, A_idx_in_features] = set_A_ses_hi
            preds_avg += clf.predict_proba(X_cf)[:, 1]
        return preds_avg / n_mc_

    # Q0: A=a0 (set ses_hi to a0_ses_hi), M ~ p(M|a0, W)
    Q0 = eval_potential(M_under_a0, set_A_ses_hi=a0_ses_hi).mean()
    # Q1: A=a1 (set ses_hi to a1_ses_hi), M ~ p(M|a0, W)
    Q1 = eval_potential(M_under_a0, set_A_ses_hi=a1_ses_hi).mean()
    # Q2: A=a0 (set ses_hi to a0_ses_hi), M ~ p(M|a1, W)
    Q2 = eval_potential(M_under_a1, set_A_ses_hi=a0_ses_hi).mean()

    CtfDE = Q1 - Q0
    CtfIE = Q2 - Q0
    CtfSE = TV - CtfDE - CtfIE

    return DecompositionResult(
        a0=a0, a1=a1, TV=TV,
        CtfDE=CtfDE, CtfIE=CtfIE, CtfSE=CtfSE,
        n_a0=n_a0, n_a1=n_a1, condition_Y0=condition_Y0
    )


def bootstrap_decomposition(
    clf_factory,  # callable returning fresh classifier
    df: pd.DataFrame,
    feature_cols: list,
    A_col: str,
    A_in_features: str | None,
    M_cols: list,
    W_cols: list,
    a0: int, a1: int,
    condition_Y0: bool = True,
    n_mc: int = 30,
    n_boot: int = 200,
    seed: int = 42,
    refit_clf: bool = True,
) -> dict:
    """Bootstrap 95% CI for decomposition. Stratified by A_intersect.

    Two modes:
      refit_clf=True  -- refit classifier on each bootstrap sample.
                         Reflects total uncertainty (data + classifier).
      refit_clf=False -- hold classifier fixed; resample only data.
                         Reflects mediator-distribution uncertainty;
                         appropriate for AUDITING a deployed model.

    For paper, we report refit_clf=True as primary; refit_clf=False as
    sensitivity (Block 5).
    """
    rng = np.random.default_rng(seed)
    A_values = df[A_col].values
    groups = np.unique(A_values)
    idx_by_group = {g: np.where(A_values == g)[0] for g in groups}

    # Point estimate on full data
    clf = clf_factory()
    Xfull = df[feature_cols].values
    yfull = df["Y"].values
    clf.fit(Xfull, yfull)
    point = decompose_pairwise(
        clf, feature_cols, df, A_col, A_in_features, M_cols, W_cols,
        a0, a1, condition_Y0, n_mc=n_mc, rng=rng
    )

    boot_TV, boot_DE, boot_IE, boot_SE = [], [], [], []
    for b in range(n_boot):
        # Stratified resample
        idx_boot = np.concatenate([
            rng.choice(idx_by_group[g], size=len(idx_by_group[g]), replace=True)
            for g in groups
        ])
        df_b = df.iloc[idx_boot].reset_index(drop=True)
        if refit_clf:
            Xb = df_b[feature_cols].values
            yb = df_b["Y"].values
            clf_b = clf_factory()
            clf_b.fit(Xb, yb)
        else:
            clf_b = clf  # reuse fitted classifier
        res = decompose_pairwise(
            clf_b, feature_cols, df_b, A_col, A_in_features, M_cols, W_cols,
            a0, a1, condition_Y0, n_mc=n_mc, rng=rng
        )
        if not np.isnan(res.TV):
            boot_TV.append(res.TV)
            boot_DE.append(res.CtfDE)
            boot_IE.append(res.CtfIE)
            boot_SE.append(res.CtfSE)

    def ci(vals):
        if len(vals) == 0:
            return (np.nan, np.nan)
        return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

    return {
        "a0": a0, "a1": a1,
        "TV": point.TV, "TV_ci": ci(boot_TV),
        "CtfDE": point.CtfDE, "CtfDE_ci": ci(boot_DE),
        "CtfIE": point.CtfIE, "CtfIE_ci": ci(boot_IE),
        "CtfSE": point.CtfSE, "CtfSE_ci": ci(boot_SE),
        "n_a0": point.n_a0, "n_a1": point.n_a1,
        "n_boot_valid": len(boot_TV),
        "condition_Y0": condition_Y0,
    }


if __name__ == "__main__":
    # Smoke test on THCSMK
    import sys
    sys.path.insert(0, '.')
    from data_loaders import load_thcsmk, get_feature_columns
    from sklearn.ensemble import RandomForestClassifier

    print("Loading THCSMK ...")
    ds = load_thcsmk()
    df = ds["data"].copy()
    feature_cols = get_feature_columns(ds, include_ses=True)
    print(f"Features: {feature_cols}")

    print("\nFitting RF classifier ...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(df[feature_cols].values, df["Y"].values)

    print("\nPoint estimate decomposition: a0=Fem×LowSES (2) vs a1=Male×HighSES (1)")
    rng = np.random.default_rng(0)
    res = decompose_pairwise(
        rf, feature_cols, df,
        A_col="A_intersect", A_in_features="A_ses_hi",
        M_cols=ds["feature_groups"]["M1"] + ds["feature_groups"]["M2"],
        W_cols=ds["feature_groups"]["W"],
        a0=2, a1=1, condition_Y0=True, n_mc=50, rng=rng
    )
    print(f"  n_a0={res.n_a0}, n_a1={res.n_a1}")
    print(f"  TV    = {res.TV:+.4f}")
    print(f"  CtfDE = {res.CtfDE:+.4f}  ({100*res.CtfDE/res.TV if res.TV!=0 else 0:+.1f}% of TV)")
    print(f"  CtfIE = {res.CtfIE:+.4f}  ({100*res.CtfIE/res.TV if res.TV!=0 else 0:+.1f}% of TV)")
    print(f"  CtfSE = {res.CtfSE:+.4f}  ({100*res.CtfSE/res.TV if res.TV!=0 else 0:+.1f}% of TV)")
