# 3. Causal Framework

This section formalizes the causal decomposition of subgroup fairness gaps in an educational early warning system (EWS). We work with a structural causal model (SCM) in which a single binary outcome $Y$ (pass/fail at the end of the academic year) is predicted by a classifier $h_\theta$ from features available at mid-semester. Our goal is not to debias the classifier, but to *audit* an already trained classifier by attributing its subgroup disparity to **direct discrimination**, **historically mediated inequality**, and **spurious confounding**.

We adopt the population-level potential outcomes notation of Pearl (2009) and the path-specific counterfactual framework of Zhang & Bareinboim (2018), whose **causal explanation formula** decomposes the total variation in a decision rule across two values of a protected attribute into three additively identifiable components. We extend their formulation to (i) intersectional subgroups defined by the joint values of gender and socioeconomic status, and (ii) decomposition conditional on the true label, which is the natural quantity when the metric of concern is the false positive rate.

## 3.1 Structural Causal Model for Mid-Semester EWS

Let $A \in \{0,1,2,3\}$ denote the intersectional protected indicator with values
$\{$Male$\times$LowSES, Male$\times$HighSES, Fem$\times$LowSES, Fem$\times$HighSES$\}$, and let $A_g, A_s$ denote its gender and SES components. We treat $A$ as exogenous and observed.

The remaining variables are:

- $W$: observed pre-enrollment confounders (school assignment, distance band, urban/rural residence, immigrant status). $W$ may causally depend on $A$ — e.g., SES correlates with neighborhood and therefore distance to school — but is determined before the school year begins.
- $M_1$: academic mediators measured at mid-semester (mid-term examination score, average formative assessment score, prior course failures).
- $M_2$: behavioral mediators measured at mid-semester (excused absences, unexcused absences, leisure time indicators).
- $Y$: true binary outcome at end of year (pass = 1, fail = 0).
- $\hat{Y} = h_\theta(A_s, W, M_1, M_2)$: classifier prediction. **Note:** $A_g$ (gender) is **not** an input to the classifier in our primary configuration — a deliberate modelling choice reflecting standard deployment practice in Vietnamese THCS systems, where gender is rarely entered into automated EWS pipelines. SES proxy variables $A_s$ are included as features in the SES-aware configuration and excluded in the SES-unaware configuration (the contrast that motivates the SES Inclusion Paradox analysis in Section 5).

The structural equations are:

$$
\begin{aligned}
W &\leftarrow f_W(A, U_W), \\
M_1 &\leftarrow f_{M_1}(A, W, U_{M_1}), \\
M_2 &\leftarrow f_{M_2}(A, W, M_1, U_{M_2}), \\
Y &\leftarrow f_Y(A, W, M_1, M_2, U_Y), \\
\hat{Y} &= h_\theta(A_s, W, M_1, M_2).
\end{aligned}
$$

The corresponding directed acyclic graph is shown in Figure 1. Three causal pathways carry the influence of $A$ on $\hat{Y}$:

1. **Direct path** $A \to \hat{Y}$ via the model input $A_s$ (active only in the SES-aware configuration). Captures *direct discrimination*: the model's use of the protected attribute itself.
2. **Indirect (mediated) path** $A \to (M_1, M_2) \to \hat{Y}$ (with possible passage through $W$). Captures *historically embedded inequality*: gender and SES gaps in formative assessments, attendance, and prior failures that the model picks up via the mediators.
3. **Spurious path** $A \leftrightarrow U \to \hat{Y}$ through unobserved confounders or selection effects. Captures correlations that are not causally attributable to $A$ but to common causes such as community-level effects on both family SES and school engagement.

The identification assumptions we require are standard in mediation analysis (Pearl 2001; VanderWeele 2015):
**(A1) No unmeasured confounding of $A \to Y$ given $W$.** Since $A$ is intersectional gender $\times$ SES and the relevant pre-enrollment confounders are observed, A1 is plausible but not guaranteed.
**(A2) No unmeasured confounding of $M \to Y$ given $(A,W)$.** This is the strongest assumption: unmeasured determinants of mid-term performance (parental support, peer effects, teacher quality) are likely correlated with the outcome.
**(A3) No unmeasured confounding of $A \to M$ given $W$.**
**(A4) Cross-world assumption (Pearl 2001):** no unmeasured confounders affect both $M$ under one assignment of $A$ and $Y$ under another. Untestable but necessary for path-specific identifiability.

Because A2 is the weakest assumption in practice, Section 3.4 specifies a sensitivity protocol (E-value plus tipping point) that we report alongside every pathway estimate. We do not present causal effects as identified facts; we present them as **estimates accompanied by a robustness range**.

## 3.2 Counterfactual Decomposition of FPR Disparity

For two intersectional subgroup labels $a_0$ (reference) and $a_1$ (comparison), we are interested in decomposing the disparity in the model's positive-prediction rate among true negatives — that is, the FPR gap. Following Zhang & Bareinboim's *causal explanation formula*, conditioned on the event $\{Y=0\}$:

$$
\text{TV}_{a_0, a_1}(\hat{Y} \mid Y{=}0)
= \underbrace{\Pr(\hat{Y}=1 \mid A{=}a_1, Y{=}0) - \Pr(\hat{Y}=1 \mid A{=}a_0, Y{=}0)}_{\text{observed FPR gap}}.
$$

Define the **counterfactual potential outcomes**:

- $\hat{Y}_{a, M_{a'}, W_{a'}}$: the prediction that would be observed if $A$ were set to $a$, while $M$ and $W$ were drawn from their conditional distributions under $A = a'$.

The three components are:

$$
\begin{aligned}
\text{CtfDE}_{a_0, a_1} &= \mathbb{E}\!\bigl[\hat{Y}_{a_1, M_{a_0}, W_{a_0}} - \hat{Y}_{a_0, M_{a_0}, W_{a_0}} \mid Y=0\bigr], \\[4pt]
\text{CtfIE}_{a_0, a_1} &= \mathbb{E}\!\bigl[\hat{Y}_{a_0, M_{a_1}, W_{a_0}} - \hat{Y}_{a_0, M_{a_0}, W_{a_0}} \mid Y=0\bigr], \\[4pt]
\text{CtfSE}_{a_0, a_1} &= \text{TV}_{a_0, a_1} - \text{CtfDE}_{a_0, a_1} - \text{CtfIE}_{a_0, a_1}.
\end{aligned}
$$

**Reading the decomposition.** In CtfDE, the mediators and confounders are held at the joint distribution of the reference subgroup $a_0$; only the protected attribute is flipped. CtfDE thus isolates the change attributable to the model's *direct* use of $A$ (in our setting, the SES proxy $A_s$). In CtfIE, $A$ is held at $a_0$ in the prediction step, but the mediators are drawn from the joint distribution under $A=a_1$. CtfIE thus isolates the change attributable to *the difference in mediator values* between the two subgroups — historically inherited inequality in academic and behavioral outcomes. CtfSE absorbs the remainder, including correlations through unobserved confounders or selection.

Under assumptions (A1)–(A4), CtfDE and CtfIE are identifiable as functionals of the observational distribution via the mediation formula (Pearl 2001). For continuous-mediator settings such as ours, the identifying expressions become:

$$
\mathbb{E}\!\bigl[\hat{Y}_{a, M_{a'}, W_{a'}} \mid Y{=}0\bigr]
= \int\!\!\int h_\theta(a_s, w, m)\, p_{M\mid A,W}(m \mid a', w)\, p_{W\mid A,Y}(w \mid a', Y{=}0)\, dm\, dw,
$$

where $a_s$ is the SES component of $a$ (when $A$ is intersectional) and $h_\theta$ may not depend on $a_g$ (gender) when gender is not a feature.

**Operational interpretation conditional on $Y=0$.** Conditioning on $Y=0$ restricts attention to students who genuinely failed. Among these, the FPR gap is the gap in the rate at which the model erroneously predicts pass. A positive CtfDE means: holding everything else equal, switching the protected indicator from $a_0$ to $a_1$ raises the false pass rate by CtfDE percentage points — the model is *directly* more lenient toward students in subgroup $a_1$. A positive CtfIE means: the mediator distribution of subgroup $a_1$ (e.g., higher mid-term scores, fewer absences) leads to a higher false pass rate. CtfIE positive does not imply the *model* is biased; it implies that the *data on which it operates* carries inequality that the model faithfully reflects.

### 3.2.1 Special case: intersectional subgroup with one fixed component

When comparing subgroups that differ only in gender (e.g., $a_0 = $ Fem$\times$LowSES and $a_1 = $ Male$\times$LowSES), and when gender is not a model feature, the CtfDE *with respect to $\hat{Y}$* is necessarily zero — the model cannot directly read gender. Any observed gender-pair TV must therefore decompose into CtfIE (mediated through gender differences in $M$ and $W$) and CtfSE (unobserved confounding). This identifiability result is automatic in our setting and provides a built-in sanity check: an estimator that returns nonzero CtfDE for a pure gender flip would be misspecified.

The same is not true for SES flips in the SES-aware configuration: SES *is* a model feature, so a direct path exists and CtfDE can be nonzero.

## 3.3 Estimation Procedure

We estimate the three potential-outcome means via a Monte Carlo plug-in (g-computation) procedure, the standard approach for path-specific effects with multiple continuous mediators (VanderWeele & Vansteelandt 2014; Lin et al. 2017).

**Step 1 — Reference sample.** Define the audit population $\mathcal{S}_{a_0, Y=0} = \{i : A_i = a_0, Y_i = 0\}$. All counterfactual quantities are averaged over this sample, so confounder values $W_i$ enter at their observed levels for the reference subgroup. This automatically handles $p_{W \mid A, Y=0}(\cdot \mid a_0, Y{=}0)$ in the identifying integral.

**Step 2 — Mediator model.** For each mediator component $M_j \in \{M_{1,1}, M_{1,2}, \ldots, M_{2,k}\}$, fit a flexible conditional mean model $\hat\mu_{M_j}(a, w) = \mathbb{E}[M_j \mid A=a, W=w]$ using gradient boosting (sklearn `GradientBoostingRegressor`, 100 trees, depth 3, learning rate 0.05). Compute residuals $\hat r_{ij} = M_{ij} - \hat\mu_{M_j}(A_i, W_i)$, retaining the empirical residual distribution stratified by $A$. This residual-stratified pool provides a non-parametric conditional density without imposing a Gaussian assumption — an important robustness choice for the THCSMK absence variables, which have heavy right tails. The independence of mediator components conditional on $(A, W)$ is the multivariate mediation assumption (VanderWeele & Vansteelandt 2014); we examine its empirical adequacy via cross-residual correlations and report results in Block 5.

**Step 3 — Monte Carlo evaluation of potential outcomes.** For each reference unit $i \in \mathcal{S}_{a_0, Y=0}$ and each scenario $(a_M, a_A)$ in $\{(a_0, a_0), (a_0, a_1), (a_1, a_0)\}$, draw $S$ mediator vectors $\tilde M^{(s)}_i \sim p_{M \mid A, W}(\cdot \mid a_M, W_i)$ and compute

$$
\hat{Q}(a_M, a_A) = \frac{1}{|\mathcal{S}_{a_0, Y=0}|} \sum_{i \in \mathcal{S}_{a_0, Y=0}} \frac{1}{S} \sum_{s=1}^{S} h_\theta\!\bigl(a_A^{(s)}_s, W_i, \tilde M^{(s)}_i\bigr),
$$

where $a_A^{(s)}_s$ is the SES component of $a_A$ in the SES-aware configuration. We use $S = 30$ throughout; larger $S$ produced changes below 0.002 in pilot tests.

**Step 4 — Plug-in decomposition.**
$$
\widehat{\text{CtfDE}} = \hat{Q}(a_0, a_1) - \hat{Q}(a_0, a_0), \quad
\widehat{\text{CtfIE}} = \hat{Q}(a_1, a_0) - \hat{Q}(a_0, a_0),
$$
$$
\widehat{\text{CtfSE}} = \widehat{\text{TV}} - \widehat{\text{CtfDE}} - \widehat{\text{CtfIE}}.
$$

**Step 5 — Bootstrap variance.** We construct 95% confidence intervals via $B = 200$ stratified bootstrap replicates, sampling within each intersectional cell to preserve cell sizes. In each replicate we refit both the classifier $h_\theta$ and the mediator model and recompute the decomposition. This reflects total uncertainty (training data + nuisance estimation). As a sensitivity check we also report intervals from the *audit-only* bootstrap, in which $h_\theta$ is held fixed at its training-data-trained version and only the mediator model is refit (Block 5). The two regimes answer different questions: the audit-only bootstrap reflects the question "is *this deployed* model unfair?", while the joint bootstrap reflects "is the *fitting procedure* unfair on this kind of data?" — both are informative.

**Step 6 — Statistical inference.** We do not perform null-hypothesis tests on CtfDE/CtfIE/CtfSE because the relevant scientific questions are about magnitudes and pathway proportions, not binary significance. We report 95% bootstrap CIs and the share of bootstrap replicates in which the dominant pathway agrees with the point estimate (a robustness summary). Where the paper makes formal comparisons (e.g., SES-aware vs SES-unaware $\Delta$CtfDE), Holm-Bonferroni-adjusted bootstrap CIs are reported.

## 3.4 Sensitivity to Unmeasured Confounding

Identifiability assumption A2 — no unmeasured confounding of $M \to Y$ given $(A, W)$ — is plausibly violated in our datasets. Parental motivation, peer effects, and teacher quality plausibly influence both mid-semester academic performance and end-of-year outcomes. We therefore report two complementary sensitivity tools for every CtfDE and CtfIE estimate.

### 3.4.1 E-value

Following VanderWeele & Ding (2017), the **E-value** is the minimum strength of association on the risk-ratio scale that an unmeasured confounder $U$ would need with both the mediator and the outcome, conditional on observed covariates, to fully explain away the estimated pathway effect. Concretely, for an estimated pathway risk ratio $\text{RR}$ on the same scale,

$$
\text{E-value}(\text{RR}) =
\begin{cases}
\text{RR} + \sqrt{\text{RR}(\text{RR} - 1)} & \text{if } \text{RR} \geq 1, \\[4pt]
\text{RR}^{*\!-1} + \sqrt{\text{RR}^{*\!-1}(\text{RR}^{*\!-1} - 1)} & \text{if } \text{RR} < 1, \, \text{RR}^* = 1/\text{RR}.
\end{cases}
$$

An E-value of, say, 1.8 means that a confounder associated with both $M$ and $Y$ by a risk ratio of at least 1.8 (above and beyond observed covariates) would be required to eliminate the estimated effect — context-specific judgment is required to assess plausibility. For our application we convert the additive CtfDE and CtfIE on the probability scale to approximate risk ratios using the marginal $\hat Y$ rate in the reference population, $\bar p_0$, via $\text{RR} \approx (\bar p_0 + \text{Ctf}) / \bar p_0$.

### 3.4.2 Tipping point analysis

For the headline subgroup pair in each dataset, we conduct a two-dimensional sensitivity sweep over hypothesized confounder strengths $(\gamma_{A,U}, \gamma_{U,Y})$ on the partial-correlation scale. For each grid point we apply a Mendelian-randomization-style adjustment (Cinelli & Hazlett 2020) to compute the bias-corrected CtfDE and CtfIE. We then trace the **tipping curve** in $(\gamma_{A,U}, \gamma_{U,Y})$ space at which the corrected CtfDE crosses zero. Findings whose tipping curves require implausibly large $\gamma$ are reported as robust; those near the origin are flagged as fragile.

## 3.5 Worked Numerical Example (THCSMK)

To make the framework concrete, consider a single comparison in the THCSMK dataset:
$a_0 = $ Fem$\times$LowSES, $a_1 = $ Male$\times$HighSES, condition $Y=0$.

The reference subgroup has $n_{a_0,Y=0} = 75$ students who failed despite their classification as female and low-SES. The classifier (Random Forest, SES-aware, 5-fold CV out-of-fold predictions) assigns this group an average false-positive prediction probability of 0.47. The comparison subgroup (Male$\times$HighSES, $Y=0$, $n = 13$) receives an average of 0.50, giving an observed FPR gap of $\widehat{\text{TV}} \approx +0.03$.

For each of the 75 reference units, we now perform three Monte Carlo simulations of $\hat{Y}$:

1. *Status quo* — predict using their actual mediator values, with $A_s = 0$ (their actual LowSES indicator). Average: $\hat{Q}(a_0, a_0) \approx 0.47$.
2. *Direct counterfactual* — same mediator values, but flip $A_s$ to 1 (the SES indicator value of $a_1$). Average: $\hat{Q}(a_0, a_1) \approx 0.48$. The difference is $\widehat{\text{CtfDE}} \approx +0.01$.
3. *Indirect counterfactual* — restore $A_s = 0$, but resample mediator values from the empirical conditional distribution observed among Male$\times$HighSES students with similar confounder profile. Average: $\hat{Q}(a_1, a_0) \approx 0.50$. The difference is $\widehat{\text{CtfIE}} \approx +0.03$.

The residual $\widehat{\text{CtfSE}} = \widehat{\text{TV}} - \widehat{\text{CtfDE}} - \widehat{\text{CtfIE}} \approx -0.01$, indicating slight spurious offset.

**The interpretation.** Of the small +0.03 FPR gap between Male$\times$HighSES and Fem$\times$LowSES on THCSMK, roughly one-third is attributable to the model directly using SES, and roughly all of the rest is attributable to the difference in academic and behavioral mediator distributions between the two subgroups. *Removing SES from the model would reduce the CtfDE component to zero but would not affect the CtfIE component*, which encodes the historical inequality embedded in mid-term scores and attendance records themselves. This is the empirical fact that motivates pathway-targeted interventions in Section 5: a fairness method that removes only the protected attribute cannot, by construction, address mediated bias.

## 3.6 Summary

We have specified a structural causal model for mid-semester EWS prediction, decomposed the observed FPR disparity between intersectional subgroups into three identifiable components corresponding to direct, mediated, and spurious causal pathways, given a doubly-robust-style Monte Carlo estimation procedure, and outlined the sensitivity protocol used to bound the role of unmeasured confounding. Section 4 specifies the three datasets and harmonization decisions; Section 5 reports the empirical decomposition together with the SES Paradox analysis and the pathway-targeted intervention comparison.
