# Causal Decomposition of Subgroup Fairness Gaps in EWS

**Target venue:** IEEE Access (Trustworthy AI / AI in Education sections)
**Datasets:** THCSMK (Vietnam, n=675), UCI-Por (Portugal, n=649), OULAD (UK, n≈32,000)

---

## NOVELTY — Quick Reference

| What others do | What this paper does |
|---|---|
| Hou & Chen (2026): aggregate DP / EO metrics; debias one attribute at a time; end-of-year features. | Per-pathway causal decomposition (CtfDE / CtfIE / CtfSE) under Zhang & Bareinboim (2018); intersectional subgroups; mid-semester features only. |
| FairAIED (2025) names "multi-level bias" as an open gap. | First **empirical** operationalisation in EDM. |
| Threshold optimisation (Hardt 2016) and reweighing (Kamiran-Calders) reported in isolation. | Each intervention is **mapped to the pathway it targets**; combination tested against KC + threshold (proxy for Hou & Chen ADRL). |
| Single-dataset fairness studies. | Three datasets, three educational systems, one harmonised pipeline. |

## CLAIM — One Paragraph

The observed false-positive-rate disparity between intersectional subgroups in a mid-semester EWS is the same number, but a fundamentally different *thing*, across datasets: in THCSMK it is mostly mediated via historical inequality in grades and attendance (CtfIE), whereas in UCI-Por the mediated pathway partially offsets a direct shortcut (CtfDE) created by the SES feature itself. The **SES Inclusion Paradox** (adding SES features improves aggregate AUC but worsens subgroup FPR) is mechanistically explained: ΔCtfDE = +0.059 (THCSMK) / +0.069 (UCI-Por) while ΔCtfIE ≈ 0 — replicated across two distinct educational systems. The decomposition makes **pathway-targeted interventions** rational: I3 (per-group thresholds) suffices on UCI-Por (FPR spread −22%); KC reweighing + thresholds is needed on THCSMK (−29%). E-values ≥ 1.48 indicate the direct-effect findings are robust to moderate unobserved confounding.

## Q1-readiness (IEEE Access)

Strong:
- Genuine technical novelty (causal pathway decomposition is new to EDM)
- 3-dataset external validity
- Explicit head-to-head with Hou & Chen (2026) on the same UCI-Por dataset
- Story-telling structure (mask → mechanism → action → robustness)
- Self-proposed quantities derive from a published framework — not invented metrics

Needs attention before submission:
- Bootstrap from n_boot=80 (sandbox-limited) → n_boot=2000 (user's machine, ~6h)
- Tipping-point sensitivity in addition to E-values
- SCM specification expert-validated by a domain expert

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional but recommended) download and harmonise OULAD (~85 MB → ~5 MB)
python code/prepare_oulad.py

# 3. Run all blocks in story order (~10 min without OULAD; ~30 min with)
python code/run_all.py

# For paper-grade bootstrap CIs (~6 hours total):
python code/run_all.py --n-boot 2000 --n-mc 30
```

THCSMK and UCI-Por CSVs ship with this bundle.

---

## Layout

```
cf_final/
├── code/
│   ├── data_loaders.py            THCSMK / UCI-Por / OULAD with common schema
│   ├── prepare_oulad.py           downloads + harmonises OULAD
│   ├── metrics.py                 AUC, FPR, ECE, DeLong, bootstrap CI
│   ├── causal_estimator.py        CtfDE/CtfIE/CtfSE via g-computation
│   ├── run_block1.py              AGGREGATE vs SUBGROUP (Act 1: mask)
│   ├── run_block2_lean.py         CAUSAL DECOMPOSITION (Act 2: mechanism)
│   ├── run_block3_lean.py         SES INCLUSION PARADOX (Act 2 climax)
│   ├── run_block4.py              PATHWAY-TARGETED INTERVENTIONS (Act 3)
│   ├── run_block5_evalue.py       E-value sensitivity (Act 3 robustness)
│   ├── make_figures.py            all 6 paper figures
│   └── run_all.py                 master runner — story-telling sequence
├── data/
│   ├── THCSMK.csv                 shipped (n=675)
│   ├── student-por.csv            shipped (UCI Portuguese, n=649)
│   └── oulad_harmonised.csv       created by prepare_oulad.py (n≈32K)
├── results/                       CSV outputs from each block
├── figures/                       6 paper figures
├── docs/
│   ├── paper_section3.md          ready-to-use Section 3 manuscript
│   └── RESULTS_SUMMARY.md         one-page summary of all numbers
├── requirements.txt
└── README.md                      this file
```

## Execution Order (Story Arc)

1. **Block 1 – Discovery (Act 1).** Aggregate metrics look fine; **per-subgroup FPR** explodes (Male×HighSES = 0.85 on THCSMK). Aggregate fairness fails.
2. **Block 2 – Mechanism (Act 2).** Causal pathway decomposition. Sanity check passes (CtfDE = 0 for pure gender flips). Datasets share an observable disparity but have **opposite causal structures**.
3. **Block 3 – The Paradox Explained (Act 2 climax).** Adding SES features increases CtfDE by ~+0.06 on both datasets, ΔCtfIE ≈ 0. The paradox is a **direct-effect** phenomenon, replicated cross-culturally.
4. **Block 4 – Action (Act 3).** Pathway-targeted interventions; the best intervention is **dataset-specific** — exactly as the decomposition predicted.
5. **Block 5 – Robustness.** E-values 1.48–1.81 → moderate unmeasured confounding cannot eliminate the direct-effect findings.

---

## Citing

Underlying frameworks:
- Zhang, J. & Bareinboim, E. (2018). *Fairness in Decision-Making: The Causal Explanation Formula*. AAAI.
- VanderWeele, T. J. & Ding, P. (2017). *Sensitivity Analysis in Observational Research: Introducing the E-value*. Annals of Internal Medicine.
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4:170171.
- Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. (UCI-Por)
# The-Socioeconomic-Feature-Paradox-in-Educational-AI
