"""Generate figures for the paper."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'figure.dpi': 110, 'savefig.dpi': 200
})

# ---------------------------------------------------------------
# Figure 1: Subgroup FPR with 95% bootstrap CI, both datasets
# ---------------------------------------------------------------
def fig1_subgroup_fpr():
    sub = pd.read_csv(RESULTS_DIR / "block1_subgroup.csv")
    sub = sub[(sub.model == "RF") & (sub.config == "SES-aware") & (sub.subgroup_code != -1)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ds_name in zip(axes, ["THCSMK", "UCI-Por"]):
        d = sub[sub.dataset == ds_name].sort_values("subgroup_code").reset_index(drop=True)
        x = np.arange(len(d))
        fpr = d["FPR"].values
        err_lo = fpr - d["FPR_lo"].values
        err_hi = d["FPR_hi"].values - fpr
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
        ax.bar(x, fpr, yerr=[err_lo, err_hi], color=colors, capsize=4, edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(d["subgroup"].values, rotation=15, ha='right')
        ax.set_title(f"{ds_name} (n={d['n'].sum()})")
        ax.set_ylabel("FPR @ τ=0.5")
        ax.axhline(d["FPR"].mean(), ls=":", color='gray', alpha=0.7, label="mean FPR")
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        # Annotate values
        for i, (f, n) in enumerate(zip(fpr, d["n"])):
            ax.text(i, f + err_hi[i] + 0.02, f"{f:.2f}\n(n={n})", ha='center', fontsize=8)
    plt.suptitle("Subgroup False Positive Rate (RF, SES-aware) with 95% Bootstrap CI", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_subgroup_fpr.png", bbox_inches='tight')
    plt.close()
    print("Saved fig1_subgroup_fpr.png")


# ---------------------------------------------------------------
# Figure 2: Causal decomposition — stacked horizontal bars
# ---------------------------------------------------------------
def fig2_decomposition():
    decomp = pd.read_csv(RESULTS_DIR / "block2_decomposition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, ds_name in zip(axes, ["THCSMK", "UCI-Por"]):
        d = decomp[decomp.dataset == ds_name].sort_values("a1_code").reset_index(drop=True)
        y = np.arange(len(d))
        labels = [f"{r.a1}\nvs {r.a0}" for _, r in d.iterrows()]
        ax.barh(y, d["CtfDE"].values, color='#d62728', label='CtfDE (direct)', alpha=0.85)
        ax.barh(y, d["CtfIE"].values, left=d["CtfDE"].values,
                color='#1f77b4', label='CtfIE (mediated)', alpha=0.85)
        ax.barh(y, d["CtfSE"].values, left=d["CtfDE"].values + d["CtfIE"].values,
                color='#2ca02c', label='CtfSE (spurious)', alpha=0.85)
        # TV markers
        for i, tv in enumerate(d["TV"].values):
            ax.plot([tv], [i], marker='D', color='black', markersize=8, zorder=5,
                    label='TV (observed)' if i == 0 else None)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title(f"{ds_name}")
        ax.set_xlabel("Effect on FPR (conditional on Y=0)")
        ax.grid(axis='x', alpha=0.3)
        if ax == axes[0]:
            ax.legend(loc='lower right', fontsize=8)
    plt.suptitle("Causal Decomposition of FPR Disparity\n(reference: Fem×LowSES)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_decomposition.png", bbox_inches='tight')
    plt.close()
    print("Saved fig2_decomposition.png")


# ---------------------------------------------------------------
# Figure 3: SES Paradox — Delta CtfDE vs Delta CtfIE
# ---------------------------------------------------------------
def fig3_ses_paradox():
    df = pd.read_csv(RESULTS_DIR / "block3_ses_paradox.csv")
    deltas = []
    for ds in df.dataset.unique():
        sub = df[df.dataset == ds]
        aware = sub[sub.config == "SES-aware"].iloc[0]
        unaware = sub[sub.config == "SES-unaware"].iloc[0]
        deltas.append({
            "dataset": ds,
            "dTV": aware.TV - unaware.TV,
            "dCtfDE": aware.CtfDE - unaware.CtfDE,
            "dCtfIE": aware.CtfIE - unaware.CtfIE,
            "dCtfSE": aware.CtfSE - unaware.CtfSE,
        })
    dd = pd.DataFrame(deltas)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(dd))
    width = 0.2
    ax.bar(x - 1.5*width, dd["dTV"],    width, label="ΔTV",    color='black',  alpha=0.7)
    ax.bar(x - 0.5*width, dd["dCtfDE"], width, label="ΔCtfDE", color='#d62728')
    ax.bar(x + 0.5*width, dd["dCtfIE"], width, label="ΔCtfIE", color='#1f77b4')
    ax.bar(x + 1.5*width, dd["dCtfSE"], width, label="ΔCtfSE", color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(dd["dataset"])
    ax.set_ylabel("Δ (SES-aware − SES-unaware)")
    ax.set_title("SES Inclusion Paradox: Where does the change go?\n"
                 "(Male×HighSES vs Fem×LowSES, conditional on Y=0)")
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    # Annotate
    for i, r in dd.iterrows():
        ax.annotate(f"{r.dCtfDE:+.3f}", (i - 0.5*width, r.dCtfDE),
                    ha='center', va='bottom' if r.dCtfDE >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_ses_paradox.png", bbox_inches='tight')
    plt.close()
    print("Saved fig3_ses_paradox.png")


# ---------------------------------------------------------------
# Figure 4: Intervention comparison — FPR spread
# ---------------------------------------------------------------
def fig4_interventions():
    df = pd.read_csv(RESULTS_DIR / "block4_interventions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    short_names = {
        "Baseline (SES-aware)": "Baseline",
        "I1: SES feature ablation": "I1: Ablation",
        "I2: Mediator reweighing": "I2: M-reweigh",
        "I1+I3: Ablation + thresholds": "I1+I3",
        "Baseline: KC Reweighing": "KC Reweigh",
    }
    def shorten(s):
        if "Per-group thresholds" in s and "KC" not in s:
            return "I3: τ-opt"
        if "ADRL" in s:
            return "KC+τ (≈ADRL)"
        return short_names.get(s, s)
    for ax, ds_name in zip(axes, ["THCSMK", "UCI-Por"]):
        d = df[df.dataset == ds_name].copy()
        d["short"] = d["method"].apply(shorten)
        x = np.arange(len(d))
        colors = []
        for s in d["short"]:
            if s.startswith("Baseline"):
                colors.append('gray')
            elif "I1" in s or "I2" in s or "I3" in s:
                colors.append('#d62728')
            else:
                colors.append('#1f77b4')
        ax.bar(x, d["FPR_spread"], color=colors, edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(d["short"], rotation=30, ha='right')
        ax.set_title(f"{ds_name}")
        ax.set_ylabel("FPR spread (max - min across subgroups)")
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(d["FPR_spread"]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=8)
    plt.suptitle("Fairness Intervention Comparison: FPR Spread Reduction\n"
                 "(Red = pathway-targeted, Blue = baseline methods)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_interventions.png", bbox_inches='tight')
    plt.close()
    print("Saved fig4_interventions.png")


# ---------------------------------------------------------------
# Figure 5: E-value sensitivity
# ---------------------------------------------------------------
def fig5_evalues():
    df = pd.read_csv(RESULTS_DIR / "block5_evalues.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{r.dataset}\n{r.a1} vs {r.a0}" for _, r in df.iterrows()]
    x = np.arange(len(df))
    width = 0.4
    ax.bar(x - width/2, df["E-value_CtfDE"], width, label="E-value(CtfDE)",
           color='#d62728', edgecolor='black', linewidth=0.6)
    ax.bar(x + width/2, df["E-value_CtfIE"], width, label="E-value(CtfIE)",
           color='#1f77b4', edgecolor='black', linewidth=0.6)
    ax.axhline(1.0, ls=":", color='black', alpha=0.6, label="E=1 (no effect)")
    ax.axhline(2.0, ls="--", color='red',   alpha=0.6, label="E=2 (moderate)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel("E-value (minimum confounder RR)")
    ax.set_title("Sensitivity to Unmeasured Confounding (VanderWeele & Ding 2017)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for i, (de, ie) in enumerate(zip(df["E-value_CtfDE"], df["E-value_CtfIE"])):
        ax.text(i - width/2, de + 0.05, f"{de:.2f}", ha='center', fontsize=8)
        ax.text(i + width/2, ie + 0.05, f"{ie:.2f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_evalues.png", bbox_inches='tight')
    plt.close()
    print("Saved fig5_evalues.png")


# ---------------------------------------------------------------
# Figure 6: DAG (manually drawn)
# ---------------------------------------------------------------
def fig6_dag():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis('off')
    # Cleaner layout: left-to-right
    nodes = {
        'A':    (1.0, 3.0, '$A$\n(gender×SES)', '#ffcccc'),
        'W':    (3.0, 5.0, '$W$\n(school, distance,\nimmigrant)', '#ddeedd'),
        'M1':   (5.0, 4.0, '$M_1$\n(academic\nmediators)',  '#cceeff'),
        'M2':   (5.0, 2.0, '$M_2$\n(behavioral\nmediators)','#cceeff'),
        'Yhat': (8.5, 3.0, r'$\hat{Y}$' + '\n(prediction)', '#eecccc'),
        'Y':    (8.5, 0.7, '$Y$\n(true outcome)', 'white'),
        'U':    (5.0, 0.3, '$U$ (unobserved)', '#dddddd'),
    }
    box_style = dict(boxstyle='round,pad=0.35', edgecolor='black', linewidth=1.2)
    for nm, (x, y, lab, col) in nodes.items():
        ax.text(x, y, lab, ha='center', va='center', fontsize=9.5,
                bbox=dict(facecolor=col, **box_style))

    def arrow(s, d, color='black', style='-', lw=1.2, shrinkA=28, shrinkB=28):
        x0, y0, _, _ = nodes[s]; x1, y1, _, _ = nodes[d]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    linestyle=style, lw=lw,
                                    shrinkA=shrinkA, shrinkB=shrinkB,
                                    connectionstyle='arc3,rad=0'))

    # Black arrows: causal paths to model prediction
    arrow('A','W'); arrow('A','M1'); arrow('A','M2')
    arrow('W','M1'); arrow('W','M2'); arrow('W','Yhat')
    arrow('M1','M2'); arrow('M1','Yhat'); arrow('M2','Yhat')
    # Red direct arrow (curved to avoid overlap)
    x0,y0,_,_ = nodes['A']; x1,y1,_,_ = nodes['Yhat']
    ax.annotate('', xy=(x1,y1-0.4), xytext=(x0,y0-0.4),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5,
                                shrinkA=28, shrinkB=28,
                                connectionstyle='arc3,rad=-0.25'))
    ax.text(4.5, 1.6, 'direct path\n(SES-aware only)', color='red',
            fontsize=8.5, ha='center', fontstyle='italic')

    # Dotted: paths to Y (true outcome)
    arrow('A','Y', color='gray', style=':', lw=0.9)
    arrow('M1','Y', color='gray', style=':', lw=0.9)
    arrow('M2','Y', color='gray', style=':', lw=0.9)
    # U → M1, M2, Y (unobserved confounder)
    arrow('U','M1', color='#999999', style='--', lw=0.9)
    arrow('U','M2', color='#999999', style='--', lw=0.9)
    arrow('U','Y',  color='#999999', style='--', lw=0.9)

    # Legend
    ax.text(0.5, 0.5,
            'Solid black: causal paths used by classifier\n'
            'Red:         direct discrimination path\n'
            'Dotted gray: paths to true outcome\n'
            'Dashed:      unobserved confounding',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor='gray',
                      facecolor='white', alpha=0.9))

    ax.set_title('Structural Causal Model for Mid-Semester EWS', fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_dag.png", bbox_inches='tight')
    plt.close()
    print("Saved fig6_dag.png")


if __name__ == "__main__":
    fig1_subgroup_fpr()
    fig2_decomposition()
    fig3_ses_paradox()
    fig4_interventions()
    fig5_evalues()
    fig6_dag()
    print(f"\nAll figures -> {FIG_DIR}")
