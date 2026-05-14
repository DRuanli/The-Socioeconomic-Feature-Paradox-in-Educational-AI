"""
MASTER EXPERIMENT RUNNER  —  story-telling sequence
====================================================

Run end-to-end:  python code/run_all.py

Each block produces results that motivate the next.  Output is saved
incrementally to results/  so a partial run is still useful.

The narrative arc:
  Act 1 (Discovery)
    Block 1: Aggregate fairness metrics look fine
           -> but subgroup decomposition reveals MASKED disparity.
  Act 2 (Mechanism)
    Block 2: Causal pathway decomposition shows different datasets have
             different causes for the same observed disparity.
    Block 3: The mysterious "SES Inclusion Paradox" is explained -
             adding SES creates a NEW direct shortcut (not a worse mediator).
  Act 3 (Action)
    Block 4: Pathway-targeted interventions outperform one-size-fits-all
             precisely BECAUSE pathway composition differs.
    Block 5: Findings robust to plausible unmeasured confounding.

Datasets:
  THCSMK  (Vietnam,   n=675,   primary)         shipped in data/
  UCI-Por (Portugal,  n=649,   head-to-head)    shipped in data/
  OULAD   (UK,        n=~32K,  external valid.) run prepare_oulad.py first
"""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)


def header(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def main(skip_oulad=False, n_boot=80, n_mc=15):
    from data_loaders import load_thcsmk, load_uci_por
    DATASETS = [(load_thcsmk, "THCSMK"), (load_uci_por, "UCI-Por")]
    if not skip_oulad:
        try:
            from data_loaders import load_oulad
            _ = load_oulad()
            DATASETS.append((load_oulad, "OULAD"))
            print("OULAD detected, will include in all blocks.")
        except FileNotFoundError as e:
            print(f"\n[OULAD skipped] {e}")
            print("Continuing with THCSMK + UCI-Por only.\n")

    t_total = time.time()

    # =============================================================
    header("ACT 1: AGGREGATE METRICS MASK SUBGROUP DISPARITY")
    # =============================================================
    print("Block 1: 5-fold CV, RF + LR, SES-aware vs SES-unaware, "
          "per-subgroup FPR/AUC with 95% bootstrap CIs and DeLong tests.")
    from run_block1 import main as block1_main
    block1_main()

    # =============================================================
    header("ACT 2 (Mechanism): CAUSAL PATHWAY DECOMPOSITION")
    # =============================================================
    print("Block 2: Decompose FPR disparity into direct + mediated + spurious "
          "(Zhang & Bareinboim 2018). Each pair of intersectional subgroups, "
          "n_boot=" + str(n_boot) + " stratified bootstrap CI.")
    import pandas as pd
    all_b2 = []
    from run_block2_lean import run_dataset
    for loader, name in DATASETS:
        rows = run_dataset(loader, name, n_boot=n_boot, n_mc=n_mc)
        all_b2.extend(rows)
        pd.DataFrame(all_b2).to_csv(RESULTS / "block2_decomposition.csv", index=False)

    print("\nBlock 3: SES Inclusion Paradox via decomposition. "
          "Re-train with vs without SES feature; show Delta CtfDE vs Delta CtfIE.")
    from run_block3_lean import run as block3_run
    all_b3 = []
    for loader, name in DATASETS:
        rows = block3_run(loader, name, n_boot=n_boot, n_mc=n_mc)
        all_b3.extend(rows)
        pd.DataFrame(all_b3).to_csv(RESULTS / "block3_ses_paradox.csv", index=False)

    # =============================================================
    header("ACT 3 (Action): PATHWAY-TARGETED INTERVENTIONS")
    # =============================================================
    print("Block 4: Compare baselines (none, Kamiran-Calders, ADRL-proxy) "
          "vs pathway-targeted interventions (I1, I2, I3 and combinations).")
    from run_block4 import run as block4_run
    all_b4 = []
    for loader, name in DATASETS:
        out = block4_run(loader, name)
        all_b4.append(out)
    pd.concat(all_b4, ignore_index=True).to_csv(RESULTS / "block4_interventions.csv", index=False)

    print("\nBlock 5: E-value sensitivity (VanderWeele & Ding 2017) for "
          "every CtfDE / CtfIE estimate.")
    from run_block5_evalue import main as block5_main
    block5_main()

    # =============================================================
    header("FIGURES")
    # =============================================================
    from make_figures import (fig1_subgroup_fpr, fig2_decomposition,
                              fig3_ses_paradox, fig4_interventions,
                              fig5_evalues, fig6_dag)
    fig1_subgroup_fpr(); fig2_decomposition(); fig3_ses_paradox()
    fig4_interventions(); fig5_evalues(); fig6_dag()

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    print(f"Results saved to {RESULTS}")
    print(f"Figures saved to {FIGS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-oulad", action="store_true",
                    help="Skip OULAD even if available")
    ap.add_argument("--n-boot", type=int, default=80,
                    help="Bootstrap replicates per decomposition (default 80; "
                         "increase to 2000+ for paper)")
    ap.add_argument("--n-mc", type=int, default=15,
                    help="Monte Carlo mediator samples (default 15; 30+ for paper)")
    args = ap.parse_args()
    main(skip_oulad=args.skip_oulad, n_boot=args.n_boot, n_mc=args.n_mc)
