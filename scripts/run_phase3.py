#!/usr/bin/env python3
"""
Phase 3 Execution Pipeline: Double Machine Learning (DML) & Interactive Regression Model (IRM)
Based on Wyrembek et al. (2025), Sections 3.4 & 4.1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure MPLCONFIGDIR is set to writable directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.models.dml_runner import DMLBatchRunner
from src.utils.logger import setup_logger

logger = setup_logger("phase3_pipeline", log_file="reports/phase3_execution.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3 Double ML IRM Pipeline")
    parser.add_argument(
        "--input-data",
        type=str,
        default="data/processed/causal_supply_chain_encoded.parquet",
        help="Path to processed causal encoded parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save DML results CSV and JSON"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/figures",
        help="Directory to save DML forest plots and propensity overlap figures"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Sample size for training nuisance models (default: 50000)"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-fitting folds (default: 5)"
    )
    return parser.parse_args()

def plot_forest_ate(results_df: pd.DataFrame, output_path: Path):
    """
    Renders publication-grade Forest Plot of Average Treatment Effects (ATE)
    with 95% confidence intervals, matching the empirical reporting in Table 5.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    y_pos = np.arange(len(results_df))
    labels = results_df["treatment_label"].values
    coefs = results_df["coef"].values
    ci_low = results_df["ci_95_lower"].values
    ci_high = results_df["ci_95_upper"].values

    # Determine colors: Green for delay-reducing (negative), Red/Orange for delay-increasing (positive)
    colors = ["#2ca02c" if c < 0 else "#d62728" for c in coefs]

    # Draw zero reference line
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Null Effect (θ = 0)")

    # Plot error bars and points
    for idx in range(len(results_df)):
        err_low = coefs[idx] - ci_low[idx]
        err_high = ci_high[idx] - coefs[idx]
        ax.errorbar(
            coefs[idx],
            y_pos[idx],
            xerr=[[err_low], [err_high]],
            fmt='o',
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=2.2,
            capsize=5,
            markersize=8,
            capthick=2
        )
        # Add value label
        offset = 0.05 if coefs[idx] >= 0 else -0.05
        ha = "left" if coefs[idx] >= 0 else "right"
        ax.text(
            coefs[idx] + offset,
            y_pos[idx] + 0.15,
            f"{coefs[idx]:+.3f}d (p={results_df['p_val'].values[idx]:.1e})",
            ha=ha,
            fontsize=9.5,
            fontweight="bold",
            color=colors[idx]
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel("Average Treatment Effect on Delay Duration (Days)", fontsize=12, fontweight="bold")
    ax.set_title("Double Machine Learning (DML-IRM): Average Treatment Effects (ATE)", fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved ATE forest plot to: {output_path}")

def plot_propensity_overlap(fitted_models: dict, output_path: Path):
    """
    Renders multi-panel propensity score overlap distributions checking
    the positivity / common support assumption for all estimated treatments.
    """
    n_treatments = len(fitted_models)
    cols = 3
    rows = (n_treatments + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
    axes = np.array(axes).flatten()

    for idx, (treatment, model) in enumerate(fitted_models.items()):
        ax = axes[idx]
        treated_ps, control_ps = model.get_propensity_scores()

        sns.kdeplot(treated_ps, ax=ax, label="Treated (D=1)", color="#d62728", fill=True, alpha=0.35, linewidth=1.5)
        sns.kdeplot(control_ps, ax=ax, label="Control (D=0)", color="#1f77b4", fill=True, alpha=0.35, linewidth=1.5)

        ax.set_title(f"Propensity Overlap: {treatment}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Estimated Propensity Score P(D=1|X)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", fontsize=9)

    # Hide unused axes
    for idx in range(n_treatments, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved propensity overlap plot to: {output_path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 3: DOUBLE MACHINE LEARNING (DML) & IRM")
    logger.info("=" * 60)

    # 1. Load preprocessed encoded dataset
    input_path = Path(args.input_data)
    if not input_path.exists():
        input_path = ROOT_DIR / "data" / "processed" / "causal_supply_chain_encoded.parquet"

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} records with {df.shape[1]} columns.")

    if args.sample_size and len(df) > args.sample_size:
        logger.info(f"Subsampling to {args.sample_size:,} records for high-fidelity cross-fitting...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # 2. Target Treatments to estimate (matching & extending Table 5)
    target_treatments = [
        "Multi",
        "season_q1",
        "season_q2",
        "season_q3",
        "season_q4",
        "Shipping_Expedited"
    ]

    # Hyperparameters from Table 4
    reg_params = {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "n_jobs": 4,
        "random_state": 42
    }
    clf_params = {
        "n_estimators": 250,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "n_jobs": 4,
        "eval_metric": "logloss",
        "random_state": 42
    }

    # 3. Execute Batch DML
    runner = DMLBatchRunner(
        n_folds=args.n_folds,
        regressor_params=reg_params,
        classifier_params=clf_params
    )

    results_df = runner.run_all_treatments(
        df=df,
        treatments=target_treatments,
        y_col="delay_days"
    )

    # 4. Save results to CSV and JSON
    csv_path = output_dir / "phase3_treatment_effects.csv"
    json_path = output_dir / "phase3_dml_results.json"

    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved treatment effects table to: {csv_path}")

    full_results = {
        "framework": "Double Machine Learning Interactive Regression Model (DML-IRM)",
        "reference_paper": "Wyrembek et al. (2025), Sections 3.4, 4.1, Table 5",
        "sample_size": len(df),
        "n_folds": args.n_folds,
        "treatment_effects": results_df.to_dict(orient="records"),
        "hyperparameters": {
            "regressor": reg_params,
            "classifier": clf_params
        }
    }
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=4)
    logger.info(f"Saved full DML JSON results to: {json_path}")

    # 5. Generate Visual Diagnostic Figures
    forest_path = reports_dir / "07_dml_ate_coefficients.png"
    overlap_path = reports_dir / "08_propensity_score_overlap.png"

    plot_forest_ate(results_df, forest_path)
    plot_propensity_overlap(runner.fitted_models, overlap_path)

    # 6. Console Summary
    logger.info("=" * 60)
    logger.info("PHASE 3 EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("Average Treatment Effects (Replicating Table 5):")
    for _, row in results_df.iterrows():
        sig = "***" if row["p_val"] < 0.001 else ("**" if row["p_val"] < 0.01 else ("*" if row["p_val"] < 0.05 else "n.s."))
        logger.info(
            f"  {row['treatment_label']:<38} | Coef: {row['coef']:+7.3f}d | "
            f"SE: {row['std_err']:6.3f} | t: {row['t_stat']:+6.2f} | p: {row['p_val']:.2e} {sig}"
        )
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
