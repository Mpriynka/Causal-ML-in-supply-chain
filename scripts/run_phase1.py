#!/usr/bin/env python3
"""
Phase 1 Execution Pipeline: Data Preparation, Feature Engineering & Causal Framing
Based on Wyrembek et al. (2025)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.causal_builder import CausalFeatureBuilder
from src.utils.logger import setup_logger

logger = setup_logger("phase1_pipeline", log_file="reports/phase1_execution.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 Causal ML Preparation Pipeline")
    parser.add_argument(
        "--raw-data",
        type=str,
        default="archive/DataCoSupplyChainDataset.csv",
        help="Path to raw DataCoSupplyChainDataset.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed causal datasets"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/figures",
        help="Directory to save diagnostic and EDA figures"
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row limit for quick testing (default: None for full dataset)"
    )
    return parser.parse_args()

def generate_eda_plots(df: pd.DataFrame, causal_df: pd.DataFrame, output_dir: Path):
    """Generates publication-quality diagnostic plots matching the paper's empirical analyses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Delay Distribution & Late Delivery Risk
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Delay Days Histogram & KDE
    sns.histplot(
        causal_df["delay_days"],
        bins=np.arange(-2.5, 5.5, 1),
        kde=False,
        ax=axes[0],
        color="#2b5c8f",
        edgecolor="black"
    )
    axes[0].set_title("Distribution of Delivery Delay (Days)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Delay Duration (Days: Real - Scheduled)")
    axes[0].set_ylabel("Order Count")
    axes[0].axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Scheduled Delivery Date")
    axes[0].legend()

    # Late Delivery Risk Proportion
    late_counts = causal_df["late_risk"].value_counts(normalize=True) * 100
    bars = axes[1].bar(["On-Time / Early (0)", "Late Delivery (1)"], late_counts.values, color=["#3ca048", "#d9534f"], edgecolor="black")
    axes[1].set_title("Proportion of Delayed Orders (Outcome Y)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Percentage (%)")
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    axes[1].set_ylim(0, 70)

    plt.tight_layout()
    fig_path = output_dir / "01_delay_distribution.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # 2. Seasonal Delay Analysis (Treatment D_season)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Delay rate by Season
    season_late = causal_df.groupby("Season")["late_risk"].mean() * 100
    sns.barplot(x=season_late.index, y=season_late.values, ax=axes[0], palette="Blues_d", edgecolor="black")
    axes[0].set_title("Late Delivery Rate by Order Season (Quarter)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Late Order Rate (%)")
    axes[0].set_xlabel("Season")
    for i, v in enumerate(season_late.values):
        axes[0].text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom', fontweight='bold')
    axes[0].set_ylim(0, 65)

    # Delay duration by Season
    sns.barplot(x="Season", y="delay_days", data=causal_df, ax=axes[1], palette="Purples_d", edgecolor="black", errorbar=None)
    axes[1].set_title("Average Delay Duration (Days) by Season", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Mean Delay (Days)")
    axes[1].set_xlabel("Season")
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    fig_path = output_dir / "02_seasonal_delay_effects.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # 3. Multi-Market Sourcing Intervention (Treatment D_multi)
    fig, ax = plt.subplots(figsize=(8, 5))
    multi_agg = causal_df.groupby("Multi").agg(
        mean_delay=("delay_days", "mean"),
        late_rate=("late_risk", lambda x: x.mean() * 100)
    )
    labels = ["Single-Market (Multi=0)", "Multi-Market Network (Multi=1)"]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, multi_agg["mean_delay"], width, label="Mean Delay (Days)", color="#4575b4", edgecolor="black")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, multi_agg["late_rate"], width, label="Late Delivery Risk (%)", color="#d73027", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Mean Delay (Days)", color="#4575b4", fontweight="bold")
    ax2.set_ylabel("Late Delivery Risk (%)", color="#d73027", fontweight="bold")
    ax.set_title("Delivery Risk Contrast: Single vs Multi-Market Sourcing", fontsize=13, fontweight="bold")
    ax.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    fig_path = output_dir / "03_multi_market_treatment.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = [
        "delay_days", "late_risk", "Multi", "season_q1", "season_q2", "season_q3", "season_q4",
        "Shipping_Expedited", "Quantity", "Price", "Discount_Rate", "Order_Total"
    ]
    corr = causal_df[numeric_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Heatmap: Outcomes, Treatments & Covariates", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = output_dir / "04_feature_correlation_matrix.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {fig_path}")

    # 5. Shipping Mode Delay Contrast
    fig, ax = plt.subplots(figsize=(10, 5))
    if "Shipping Mode" in df.columns:
        sns.boxplot(
            x="Shipping Mode",
            y="delay_days",
            data=df,
            ax=ax,
            palette="Set2"
        )
        ax.set_title("Shipping Duration Variance across Shipping Modes", fontsize=13, fontweight="bold")
        ax.set_ylabel("Delay Duration (Days)")
        plt.tight_layout()
        fig_path = output_dir / "05_shipping_mode_delay.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        logger.info(f"Saved: {fig_path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 1: DATA PREPARATION & CAUSAL FRAMING")
    logger.info("=" * 60)

    # 1. Load raw data
    raw_path = Path(args.raw_data)
    if not raw_path.exists():
        # Fallback to archive path if relative path differs
        raw_path = ROOT_DIR / "archive" / "DataCoSupplyChainDataset.csv"
    
    loader = DataLoader(str(raw_path))
    raw_df, audit_info = loader.load_raw_data(nrows=args.sample_rows)

    # 2. Clean and preprocess
    preprocessor = DataPreprocessor(keep_outliers=True)
    cleaned_df, cleaning_stats = preprocessor.clean_and_transform(raw_df)

    # 3. Build Causal Features
    builder = CausalFeatureBuilder()
    causal_df, causal_meta = builder.build_causal_dataset(cleaned_df)
    encoded_df, encoding_meta = builder.build_encoded_feature_matrix(causal_df)

    # 4. Generate EDA & Diagnostic Plots
    logger.info("Generating EDA and diagnostic causal figures...")
    generate_eda_plots(cleaned_df, causal_df, reports_dir)

    # 5. Save Processed Artifacts
    logger.info(f"Saving processed datasets to: {output_dir}")
    parquet_path = output_dir / "causal_supply_chain.parquet"
    encoded_parquet_path = output_dir / "causal_supply_chain_encoded.parquet"
    sample_csv_path = output_dir / "causal_dataset_sample.csv"
    meta_json_path = output_dir / "causal_features_summary.json"

    causal_df.to_parquet(parquet_path, index=False)
    encoded_df.to_parquet(encoded_parquet_path, index=False)
    causal_df.head(1000).to_csv(sample_csv_path, index=False)

    full_summary = {
        "pipeline_phase": "Phase 1: Preparation & Causal Framing",
        "reference_paper": "Wyrembek, Baryannis, & Brintrup (2025), IJPR",
        "dataset_name": "DataCo Smart Supply Chain Dataset",
        "audit_info": audit_info,
        "cleaning_stats": cleaning_stats,
        "causal_metadata": causal_meta,
        "encoding_metadata": encoding_meta,
        "statistical_summary": {
            "total_records": len(causal_df),
            "delay_days_mean": float(causal_df["delay_days"].mean()),
            "delay_days_std": float(causal_df["delay_days"].std()),
            "delay_days_min": float(causal_df["delay_days"].min()),
            "delay_days_max": float(causal_df["delay_days"].max()),
            "late_risk_rate": float(causal_df["late_risk"].mean()),
            "multi_treatment_mean": float(causal_df["Multi"].mean()),
            "shipping_expedited_mean": float(causal_df["Shipping_Expedited"].mean()),
            "seasonal_distribution": causal_df["Season"].value_counts().to_dict(),
            "mean_delay_by_season": causal_df.groupby("Season")["delay_days"].mean().to_dict(),
            "mean_delay_by_multi": causal_df.groupby("Multi")["delay_days"].mean().to_dict(),
            "mean_delay_by_shipping": causal_df.groupby("Shipping_Expedited")["delay_days"].mean().to_dict()
        },
        "output_artifacts": {
            "causal_dataset_parquet": str(parquet_path),
            "encoded_dataset_parquet": str(encoded_parquet_path),
            "sample_csv": str(sample_csv_path),
            "summary_json": str(meta_json_path)
        }
    }

    with open(meta_json_path, "w") as f:
        json.dump(full_summary, f, indent=4)
    logger.info(f"Saved full metadata summary to: {meta_json_path}")

    logger.info("=" * 60)
    logger.info("PHASE 1 EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Processed: {len(causal_df):,} records")
    logger.info(f"Delayed rate: {cleaning_stats['late_delivery_pct']}%")
    logger.info(f"Average delay duration: {cleaning_stats['mean_delay_days']} days")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
