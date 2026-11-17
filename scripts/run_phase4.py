#!/usr/bin/env python3
"""
Phase 4 Execution Pipeline: Granular CATE (B-Splines), Policy Trees & Decision Support
Based on Wyrembek et al. (2025), Sections 3.4 & 4.2, Figures 5 & 6
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
from sklearn.tree import plot_tree
import seaborn as sns
import pandas as pd
import numpy as np

from doubleml import DoubleMLData, DoubleMLIRM
from xgboost import XGBClassifier, XGBRegressor

from src.models.cate_splines import CATESplineEstimator
from src.models.policy_tree import PolicyTreeEstimator
from src.utils.logger import setup_logger

logger = setup_logger("phase4_pipeline", log_file="reports/phase4_execution.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 4 Granular CATE & Policy Tree Pipeline")
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
        help="Directory to save Phase 4 models and JSON"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/figures",
        help="Directory to save CATE curves and Policy Tree figures"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25000,
        help="Sample size for training (default: 25000)"
    )
    return parser.parse_args()

def plot_cate_curves(curves_dict: dict, output_path: Path):
    """
    Renders 4-panel publication-grade 1-D CATE curves with 95% confidence bands
    reproducing Figure 5 of Wyrembek et al. (2025).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid", font_scale=1.05)

    subtitles = [
        "(a) CATE θ(Quantity) for Shipping Mode Intervention",
        "(b) CATE θ(Quantity) for Multi-Market Sourcing",
        "(c) CATE θ(Quantity) for Q1 vs Q4 Seasonal Timing",
        "(d) CATE θ(Price) for Shipping Mode Intervention"
    ]

    keys = ["shipping_qty", "multi_qty", "season_qty", "shipping_price"]

    for idx, key in enumerate(keys):
        ax = axes[idx]
        df_curve = curves_dict[key]
        x_col = [c for c in df_curve.columns if c not in ["cate", "se", "ci_lower", "ci_upper"]][0]

        ax.plot(df_curve[x_col], df_curve["cate"], color="#1f77b4", linewidth=2.5, label="CATE Estimate θ(x)")
        ax.fill_between(
            df_curve[x_col],
            df_curve["ci_lower"],
            df_curve["ci_upper"],
            color="#1f77b4",
            alpha=0.2,
            label="95% Asymptotic Confidence Band"
        )
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8, label="Null Effect")

        ax.set_title(subtitles[idx], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(f"Order {x_col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Treatment Effect on Delay (Days)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved 4-panel CATE B-splines curves to: {output_path}")

def plot_policy_trees(ptree_shipping: PolicyTreeEstimator, ptree_multi: PolicyTreeEstimator, output_path: Path):
    """
    Renders 2-panel policy trees reproducing Figure 6 of Wyrembek et al. (2025).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Shipping Mode Policy Tree
    tree1 = ptree_shipping.policy_tree_obj.policy_tree
    plot_tree(
        tree1,
        feature_names=ptree_shipping.feature_names,
        class_names=["Standard Class (0)", "Expedited Mode (1)"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=axes[0]
    )
    axes[0].set_title("(a) Policy Tree: Shipping Mode Allocation Rule", fontsize=12, fontweight="bold", pad=12)

    # Panel 2: Sourcing Network Policy Tree
    tree2 = ptree_multi.policy_tree_obj.policy_tree
    plot_tree(
        tree2,
        feature_names=ptree_multi.feature_names,
        class_names=["Single-Market (0)", "Multi-Market (1)"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=axes[1]
    )
    axes[1].set_title("(b) Policy Tree: Sourcing Network Allocation Rule", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved 2-panel Policy Trees figure to: {output_path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 4: GRANULAR CATE, POLICY TREES & DECISION SUPPORT")
    logger.info("=" * 60)

    # 1. Load preprocessed encoded dataset
    input_path = Path(args.input_data)
    if not input_path.exists():
        input_path = ROOT_DIR / "data" / "processed" / "causal_supply_chain_encoded.parquet"

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)

    if args.sample_size and len(df) > args.sample_size:
        logger.info(f"Subsampling to {args.sample_size:,} records for CATE and Policy Tree optimization...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    all_treatments = ["Multi", "season_q1", "season_q2", "season_q3", "season_q4", "Shipping_Expedited"]
    outcomes = ["delay_days", "late_risk"]
    base_covs = [c for c in df.columns if c not in outcomes and c not in all_treatments]

    # Fast nuisance parameters for DoubleMLIRM
    reg_params = {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.7, "n_jobs": 4, "random_state": 42}
    clf_params = {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.7, "n_jobs": 4, "eval_metric": "logloss", "random_state": 42}

    # 2. Fit Base IRM for Shipping_Expedited
    logger.info("--- Fitting IRM for Shipping Mode Intervention ---")
    data_shipping = DoubleMLData(df, y_col="delay_days", d_cols="Shipping_Expedited", x_cols=base_covs + ["Multi", "season_q1"])
    irm_shipping = DoubleMLIRM(data_shipping, XGBRegressor(**reg_params), XGBClassifier(**clf_params), n_folds=3)
    irm_shipping.fit(store_predictions=True)

    # 3. Fit Base IRM for Multi Sourcing
    logger.info("--- Fitting IRM for Multi-Market Sourcing Intervention ---")
    data_multi = DoubleMLData(df, y_col="delay_days", d_cols="Multi", x_cols=base_covs + ["Shipping_Expedited", "season_q1"])
    irm_multi = DoubleMLIRM(data_multi, XGBRegressor(**reg_params), XGBClassifier(**clf_params), n_folds=3)
    irm_multi.fit(store_predictions=True)

    # 4. Fit Base IRM for Season (Q1 vs Rest)
    logger.info("--- Fitting IRM for Seasonal Timing Intervention ---")
    data_season = DoubleMLData(df, y_col="delay_days", d_cols="season_q1", x_cols=base_covs + ["Shipping_Expedited", "Multi"])
    irm_season = DoubleMLIRM(data_season, XGBRegressor(**reg_params), XGBClassifier(**clf_params), n_folds=3)
    irm_season.fit(store_predictions=True)

    # 5. Granular CATE Estimation with B-Splines (Figure 5)
    logger.info("--- Estimating B-Splines CATE Curves across Quantity & Price ---")
    spline_shipping_qty = CATESplineEstimator(df_degrees=5, spline_degree=3).fit(irm_shipping, df, "Quantity")
    curve_shipping_qty = spline_shipping_qty.predict_curve(grid_values=np.linspace(1, 5, 50))

    spline_multi_qty = CATESplineEstimator(df_degrees=5, spline_degree=3).fit(irm_multi, df, "Quantity")
    curve_multi_qty = spline_multi_qty.predict_curve(grid_values=np.linspace(1, 5, 50))

    spline_season_qty = CATESplineEstimator(df_degrees=5, spline_degree=3).fit(irm_season, df, "Quantity")
    curve_season_qty = spline_season_qty.predict_curve(grid_values=np.linspace(1, 5, 50))

    spline_shipping_price = CATESplineEstimator(df_degrees=5, spline_degree=3).fit(irm_shipping, df, "Price")
    curve_shipping_price = spline_shipping_price.predict_curve(grid_values=np.linspace(10, 400, 50))

    curves_dict = {
        "shipping_qty": curve_shipping_qty,
        "multi_qty": curve_multi_qty,
        "season_qty": curve_season_qty,
        "shipping_price": curve_shipping_price
    }

    # Render Figure 5
    cate_fig_path = reports_dir / "09_cate_bsplines_curves.png"
    plot_cate_curves(curves_dict, cate_fig_path)

    # 6. Induce Deterministic Policy Trees (Figure 6)
    logger.info("--- Inducing Deterministic Binary Treatment Policy Trees ---")
    policy_features = df[["Quantity", "Price"]]

    ptree_shipping = PolicyTreeEstimator(depth=2, ccp_alpha=0.005).fit(
        irm_shipping, policy_features, treatment_name="Shipping_Expedited"
    )
    ptree_multi = PolicyTreeEstimator(depth=2, ccp_alpha=0.005).fit(
        irm_multi, policy_features, treatment_name="Multi_Sourcing"
    )

    # Render Figure 6
    tree_fig_path = reports_dir / "10_policy_trees_visualization.png"
    plot_policy_trees(ptree_shipping, ptree_multi, tree_fig_path)

    # 7. Save Phase 4 Models and Metadata
    json_path = output_dir / "phase4_cate_models.json"
    phase4_results = {
        "phase": "Phase 4: Granular CATE, Policy Trees & Decision Support",
        "sample_size": len(df),
        "cate_splines": {
            "shipping_qty": spline_shipping_qty.get_summary(),
            "multi_qty": spline_multi_qty.get_summary(),
            "season_qty": spline_season_qty.get_summary(),
            "shipping_price": spline_shipping_price.get_summary()
        },
        "policy_trees": {
            "shipping": ptree_shipping.get_tree_structure(),
            "multi": ptree_multi.get_tree_structure()
        },
        "artifacts": {
            "cate_curves_png": str(cate_fig_path),
            "policy_trees_png": str(tree_fig_path)
        }
    }
    with open(json_path, "w") as f:
        json.dump(phase4_results, f, indent=4)
    logger.info(f"Saved Phase 4 metadata to: {json_path}")

    # 8. Print Rules Summary
    logger.info("=" * 60)
    logger.info("PHASE 4 EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("Prescriptive Decision Policy Rules:")
    logger.info(f"\nPolicy Tree 1 (Shipping Mode):\n{ptree_shipping.export_text_rules()}")
    logger.info(f"\nPolicy Tree 2 (Multi-Market Sourcing):\n{ptree_multi.export_text_rules()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
