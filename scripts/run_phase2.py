#!/usr/bin/env python3
"""
Phase 2 Execution Pipeline: Causal Discovery & Directed Acyclic Graph (DAG) Generation
Based on Wyrembek et al. (2025), Section 3.3
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
import networkx as nx
import pandas as pd
import numpy as np

from src.models.dag_expert import ExpertDAGBuilder
from src.models.dag_hill_climbing import HillClimbingDAG
from src.models.dag_tabu_search import TabuSearchDAG
from src.models.dag_pc import PCDAG
from src.models.dag_evaluator import DAGEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("phase2_pipeline", log_file="reports/phase2_execution.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 2 Causal DAG Discovery Pipeline")
    parser.add_argument(
        "--input-data",
        type=str,
        default="data/processed/causal_supply_chain.parquet",
        help="Path to processed causal parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save DAG structures JSON"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/figures",
        help="Directory to save DAG visualization figures"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Sample size for score/constraint structure search (default: 10000)"
    )
    return parser.parse_args()

def visualize_dag_comparison(dags: dict, output_path: Path):
    """
    Renders a publication-quality 4-panel comparison figure matching Figures 3 & 4
    of Wyrembek et al. (2025).
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    titles = [
        "(a) Domain Expert Knowledge DAG (Figure 3)",
        "(b) Hill Climbing autoDAG [K2 Score] (Figure 4a)",
        "(c) Tabu Search autoDAG [BIC Score] (Figure 4b)",
        "(d) Peter-Clark (PC) autoDAG [Conditional Independence] (Figure 4c)"
    ]

    keys = ["Expert_DAG", "Hill_Climbing", "Tabu_Search", "PC_Algorithm"]

    # Consistent layout coordinates for comparable visualization
    pos_dict = {
        "Multi": (-1.0, 0.8),
        "Season": (0.0, 0.9),
        "Shipping": (1.0, 0.8),
        "Shipping_Expedited": (1.0, 0.8),
        "Market": (-0.8, 0.1),
        "Category": (-0.3, 0.0),
        "Customer_Segment": (0.3, 0.0),
        "Segment": (0.3, 0.0),
        "Quantity": (-0.6, -0.4),
        "Price": (0.6, -0.4),
        "Delay": (0.0, -0.9)
    }

    # Color scheme
    # Treatments: light green (#a1d99b), Outcome: light red/coral (#fc9272), Covariates: light blue (#9ecae1)
    def get_node_color(node):
        if "Delay" in node:
            return "#fc9272"  # Outcome
        elif node in ["Multi", "Season", "Shipping", "Shipping_Expedited"]:
            return "#a1d99b"  # Treatments
        else:
            return "#9ecae1"  # Covariates

    for idx, key in enumerate(keys):
        ax = axes[idx]
        G = dags[key]

        # Use predefined positions or fallback to spring layout
        pos = {node: pos_dict.get(node, (np.random.uniform(-1, 1), np.random.uniform(-1, 1))) for node in G.nodes()}

        node_colors = [get_node_color(n) for n in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=2800,
            edgecolors="black",
            linewidths=1.5
        )

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=10,
            font_weight="bold",
            font_family="sans-serif"
        )

        # Draw directed edges
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="#333333",
            width=1.6,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            min_source_margin=18,
            min_target_margin=18
        )

        ax.set_title(titles[idx], fontsize=13, fontweight="bold", pad=12)
        ax.axis("off")

        # Subtitle metrics
        shd_text = ""
        if key != "Expert_DAG":
            shd = DAGEvaluator.compute_shd(dags["Expert_DAG"], G)
            shd_text = f" | SHD to Expert: {shd}"
        ax.text(
            0.5, -0.02,
            f"Nodes: {G.number_of_nodes()} | Directed Edges: {G.number_of_edges()}{shd_text}",
            ha="center",
            transform=ax.transAxes,
            fontsize=10,
            fontstyle="italic"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved 4-panel DAG comparison figure to: {output_path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 2: CAUSAL DISCOVERY & DAG GENERATION")
    logger.info("=" * 60)

    # 1. Load preprocessed canonical data
    input_path = Path(args.input_data)
    if not input_path.exists():
        input_path = ROOT_DIR / "data" / "processed" / "causal_supply_chain.parquet"

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} records for structure discovery.")

    # 2. Prepare discrete dataset for structure learning
    discrete_data = HillClimbingDAG.prepare_discrete_data(
        df,
        sample_size=args.sample_size,
        random_state=42
    )
    logger.info(f"Discretized feature matrix shape: {discrete_data.shape}")

    # 3. Model 1: Domain Expert Knowledge DAG
    logger.info("--- Model 1: Constructing Domain Expert Knowledge DAG ---")
    expert_builder = ExpertDAGBuilder()
    # Align node naming with discrete dataset
    expert_dag_raw = expert_builder.build_dag()
    # Create aligned expert DAG matching discrete_data node names
    name_map = {
        "Shipping_Expedited": "Shipping",
        "Customer_Segment": "Segment"
    }
    expert_dag = nx.relabel_nodes(expert_dag_raw, name_map)
    # Ensure all nodes in discrete_data exist in expert_dag
    for node in discrete_data.columns:
        if node not in expert_dag:
            expert_dag.add_node(node)

    # 4. Model 2: Hill Climbing autoDAG (K2 score)
    logger.info("--- Model 2: Learning Hill Climbing autoDAG [K2 Score] ---")
    hc_learner = HillClimbingDAG(max_iter=30, scoring_method="k2")
    hc_dag = hc_learner.fit(discrete_data)

    # 5. Model 3: Tabu Search autoDAG (BIC score)
    logger.info("--- Model 3: Learning Tabu Search autoDAG [BIC Score] ---")
    tabu_learner = TabuSearchDAG(tabu_size=15, max_iter=25, max_no_improvement=6)
    # Optional seed edges matching paper: edges from treatment to outcome can be pre-defined
    tabu_dag = tabu_learner.fit(discrete_data, initial_edges=[("Shipping", "Delay")])

    # 6. Model 4: Peter-Clark (PC) autoDAG (Constraint-based)
    logger.info("--- Model 4: Learning Peter-Clark (PC) autoDAG [Conditional Independence] ---")
    pc_learner = PCDAG(alpha=0.05, indep_test="chisq")
    pc_dag = pc_learner.fit(discrete_data)

    # 7. Evaluate and Compare Graphs
    logger.info("--- Evaluating Graphs, Computing SHD & Checking Causal Identifiability ---")
    learned_dags = {
        "Hill_Climbing": hc_dag,
        "Tabu_Search": tabu_dag,
        "PC_Algorithm": pc_dag
    }
    all_dags = {"Expert_DAG": expert_dag, **learned_dags}

    evaluation_report = DAGEvaluator.evaluate_all(
        expert_dag=expert_dag,
        learned_dags=learned_dags,
        treatments=["Multi", "Season", "Shipping"],
        outcome="Delay"
    )

    # 8. Save DAG Structures and Evaluation JSON
    json_output_path = output_dir / "dag_structures.json"
    with open(json_output_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)
    logger.info(f"Saved DAG structural evaluation to: {json_output_path}")

    # 9. Render Comparative Visualization Figure
    fig_output_path = reports_dir / "06_causal_dag_comparison.png"
    visualize_dag_comparison(all_dags, fig_output_path)

    # 10. Summary Log
    logger.info("=" * 60)
    logger.info("PHASE 2 EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Expert DAG edges: {expert_dag.number_of_edges()}")
    logger.info(f"Hill Climbing DAG edges: {hc_dag.number_of_edges()} (SHD to Expert: {evaluation_report['expert_comparison']['Hill_Climbing']['shd_to_expert']})")
    logger.info(f"Tabu Search DAG edges: {tabu_dag.number_of_edges()} (SHD to Expert: {evaluation_report['expert_comparison']['Tabu_Search']['shd_to_expert']})")
    logger.info(f"PC Algorithm DAG edges: {pc_dag.number_of_edges()} (SHD to Expert: {evaluation_report['expert_comparison']['PC_Algorithm']['shd_to_expert']})")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
