from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from src.utils.logger import setup_logger

logger = setup_logger("dag_pc")

class PCDAG:
    """
    Learns causal DAG structure using constraint-based Peter-Clark (PC) algorithm
    via conditional independence testing and Meek orientation rules, as specified in
    Section 3.3.3 of Wyrembek et al. (2025).
    """

    def __init__(self, alpha: float = 0.05, indep_test: str = "chisq"):
        self.alpha = alpha
        self.indep_test = indep_test
        self.graph = nx.DiGraph()

    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Executes PC algorithm:
        1. Complete undirected skeleton.
        2. Conditional independence testing to prune indirect edges.
        3. V-structure detection and orientation rules.
        4. Resolves any remaining undirected edges into a valid DAG.
        """
        node_names = list(data.columns)
        logger.info(
            f"Fitting PC algorithm on {data.shape[0]} samples with {len(node_names)} nodes, "
            f"alpha={self.alpha}, indep_test='{self.indep_test}'..."
        )

        # Convert categorical data to integer codes for causal-learn
        encoded_mat = np.zeros(data.shape, dtype=int)
        for col_idx, col in enumerate(node_names):
            if data[col].dtype.name == "category" or data[col].dtype == object:
                encoded_mat[:, col_idx] = pd.Categorical(data[col]).codes
            else:
                encoded_mat[:, col_idx] = data[col].values

        # Run PC
        cg = pc(
            encoded_mat,
            alpha=self.alpha,
            indep_test=self.indep_test,
            node_names=node_names,
            show_progress=False
        )

        # Extract graph edges from causal-learn Graph object
        dag = nx.DiGraph()
        dag.add_nodes_from(node_names)

        # In causal-learn:
        # endpoint == -1 (tail), 1 (arrow)
        # edge.get_endpoint1() and edge.get_endpoint2()
        graph_edges = cg.G.get_graph_edges()
        for edge in graph_edges:
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            ep1 = edge.get_endpoint1()
            ep2 = edge.get_endpoint2()

            # Arrow pointing to node 2: u -> v
            if ep1 == -1 and ep2 == 1:
                dag.add_edge(u, v)
            # Arrow pointing to node 1: v -> u
            elif ep1 == 1 and ep2 == -1:
                dag.add_edge(v, u)
            # Undirected edge (ep1 == -1 and ep2 == -1): orient toward delay/downstream
            else:
                if v == "Delay":
                    dag.add_edge(u, v)
                elif u == "Delay":
                    dag.add_edge(v, u)
                else:
                    # Orient without creating cycles
                    dag.add_edge(u, v)
                    if not nx.is_directed_acyclic_graph(dag):
                        dag.remove_edge(u, v)
                        dag.add_edge(v, u)

        # Final acyclicity safety check
        if not nx.is_directed_acyclic_graph(dag):
            # Break any remaining cycles by removing feedback edges
            cycles = list(nx.simple_cycles(dag))
            for cycle in cycles:
                if dag.has_edge(cycle[-1], cycle[0]):
                    dag.remove_edge(cycle[-1], cycle[0])

        self.graph = dag
        logger.info(
            f"PC DAG learned: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} directed edges. Is DAG: {nx.is_directed_acyclic_graph(self.graph)}"
        )
        return self.graph
