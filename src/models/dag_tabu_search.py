from typing import Dict, List, Tuple, Set, Optional
from collections import deque
import itertools
import pandas as pd
import networkx as nx
from pgmpy.structure_score import BIC
from src.utils.logger import setup_logger

logger = setup_logger("dag_tabu_search")

class TabuSearchDAG:
    """
    Learns Bayesian Network DAG structure using Tabu Search metaheuristic with BIC score,
    implementing the 8-step algorithm from Section 3.3.2 of Wyrembek et al. (2025).
    """

    def __init__(
        self,
        tabu_size: int = 15,
        max_iter: int = 30,
        max_no_improvement: int = 8
    ):
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.graph = nx.DiGraph()

    def _get_legal_moves(self, current_dag: nx.DiGraph, nodes: List[str]) -> List[Tuple[str, Tuple[str, str]]]:
        """
        Generates valid DAG neighborhood moves:
        - ('+', (u, v)): add directed edge u -> v
        - ('-', (u, v)): remove directed edge u -> v
        - ('flip', (u, v)): reverse edge u -> v to v -> u
        Ensures acyclicity is strictly preserved.
        """
        moves = []
        existing_edges = set(current_dag.edges())

        # 1. Edge removals
        for u, v in existing_edges:
            moves.append(("-", (u, v)))

        # 2. Edge reversals
        for u, v in existing_edges:
            # Check if reversing creates cycle
            current_dag.remove_edge(u, v)
            current_dag.add_edge(v, u)
            if nx.is_directed_acyclic_graph(current_dag):
                moves.append(("flip", (u, v)))
            # Revert
            current_dag.remove_edge(v, u)
            current_dag.add_edge(u, v)

        # 3. Edge additions
        for u, v in itertools.permutations(nodes, 2):
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                current_dag.add_edge(u, v)
                if nx.is_directed_acyclic_graph(current_dag):
                    moves.append(("+", (u, v)))
                current_dag.remove_edge(u, v)

        return moves

    def _apply_move(self, dag: nx.DiGraph, move: Tuple[str, Tuple[str, str]]) -> nx.DiGraph:
        """Applies a candidate move to produce a new candidate DAG."""
        new_dag = dag.copy()
        op, (u, v) = move
        if op == "+":
            new_dag.add_edge(u, v)
        elif op == "-":
            new_dag.remove_edge(u, v)
        elif op == "flip":
            new_dag.remove_edge(u, v)
            new_dag.add_edge(v, u)
        return new_dag

    def fit(
        self,
        data: pd.DataFrame,
        initial_edges: Optional[List[Tuple[str, str]]] = None
    ) -> nx.DiGraph:
        """
        Executes the 8-step Tabu Search optimization maximizing BIC score:
        1. Initialize G0 and Tabu list.
        2. Create neighborhood of candidate DAGs.
        3. Calculate BIC scores.
        4. Select maximizing candidate.
        5. Update Tabu list (FIFO).
        6. Apply Aspiration Criterion for exceptional moves.
        7. Repeat until stopping criterion (convergence / max_iter).
        8. Return the best global DAG encountered.
        """
        nodes = list(data.columns)
        scorer = BIC(data)

        # Step 1: Initialize graph
        current_dag = nx.DiGraph()
        current_dag.add_nodes_from(nodes)
        if initial_edges:
            for u, v in initial_edges:
                if u in nodes and v in nodes:
                    current_dag.add_edge(u, v)

        current_score = scorer.score(current_dag)
        best_dag = current_dag.copy()
        best_score = current_score

        tabu_list = deque(maxlen=self.tabu_size)
        no_improvement_count = 0

        logger.info(
            f"Starting Tabu Search on {len(nodes)} nodes, initial BIC score: {current_score:.2f}, "
            f"tabu_size: {self.tabu_size}, max_iter: {self.max_iter}"
        )

        for iteration in range(1, self.max_iter + 1):
            legal_moves = self._get_legal_moves(current_dag, nodes)
            if not legal_moves:
                logger.info("No further legal DAG moves found. Stopping search.")
                break

            best_candidate_move = None
            best_candidate_dag = None
            best_candidate_score = -float("inf")

            for move in legal_moves:
                cand_dag = self._apply_move(current_dag, move)
                cand_score = scorer.score(cand_dag)

                # Step 6: Aspiration Criterion (allow move if it beats global best)
                is_tabu = move in tabu_list
                if is_tabu and cand_score > best_score:
                    is_tabu = False  # Override tabu restriction

                if not is_tabu and cand_score > best_candidate_score:
                    best_candidate_score = cand_score
                    best_candidate_move = move
                    best_candidate_dag = cand_dag

            if best_candidate_move is None:
                logger.info("All candidate moves are tabu. Stopping search.")
                break

            # Step 4 & 5: Update current state and Tabu list
            current_dag = best_candidate_dag
            current_score = best_candidate_score
            tabu_list.append(best_candidate_move)

            # Update historical best
            if current_score > best_score:
                best_score = current_score
                best_dag = current_dag.copy()
                no_improvement_count = 0
                logger.info(
                    f"Iteration {iteration}: New best BIC score = {best_score:.2f} "
                    f"via move {best_candidate_move}"
                )
            else:
                no_improvement_count += 1

            # Step 7: Stopping criteria check
            if no_improvement_count >= self.max_no_improvement:
                logger.info(
                    f"Convergence reached: No improvement for {self.max_no_improvement} "
                    f"consecutive iterations. Stopping at iteration {iteration}."
                )
                break

        # Step 8: Return optimal DAG
        self.graph = best_dag
        logger.info(
            f"Tabu Search completed. Best BIC score: {best_score:.2f}, "
            f"Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}"
        )
        return self.graph
