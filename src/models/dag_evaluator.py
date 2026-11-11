from typing import Dict, List, Tuple, Set, Any
import networkx as nx
from src.utils.logger import setup_logger

logger = setup_logger("dag_evaluator")

class DAGEvaluator:
    """
    Evaluates and compares learned causal DAGs against the Domain Expert Knowledge DAG.
    Computes Structural Hamming Distance (SHD), edge metrics, and d-separation paths.
    """

    @staticmethod
    def compute_shd(dag1: nx.DiGraph, dag2: nx.DiGraph) -> int:
        """
        Calculates the Structural Hamming Distance (SHD) between two DAGs on the same node set.
        SHD counts:
        - Missing edges (in dag1 but not dag2)
        - Extra edges (in dag2 but not dag1)
        - Reversed edges (u->v in dag1, v->u in dag2)
        """
        nodes = set(dag1.nodes()).union(set(dag2.nodes()))
        edges1 = set(dag1.edges())
        edges2 = set(dag2.edges())

        shd = 0
        checked_pairs = set()

        for u in nodes:
            for v in nodes:
                if u >= v:
                    continue
                pair = (u, v)
                pair_rev = (v, u)

                has_e1 = pair in edges1
                has_e1_rev = pair_rev in edges1
                has_e2 = pair in edges2
                has_e2_rev = pair_rev in edges2

                # If reversed: 1 modification
                if (has_e1 and has_e2_rev) or (has_e1_rev and has_e2):
                    shd += 1
                # If one has edge and other has none: 1 modification
                elif (has_e1 or has_e1_rev) != (has_e2 or has_e2_rev):
                    shd += 1

        return shd

    @staticmethod
    def compute_jaccard_similarity(dag1: nx.DiGraph, dag2: nx.DiGraph) -> float:
        """Computes Jaccard similarity between directed edge sets."""
        e1 = set(dag1.edges())
        e2 = set(dag2.edges())
        if not e1 and not e2:
            return 1.0
        union_size = len(e1.union(e2))
        if union_size == 0:
            return 0.0
        return round(len(e1.intersection(e2)) / union_size, 4)

    @classmethod
    def check_causal_identifiability(
        cls,
        dag: nx.DiGraph,
        treatment: str,
        outcome: str = "Delay",
        covariates: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes causal paths, d-separation, and backdoor criteria from treatment to outcome.
        """
        if treatment not in dag.nodes() or outcome not in dag.nodes():
            return {
                "has_directed_path": False,
                "paths": [],
                "backdoor_admissible": False,
                "parents_of_treatment": []
            }

        has_path = nx.has_path(dag, treatment, outcome)
        all_paths = []
        if has_path:
            all_paths = list(nx.all_simple_paths(dag, treatment, outcome, cutoff=5))

        # Backdoor path evaluation:
        # A set of covariates X satisfies the backdoor criterion relative to (D, Y) if:
        # 1. No node in X is a descendant of D.
        # 2. X blocks every backdoor path between D and Y.
        descendants_of_treatment = nx.descendants(dag, treatment)
        parents_of_treatment = list(dag.predecessors(treatment))

        # Check if covariates contain descendants
        covs = covariates or []
        invalid_covariates = [c for c in covs if c in descendants_of_treatment and c != outcome]

        return {
            "has_directed_path": has_path,
            "path_count": len(all_paths),
            "sample_paths": [" -> ".join(p) for p in all_paths[:3]],
            "parents_of_treatment": parents_of_treatment,
            "descendants_of_treatment": list(descendants_of_treatment),
            "backdoor_clean": len(invalid_covariates) == 0,
            "collider_risks": invalid_covariates
        }

    @classmethod
    def evaluate_all(
        cls,
        expert_dag: nx.DiGraph,
        learned_dags: Dict[str, nx.DiGraph],
        treatments: List[str] = ["Multi", "Season", "Shipping"],
        outcome: str = "Delay"
    ) -> Dict[str, Any]:
        """Runs full comparison across all learned autoDAGs against the Expert DAG."""
        results = {
            "graph_summaries": {},
            "expert_comparison": {},
            "identifiability_analysis": {}
        }

        # Add expert summary
        results["graph_summaries"]["Expert_DAG"] = {
            "num_nodes": expert_dag.number_of_nodes(),
            "num_edges": expert_dag.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(expert_dag),
            "density": round(nx.density(expert_dag), 4),
            "edges": list(expert_dag.edges())
        }

        for name, dag in learned_dags.items():
            results["graph_summaries"][name] = {
                "num_nodes": dag.number_of_nodes(),
                "num_edges": dag.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(dag),
                "density": round(nx.density(dag), 4),
                "edges": list(dag.edges())
            }

            shd = cls.compute_shd(expert_dag, dag)
            jaccard = cls.compute_jaccard_similarity(expert_dag, dag)
            shared_edges = list(set(expert_dag.edges()).intersection(set(dag.edges())))

            results["expert_comparison"][name] = {
                "shd_to_expert": shd,
                "jaccard_similarity": jaccard,
                "shared_edges_count": len(shared_edges),
                "shared_edges": shared_edges
            }

        # Causal identifiability for each DAG across treatments
        all_dags = {"Expert_DAG": expert_dag, **learned_dags}
        for dag_name, dag in all_dags.items():
            results["identifiability_analysis"][dag_name] = {}
            for t in treatments:
                if t in dag.nodes():
                    results["identifiability_analysis"][dag_name][t] = cls.check_causal_identifiability(
                        dag, treatment=t, outcome=outcome
                    )

        return results
