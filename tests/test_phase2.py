import unittest
from pathlib import Path
import sys
import networkx as nx
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.dag_expert import ExpertDAGBuilder
from src.models.dag_hill_climbing import HillClimbingDAG
from src.models.dag_tabu_search import TabuSearchDAG
from src.models.dag_pc import PCDAG
from src.models.dag_evaluator import DAGEvaluator

PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "causal_supply_chain.parquet"

class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROCESSED_DATA_PATH.exists():
            raise unittest.SkipTest(f"Processed data file not found at {PROCESSED_DATA_PATH}")
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        cls.discrete_sample = HillClimbingDAG.prepare_discrete_data(
            df, sample_size=500, random_state=42
        )

    def test_01_expert_dag(self):
        """Verifies construction and acyclicity of Domain Expert DAG."""
        builder = ExpertDAGBuilder()
        dag = builder.build_dag()
        self.assertEqual(dag.number_of_nodes(), 9)
        self.assertEqual(dag.number_of_edges(), 15)
        self.assertTrue(nx.is_directed_acyclic_graph(dag))

        # Check key causal paths
        self.assertTrue(dag.has_edge("Shipping_Expedited", "Delay"))
        self.assertTrue(dag.has_edge("Multi", "Delay"))
        self.assertTrue(dag.has_edge("Season", "Quantity"))

        eqs = builder.get_structural_equations()
        self.assertIn("Delay", eqs)
        self.assertGreaterEqual(len(eqs["Delay"]), 5)

    def test_02_hill_climbing_dag(self):
        """Verifies Hill Climbing with K2 score produces a valid DAG."""
        hc = HillClimbingDAG(max_iter=10, scoring_method="k2")
        dag = hc.fit(self.discrete_sample)

        self.assertTrue(nx.is_directed_acyclic_graph(dag))
        self.assertEqual(dag.number_of_nodes(), self.discrete_sample.shape[1])
        self.assertGreater(dag.number_of_edges(), 0)

    def test_03_tabu_search_dag(self):
        """Verifies Tabu Search with BIC score produces a valid DAG."""
        tabu = TabuSearchDAG(tabu_size=5, max_iter=8, max_no_improvement=4)
        dag = tabu.fit(self.discrete_sample, initial_edges=[("Shipping", "Delay")])

        self.assertTrue(nx.is_directed_acyclic_graph(dag))
        self.assertEqual(dag.number_of_nodes(), self.discrete_sample.shape[1])
        self.assertGreater(dag.number_of_edges(), 0)
        self.assertTrue(dag.has_edge("Shipping", "Delay"))

    def test_04_pc_dag(self):
        """Verifies Peter-Clark algorithm produces a valid DAG."""
        pc_model = PCDAG(alpha=0.05, indep_test="chisq")
        dag = pc_model.fit(self.discrete_sample)

        self.assertTrue(nx.is_directed_acyclic_graph(dag))
        self.assertEqual(dag.number_of_nodes(), self.discrete_sample.shape[1])

    def test_05_evaluator_metrics(self):
        """Verifies Structural Hamming Distance and identifiability checks."""
        g1 = nx.DiGraph([("A", "B"), ("B", "C")])
        g2 = nx.DiGraph([("A", "B"), ("B", "C")])
        # Identical graphs: SHD = 0
        self.assertEqual(DAGEvaluator.compute_shd(g1, g2), 0)

        # One reversed edge: SHD = 1
        g3 = nx.DiGraph([("B", "A"), ("B", "C")])
        self.assertEqual(DAGEvaluator.compute_shd(g1, g3), 1)

        # One missing edge: SHD = 1
        g4 = nx.DiGraph([("A", "B")])
        self.assertEqual(DAGEvaluator.compute_shd(g1, g4), 1)

        # Test identifiability
        check = DAGEvaluator.check_causal_identifiability(g1, treatment="A", outcome="C")
        self.assertTrue(check["has_directed_path"])
        self.assertEqual(check["path_count"], 1)

if __name__ == "__main__":
    unittest.main()
