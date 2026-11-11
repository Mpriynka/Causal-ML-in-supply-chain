from typing import Dict, List, Tuple
import networkx as nx
from src.utils.logger import setup_logger

logger = setup_logger("dag_expert")

class ExpertDAGBuilder:
    """
    Constructs the Domain Expert Knowledge DAG from Figure 3 of Wyrembek et al. (2025).
    Translates maritime supply chain domain knowledge onto the standardized DataCo schema.
    """

    CORE_NODES = [
        "Multi",
        "Season",
        "Category",
        "Customer_Segment",
        "Market",
        "Quantity",
        "Price",
        "Shipping_Expedited",
        "Delay"
    ]

    # Explicit structural edges defined in Figure 3 of Wyrembek et al. (2025)
    EXPERT_EDGES = [
        ("Multi", "Market"),
        ("Multi", "Delay"),
        ("Season", "Quantity"),
        ("Season", "Price"),
        ("Market", "Category"),
        ("Market", "Customer_Segment"),
        ("Market", "Quantity"),
        ("Market", "Price"),
        ("Market", "Delay"),
        ("Category", "Price"),
        ("Category", "Delay"),
        ("Customer_Segment", "Delay"),
        ("Quantity", "Delay"),
        ("Price", "Delay"),
        ("Shipping_Expedited", "Delay")
    ]

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_dag(self) -> nx.DiGraph:
        """Builds and validates the theoretical domain expert DAG."""
        self.graph.clear()
        self.graph.add_nodes_from(self.CORE_NODES)
        self.graph.add_edges_from(self.EXPERT_EDGES)

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Constructed Expert Graph contains cycles, not a valid DAG!")

        logger.info(
            f"Expert DAG built successfully with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} directed edges."
        )
        return self.graph

    def get_structural_equations(self) -> Dict[str, List[str]]:
        """Returns the structural causal equations (parents for each node)."""
        if self.graph.number_of_nodes() == 0:
            self.build_dag()
        return {node: list(self.graph.predecessors(node)) for node in self.graph.nodes()}
