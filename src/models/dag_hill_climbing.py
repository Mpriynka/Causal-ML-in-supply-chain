from typing import Dict, List, Optional
import pandas as pd
import networkx as nx
from pgmpy.estimators import HillClimbSearch
from src.utils.logger import setup_logger

logger = setup_logger("dag_hill_climbing")

class HillClimbingDAG:
    """
    Learns Bayesian Network DAG structure using greedy Hill Climbing with K2 score,
    as specified in Section 3.3.1 of Wyrembek et al. (2025).
    """

    def __init__(self, max_iter: int = 50, scoring_method: str = "k2"):
        self.max_iter = max_iter
        self.scoring_method = scoring_method
        self.graph = nx.DiGraph()

    @staticmethod
    def prepare_discrete_data(
        df: pd.DataFrame,
        sample_size: Optional[int] = 10000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Transforms continuous and high-cardinality features into discrete categories
        required for exact K2 Dirichlet score calculation.
        """
        disc_df = pd.DataFrame()

        # Target: Delay (late risk binary)
        disc_df["Delay"] = df["late_risk"].astype(str)

        # Treatments
        disc_df["Multi"] = df["Multi"].astype(str)
        disc_df["Season"] = df["Season"].astype(str)
        disc_df["Shipping"] = df["Shipping_Expedited"].astype(str)

        # Covariates
        disc_df["Quantity"] = pd.qcut(df["Quantity"], q=3, duplicates="drop").astype(str)
        disc_df["Price"] = pd.qcut(df["Price"], q=4, duplicates="drop").astype(str)
        disc_df["Segment"] = df["Customer_Segment"].astype(str)
        disc_df["Market"] = df["Market"].astype(str)

        # Top categories binned to avoid extreme cardinality
        top_cats = df["Category"].value_counts().nlargest(10).index
        disc_df["Category"] = df["Category"].apply(lambda x: x if x in top_cats else "Other").astype(str)

        disc_df = disc_df.astype("category")

        if sample_size and len(disc_df) > sample_size:
            disc_df = disc_df.sample(n=sample_size, random_state=random_state)

        return disc_df

    def fit(self, data: pd.DataFrame) -> nx.DiGraph:
        """Executes greedy hill climbing search maximizing the K2 score."""
        logger.info(
            f"Fitting Hill Climbing DAG on {data.shape[0]} samples with scoring='{self.scoring_method}'..."
        )
        hc = HillClimbSearch(data)
        learned_model = hc.estimate(
            scoring_method=self.scoring_method,
            max_iter=self.max_iter,
            show_progress=False
        )

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(data.columns)
        self.graph.add_edges_from(learned_model.edges())

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Learned Hill Climbing graph is not acyclic!")

        logger.info(
            f"Hill Climbing DAG learned: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} directed edges."
        )
        return self.graph
