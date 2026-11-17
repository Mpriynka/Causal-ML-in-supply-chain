from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from doubleml import DoubleMLIRM
from src.utils.logger import setup_logger

logger = setup_logger("policy_tree")

class PolicyTreeEstimator:
    """
    Estimates deterministic binary treatment policies using weighted classification trees,
    implementing the policy learning framework of Athey & Wager (2021) as described in
    Wyrembek et al. (2025), Section 3.4 (Equation 6) and Section 4.2 (Figure 6).
    """

    def __init__(self, depth: int = 2, ccp_alpha: float = 0.005):
        self.depth = depth
        self.ccp_alpha = ccp_alpha
        self.feature_names: List[str] = []
        self.policy_tree_obj = None
        self.treatment_name: Optional[str] = None

    def fit(
        self,
        irm_model: DoubleMLIRM,
        features: pd.DataFrame,
        treatment_name: str = "Treatment"
    ) -> "PolicyTreeEstimator":
        """
        Estimates the optimal policy tree by passing observable covariates
        and orthogonal score components to DoubleML's policy_tree method.
        """
        self.treatment_name = treatment_name
        self.feature_names = list(features.columns)

        logger.info(
            f"Fitting Policy Tree for treatment '{treatment_name}' with depth={self.depth}, "
            f"features: {self.feature_names}"
        )

        # Estimate policy tree
        self.policy_tree_obj = irm_model.policy_tree(
            features=features,
            depth=self.depth,
            ccp_alpha=self.ccp_alpha
        )

        tree = self.policy_tree_obj.policy_tree
        logger.info(
            f"Policy tree induced successfully: {tree.get_n_leaves()} leaves, "
            f"depth {tree.get_depth()}."
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predicts the optimal treatment assignment (0 or 1) for new observations."""
        if self.policy_tree_obj is None:
            raise ValueError("Policy Tree has not been fitted yet!")
        res = self.policy_tree_obj.predict(features[self.feature_names])
        return res["pred_treatment"].values

    def export_text_rules(self) -> str:
        """Returns human-readable prescriptive intervention rules."""
        if self.policy_tree_obj is None:
            raise ValueError("Policy Tree has not been fitted yet!")
        tree = self.policy_tree_obj.policy_tree
        rules = export_text(tree, feature_names=self.feature_names)
        return rules

    def get_tree_structure(self) -> Dict[str, Any]:
        """Returns machine-readable tree node properties and leaf decisions."""
        if self.policy_tree_obj is None:
            raise ValueError("Policy Tree has not been fitted yet!")

        tree = self.policy_tree_obj.policy_tree
        return {
            "treatment": self.treatment_name,
            "depth": int(tree.get_depth()),
            "n_leaves": int(tree.get_n_leaves()),
            "features": self.feature_names,
            "rules_text": self.export_text_rules()
        }
