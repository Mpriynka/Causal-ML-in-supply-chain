import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLIRM
from xgboost import XGBClassifier, XGBRegressor

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.cate_splines import CATESplineEstimator
from src.models.policy_tree import PolicyTreeEstimator

ENCODED_DATA_PATH = ROOT_DIR / "data" / "processed" / "causal_supply_chain_encoded.parquet"

class TestPhase4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENCODED_DATA_PATH.exists():
            raise unittest.SkipTest(f"Data file not found at {ENCODED_DATA_PATH}")
        df = pd.read_parquet(ENCODED_DATA_PATH)
        cls.sample = df.sample(n=600, random_state=42).reset_index(drop=True)
        covariates = [c for c in df.columns if c not in ['delay_days', 'late_risk', 'Multi', 'season_q1', 'season_q2', 'season_q3', 'season_q4', 'Shipping_Expedited']][:6]

        dml_data = DoubleMLData(cls.sample, y_col='delay_days', d_cols='Shipping_Expedited', x_cols=covariates)
        ml_g = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        ml_m = XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric='logloss')

        cls.irm = DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2)
        cls.irm.fit(store_predictions=True)

    def test_01_cate_spline_fit_predict(self):
        """Verifies CATESplineEstimator fits B-spline basis and evaluates smooth curves with CIs."""
        spline_est = CATESplineEstimator(df_degrees=5, spline_degree=3)
        spline_est.fit(self.irm, self.sample, "Quantity")

        grid = np.linspace(1, 5, 20)
        curve_df = spline_est.predict_curve(grid_values=grid)

        self.assertEqual(len(curve_df), 20)
        self.assertIn("Quantity", curve_df.columns)
        self.assertIn("cate", curve_df.columns)
        self.assertIn("se", curve_df.columns)
        self.assertIn("ci_lower", curve_df.columns)
        self.assertIn("ci_upper", curve_df.columns)

        # Standard error non-negative
        self.assertTrue(np.all(curve_df["se"] >= 0))
        # Confidence interval lower <= upper
        self.assertTrue(np.all(curve_df["ci_lower"] <= curve_df["ci_upper"]))

        summary = spline_est.get_summary()
        self.assertEqual(summary["feature"], "Quantity")
        self.assertEqual(len(summary["coefficients"]), 5)

    def test_02_policy_tree_fit_predict(self):
        """Verifies PolicyTreeEstimator induces valid tree rules and makes binary predictions."""
        policy_features = self.sample[["Quantity", "Price"]]
        ptree = PolicyTreeEstimator(depth=2, ccp_alpha=0.01)
        ptree.fit(self.irm, policy_features, treatment_name="Shipping_Expedited")

        # Predictions
        preds = ptree.predict(policy_features.head(10))
        self.assertEqual(len(preds), 10)
        self.assertTrue(set(preds).issubset({0, 1}))

        # Text rules
        rules = ptree.export_text_rules()
        self.assertIsInstance(rules, str)
        self.assertGreater(len(rules), 10)

        # Tree structure
        struct = ptree.get_tree_structure()
        self.assertEqual(struct["treatment"], "Shipping_Expedited")
        self.assertLessEqual(struct["depth"], 2)
        self.assertGreaterEqual(struct["n_leaves"], 1)

if __name__ == "__main__":
    unittest.main()
