import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.dml_irm import DMLInteractiveRegression
from src.models.dml_runner import DMLBatchRunner

ENCODED_DATA_PATH = ROOT_DIR / "data" / "processed" / "causal_supply_chain_encoded.parquet"

class TestPhase3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENCODED_DATA_PATH.exists():
            raise unittest.SkipTest(f"Data file not found at {ENCODED_DATA_PATH}")
        df = pd.read_parquet(ENCODED_DATA_PATH)
        # Use a small sample for rapid testing
        cls.sample = df.sample(n=600, random_state=42).reset_index(drop=True)
        outcomes_and_treatments = {"delay_days", "late_risk", "Multi", "season_q1", "season_q2", "season_q3", "season_q4", "Shipping_Expedited"}
        cls.covariates = [c for c in df.columns if c not in outcomes_and_treatments][:8]

    def test_01_dml_initialization(self):
        """Verifies DMLInteractiveRegression initializes properly with defaults."""
        estimator = DMLInteractiveRegression(n_folds=3)
        self.assertEqual(estimator.n_folds, 3)
        self.assertEqual(estimator.score, "ATE")
        self.assertIn("n_estimators", estimator.default_regressor_params)
        self.assertIn("max_depth", estimator.default_classifier_params)

    def test_02_dml_fit_sample(self):
        """Verifies DMLInteractiveRegression fits properly and computes ATE statistics."""
        fast_reg_params = {"n_estimators": 15, "max_depth": 3, "n_jobs": 2, "random_state": 42}
        fast_clf_params = {"n_estimators": 15, "max_depth": 3, "n_jobs": 2, "eval_metric": "logloss", "random_state": 42}

        estimator = DMLInteractiveRegression(
            n_folds=2,
            regressor_params=fast_reg_params,
            classifier_params=fast_clf_params
        )

        estimator.fit(
            df=self.sample,
            y_col="delay_days",
            d_col="Shipping_Expedited",
            x_cols=self.covariates
        )

        summary = estimator.get_summary()
        self.assertEqual(summary["treatment"], "Shipping_Expedited")
        self.assertEqual(summary["outcome"], "delay_days")
        self.assertIsInstance(summary["coef"], float)
        self.assertGreater(summary["std_err"], 0)
        self.assertIsInstance(summary["t_stat"], float)
        self.assertGreaterEqual(summary["p_val"], 0.0)
        self.assertLessEqual(summary["p_val"], 1.0)
        self.assertLessEqual(summary["ci_95_lower"], summary["ci_95_upper"])

    def test_03_propensity_scores(self):
        """Verifies extraction and valid range of out-of-fold propensity scores."""
        fast_reg_params = {"n_estimators": 10, "max_depth": 3, "n_jobs": 2, "random_state": 42}
        fast_clf_params = {"n_estimators": 10, "max_depth": 3, "n_jobs": 2, "eval_metric": "logloss", "random_state": 42}

        estimator = DMLInteractiveRegression(
            n_folds=2,
            regressor_params=fast_reg_params,
            classifier_params=fast_clf_params
        )
        estimator.fit(
            df=self.sample,
            y_col="delay_days",
            d_col="Shipping_Expedited",
            x_cols=self.covariates
        )

        treated_ps, control_ps = estimator.get_propensity_scores()
        self.assertGreater(len(treated_ps), 0)
        self.assertGreater(len(control_ps), 0)
        # All propensity scores must be bounded in [0, 1]
        self.assertTrue(np.all((treated_ps >= 0.0) & (treated_ps <= 1.0)))
        self.assertTrue(np.all((control_ps >= 0.0) & (control_ps <= 1.0)))

    def test_04_batch_runner(self):
        """Verifies DMLBatchRunner executes across multiple treatments."""
        fast_reg_params = {"n_estimators": 10, "max_depth": 3, "n_jobs": 2, "random_state": 42}
        fast_clf_params = {"n_estimators": 10, "max_depth": 3, "n_jobs": 2, "eval_metric": "logloss", "random_state": 42}

        runner = DMLBatchRunner(
            n_folds=2,
            regressor_params=fast_reg_params,
            classifier_params=fast_clf_params
        )

        results_df = runner.run_all_treatments(
            df=self.sample,
            treatments=["Shipping_Expedited", "season_q1"],
            y_col="delay_days",
            covariates=self.covariates
        )

        self.assertEqual(len(results_df), 2)
        expected_cols = {"treatment", "treatment_label", "coef", "std_err", "t_stat", "p_val"}
        self.assertTrue(expected_cols.issubset(set(results_df.columns)))

if __name__ == "__main__":
    unittest.main()
