import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.causal_builder import CausalFeatureBuilder

RAW_DATA_PATH = ROOT_DIR / "archive" / "DataCoSupplyChainDataset.csv"

class TestPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RAW_DATA_PATH.exists():
            raise unittest.SkipTest(f"Raw data file not found at {RAW_DATA_PATH}")
        loader = DataLoader(str(RAW_DATA_PATH))
        cls.raw_df, cls.audit_info = loader.load_raw_data(nrows=500)

    def test_01_loader_schema(self):
        """Verifies that the DataLoader extracts core columns properly."""
        self.assertEqual(len(self.raw_df), 500)
        for col in DataLoader.EXPECTED_CORE_COLUMNS:
            self.assertIn(col, self.raw_df.columns, f"Expected column {col} missing")

    def test_02_preprocessor_transform(self):
        """Verifies data cleaning, timestamp parsing, delay target, and treatment indicators."""
        preprocessor = DataPreprocessor(keep_outliers=True)
        df_clean, stats = preprocessor.clean_and_transform(self.raw_df)

        self.assertGreater(len(df_clean), 0)
        self.assertIn("delay_days", df_clean.columns)
        self.assertIn("late_risk", df_clean.columns)
        self.assertIn("Multi", df_clean.columns)
        self.assertIn("Season", df_clean.columns)
        self.assertIn("Shipping_Expedited", df_clean.columns)

        # Verify delay_days calculation: real - scheduled
        expected_delay = df_clean["Days for shipping (real)"] - df_clean["Days for shipment (scheduled)"]
        pd.testing.assert_series_equal(df_clean["delay_days"], expected_delay.astype(float), check_names=False)

        # Verify late_risk values are binary 0 or 1
        self.assertTrue(set(df_clean["late_risk"].unique()).issubset({0, 1}))

        # Verify Multi values are binary 0 or 1
        self.assertTrue(set(df_clean["Multi"].unique()).issubset({0, 1}))

        # Verify Seasons are in Q1, Q2, Q3, Q4
        self.assertTrue(set(df_clean["Season"].unique()).issubset({"Q1", "Q2", "Q3", "Q4"}))

        # Verify no nulls in target variables
        self.assertEqual(df_clean["delay_days"].isnull().sum(), 0)
        self.assertEqual(df_clean["late_risk"].isnull().sum(), 0)

    def test_03_causal_builder(self):
        """Verifies CausalFeatureBuilder creates canonical and encoded matrices."""
        preprocessor = DataPreprocessor()
        df_clean, _ = preprocessor.clean_and_transform(self.raw_df)

        builder = CausalFeatureBuilder()
        causal_df, meta = builder.build_causal_dataset(df_clean)

        # Check outcomes, treatments, and covariates exist
        self.assertIn(meta["outcome_continuous"], causal_df.columns)
        self.assertIn(meta["outcome_binary"], causal_df.columns)
        for t in meta["treatments"]:
            self.assertIn(t, causal_df.columns)
        for c in meta["continuous_covariates"]:
            self.assertIn(c, causal_df.columns)
        for cat in meta["categorical_covariates"]:
            self.assertIn(cat, causal_df.columns)

        # Verify no nulls in causal dataset
        self.assertEqual(causal_df.isnull().sum().sum(), 0)

        # Check encoded matrix
        encoded_df, enc_meta = builder.build_encoded_feature_matrix(causal_df)
        self.assertEqual(len(encoded_df), len(causal_df))
        self.assertEqual(encoded_df.isnull().sum().sum(), 0)
        self.assertIn("delay_days", encoded_df.columns)
        self.assertIn("Multi", encoded_df.columns)
        self.assertIn("season_q1", encoded_df.columns)

if __name__ == "__main__":
    unittest.main()
