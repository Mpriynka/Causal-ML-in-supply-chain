from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("data_preprocessor")

class DataPreprocessor:
    """Preprocesses raw supply chain data and constructs domain variables."""

    DROP_COLUMNS = [
        "Product Description",  # 100% missing
        "Order Zipcode",        # >85% missing
        "Product Image",        # URL not needed for tabular modeling
        "Customer Email",       # PII masked
        "Customer Password"     # PII masked
    ]

    def __init__(self, keep_outliers: bool = True):
        self.keep_outliers = keep_outliers

    def clean_and_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Executes full cleaning, date parsing, delay calculation, and causal indicator generation."""
        logger.info("Starting data cleaning and transformation pipeline...")
        df_clean = df.copy()

        initial_rows = len(df_clean)

        # 1. Drop irrelevant/uninformative columns
        cols_to_drop = [c for c in self.DROP_COLUMNS if c in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} uninformative columns: {cols_to_drop}")

        # 2. Handle duplicates
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Dropped {duplicates} duplicate records.")

        # 3. Handle missing values
        if "Customer Lname" in df_clean.columns:
            df_clean["Customer Lname"] = df_clean["Customer Lname"].fillna("Unknown")
        if "Customer Zipcode" in df_clean.columns:
            df_clean["Customer Zipcode"] = df_clean["Customer Zipcode"].fillna(0)

        # 4. Parse dates
        logger.info("Parsing datetime attributes...")
        df_clean["order_date"] = pd.to_datetime(df_clean["order date (DateOrders)"], errors="coerce")
        df_clean["shipping_date"] = pd.to_datetime(df_clean["shipping date (DateOrders)"], errors="coerce")

        # 5. Extract temporal features matching paper definition
        df_clean["order_year"] = df_clean["order_date"].dt.year
        df_clean["order_month"] = df_clean["order_date"].dt.month
        df_clean["order_day"] = df_clean["order_date"].dt.day
        df_clean["order_dayofweek"] = df_clean["order_date"].dt.day_of_week
        df_clean["order_hour"] = df_clean["order_date"].dt.hour
        df_clean["quarter"] = df_clean["order_date"].dt.quarter
        df_clean["Season"] = "Q" + df_clean["quarter"].astype(str)

        # Create one-hot indicators for seasons (Q1 to Q4) matching paper's treatment analysis
        for q in [1, 2, 3, 4]:
            df_clean[f"season_q{q}"] = (df_clean["quarter"] == q).astype(int)

        # 6. Calculate Delay Targets
        # Continuous Delay (days) = Actual shipping days - Scheduled shipping days
        df_clean["delay_days"] = (
            df_clean["Days for shipping (real)"] - df_clean["Days for shipment (scheduled)"]
        ).astype(float)

        # Binary Delay Risk (1 = delayed, 0 = on-time / advance)
        df_clean["late_risk"] = df_clean["Late_delivery_risk"].astype(int)

        # 7. Construct 'Multi' Network Treatment Indicator
        # Paper: "whether a supplier supplies to another buyer in the network (1 = supplies to >1, 0 = only 1)"
        # In DataCo: Products / Categories supplying to multiple geographic markets
        product_market_counts = df_clean.groupby("Product Card Id")["Market"].nunique()
        df_clean["product_market_count"] = df_clean["Product Card Id"].map(product_market_counts)
        df_clean["Multi"] = (df_clean["product_market_count"] > 1).astype(int)

        # 8. Operational Treatment Indicator (Expedited vs Standard Shipping)
        df_clean["Shipping_Expedited"] = df_clean["Shipping Mode"].isin(
            ["First Class", "Same Day", "Second Class"]
        ).astype(int)

        # 9. Quality validations
        # Positive quantities and prices
        valid_mask = (df_clean["Order Item Quantity"] > 0) & (df_clean["Order Item Product Price"] > 0)
        df_clean = df_clean[valid_mask].copy()

        logger.info(f"Preprocessing complete. Retained {len(df_clean):,} of {initial_rows:,} records.")

        stats_summary = {
            "initial_rows": initial_rows,
            "final_rows": len(df_clean),
            "late_delivery_count": int((df_clean["late_risk"] == 1).sum()),
            "late_delivery_pct": round(float((df_clean["late_risk"] == 1).mean() * 100), 2),
            "mean_delay_days": round(float(df_clean["delay_days"].mean()), 4),
            "std_delay_days": round(float(df_clean["delay_days"].std()), 4),
            "multi_market_treatment_pct": round(float(df_clean["Multi"].mean() * 100), 2),
            "shipping_expedited_treatment_pct": round(float(df_clean["Shipping_Expedited"].mean() * 100), 2),
            "season_distribution": df_clean["Season"].value_counts().to_dict()
        }

        return df_clean, stats_summary
