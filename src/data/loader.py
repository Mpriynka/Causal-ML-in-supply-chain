from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

class DataLoader:
    """Handles raw supply chain data loading, schema inspection, and verification."""

    EXPECTED_CORE_COLUMNS = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Late_delivery_risk",
        "Delivery Status",
        "order date (DateOrders)",
        "shipping date (DateOrders)",
        "Order Item Quantity",
        "Order Item Product Price",
        "Category Name",
        "Department Name",
        "Customer Segment",
        "Market",
        "Order Region",
        "Shipping Mode"
    ]

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")

    def load_raw_data(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads raw CSV data with proper encoding and validates schema."""
        logger.info(f"Loading raw data from: {self.data_path}")
        
        # Try latin1 encoding as required for DataCo
        try:
            df = pd.read_csv(self.data_path, encoding="latin1", nrows=nrows)
        except Exception as e:
            logger.warning(f"Failed loading with latin1 ({e}), attempting utf-8")
            df = pd.read_csv(self.data_path, encoding="utf-8", nrows=nrows)

        logger.info(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

        # Validate core columns
        missing_core = [col for col in self.EXPECTED_CORE_COLUMNS if col not in df.columns]
        if missing_core:
            logger.warning(f"Missing expected core columns: {missing_core}")
        else:
            logger.info("All expected core supply chain columns are present.")

        audit_info = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "file_size_mb": round(self.data_path.stat().st_size / (1024 * 1024), 2),
            "columns": list(df.columns),
            "missing_values_by_col": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "duplicate_rows": int(df.duplicated().sum())
        }

        return df, audit_info
