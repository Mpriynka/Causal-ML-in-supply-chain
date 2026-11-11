from typing import Dict, List, Tuple, Any, Optional
import json
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("causal_builder")

class CausalFeatureBuilder:
    """
    Constructs causal datasets matching the specification in Wyrembek et al. (2025).
    Maps raw features to Outcomes (Y), Treatments (D), and Covariates (X),
    and creates both tabular interpretable and one-hot encoded matrices.
    """

    # Direct mapping corresponding to Table 3 in Wyrembek et al. (2025)
    FEATURE_MAPPING = {
        "Outcome_Continuous": "delay_days",              # Target Y: delay duration in days
        "Outcome_Binary": "late_risk",                   # Target Y_bin: late delivery risk indicator
        "Treatment_Multi": "Multi",                      # Treatment D_multi: supplier/product network degree
        "Treatment_Season_Q1": "season_q1",              # Treatment D_season_q1
        "Treatment_Season_Q2": "season_q2",              # Treatment D_season_q2
        "Treatment_Season_Q3": "season_q3",              # Treatment D_season_q3
        "Treatment_Season_Q4": "season_q4",              # Treatment D_season_q4
        "Treatment_Shipping": "Shipping_Expedited",      # Operational intervention treatment
        "Covariate_Quantity": "Order Item Quantity",     # Covariate: Quantity
        "Covariate_Price": "Order Item Product Price",   # Covariate: Price
        "Covariate_Category": "Category Name",           # Covariate: Part Description proxy
        "Covariate_Department": "Department Name",       # Covariate: Store Department
        "Covariate_Segment": "Customer Segment",         # Covariate: Project / Customer tier proxy
        "Covariate_Market": "Market",                    # Covariate: Destination Market
        "Covariate_Region": "Order Region",              # Covariate: Destination World Region
        "Covariate_Payment": "Type"                      # Covariate: Transaction Type
    }

    NUMERICAL_COVARIATES = [
        "Order Item Quantity",
        "Order Item Product Price",
        "Order Item Discount Rate",
        "Order Item Total"
    ]

    CATEGORICAL_COVARIATES = [
        "Category Name",
        "Department Name",
        "Customer Segment",
        "Market",
        "Order Region",
        "Type"
    ]

    def __init__(self):
        pass

    def build_causal_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts and structures the causal dataset with cleanly named columns
        matching the paper's theoretical framework.
        """
        logger.info("Building canonical causal dataset...")

        # Rename core columns to standardized names
        causal_df = pd.DataFrame()

        # Outcomes
        causal_df["delay_days"] = df["delay_days"].astype(float)
        causal_df["late_risk"] = df["late_risk"].astype(int)

        # Treatments
        causal_df["Multi"] = df["Multi"].astype(int)
        causal_df["Season"] = df["Season"].astype(str)
        causal_df["season_q1"] = df["season_q1"].astype(int)
        causal_df["season_q2"] = df["season_q2"].astype(int)
        causal_df["season_q3"] = df["season_q3"].astype(int)
        causal_df["season_q4"] = df["season_q4"].astype(int)
        causal_df["Shipping_Expedited"] = df["Shipping_Expedited"].astype(int)

        # Covariates - Continuous
        causal_df["Quantity"] = df["Order Item Quantity"].astype(float)
        causal_df["Price"] = df["Order Item Product Price"].astype(float)
        if "Order Item Discount Rate" in df.columns:
            causal_df["Discount_Rate"] = df["Order Item Discount Rate"].astype(float)
        if "Order Item Total" in df.columns:
            causal_df["Order_Total"] = df["Order Item Total"].astype(float)

        # Covariates - Categorical
        causal_df["Category"] = df["Category Name"].astype(str)
        causal_df["Department"] = df["Department Name"].astype(str)
        causal_df["Customer_Segment"] = df["Customer Segment"].astype(str)
        causal_df["Market"] = df["Market"].astype(str)
        causal_df["Order_Region"] = df["Order Region"].astype(str)
        causal_df["Payment_Type"] = df["Type"].astype(str)

        metadata = {
            "total_samples": len(causal_df),
            "outcome_continuous": "delay_days",
            "outcome_binary": "late_risk",
            "treatments": ["Multi", "season_q1", "season_q2", "season_q3", "season_q4", "Shipping_Expedited"],
            "continuous_covariates": ["Quantity", "Price", "Discount_Rate", "Order_Total"],
            "categorical_covariates": ["Category", "Department", "Customer_Segment", "Market", "Order_Region", "Payment_Type"]
        }

        logger.info(f"Causal dataset constructed: {causal_df.shape[0]} rows, {causal_df.shape[1]} columns.")
        return causal_df, metadata

    def build_encoded_feature_matrix(
        self,
        causal_df: pd.DataFrame,
        top_k_categories: int = 20
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates one-hot encoded and standardized feature matrix for DoubleML and XGBoost nuisance models.
        High-cardinality categories are capped at top_k to preserve statistical power and prevent sparsity issues.
        """
        logger.info("Building one-hot encoded matrix for machine learning estimators...")
        df_encoded = causal_df.copy()

        # Handle high-cardinality Category: keep top_k, group rest into 'Other'
        top_cats = df_encoded["Category"].value_counts().nlargest(top_k_categories).index
        df_encoded["Category_Binned"] = df_encoded["Category"].apply(lambda x: x if x in top_cats else "Other")

        cat_cols = ["Category_Binned", "Department", "Customer_Segment", "Market", "Order_Region", "Payment_Type"]
        
        # One-hot encoding
        encoded_dummies = pd.get_dummies(df_encoded[cat_cols], drop_first=True, dtype=int)

        # Combine with outcomes, treatments, and continuous covariates
        base_cols = [
            "delay_days", "late_risk",
            "Multi", "season_q1", "season_q2", "season_q3", "season_q4", "Shipping_Expedited",
            "Quantity", "Price", "Discount_Rate", "Order_Total"
        ]

        matrix_df = pd.concat([df_encoded[base_cols], encoded_dummies], axis=1)

        encoding_meta = {
            "matrix_shape": list(matrix_df.shape),
            "num_features": matrix_df.shape[1] - 8,  # excluding 2 outcomes and 6 treatments
            "encoded_columns": list(encoded_dummies.columns)
        }

        logger.info(f"Encoded matrix created with shape {matrix_df.shape}.")
        return matrix_df, encoding_meta
