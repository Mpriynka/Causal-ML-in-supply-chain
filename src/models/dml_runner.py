from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from src.models.dml_irm import DMLInteractiveRegression
from src.utils.logger import setup_logger

logger = setup_logger("dml_runner")

class DMLBatchRunner:
    """
    Orchestrates batch Double Machine Learning estimation across all treatment variables,
    reproducing Table 5 from Wyrembek et al. (2025).
    """

    TREATMENT_LABELS = {
        "Multi": "Multi (Network Sourcing)",
        "season_q1": "First quarter (Q1 Order)",
        "season_q2": "Second quarter (Q2 Order)",
        "season_q3": "Third quarter (Q3 Order)",
        "season_q4": "Fourth quarter (Q4 Order)",
        "Shipping_Expedited": "Shipping Mode (Expedited vs Standard)"
    }

    def __init__(
        self,
        n_folds: int = 5,
        regressor_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None
    ):
        self.n_folds = n_folds
        self.regressor_params = regressor_params
        self.classifier_params = classifier_params
        self.results: List[Dict[str, Any]] = []
        self.fitted_models: Dict[str, DMLInteractiveRegression] = {}

    def run_all_treatments(
        self,
        df: pd.DataFrame,
        treatments: List[str],
        y_col: str = "delay_days",
        covariates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Runs DML IRM across each specified treatment, controlling for all other covariates.
        """
        all_treatments_set = set(self.TREATMENT_LABELS.keys()).union(set(treatments))
        outcomes_set = {"delay_days", "late_risk"}

        if covariates is None:
            # Exclude outcomes and all treatments from default covariate set
            base_covs = [c for c in df.columns if c not in outcomes_set and c not in all_treatments_set]
        else:
            base_covs = covariates

        self.results = []
        self.fitted_models = {}

        logger.info(f"Starting batch DML IRM estimation across {len(treatments)} treatments...")

        for treatment in treatments:
            if treatment not in df.columns:
                logger.warning(f"Treatment column '{treatment}' not found in dataset. Skipping.")
                continue

            # Confounders for this treatment include base covariates plus other non-collinear treatments
            # (e.g. when estimating season_q1, controlling for Multi and other seasons)
            other_treatments = [t for t in treatments if t != treatment and not (treatment.startswith("season_") and t.startswith("season_"))]
            x_cols = base_covs + other_treatments

            estimator = DMLInteractiveRegression(
                n_folds=self.n_folds,
                regressor_params=self.regressor_params,
                classifier_params=self.classifier_params
            )

            estimator.fit(df=df, y_col=y_col, d_col=treatment, x_cols=x_cols)
            summary = estimator.get_summary()

            # Enrich with human-readable label
            summary["treatment_label"] = self.TREATMENT_LABELS.get(treatment, treatment)
            self.results.append(summary)
            self.fitted_models[treatment] = estimator

        results_df = pd.DataFrame(self.results)
        # Format table columns to match Table 5
        cols_order = [
            "treatment_label", "treatment", "coef", "std_err", "t_stat", "p_val",
            "ci_95_lower", "ci_95_upper", "statistically_significant"
        ]
        results_df = results_df[[c for c in cols_order if c in results_df.columns]]
        return results_df
