from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLIRM
from xgboost import XGBClassifier, XGBRegressor
from src.utils.logger import setup_logger

logger = setup_logger("dml_irm")

class DMLInteractiveRegression:
    """
    Implements the Interactive Regression Model (IRM) using Double Machine Learning (DML)
    with XGBoost nuisance learners as described in Wyrembek et al. (2025), Sections 3.4 & 4.1.
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_rep: int = 1,
        score: str = "ATE",
        trimming_threshold: float = 1e-2,
        regressor_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.score = score
        self.trimming_threshold = trimming_threshold
        self.random_state = random_state

        # Optimal hyperparameters from Table 4 of Wyrembek et al. (2025)
        self.default_regressor_params = {
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "n_jobs": 4,
            "random_state": random_state
        }
        self.default_classifier_params = {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "n_jobs": 4,
            "eval_metric": "logloss",
            "random_state": random_state
        }

        if regressor_params:
            self.default_regressor_params.update(regressor_params)
        if classifier_params:
            self.default_classifier_params.update(classifier_params)

        self.model: Optional[DoubleMLIRM] = None
        self.treatment_name: Optional[str] = None
        self.outcome_name: Optional[str] = None

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str,
        d_col: str,
        x_cols: List[str]
    ) -> "DMLInteractiveRegression":
        """
        Fits the DoubleML IRM model using 5-fold cross-fitting and Neyman-orthogonal score.
        """
        self.treatment_name = d_col
        self.outcome_name = y_col

        logger.info(
            f"Initializing DoubleML IRM | Outcome: '{y_col}', Treatment: '{d_col}', "
            f"Covariates: {len(x_cols)} features, Folds: {self.n_folds}"
        )

        dml_data = DoubleMLData(df, y_col=y_col, d_cols=d_col, x_cols=x_cols)

        # Initialize nuisance estimators with Table 4 hyperparameters
        ml_g = XGBRegressor(**self.default_regressor_params)
        ml_m = XGBClassifier(**self.default_classifier_params)

        self.model = DoubleMLIRM(
            obj_dml_data=dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            n_folds=self.n_folds,
            n_rep=self.n_rep,
            score=self.score,
            trimming_threshold=self.trimming_threshold
        )

        logger.info(f"Fitting nuisance models with {self.n_folds}-fold cross-fitting...")
        self.model.fit(store_predictions=True)

        summary = self.get_summary()
        logger.info(
            f"ATE Estimation Complete for '{d_col}': Coef = {summary['coef']:.4f}, "
            f"Std Err = {summary['std_err']:.4f}, t = {summary['t_stat']:.2f}, "
            f"p-val = {summary['p_val']:.2e}"
        )
        return self

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured ATE estimation results matching Table 5 in the paper."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet!")

        coef = float(self.model.coef[0])
        std_err = float(self.model.se[0])
        t_stat = float(self.model.t_stat[0])
        p_val = float(self.model.pval[0])
        ci = self.model.confint(level=0.95)
        ci_lower = float(ci.iloc[0, 0])
        ci_upper = float(ci.iloc[0, 1])

        return {
            "treatment": self.treatment_name,
            "outcome": self.outcome_name,
            "coef": round(coef, 4),
            "std_err": round(std_err, 4),
            "t_stat": round(t_stat, 2),
            "p_val": p_val,
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "statistically_significant": bool(p_val < 0.05)
        }

    def get_propensity_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts out-of-fold propensity scores m(X) for treated and control units
        to verify common support / positivity assumption.
        """
        if self.model is None or self.model.predictions is None:
            raise ValueError("Predictions are not stored. Fit model first.")

        d_vals = self.model._dml_data.d.flatten()
        m_hat = self.model.predictions["ml_m"].flatten()

        treated_ps = m_hat[d_vals == 1]
        control_ps = m_hat[d_vals == 0]

        return treated_ps, control_ps
