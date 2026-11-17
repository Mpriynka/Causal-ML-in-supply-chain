from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import patsy
from doubleml import DoubleMLIRM
from src.utils.logger import setup_logger

logger = setup_logger("cate_splines")

class CATESplineEstimator:
    """
    Estimates Conditional Average Treatment Effects (CATE) theta_0(x) = E[Y(1) - Y(0) | X = x]
    using a B-splines basis with 5 degrees of freedom, reproducing Section 4.2 & Figure 5
    from Wyrembek et al. (2025).
    """

    def __init__(self, df_degrees: int = 5, spline_degree: int = 3):
        self.df_degrees = df_degrees
        self.spline_degree = spline_degree
        self.feature_name: Optional[str] = None
        self.cate_blp = None
        self.design_info = None
        self.feature_min: float = 0.0
        self.feature_max: float = 1.0

    def fit(
        self,
        irm_model: DoubleMLIRM,
        df: pd.DataFrame,
        feature_name: str
    ) -> "CATESplineEstimator":
        """
        Constructs the B-spline basis matrix and fits the best linear predictor on the IRM scores.
        """
        self.feature_name = feature_name
        self.feature_min = float(df[feature_name].min())
        self.feature_max = float(df[feature_name].max())

        logger.info(
            f"Fitting B-spline CATE on feature '{feature_name}' (df={self.df_degrees}, "
            f"degree={self.spline_degree}) over range [{self.feature_min:.2f}, {self.feature_max:.2f}]"
        )

        # Build cubic B-spline basis formula without intercept
        formula = f"bs({feature_name}, df={self.df_degrees}, degree={self.spline_degree}) - 1"
        basis = patsy.dmatrix(formula, df, return_type="dataframe")
        self.design_info = basis.design_info

        # Estimate CATE via best linear predictor on orthogonal DML score
        self.cate_blp = irm_model.cate(basis)

        logger.info(f"CATE spline model fitted successfully. Coef count: {len(self.cate_blp.coef)}")
        return self

    def predict_curve(
        self,
        grid_values: Optional[np.ndarray] = None,
        num_points: int = 100,
        ci_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Evaluates point estimates and asymptotic confidence intervals across a continuous grid.
        """
        if self.cate_blp is None or self.design_info is None:
            raise ValueError("CATE Spline estimator has not been fitted yet!")

        if grid_values is None:
            grid_values = np.linspace(self.feature_min, self.feature_max, num_points)

        eval_df = pd.DataFrame({self.feature_name: grid_values})
        grid_basis = patsy.build_design_matrices([self.design_info], eval_df, return_type="dataframe")[0]
        grid_mat = np.array(grid_basis)

        coef = np.array(self.cate_blp.coef)
        # Extract covariance matrix of parameters
        if hasattr(self.cate_blp, "blp_omega") and self.cate_blp.blp_omega is not None:
            cov_mat = np.array(self.cate_blp.blp_omega)[:, :, 0]
        else:
            cov_mat = self.cate_blp.blp_model[0].cov_params()

        pred_cate = np.dot(grid_mat, coef)
        pred_var = np.diag(np.dot(np.dot(grid_mat, cov_mat), grid_mat.T))
        # Ensure numerical non-negativity
        pred_se = np.sqrt(np.maximum(pred_var, 0.0))

        # Critical value for confidence level (e.g., 1.96 for 95%)
        z_crit = 1.96 if ci_level == 0.95 else 1.645
        ci_lower = pred_cate - z_crit * pred_se
        ci_upper = pred_cate + z_crit * pred_se

        curve_df = pd.DataFrame({
            self.feature_name: grid_values,
            "cate": pred_cate,
            "se": pred_se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
        return curve_df

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured metadata of the fitted spline basis."""
        if self.cate_blp is None:
            raise ValueError("Model not fitted.")
        return {
            "feature": self.feature_name,
            "df_degrees": self.df_degrees,
            "spline_degree": self.spline_degree,
            "coefficients": [float(c) for c in self.cate_blp.coef],
            "standard_errors": [float(s) for s in self.cate_blp.se],
            "feature_domain": [self.feature_min, self.feature_max]
        }
