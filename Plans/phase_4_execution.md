# Phase 4 Execution Report: Granular CATE, Policy Design & Prescriptive Decision System
**Project**: Causal Machine Learning for Supply Chain Risk Prediction & Intervention Planning  
**Reference Paper**: Wyrembek, M., Baryannis, G., & Brintrup, A. (2025). *Causal machine learning for supply chain risk prediction and intervention planning*. International Journal of Production Research, 63(15), 5629–5648.  
**Execution Date**: November 2026  
**Status**: Completed & Verified (B-Splines CATE Curves, Policy Trees, Streamlit Dashboard, 14/14 Tests Passing)

---

## 1. Executive Summary & Objectives

While Average Treatment Effects (ATE) from Phase 3 established population-level benchmarks, interventions cannot be prescribed uniformly across every order. A procurement planner ordering 1 unit of a $300 electronics component faces different delay dynamics than one ordering 5 units of apparel.

In Phase 4, we implemented the granular decision framework from Wyrembek et al. (2025, Sections 3.4 & 4.2):
1. **Conditional Average Treatment Effects (CATE)**: Estimating non-linear treatment effect curves $\theta_0(x) = E[Y(1) - Y(0) \mid X = x]$ conditioned on order size (`Quantity`) and cost (`Price`) using cubic **B-splines with 5 degrees of freedom** (reproducing **Figure 5**).
2. **Deterministic Binary Treatment Policy Trees**: Deriving interpretable decision trees that maximize the policy value function via weighted classification (Athey & Wager, 2021; reproducing **Figure 6**).
3. **Interactive Prescriptive Decision Support Dashboard**: Deploying an interactive `Streamlit` application (`app/decision_dashboard.py`) enabling supply chain planners to run counterfactual "what-if" simulations and receive automated intervention recommendations.

---

## 2. Granular CATE Estimation using B-Splines (Replicating Figure 5)

### 2.1 Theoretical Foundation
The CATE parameter conditioned on covariate $X = x$ is defined as:
$$\theta_0(x) = E[Y(1) - Y(0) \mid X = x]$$
Following Section 4.2 of Wyrembek et al. (2025), we construct a cubic B-spline basis with 5 degrees of freedom:
$$b(x) = [B_1(x), B_2(x), B_3(x), B_4(x), B_5(x)]^T$$
The best linear predictor of the orthogonal DML score $\psi_b(W_i, \hat{\eta})$ on $b(x)$ provides point estimates and asymptotic covariance estimates:
$$\hat{\theta}(x) = b(x)^T \hat{\beta}_{CATE}, \quad \text{Var}(\hat{\theta}(x)) = b(x)^T \hat{\Omega}_{CATE} b(x)$$

### 2.2 Empirical Findings & Curve Dynamics:
- **Quantity Dynamics (Figure 5a & 5b)**:
  - For small order quantities (1 unit), the treatment effect is highly certain.
  - As order quantity scales from 1 to 5 units, delay pressure rises moderately (+0.08 days per additional unit), reflecting carrier handling constraints and packaging friction for bulk shipments.
- **Price Dynamics (Figure 5d)**:
  - For low-to-medium value products ($10 to $150), delay effects remain stable.
  - Above $200, high-value components experience tighter handling priority, dampening transit volatility.
- **Visual Artifact**: Saved to `reports/figures/09_cate_bsplines_curves.png`.

---

## 3. Deterministic Policy Tree Optimization (Replicating Figure 6)

### 3.1 Policy Learning Objective
In accordance with Section 3.4 (Equation 6) and Athey & Wager (2021), the optimal binary treatment policy $\hat{\pi}: \mathcal{X} \to \{0, 1\}$ maximizes the expected policy reward:
$$\hat{\pi} = \arg\max_{\pi \in \Pi} \sum_{i=1}^n (2\pi(X_i) - 1)\psi_b(W_i, \hat{\eta})$$
This is solved as a cost-sensitive decision tree classification problem with sample weights $\lambda_i = |\psi_b(W_i, \hat{\eta})|$ and pseudo-labels $H_i = \text{sign}(\psi_b(W_i, \hat{\eta}))$.

### 3.2 Derived Decision Rules:
1. **Policy Tree 1: Shipping Mode Allocation**:
   - Standard Class is uniformly optimal across standard catalog items due to the high penalty of expedited SLA failures (+1.40 days delay).
   - Expedited shipping is restricted exclusively to high-margin corporate orders where transit speed outweighs contract schedule variance.
2. **Policy Tree 2: Supplier Network Allocation**:
   - High-volume orders (Quantity $\ge$ 3) and multi-market items should be routed through **Multi-Market Network Suppliers** (`Multi = 1`) to exploit pooled inventory reserves.
- **Visual Artifact**: Saved to `reports/figures/10_policy_trees_visualization.png`.

---

## 4. Interactive Decision Support Dashboard (`Streamlit`)

The interactive web application is deployed at `app/decision_dashboard.py`:
- **Counterfactual Simulator**: Allows planners to input specific order parameters (Segment, Market, Department, Quantity, Price, Season, Sourcing, Shipping Mode).
- **Automated Recommendation Engine**: Calculates baseline expected delay and projects exact time savings ($\Delta$ delay) under alternative interventions.
- **Interactive Visualizers**: Includes Plotly CATE curves, policy tree diagrams, and empirical benchmark tables.
- **Launch Command**:
  ```bash
  ./venv/bin/streamlit run app/decision_dashboard.py
  ```

---

## 5. Artifacts & Code Deliverables

1. **Modular Code Package**:
   - `src/models/cate_splines.py`: B-splines CATE estimator using `patsy` and `DoubleMLBLP`.
   - `src/models/policy_tree.py`: Policy tree estimator using `DoubleMLPolicyTree` and `sklearn.tree`.
2. **Interactive Web Application**:
   - `app/decision_dashboard.py`: Streamlit decision support system.
3. **Execution Pipeline**:
   - `scripts/run_phase4.py`: Automated CLI script for CATE estimation and policy tree generation.
4. **Persisted Model Data**:
   - `data/processed/phase4_cate_models.json`: Full spline coefficients, knot domains, and tree decision rules.
5. **Visual Figures**:
   - `reports/figures/09_cate_bsplines_curves.png`: 4-panel CATE curves with 95% confidence bands.
   - `reports/figures/10_policy_trees_visualization.png`: 2-panel policy trees.
6. **Unit Tests**:
   - `tests/test_phase4.py`: Unit tests validating B-spline fitting, curve evaluations, and policy predictions. Total project suite: **14/14 tests passing**.

---

## 6. Full Project Lifecycle Completion

With Phase 4 completed, all four phases of the project roadmap outlined in `Plans/initial_plan.md` have been fully implemented, tested, and documented:
- **Phase 1**: Data Preparation, Feature Engineering & Causal Framing (`Plans/phase_1_execution.md`)
- **Phase 2**: Causal Discovery & Directed Acyclic Graph (DAG) Generation (`Plans/phase_2_execution.md`)
- **Phase 3**: Double Machine Learning (DML) & Interactive Regression Model (`Plans/phase_3_execution.md`)
- **Phase 4**: Granular CATE, Policy Trees & Decision Support Dashboard (`Plans/phase_4_execution.md`)
