# Causal Machine Learning in Supply Chain Risk Prediction & Intervention Planning
## Phase-Wise Implementation Roadmap

Based on: *Wyrembek, M., Baryannis, G., & Brintrup, A. (2025). Causal machine learning for supply chain risk prediction and intervention planning. International Journal of Production Research, 63(15), 5629–5648.*

---

## Executive Summary & Research Framework

Traditional machine learning in supply chain risk management (SCRM) relies on correlation-based pattern recognition, which is insufficient for prescriptive intervention planning. A planner predicting a supplier delivery delay needs to know not merely *that* a delay will occur, but *what intervention* (e.g. altering order season, choosing single vs. multi-destination suppliers, or expediting shipment mode) will minimize delivery delay duration.

This project implements the end-to-end Causal Machine Learning (CML) framework introduced by Wyrembek et al. (2025), structured across four comprehensive phases:

```
+-----------------------------------------------------------------------------------+
|                           PHASE 1: PREPARATION & PROBLEM SETUP                     |
|  * Problem Definition & Causal Framing                                           |
|  * Data Ingestion, Profiling, Cleaning & Imputation                                |
|  * Causal Variable Formulation: Target (Y), Treatments (D), Covariates (X)       |
|  * Exploratory Causal & Statistical Analysis                                      |
|  * Modular Codebase Architecture & Data Persistence                               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           PHASE 2: CAUSAL DISCOVERY & DAG GENERATION              |
|  * Heuristic Score-Based autoDAGs: Hill Climbing (K2 Score)                       |
|  * Metaheuristic Score-Based autoDAGs: Tabu Search (BIC Score)                    |
|  * Constraint-Based autoDAG: Peter-Clark (PC) Algorithm (Conditional Independence)|
|  * Domain Expert Knowledge DAG Construction & Causal Channel Validation           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           PHASE 3: DOUBLE MACHINE LEARNING (DML) & IRM            |
|  * Interactive Regression Model (IRM) formulation                                 |
|  * High-capacity Nuisance Estimators (XGBClassifier for m(X), XGBRegressor for g(X))|
|  * Neyman-Orthogonal Score Functions & K-Fold Cross-Fitting                       |
|  * Average Treatment Effect (ATE) Estimation for All Intervention Channels        |
|  * Statistical Significance Testing (t-statistics, standard errors, p-values)     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           PHASE 4: GRANULAR CATE, POLICY TREES & DECISION SYSTEM  |
|  * Conditional Average Treatment Effects (CATE) using B-Spline Basis Functions    |
|  * Heterogeneity Analysis across Covariates (Order Quantity, Price, Product Tier) |
|  * Deterministic Binary Policy Tree Induction for Prescriptive Intervention Rules |
|  * Interactive Streamlit Decision Support Application for Supply Chain Planners    |
+-----------------------------------------------------------------------------------+
```

---

## Detailed Phase Breakdown

### Phase 1: Preparation & Causal Problem Formulation (Current Phase)
- **Objective**: Establish the causal objective, clean and preprocess raw supply chain records, map domain entities to causal variables, perform exploratory causal data analysis, and establish a modular pipeline.
- **Key Deliverables**:
  1. Complete ingestion and audit of `archive/DataCoSupplyChainDataset.csv` (180,519 records).
  2. Cleaning pipeline: handling missing values, parsing order and shipping timestamps, calculating continuous delay days (`Days for shipping (real) - Days for shipment (scheduled)`) and binary late risk (`Late_delivery_risk`).
  3. Causal variable mapping matching Table 3 of the paper:
     - **Outcome ($Y$)**: Continuous delivery delay duration in days, and binary delay risk.
     - **Treatments ($D$)**:
       - `Season`: Quarterly indicators (`Q1`, `Q2`, `Q3`, `Q4`) capturing seasonal order variability.
       - `Multi`: Binary indicator representing multi-market fulfillment / network congestion.
       - `Shipping_Mode_Expedited`: Operational treatment (expedited vs. standard shipping).
     - **Confounders / Covariates ($X$)**: Order Item Quantity, Product Price, Product Category / Department, Customer Segment, Market, Order Region.
  4. Exploratory Data Analysis & Causal Association Reports.
  5. Clean modular codebase (`src/data/`, `src/features/`, `src/utils/`, `scripts/`, `tests/`) and exported datasets in `data/processed/`.

---

### Phase 2: Causal Discovery & Directed Acyclic Graph (DAG) Generation
- **Objective**: Learn the causal graph structure from observational data to make causal quantities identifiable and compare data-driven autoDAGs against domain-knowledge DAGs.
- **Key Deliverables**:
  1. Score-based Bayesian Network structure learning:
     - **Hill Climbing**: Local greedy optimization maximizing K2 score (`pgmpy`).
     - **Tabu Search**: Metaheuristic with memory tabu list maximizing BIC score (`pgmpy`).
  2. Constraint-based structure learning:
     - **Peter-Clark (PC) Algorithm**: Independence testing using Fisher z-test / Chi-square test, v-structure orientation, and Meek rules (`causal-learn`).
  3. Domain Expert DAG construction reflecting the structural equations in Wyrembek et al. (Figure 3).
  4. Structural equation analysis and d-separation checks to confirm causal channels from treatments to delay.

---

### Phase 3: Double/Debiased Machine Learning (DML) & Interactive Regression Model
- **Objective**: Overcome regularization bias and confounding through Neyman-orthogonalized score functions and sample splitting to estimate Average Treatment Effects (ATE).
- **Key Deliverables**:
  1. Interactive Regression Model (IRM) specification:
     $$Y = g_0(D, X) + U, \quad E[U \mid X, D] = 0$$
     $$D = m_0(X) + V, \quad E[V \mid X] = 0$$
  2. Nuisance model training with 5-fold cross-fitting using `XGBClassifier` and `XGBRegressor` with hyperparameters tuned via GridSearchCV (max_depth=7-8, n_estimators=300-400, lr=0.05, subsample=0.7).
  3. Computation of ATE ($\theta_0$):
     $$\theta_0 = E[g_0(1, X) - g_0(0, X)]$$
  4. Robust statistical inference: confidence intervals, standard errors, and p-values for all treatments.

---

### Phase 4: Granular CATE, Policy Design & Intervention Decision Support
- **Objective**: Estimate heterogeneous treatment effects across order quantities and prices, learn optimal decision trees for interventions, and deploy an interactive decision support tool.
- **Key Deliverables**:
  1. Conditional Average Treatment Effect (CATE) estimation:
     $$\theta_0(x) = E[Y(1) - Y(0) \mid X = x]$$
     using B-splines basis functions (5 degrees of freedom) over `Quantity` and `Price`.
  2. Deterministic binary policy tree induction maximizing expected reward:
     $$\hat{\pi} = \arg\max_{\pi \in \Pi} \sum_{i=1}^n (2\pi(X_i) - 1)\psi_b(W_i, \hat{\eta})$$
  3. Interactive Streamlit dashboard enabling supply chain planners to run "what-if" counterfactual simulations and receive intervention recommendations.
