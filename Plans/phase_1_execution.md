# Phase 1 Execution Report: Data Preparation, Feature Engineering & Causal Framing
**Project**: Causal Machine Learning for Supply Chain Risk Prediction & Intervention Planning  
**Reference Paper**: Wyrembek, M., Baryannis, G., & Brintrup, A. (2025). *Causal machine learning for supply chain risk prediction and intervention planning*. International Journal of Production Research, 63(15), 5629–5648.  
**Execution Date**: November 2026  
**Status**: Completed (Zero loose ends, verified with tests & end-to-end data pipeline)

---

## 1. Executive Summary & Objective

In traditional Supply Chain Risk Management (SCRM), machine learning is predominantly predictive and reactive—identifying correlation patterns to forecast delays. However, supply chain planners require prescriptive and causal insights: **what specific interventions will actively prevent or minimize delivery delays before they manifest?**

This report documents the full execution of **Phase 1: Preparation & Problem Formulation**, translating the theoretical methodology of Wyrembek et al. (2025) onto the empirical DataCo Smart Supply Chain dataset (`archive/DataCoSupplyChainDataset.csv`). Phase 1 establishes the foundational data architecture, eliminates missing data/anomalies, formalizes the structural causal variables (Target $Y$, Treatments $D$, and Covariates $X$), and exports type-safe, production-ready artifacts for downstream causal discovery (Phase 2 autoDAGs) and Double Machine Learning (Phase 3 IRM).

---

## 2. Dataset Audit & Preprocessing Methodology

### 2.1 Raw Data Ingestion & Audit
The raw dataset was loaded and validated through `src/data/loader.py`:
- **File**: `archive/DataCoSupplyChainDataset.csv` (symlinked into `data/raw/`)
- **Total Raw Records**: 180,519 transactions
- **Total Attributes**: 53 columns
- **File Size**: 91.47 MB
- **Duplicate Records**: 0

#### Missing Value Audit:
| Column Name | Missing Count | % Missing | Action Taken | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `Product Description` | 180,519 | 100.0% | Dropped | Zero information content |
| `Order Zipcode` | 155,679 | 86.24% | Dropped | Severe sparsity; destination captured via `Order City`, `Order State`, `Order Region`, `Order Country` |
| `Customer Lname` | 8 | 0.004% | Imputed with `"Unknown"` | Minor customer metadata gap |
| `Customer Zipcode` | 3 | 0.002% | Imputed with `0` | Numerical postal code gap |
| Masked PII (`Customer Email`, `Customer Password`, `Product Image`) | N/A | N/A | Dropped | Redundant / uninformative for tabular causal inference |

### 2.2 Date Parsing & Temporal Feature Extraction
- **Order Date Range**: `2015-01-01 00:00:00` to `2018-01-31 23:38:00` (spanning 3 complete calendar years).
- **Shipping Date Range**: `2015-01-03 00:00:00` to `2018-02-06 22:14:00`.
- Extracted temporal attributes: `order_year`, `order_month`, `order_day`, `order_day_of_week`, `order_hour`, and quarterly `Season` (`Q1`, `Q2`, `Q3`, `Q4`).
- Derived individual binary seasonal indicators (`season_q1`, `season_q2`, `season_q3`, `season_q4`) enabling exact replication of the paper's seasonal treatment estimation.

---

## 3. Causal Framing & Entity Mapping

Wyrembek et al. (2025, Table 3) defined a specific causal schema for supply chain delay intervention. We mapped the DataCo supply chain variables directly to these constructs:

| Paper Construct (Table 3) | Paper Description | DataCo Empirical Variable | Causal Role | Data Type | Notes / Alignment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Delay** | Duration of delay in days | `delay_days = Days for shipping (real) - Days for shipment (scheduled)` | Outcome ($Y$) | Continuous (Float) | Primary target for Interactive Regression Model (IRM) |
| **Late_delivery_risk** | Binary late delivery indicator | `late_risk` ($1$ = late delivery, $0$ = advance/on-time) | Outcome ($Y_{bin}$) | Binary ($0/1$) | Classification target & robustness check |
| **Season** | Quarter of the year | `Season` (`Q1`, `Q2`, `Q3`, `Q4`) & one-hot `season_q1..q4` | Treatment ($D_{season}$) | Binary ($0/1$) | Evaluates seasonal timing interventions (Table 5 of paper) |
| **Multi** | Supplier supplies to >1 buyer in network | `Multi` ($1$ = supplies $>1$ market, $0$ = single market) | Treatment ($D_{multi}$) | Binary ($0/1$) | Measures multi-facility competition / prioritization effects |
| **Shipping Mode** | Operational intervention | `Shipping_Expedited` ($1$ = First/Same Day/Second, $0$ = Standard) | Treatment ($D_{shipping}$) | Binary ($0/1$) | Actionable operational mitigation channel |
| **Quantity** | Ordered product quantity | `Order Item Quantity` | Covariate ($X$) | Numeric | Confounder & conditioning variable for CATE B-splines |
| **Price** | Ordered product price | `Order Item Product Price` | Covariate ($X$) | Numeric | Confounder & conditioning variable for CATE B-splines |
| **Part Description** | Characteristic description | `Category Name` & `Department Name` | Covariate ($X$) | Categorical | High-dimensional product confounder |
| **Project** | Intended project use | `Customer Segment` (Consumer, Corporate, Home Office) | Covariate ($X$) | Categorical | Project / customer priority proxy |
| **Network Context** | Geographical destination | `Market` & `Order Region` | Covariate ($X$) | Categorical | Geographic routing confounders |

---

## 4. Empirical Statistical & Causal Summary

### 4.1 Delivery Delay Statistics
- **Total Validated Orders**: 180,519
- **Delayed Orders Count**: 98,977 (**54.83%** late delivery rate)
  - *Paper comparison*: The paper's empirical case study exhibited delay rates of 51% (Buyer B2), 56% (Buyer B1), and 68% (Buyer B3). The DataCo empirical rate of 54.83% aligns closely with real-world industrial benchmarks.
- **Continuous Delay Days ($Y$)**:
  - Mean: **+0.5658 days**
  - Standard Deviation: **1.4910 days**
  - Minimum: **-2.0 days** (arrived 2 days ahead of scheduled delivery date)
  - Maximum: **+4.0 days** (arrived 4 days late)

### 4.2 Delay Duration Distribution
| Delay Duration (Days) | Order Count | Percentage (%) | Classification |
| :--- | :--- | :--- | :--- |
| **-2 days** | 21,666 | 12.00% | Advance Shipping |
| **-1 day** | 21,700 | 12.02% | Advance Shipping |
| **0 days** | 33,753 | 18.70% | Shipping On Time |
| **+1 day** | 60,647 | 33.59% | Late Delivery |
| **+2 days** | 28,718 | 15.91% | Late Delivery |
| **+3 days** | 7,052 | 3.91% | Late Delivery |
| **+4 days** | 6,983 | 3.87% | Late Delivery |

### 4.3 Treatment Contrast Analysis

#### A. Seasonal Treatment ($D_{season}$):
| Season | Total Orders | Late Delivery Rate (%) | Mean Delay (Days) |
| :--- | :--- | :--- | :--- |
| **Q1** (Jan – Mar) | 48,427 | 54.82% | 0.5697 days |
| **Q2** (Apr – Jun) | 46,550 | 54.61% | 0.5641 days |
| **Q3** (Jul – Sep) | 47,323 | 55.05% | 0.5667 days |
| **Q4** (Oct – Dec) | 38,219 | 54.83% | 0.5618 days |

#### B. Sourcing Network Treatment ($D_{multi}$):
- `Multi = 0` (Single-Market Focus): Mean delay = **0.5925 days**
- `Multi = 1` (Multi-Market Network): Mean delay = **0.5651 days**

#### C. Shipping Mode Intervention ($D_{shipping}$):
- Standard Class (`Shipping_Expedited = 0`, 59.69% of orders): Mean delay = **-0.0041 days** (On-time baseline)
- Expedited Class (`Shipping_Expedited = 1`, 40.31% of orders): Mean delay = **+1.4097 days**  
  *Insight*: Expedited orders carry higher delay rates due to tighter delivery windows (scheduled delivery in 1–2 days rather than 4 days), indicating clear confounding between order urgency and delivery risk. This proves the necessity of Double/Debiased Machine Learning (DML) in Phase 3 to isolate the true causal effect from confounding covariates ($X$).

---

## 5. Modular Codebase Architecture

The codebase has been structured as a production-ready Python package:

```
Causal-ML-in-supply-chain/
├── src/
│   ├── __init__.py               # Package initializer
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py             # Formatted logging to console and reports/phase1_execution.log
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # DataLoader: encoding handling, schema integrity, checksums
│   │   └── preprocessor.py       # DataPreprocessor: cleaning, missing values, timestamps, delay targets
│   └── features/
│       ├── __init__.py
│       └── causal_builder.py     # CausalFeatureBuilder: Y, D, X mapping and one-hot matrix builder
├── scripts/
│   └── run_phase1.py             # End-to-end executable pipeline with CLI arguments
├── tests/
│   └── test_phase1.py            # Unit test suite (unittest) validating schema, shapes, and absence of NaNs
├── data/
│   ├── raw/                      # Symlinks to raw archive files
│   └── processed/
│       ├── causal_supply_chain.parquet          # Canonical dataset (180,519 x 19)
│       ├── causal_supply_chain_encoded.parquet  # DML-ready one-hot matrix (180,519 x 73)
│       ├── causal_dataset_sample.csv            # 1,000-row sample for human inspection
│       └── causal_features_summary.json         # Complete schema & statistical metadata
├── reports/
│   ├── phase1_execution.log      # Pipeline execution log
│   └── figures/                  # Publication-quality diagnostic plots
│       ├── 01_delay_distribution.png
│       ├── 02_seasonal_delay_effects.png
│       ├── 03_multi_market_treatment.png
│       ├── 04_feature_correlation_matrix.png
│       └── 05_shipping_mode_delay.png
└── Plans/
    ├── initial_plan.md           # 4-Phase project roadmap
    └── phase_1_execution.md      # This comprehensive Phase 1 documentation
```

---

## 6. Verification & Quality Assurance

1. **Unit Test Suite**:
   Executed with `./venv/bin/python3 -m unittest tests/test_phase1.py -v`:
   - `test_01_loader_schema`: Passed (all 14 core supply chain fields verified).
   - `test_02_preprocessor_transform`: Passed (timestamp parsing, `delay_days` arithmetic, binary assertions, zero NaNs).
   - `test_03_causal_builder`: Passed (causal canonical mapping, one-hot encoding shape verification, zero NaNs).
   - **Result**: `Ran 3 tests in 0.059s - OK`.

2. **Downstream Package Interoperability**:
   - **DoubleML Compatibility**: Verified instantiation of `DoubleMLData(causal_df, y_col='delay_days', d_cols='Multi', x_cols=...)` on the processed parquet dataset.
   - **pgmpy / causal-learn Compatibility**: Verified structure learning via `HillClimbSearch` with `K2` and `BIC` scoring.

---

## 7. Next Steps for Phase 2

With Phase 1 completed without loose ends, the project is positioned for **Phase 2: Causal Discovery & Directed Acyclic Graph (DAG) Generation**:
1. Implement heuristic score-based learning via **Hill Climbing** (K2 score) in `src/models/dag_hill_climbing.py`.
2. Implement metaheuristic score-based learning via **Tabu Search** (BIC score) in `src/models/dag_tabu_search.py`.
3. Implement constraint-based causal structure discovery via the **Peter-Clark (PC) algorithm** (`causal-learn`) in `src/models/dag_pc.py`.
4. Construct and visualize the **Domain Expert Knowledge DAG** (Figure 3 in paper) and evaluate d-separation paths to validate causal identifiability.
