# Credit Card Fraud Detection 2.0

A reproducible, leakage-aware binary classification project for detecting fraudulent transactions in a highly imbalanced dataset. The original notebook and Phase 1 audit notebook are separate from V2 artifacts. The V2 training logic is in `fraud_pipeline_v2.py`; `credit_card_fraud_detection_2_0.ipynb` provides an explained notebook workflow that calls the same functions.

## Dataset

The local `creditcard.csv` contains 284,807 transactions, 30 numeric input features and the `Class` label. It has no missing cells. Fraud accounts for 492 rows (0.173%). The anonymized `V1`–`V28` columns are already transformed. `Amount` is strongly right-skewed. There are 1,081 extra exact duplicate copies across 773 repeated feature patterns; there are no repeated feature patterns with conflicting labels. No rows are silently removed.

## Leakage-aware split

Exact feature rows (all predictors except `Class`) are grouped first. Unique groups are then split into training, validation, and test partitions with stratification at the group level and fixed seeds. No feature group crosses a partition. The measured row-level fraud counts are:

| Partition | Rows | Fraud | Fraud share |
|---|---:|---:|---:|
| Train | 182,263 | 318 | 0.1745% |
| Validation | 45,596 | 78 | 0.1711% |
| Test | 56,948 | 96 | 0.1686% |

Because repeated groups can have different sizes, group-level stratification may not make row percentages exactly equal. Here the largest difference from the dataset rate is only 0.0042 percentage points and every partition contains fraud cases, so the group-aware split is suitable.

## Preprocessing and models

All learned preprocessing lives inside each scikit-learn `Pipeline` and is fitted on training data only. `Amount` is transformed with `log1p` and then standardized; `Time` is standardized. `V1`–`V28` pass through unchanged because they are already transformed numeric variables.

The workflow compares:

- Logistic Regression with balanced class weights.
- Random Forest with balanced subsample weights.
- XGBoost with `scale_pos_weight` computed from training labels.

The models use fixed, moderate settings; no expensive hyperparameter search is run.

## Evaluation and threshold policy

All three models are compared on validation data. The CSV reports Precision, Recall, F1, ROC-AUC, PR-AUC (average precision), confusion-matrix counts, false positives, false negatives, and alert counts at the default 0.5 threshold and each model's validation F1-maximizing threshold. The selected model is chosen by validation PR-AUC, not accuracy.

Random Forest has the best validation PR-AUC (0.8412). Its default threshold (0.5) yields validation Precision 0.9524, Recall 0.7692, and F1 0.8511. Maximizing validation F1 selects threshold 0.339756; this raises validation Recall to 0.8333, with 5 false-positive alerts and 13 missed fraud cases. This illustrates the tradeoff: lowering a threshold often catches more fraud but may flag more legitimate transactions.

Only after model and threshold selection, the chosen Random Forest and its validation-selected threshold are evaluated on the untouched test partition once:

| Metric | Final test result |
|---|---:|
| Precision | 0.9474 |
| Recall | 0.7500 |
| F1-score | 0.8372 |
| ROC-AUC | 0.9388 |
| PR-AUC (average precision) | 0.8182 |
| False positives | 4 |
| False negatives | 24 |
| Confusion matrix (TN, FP, FN, TP) | (56,848, 4, 24, 72) |

At the validation-selected threshold, the final test set contains 76 alerts: 72 fraud cases and 4 legitimate transactions. It still misses 24 of 96 fraud cases. This is a benchmark result, not a production guarantee; real deployment needs an agreed alert capacity, cost model, monitoring, and stronger time-based validation.

## Run on Windows

Use Python 3.13, VS Code, and the project `.venv`. Detailed commands are in [WINDOWS_EXECUTION_GUIDE_V2.md](WINDOWS_EXECUTION_GUIDE_V2.md). From the project folder:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe fraud_pipeline_v2.py
```

The script refuses to overwrite existing V2 output artifacts. It writes the comparison table, V2 model and metadata, and V2 charts.

## V2 outputs

- `model_comparison_v2.csv` — validation comparison and the selected-model test result.
- `models/fraud_detection_pipeline_v2.joblib` — complete fitted preprocessing + model pipeline.
- `models/fraud_detection_metadata_v2.json` — threshold, metrics, split counts, package versions, and dataset SHA-256.
- `figures/model_comparison_dashboard_v2.png` — readable two-panel comparison of ROC/PR-AUC and default-threshold metrics.
- `figures/model_comparison_v2.png` — initial comparison chart.
- `figures/validation_threshold_tradeoff_v2.png`
- `figures/validation_confusion_matrices_v2.png`
- `figures/test_confusion_matrix_v2.png`

The Phase 1 audit charts are also retained in `figures/` with their `_v2.png` names. V2 outputs use distinct filenames and do not overwrite the original notebook or `fraud_distribution.png`.


## Streamlit dashboard

The local dashboard loads the saved Random Forest pipeline and validation-selected threshold. It does not retrain the model. It supports one-row scoring and in-memory CSV batch scoring, shows fraud probabilities and threshold-based decisions, and explains false positives and false negatives. Uploaded rows are not saved by the app.

Install dependencies in the project virtual environment and launch the dashboard from the project folder:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Batch CSV files must include all 30 model input columns listed in `models/fraud_detection_metadata_v2.json`; a `Class` column is ignored. Unknown extra columns are carried through as context in the result but are not passed to the model.

The saved benchmark test result at the fixed validation-selected threshold (0.33975597) is Precision 0.9474, Recall 0.7500, F1 0.8372, ROC-AUC 0.9388, and PR-AUC 0.8182, with 4 false positives and 24 false negatives. These historical benchmark figures do not establish production performance. The dataset has 492 frauds across a roughly two-day period, the features are anonymized, and a real deployment would require future-period validation, an agreed review-capacity policy, privacy/security review, monitoring, and human oversight.

Run the dashboard helper tests with:

```cmd
.venv\Scripts\python.exe -m unittest -v test_dashboard_v2.py
```
