# CardGuard — Credit Card Fraud Detection

CardGuard is an educational, reproducible machine-learning project for identifying rare fraudulent transactions in the ULB credit-card benchmark dataset. The current implementation is **V2**, built around grouped train/validation/test partitions, training-only preprocessing, validation-based model/threshold selection, and a Streamlit scoring dashboard.

## Dataset and attribution

Download `creditcard.csv` from the [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the repository root beside `fraud_pipeline_v2.py`. Kaggle requires an account to download. The dataset was collected and analyzed through a research collaboration between Worldline and the Machine Learning Group at Université Libre de Bruxelles (ULB); review Kaggle's dataset terms before reuse or redistribution.

The supplied local file has 284,807 transactions, 30 numeric input features (`Time`, `Amount`, and anonymized `V1`–`V28`) and target `Class`. There are 492 fraud labels (0.173%), no missing cells, 1,081 repeated copies across 773 exact feature patterns, and no feature patterns with conflicting labels. The file is intentionally excluded from Git. The saved metadata records the SHA-256 of the dataset used for the measured run.

## V2 method

V2 is implemented in `fraud_pipeline_v2.py`; `credit_card_fraud_detection_2_0.ipynb` explains and calls the same workflow. It groups identical predictor rows before a fixed-seed stratified split so exact feature groups cannot cross partitions. The observed split is:

| Partition | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| Train | 182,263 | 318 | 0.1745% |
| Validation | 45,596 | 78 | 0.1711% |
| Test | 56,948 | 96 | 0.1686% |

Preprocessing is part of each scikit-learn pipeline and is fit on training rows only: `log1p(Amount)` followed by standardization, standardization of `Time`, and pass-through of the already transformed `V1`–`V28`. The workflow compares class-weighted Logistic Regression, balanced Random Forest, and XGBoost with training-derived `scale_pos_weight`. It selects the model by validation PR-AUC and selects its operating threshold by validation F1; the test partition is held out until the final evaluation. The fixed seed is 42 (the second group split uses seed 43).

The selected model is a Random Forest with a validation-selected threshold of 0.33975597. Verified held-out results from the saved run are:

| Test metric | Result |
|---|---:|
| Precision | 0.9474 |
| Recall | 0.7500 |
| F1 | 0.8372 |
| ROC-AUC | 0.9388 |
| PR-AUC (average precision) | 0.8182 |
| Confusion matrix (TN, FP, FN, TP) | (56,848, 4, 24, 72) |

These are benchmark results on this fixed split, not production guarantees. At the selected threshold the model raises 76 alerts and misses 24 of 96 test fraud cases. The dataset covers roughly two days and has anonymized features; temporal/source shift, alert capacity, costs, privacy, monitoring, and human review need separate treatment before any operational use.

## Requirements and setup

Use Python 3.13 on Windows. Check the interpreter first; the pinned package versions are in `requirements.txt`. From the repository root in PowerShell:

```powershell
python --version  # confirm Python 3.13
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dataset file `creditcard.csv` is ignored by Git and must be downloaded before evaluation or before loading dashboard demo rows. The saved V2 model, metadata, comparison CSV, and evaluation figures are tracked in the repository, so a fresh clone can launch the dashboard without training.

## Run evaluation

From the repository root with the environment activated:

```powershell
python fraud_pipeline_v2.py
```

The script requires `creditcard.csv` at the repository root and refuses to overwrite the tracked V2 outputs. To reproduce the evaluation, use a separate scratch copy that contains the source files and dataset but not the existing V2 output files; the script writes `model_comparison_v2.csv`, the model and metadata under `models/`, and V2 figures under `figures/`. The metrics in this README come from the saved tracked evaluation artifacts and were not regenerated as part of this documentation update.

## Launch the Streamlit dashboard

The dashboard loads the saved pipeline and selected threshold; it does not train. From the repository root:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Single-row scoring needs all 30 feature inputs. Batch CSV scoring expects those same model columns; an optional `Class` column is ignored. The app can load demonstration rows from `creditcard.csv`; manual single-row input and batch scoring do not require that source CSV. Uploaded transaction rows are handled in memory and are not saved by the app. The committed V2 model is available immediately after cloning.

## Tests

The dashboard tests use the saved model and, for dataset-backed examples, the local CSV:

```powershell
python -m unittest -v test_dashboard_v2.py
```

## Project structure

```text
.
├── app.py                                  # Streamlit dashboard
├── fraud_pipeline_v2.py                   # V2 data, split, train, evaluate, save
├── credit_card_fraud_detection_2_0.ipynb  # V2 explained notebook
├── credit_card_fraud_detection_v2.ipynb   # V2 development notebook
├── credit_card_fraud_detection.ipynb      # original notebook, preserved
├── README.md                               # current V2 guide
├── README_V2.md                            # retained prior V2 documentation
├── requirements.txt
├── test_dashboard_v2.py
├── creditcard.csv                          # local download; Git-ignored
├── model_comparison_v2.csv                 # tracked measured comparison
├── models/                                 # tracked fitted pipeline and metadata
└── figures/                                # V2 evaluation plots
```

The original notebook's unstratified Logistic Regression output (about 97.47% accuracy and 0.10 fraud-class F1) is a historical, imbalanced-baseline experiment, not the V2 grouped held-out benchmark above. The notebook and plot intent remain preserved; the plot asset was regenerated from the same class labels with corrected layout. `README.md` is the current entry point; `README_V2.md` is retained as historical V2 documentation.

## Limitations

- The model is trained and evaluated on a public historical benchmark, not a live bank stream.
- A random grouped split avoids exact feature-row overlap but does not provide future-period validation.
- Fraud is extremely rare, so precision/recall tradeoffs depend on real review capacity and error costs.
- The anonymized PCA-like features limit interpretation, and predictions should support human investigation rather than automate adverse decisions.
- Dataset license/terms and local data security must be checked before redistribution or deployment.
