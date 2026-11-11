# Windows execution guide — Fraud Detection 2.0

## Requirements and fresh-clone setup

Use Python 3.13 and VS Code with the Microsoft Python and Jupyter extensions. Open the project folder in VS Code, then open an integrated terminal using **Command Prompt**. From the project root, create and activate the local virtual environment and install the pinned requirements:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `.venv` directory is local and ignored by Git, so create it again for each fresh clone. If you open a new terminal later, activate it with `.venv\Scripts\activate.bat` before running the project.

## Run the complete ML workflow

```cmd
.venv\Scripts\python.exe fraud_pipeline_v2.py
```

The workflow uses fixed seeds, creates feature-group-disjoint train/validation/test partitions, trains three models, selects by validation PR-AUC, selects the winning model threshold using validation F1, and evaluates only that model/threshold on the untouched test set. It creates the comparison CSV, figures, model, and metadata. Existing V1/Phase 1 files are not used as output paths.

## Run the Streamlit dashboard

In VS Code, open the project folder and use its integrated Command Prompt terminal. Run these commands from the project root:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local address when the app starts. Open this URL in your browser:

```text
http://localhost:8501
```

The dashboard loads the saved Random Forest pipeline and validation-selected threshold; it does not train a model. Keep the terminal open while using the dashboard. Press `Ctrl+C` in that terminal to stop it.

## Open the V2 notebook

Open `credit_card_fraud_detection_2_0.ipynb` and select `.venv\Scripts\python.exe` as the kernel. Run its cells in order. The notebook re-runs the workflow in memory without replacing saved artifacts.

## Generated outputs

- `model_comparison_v2.csv` — validation metrics for each model at 0.5 and its validation-F1 threshold, plus the selected model's one final test row.
- `models\fraud_detection_pipeline_v2.joblib` — complete preprocessing and selected classifier pipeline.
- `models\fraud_detection_metadata_v2.json` — selected threshold, split counts, versions, and final metrics.
- `figures\*..._v2.png` — comparison, threshold, and confusion-matrix charts.

## Reproducibility

Seeds are fixed in the code and dataset SHA-256 is recorded in the metadata. The script refuses to replace existing V2 artifacts. To preserve them and run a second experiment, copy the project to a new folder or create a separately named V2 experiment/output set.
