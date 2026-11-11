"""Local Streamlit dashboard for Credit Card Fraud Detection V2."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_detection_pipeline_v2.joblib"
METADATA_PATH = ROOT / "models" / "fraud_detection_metadata_v2.json"


def _load_assets(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    """Load and sanity-check the saved pipeline and its sidecar metadata."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not support probability predictions.")
    if not isinstance(metadata.get("feature_columns"), list) or not metadata["feature_columns"]:
        raise ValueError("Metadata is missing the required feature_columns list.")
    threshold = float(metadata.get("selected_threshold", float("nan")))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Metadata contains an invalid decision threshold.")
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features and model_features != metadata["feature_columns"]:
        raise ValueError("Model feature order does not match metadata.")
    estimator = getattr(model, "named_steps", {}).get("model")
    if estimator is None:
        raise ValueError("Expected a saved preprocessing + estimator Pipeline.")
    return model, metadata


@st.cache_resource
def load_assets():
    """Cache the model so Streamlit does not reload it on every interaction."""
    return _load_assets()


def prepare_features(data: pd.DataFrame, feature_columns: list[str]):
    """Validate input and return model columns in the exact saved order."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("The input contains no transactions.")
    frame = data.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique after trimming spaces.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    notices = []
    if "Class" in frame.columns:
        notices.append("The `Class` target column was ignored; predictions use input features only.")
    extras = [column for column in frame.columns if column not in feature_columns and column != "Class"]
    if extras:
        notices.append("Ignored non-model columns: " + ", ".join(extras[:12]) + (" …" if len(extras) > 12 else ""))
    features = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = converted.isna()
        if invalid.any():
            rows = [str(i) for i in features.index[invalid][:5].tolist()]
            raise ValueError(f"Column `{column}` contains missing or non-numeric values (data row indices: {', '.join(rows)}).")
        features[column] = converted.astype(float)
    numeric = features.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        row_positions, col_positions = np.where(~np.isfinite(numeric))
        examples = ", ".join(f"row {int(r)}, {feature_columns[int(c)]}" for r,c in zip(row_positions[:5], col_positions[:5]))
        raise ValueError("Inputs must be finite numbers. Invalid values found at " + examples + ".")
    for nonnegative in ("Time", "Amount"):
        if nonnegative in features and (features[nonnegative] < 0).any():
            raise ValueError(f"`{nonnegative}` cannot be negative.")
    return features, notices, extras


def predict_transactions(data: pd.DataFrame, model, metadata: dict):
    """Score valid transactions and apply the saved validation-selected threshold."""
    features, notices, extras = prepare_features(data, metadata["feature_columns"])
    probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if len(probability) != len(features) or not np.isfinite(probability).all():
        raise RuntimeError("The model returned invalid fraud probabilities.")
    threshold = float(metadata["selected_threshold"])
    decisions = np.where(probability >= threshold, "Fraud", "Legitimate")
    context_columns = [column for column in extras if column != "Class"]
    result = data.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if "Class" in result:
        result = result.drop(columns="Class")
    result = result.loc[:, context_columns].copy() if context_columns else pd.DataFrame(index=features.index)
    for column in metadata["feature_columns"]:
        result[column] = features[column]
    result["Fraud Probability"] = probability
    result["Decision"] = decisions
    result["Decision Threshold"] = threshold
    return result, notices


def predict_csv_bytes(csv_bytes: bytes, model, metadata: dict):
    """Parse an uploaded CSV in memory and return predictions without saving it."""
    try:
        data = pd.read_csv(io.BytesIO(csv_bytes))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read this CSV: {exc}") from exc
    return predict_transactions(data, model, metadata)


def _render_results(results: pd.DataFrame, threshold: float, key: str):
    fraud_count = int((results["Decision"] == "Fraud").sum())
    legitimate_count = int(len(results) - fraud_count)
    left, right, third = st.columns(3)
    left.metric("Transactions scored", f"{len(results):,}")
    right.metric("Flagged as fraud", f"{fraud_count:,}")
    third.metric("Classified legitimate", f"{legitimate_count:,}")
    st.caption(f"A transaction is flagged when its fraud probability is at least the saved threshold ({threshold:.4f}).")

    counts = pd.DataFrame({"Transactions": [fraud_count, legitimate_count]}, index=["Fraud", "Legitimate"])
    st.subheader("Prediction distribution")
    st.bar_chart(counts, y="Transactions", height=220)
    bins = np.linspace(0, 1, 21)
    values = results["Fraud Probability"].to_numpy()
    hist, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges)-1)]
    probability_counts = pd.DataFrame({"Transactions": hist}, index=labels)
    st.caption("Fraud probability distribution; the saved decision threshold is shown in the caption above.")
    st.bar_chart(probability_counts, y="Transactions", height=240)

    st.subheader("Interactive prediction results")
    st.dataframe(results, hide_index=True, width="stretch", height=360)
    st.download_button(
        "Download prediction results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"fraud_predictions_{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )
    with st.expander("What do false positives and false negatives mean?"):
        st.markdown(
            "- **False positive:** a legitimate transaction is flagged for review. This adds investigation work and can inconvenience a customer.\n"
            "- **False negative:** a fraudulent transaction is classified as legitimate. This can allow a suspicious transaction to be missed.\n\n"
            "The threshold controls this trade-off. The dashboard uses the threshold selected on validation data; it is shown for transparency and is not tuned from the uploaded rows."
        )


def _single_transaction_tab(model, metadata):
    features = metadata["feature_columns"]
    st.write("Enter all model features. `V1`–`V28` are anonymized transformed inputs from the source dataset.")
    st.info("The zero values below are form defaults only, not a sample of a typical transaction. Replace them with the transaction's actual feature values before scoring.")
    values = {}
    with st.form("single_transaction_form"):
        first = st.columns(2)
        values["Time"] = first[0].number_input("Time (seconds from dataset start)", min_value=0.0, value=0.0, step=60.0)
        values["Amount"] = first[1].number_input("Transaction amount", min_value=0.0, value=0.0, step=1.0)
        with st.expander("Anonymized transformed features (V1–V28)", expanded=False):
            cols = st.columns(4)
            for i, column in enumerate(c for c in features if c not in {"Time", "Amount"}):
                values[column] = cols[i % len(cols)].number_input(column, value=0.0, step=0.1, format="%.6f", key=f"single_{column}")
        submitted = st.form_submit_button("Score transaction", type="primary", width="stretch")
    if submitted:
        row = pd.DataFrame([{column: values[column] for column in features}])
        try:
            result, notices = predict_transactions(row, model, metadata)
            for notice in notices: st.info(notice)
            probability = float(result["Fraud Probability"].iloc[0])
            decision = result["Decision"].iloc[0]
            if decision == "Fraud": st.error(f"Flagged as fraud — estimated probability {probability:.1%}")
            else: st.success(f"Classified as legitimate — estimated fraud probability {probability:.1%}")
            st.progress(probability, text="Estimated fraud probability")
            _render_results(result, float(metadata["selected_threshold"]), "single")
        except (TypeError, ValueError, RuntimeError) as exc:
            st.error(str(exc))


def _batch_tab(model, metadata):
    features = metadata["feature_columns"]
    st.write("Upload a CSV with one row per transaction and all required model features. Files are parsed in memory and are not saved by this application.")
    template = pd.DataFrame(columns=features).to_csv(index=False).encode("utf-8")
    st.download_button("Download a blank CSV template", template, "transaction_template.csv", "text/csv", key="template_download")
    upload = st.file_uploader("Choose a transaction CSV", type=["csv"], key="batch_csv")
    if upload is None:
        st.caption("Required columns: " + ", ".join(features))
        return
    raw = upload.getvalue()
    upload_id = hashlib.sha256(raw).hexdigest()
    try:
        preview = pd.read_csv(io.BytesIO(raw), nrows=5)
        st.caption(f"File: {upload.name} · {len(raw):,} bytes · preview of up to 5 rows")
        st.dataframe(preview, hide_index=True, width="stretch")
        if st.button("Predict uploaded transactions", type="primary", key="run_batch"):
            results, notices = predict_csv_bytes(raw, model, metadata)
            st.session_state["batch_upload_id"] = upload_id
            st.session_state["batch_results"] = results
            st.session_state["batch_notices"] = notices
    except UnicodeDecodeError:
        st.error("This CSV could not be decoded as UTF-8. Save or export it as a UTF-8 encoded CSV, then upload it again.")
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_upload_id", None)
        return
    except (TypeError, ValueError, RuntimeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(str(exc))
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_upload_id", None)
        return
    if st.session_state.get("batch_upload_id") == upload_id and "batch_results" in st.session_state:
        for notice in st.session_state.get("batch_notices", []): st.warning(notice)
        _render_results(st.session_state["batch_results"], float(metadata["selected_threshold"]), "batch")


def main():
    st.set_page_config(page_title="CardGuard · Fraud Detection", page_icon="🛡️", layout="wide")
    st.title("🛡️ CardGuard")
    st.caption("Credit Card Fraud Detection · Local ML dashboard")
    try:
        model, metadata = load_assets()
    except Exception as exc:
        st.error(f"Could not load the saved V2 model: {exc}")
        st.info("Run this app from the project folder and confirm the saved files exist under `models/`.")
        st.stop()

    test_metrics = metadata["test_metrics"]
    top = st.columns(4)
    top[0].metric("Selected model", metadata["selected_model"])
    top[1].metric("Saved threshold", f"{metadata['selected_threshold']:.4f}")
    top[2].metric("Test PR-AUC", f"{test_metrics['pr_auc_average_precision']:.3f}")
    top[3].metric("Test recall", f"{test_metrics['recall']:.1%}")
    st.caption("The dashboard loads the existing saved pipeline. It does not retrain or change the threshold.")

    overview, single, batch, model_info = st.tabs(["Overview", "Single transaction", "Batch CSV", "Model & limitations"])
    with overview:
        st.subheader("A review aid for a highly imbalanced benchmark")
        st.markdown(
            "This project scores transaction features using a previously trained Random Forest. "
            "It returns an estimated fraud probability and applies the threshold selected on validation data. "
            "A flag is a model output, not a finding of fraud."
        )
        st.info("Only 492 of 284,807 benchmark transactions were fraud (about 0.173%). A model can miss fraud or flag legitimate transactions; review both error types before using any score.")
        st.markdown("Use **Single transaction** to enter one row, or **Batch CSV** to score a local CSV. The app reads only the saved model and user-provided rows; it does not load `creditcard.csv`.")
    with single:
        _single_transaction_tab(model, metadata)
    with batch:
        _batch_tab(model, metadata)
    with model_info:
        st.subheader("Saved model")
        st.write(f"**Model:** {metadata['selected_model']} · **Threshold:** {metadata['selected_threshold']:.8f} · **Seed:** {metadata['seed']}")
        st.write("**Threshold rule:** " + metadata["threshold_selection"])
        st.write("**Model selection:** " + metadata["model_selection_metric"])
        st.subheader("Final held-out test metrics")
        st.dataframe(pd.DataFrame([{
            "Precision":test_metrics["precision"], "Recall":test_metrics["recall"],
            "F1":test_metrics["f1_score"], "ROC-AUC":test_metrics["roc_auc"],
            "PR-AUC":test_metrics["pr_auc_average_precision"],
            "False positives":test_metrics["false_positive"], "False negatives":test_metrics["false_negative"],
        }]), hide_index=True, width="stretch")
        with st.expander("Interpret the errors"):
            st.markdown(
                f"On the held-out test set, the selected threshold produced **{test_metrics['false_positive']} false positives** "
                f"and **{test_metrics['false_negative']} false negatives**. A false positive is a legitimate transaction sent for review; "
                "a false negative is fraud the model failed to flag. Lower thresholds can increase recall and review volume."
            )
        st.subheader("Limitations")
        st.markdown(
            "- This is a small portfolio benchmark using anonymized features, not a bank's live transaction stream.\n"
            "- The data covers a short historical period; real-world drift and future-period performance are not established.\n"
            "- The threshold maximizes validation F1. A real team should select an operating point using review capacity and fraud/alert costs.\n"
            "- The saved test metrics are benchmark results, not a guarantee for new transactions. Human review and monitoring would be required for any real use."
        )
    st.divider()
    st.caption("Local demo · Uploaded data is processed in memory by this app and is not written to project files.")


if __name__ == "__main__":
    main()
