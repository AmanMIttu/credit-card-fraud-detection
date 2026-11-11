"""Reproducible V2 fraud-model workflow. Run: python fraud_pipeline_v2.py"""
from __future__ import annotations
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, precision_recall_fscore_support)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "creditcard.csv"
SEED = 42
TEST_SIZE = 0.20
VALIDATION_OF_REMAINING = 0.20
BASE_OUTPUTS = [
    ROOT / "model_comparison_v2.csv",
    ROOT / "models" / "fraud_detection_pipeline_v2.joblib",
    ROOT / "models" / "fraud_detection_metadata_v2.json",
]
FIGURE_NAMES = [
    "model_comparison_v2.png",
    "model_comparison_dashboard_v2.png",
    "validation_threshold_tradeoff_v2.png",
    "validation_confusion_matrices_v2.png",
    "test_confusion_matrix_v2.png",
]


def _make_splits(frame: pd.DataFrame):
    """Group exact feature rows, then stratify the unique groups into partitions."""
    feature_columns = [c for c in frame.columns if c != "Class"]
    y = frame["Class"].astype(int)
    # ngroup assigns the same exact ID to rows with the same feature values;
    # Class is intentionally excluded so identical inputs cannot cross splits.
    group_ids = frame[feature_columns].groupby(
        feature_columns, sort=False, dropna=False
    ).ngroup().to_numpy()
    group_table = pd.DataFrame({"group_id": group_ids, "target": y.to_numpy()})
    conflict_count = int(group_table.groupby("group_id")["target"].nunique().gt(1).sum())
    if conflict_count:
        raise ValueError(f"Found {conflict_count} identical feature groups with conflicting labels.")
    group_table = group_table.drop_duplicates("group_id", keep="first")
    unique_groups = group_table["group_id"].to_numpy()
    group_targets = group_table["target"].to_numpy()
    train_val_groups, test_groups = train_test_split(
        unique_groups, test_size=TEST_SIZE, random_state=SEED,
        stratify=group_targets,
    )
    target_by_group = group_table.set_index("group_id")["target"]
    train_val_targets = target_by_group.loc[train_val_groups].to_numpy()
    train_groups, validation_groups = train_test_split(
        train_val_groups, test_size=VALIDATION_OF_REMAINING,
        random_state=SEED + 1, stratify=train_val_targets,
    )
    partitions = {
        "train": np.flatnonzero(np.isin(group_ids, train_groups)),
        "validation": np.flatnonzero(np.isin(group_ids, validation_groups)),
        "test": np.flatnonzero(np.isin(group_ids, test_groups)),
    }
    group_sets = [set(train_groups), set(validation_groups), set(test_groups)]
    if any(group_sets[i] & group_sets[j] for i, j in [(0,1),(0,2),(1,2)]):
        raise AssertionError("Feature groups crossed between partitions.")
    if sum(len(ix) for ix in partitions.values()) != len(frame):
        raise AssertionError("Split row counts do not cover the complete dataset.")
    total_rate = float(y.mean())
    split_summary = {}
    for name, ix in partitions.items():
        fraud_count = int(y.iloc[ix].sum())
        rate = fraud_count / len(ix)
        split_summary[name] = {
            "rows": int(len(ix)), "fraud": fraud_count,
            "legitimate": int(len(ix) - fraud_count),
            "fraud_rate": float(rate),
            "fraud_rate_difference_percentage_points": float((rate-total_rate)*100),
            "groups": int(len(set(group_ids[ix]))),
        }
        if fraud_count == 0:
            raise ValueError(f"{name} contains no fraud cases; group split is unsuitable.")
    max_rate_gap = max(abs(v["fraud_rate"] - total_rate) for v in split_summary.values())
    if max_rate_gap > 0.005:
        raise ValueError(
            "Group split class balance differs from the dataset by over 0.5 percentage points. "
            "Do not train on this split; use a deterministic group-level rebalancing strategy."
        )
    return partitions, group_ids, split_summary, feature_columns, conflict_count


def _build_pipeline(model, feature_columns):
    # Amount is often strongly right-skewed, so log1p is applied first. Both
    # transforms are fitted only on training rows because they live in a Pipeline.
    amount_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    transformers = [
        ("amount", amount_pipe, ["Amount"]),
        ("time", StandardScaler(), ["Time"]),
    ]
    preprocessor = ColumnTransformer(
        transformers, remainder="passthrough", sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    # V1-V28 are already transformed, anonymized numeric features and are passed
    # through unchanged. The estimator and all preprocessing are saved together.
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _metrics(y_true, probability, threshold):
    predicted = (np.asarray(probability) >= float(threshold)).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, labels=[1], average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0,1]).ravel()
    return {
        "threshold": float(threshold), "precision": float(precision),
        "recall": float(recall), "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "alerts": int(predicted.sum()),
    }


def _best_f1_threshold(y_true, probability):
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=1e-12, atol=1e-12))
    # In a tie, use the higher threshold to avoid unnecessary alerts.
    best_idx = best[-1]
    return float(thresholds[best_idx])


def _plot_outputs(validation_rows, validation_probabilities, y_validation,
                  selected_name, selected_threshold, test_probability, y_test,
                  save_figures):
    sns.set_theme(style="whitegrid", context="notebook")
    figures_dir = ROOT / "figures"
    if save_figures:
        figures_dir.mkdir(exist_ok=True)
    # Validation comparison, with threshold-free PR-AUC and operating metrics at 0.5.
    default_rows = [r for r in validation_rows if r["evaluation"] == "validation_default_0.5"]
    plot_df = pd.DataFrame(default_rows).melt(
        id_vars="model", value_vars=["pr_auc_average_precision", "precision", "recall", "f1_score"],
        var_name="metric", value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set(title="Model comparison on validation data", xlabel="Model", ylabel="Score")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=8)
    fig.tight_layout()
    if save_figures: fig.savefig(figures_dir / FIGURE_NAMES[0], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Cleaner two-panel comparison keeps long metric names out of the legend.
    model_names = [r["model"] for r in default_rows]
    metric_sets = [("PR-AUC", "pr_auc_average_precision"), ("ROC-AUC", "roc_auc")]
    threshold_metrics = [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1_score")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    x = np.arange(len(model_names))
    width = 0.34
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    for j, (label, key) in enumerate(metric_sets):
        values = [r[key] for r in default_rows]
        bars = axes[0].bar(x + (j - 0.5) * width, values, width, label=label, color=palette[j])
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].set(title="Ranking quality (threshold-independent)", ylabel="Score", xticks=x, xticklabels=model_names, ylim=(0, 1.08))
    axes[0].legend(frameon=False, ncols=2, loc="upper center")
    for j, (label, key) in enumerate(threshold_metrics):
        values = [r[key] for r in default_rows]
        bars = axes[1].bar(x + (j - 1) * width, values, width, label=label, color=palette[j])
        axes[1].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[1].set(title="Alert metrics at the default 0.5 threshold", ylabel="Score", xticks=x, xticklabels=model_names, ylim=(0, 1.12))
    axes[1].legend(frameon=False, ncols=3, loc="upper center")
    fig.suptitle("Validation comparison: ranking quality and default-threshold behavior", y=1.03, fontsize=14)
    fig.tight_layout()
    if save_figures: fig.savefig(figures_dir / "model_comparison_dashboard_v2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Threshold trade-off for the selected model, using validation predictions only.
    p, r, thresholds = precision_recall_curve(y_validation, validation_probabilities[selected_name])
    f1 = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-15)
    if len(thresholds) > 900:
        sample_idx = np.unique(np.linspace(0, len(thresholds)-1, 900).astype(int))
    else:
        sample_idx = np.arange(len(thresholds))
    order = np.argsort(thresholds[sample_idx])
    si = sample_idx[order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds[si], p[:-1][si], label="Precision")
    ax.plot(thresholds[si], r[:-1][si], label="Recall")
    ax.plot(thresholds[si], f1[si], label="F1-score")
    ax.axvline(0.5, color="gray", linestyle="--", label="Default threshold 0.5")
    ax.axvline(selected_threshold, color="#E45756", linestyle=":", label=f"Selected threshold {selected_threshold:.4f}")
    ax.set(title=f"Validation threshold trade-off — {selected_name}", xlabel="Decision threshold", ylabel="Score", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="best")
    fig.tight_layout()
    if save_figures: fig.savefig(figures_dir / FIGURE_NAMES[1], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices for all models are validation-only; thresholds are validation-tuned.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, row in zip(axes, [r for r in validation_rows if r["evaluation"] == "validation_best_f1"]):
        name = row["model"]
        pred = (validation_probabilities[name] >= row["threshold"]).astype(int)
        sns.heatmap(confusion_matrix(y_validation, pred, labels=[0,1]), annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.suptitle("Validation confusion matrices at each model’s F1-selected threshold")
    fig.tight_layout()
    if save_figures: fig.savefig(figures_dir / FIGURE_NAMES[2], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # The untouched test data is plotted only for the model/threshold selected on validation.
    test_pred = (test_probability >= selected_threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(confusion_matrix(y_test, test_pred, labels=[0,1]), annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    ax.set(title=f"Final test confusion matrix — {selected_name}", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    if save_figures: fig.savefig(figures_dir / FIGURE_NAMES[3], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _assert_new_outputs_available():
    outputs = BASE_OUTPUTS + [ROOT / "figures" / n for n in FIGURE_NAMES]
    existing = [str(p.relative_to(ROOT)) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing V2 artifact(s): " + ", ".join(existing))


def run_workflow(save_artifacts=True, make_figures=True):
    if save_artifacts:
        _assert_new_outputs_available()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    if "Class" not in frame.columns or set(frame["Class"].unique()) != {0,1}:
        raise ValueError("Expected binary Class labels 0 and 1.")
    partitions, group_ids, split_summary, features, conflicting_groups = _make_splits(frame)
    X = frame[features]
    y = frame["Class"].astype(int)
    train_i, val_i, test_i = (partitions[k] for k in ["train", "validation", "test"])
    X_train, y_train = X.iloc[train_i], y.iloc[train_i]
    X_val, y_val = X.iloc[val_i], y.iloc[val_i]
    X_test, y_test = X.iloc[test_i], y.iloc[test_i]
    print("Group-aware stratified split (fixed seeds):")
    print(json.dumps(split_summary, indent=2))
    print("Feature groups with conflicting labels:", conflicting_groups)

    neg, pos = np.bincount(y_train.to_numpy(), minlength=2)
    threads = max(1, min(4, os.cpu_count() or 1))
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0,
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=threads, random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
            reg_lambda=2.0, scale_pos_weight=float(neg/pos),
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", n_jobs=threads, random_state=SEED,
        ),
    }
    fitted = {}
    validation_probabilities = {}
    rows = []
    for name, estimator in candidates.items():
        print(f"Training {name} ...", flush=True)
        pipeline = _build_pipeline(estimator, features)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        probability = pipeline.predict_proba(X_val)[:,1]
        validation_probabilities[name] = probability
        threshold = _best_f1_threshold(y_val, probability)
        default_metrics = _metrics(y_val, probability, 0.5)
        tuned_metrics = _metrics(y_val, probability, threshold)
        for evaluation, metrics in [("validation_default_0.5", default_metrics), ("validation_best_f1", tuned_metrics)]:
            rows.append({"model":name, "evaluation":evaluation, **metrics})
        print(f"  validation PR-AUC/AP={default_metrics['pr_auc_average_precision']:.6f}; "
              f"F1@0.5={default_metrics['f1_score']:.4f}; best-F1 threshold={threshold:.6f}; "
              f"F1@selected={tuned_metrics['f1_score']:.4f}", flush=True)

    # Choose model by validation PR-AUC/AP, never by accuracy or test results.
    default_rows = [r for r in rows if r["evaluation"] == "validation_default_0.5"]
    selected_row = max(default_rows, key=lambda r: (r["pr_auc_average_precision"], r["f1_score"]))
    selected_name = selected_row["model"]
    selected_threshold = next(r["threshold"] for r in rows if r["model"] == selected_name and r["evaluation"] == "validation_best_f1")
    print(f"Selected on validation PR-AUC: {selected_name}")
    print(f"Selected threshold from validation F1 analysis: {selected_threshold:.8f}")

    # First and only final holdout prediction: selected model, selected validation threshold.
    test_probability = fitted[selected_name].predict_proba(X_test)[:,1]
    test_metrics = _metrics(y_test, test_probability, selected_threshold)
    rows.append({"model":selected_name, "evaluation":"test_selected_validation_threshold", **test_metrics})
    comparison = pd.DataFrame(rows)
    print("Final selected-model test metrics:")
    print(pd.Series(test_metrics).to_string())
    print("Test metrics were calculated only for the validation-selected model and threshold.")

    if make_figures:
        _plot_outputs(rows, validation_probabilities, y_val, selected_name,
                      selected_threshold, test_probability, y_test, save_artifacts)
    metadata = {
        "project":"Credit Card Fraud Detection 2.0",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "python_version":os.sys.version.split()[0],
        "seed":SEED,
        "dataset_file":DATA_PATH.name,
        "dataset_sha256":hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "dataset_rows":int(len(frame)),
        "feature_columns":features,
        "target_column":"Class",
        "grouping":"Exact equality across all feature columns, excluding Class; stratified split performed on unique group IDs.",
        "feature_groups_with_conflicting_labels":conflicting_groups,
        "split_summary":split_summary,
        "selected_model":selected_name,
        "model_selection_metric":"Validation PR-AUC (average precision); accuracy was not used for selection.",
        "threshold_selection":"Maximize validation F1; ties use the highest threshold.",
        "selected_threshold":float(selected_threshold),
        "threshold_default":0.5,
        "test_metrics":test_metrics,
        "preprocessing":"log1p(Amount) then StandardScaler; StandardScaler(Time); V1-V28 passed through unchanged. All fitted transforms are inside the saved sklearn Pipeline.",
        "model_parameters":{
            "Logistic Regression":"class_weight=balanced, C=1, max_iter=2000",
            "Random Forest":"250 trees, class_weight=balanced_subsample, min_samples_leaf=2",
            "XGBoost":f"250 trees, max_depth=4, learning_rate=0.05, scale_pos_weight={float(neg/pos):.6f}",
        },
        "versions":{"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"xgboost":xgboost.__version__,"joblib":joblib.__version__},
    }
    if save_artifacts:
        (ROOT / "models").mkdir(exist_ok=True)
        comparison.to_csv(ROOT / "model_comparison_v2.csv", index=False, float_format="%.8g")
        model_path = ROOT / "models" / "fraud_detection_pipeline_v2.joblib"
        joblib.dump(fitted[selected_name], model_path, compress=3)
        metadata_path = ROOT / "models" / "fraud_detection_metadata_v2.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Saved model:", model_path.relative_to(ROOT))
        print("Saved metadata:", metadata_path.relative_to(ROOT))
        print("Saved comparison:", (ROOT / "model_comparison_v2.csv").relative_to(ROOT))
        if make_figures:
            for name in FIGURE_NAMES: print("Saved figure:", (ROOT / "figures" / name).relative_to(ROOT))
        # Verify the persisted pipeline immediately on five held-out sample transactions.
        loaded = joblib.load(model_path)
        sample_probability = loaded.predict_proba(X_test.iloc[:5])[:,1]
        sample_prediction = (sample_probability >= selected_threshold).astype(int)
        if len(sample_probability) != min(5, len(X_test)) or not np.isfinite(sample_probability).all():
            raise AssertionError("Persisted model sample predictions failed validation.")
        print("Reloaded model sample predictions (probability, threshold decision):")
        print(list(zip(np.round(sample_probability, 6).tolist(), sample_prediction.tolist())))
    return {"comparison":comparison,"metadata":metadata,"split_summary":split_summary,
            "validation_probabilities":validation_probabilities,"selected_pipeline":fitted[selected_name],
            "selected_model":selected_name,"selected_threshold":selected_threshold,
            "test_probability":test_probability,"test_metrics":test_metrics}

if __name__ == "__main__":
    run_workflow(save_artifacts=True, make_figures=True)
