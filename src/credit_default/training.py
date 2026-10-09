"""Optuna tuning, cross-validation, MLflow tracking, and model export."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_default.database import make_engine, read_training_features
from credit_default.features import engineer_features
from credit_default.schema import MODEL_FEATURES, RAW_FEATURES


def _build_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "xgboost":
        classifier = XGBClassifier(
            **params,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )
    elif model_name == "lightgbm":
        classifier = LGBMClassifier(
            **params,
            objective="binary",
            verbosity=-1,
            n_jobs=1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", classifier),
        ]
    )


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if model_name == "xgboost":
        params["min_child_weight"] = trial.suggest_float(
            "min_child_weight", 1.0, 10.0
        )
    else:
        params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 50)
    return params


def train_model(
    database_url: str,
    model_output: str | Path,
    n_trials: int = 30,
    model_name: str = "xgboost",
    tracking_uri: str = "./mlruns",
) -> dict[str, float]:
    """Tune and train a model, then log metrics, SHAP, and a serving artifact."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    engine = make_engine(database_url)
    try:
        frame, labels = read_training_features(engine)
    finally:
        engine.dispose()

    if labels.value_counts().min() < 10:
        raise ValueError("At least ten examples from each class are required for a stratified holdout and 5-fold CV")
    missing = sorted(set(RAW_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Training data is missing raw features: {', '.join(missing)}")

    model_frame = engineer_features(frame)
    model_frame = model_frame.loc[:, MODEL_FEATURES]
    tuning_features, holdout_features, tuning_labels, holdout_labels = train_test_split(
        model_frame,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        pipeline = _build_pipeline(model_name, _suggest_params(trial, model_name))
        scores = cross_val_score(
            pipeline,
            tuning_features,
            tuning_labels,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    if "://" not in tracking_uri and not tracking_uri.startswith("file:"):
        tracking_uri = Path(tracking_uri).expanduser().resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-default-prediction")
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        study.optimize(objective, n_trials=n_trials)
        holdout_pipeline = _build_pipeline(model_name, study.best_params)
        holdout_pipeline.fit(tuning_features, tuning_labels)
        holdout_probabilities = holdout_pipeline.predict_proba(holdout_features)[:, 1]
        metrics = {
            "holdout_roc_auc": float(roc_auc_score(holdout_labels, holdout_probabilities)),
            "holdout_average_precision": float(
                average_precision_score(holdout_labels, holdout_probabilities)
            ),
            "holdout_brier_score": float(
                brier_score_loss(holdout_labels, holdout_probabilities)
            ),
        }
        best_pipeline = _build_pipeline(model_name, study.best_params)
        best_pipeline.fit(model_frame, labels)
        mlflow.log_params({"model": model_name, "n_trials": n_trials, **study.best_params})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_mean_fold_roc_auc", float(study.best_value))
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        artifact = {
            "pipeline": best_pipeline,
            "feature_columns": list(MODEL_FEATURES),
            "model_version": run.info.run_id,
        }
        joblib.dump(artifact, output_path)
        mlflow.log_artifact(str(output_path), artifact_path="serving")

        transformed = best_pipeline.named_steps["imputer"].transform(model_frame)
        sample = transformed[: min(500, len(transformed))]
        explainer = shap.TreeExplainer(best_pipeline.named_steps["model"])
        shap_values = explainer.shap_values(sample)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            sample,
            feature_names=list(MODEL_FEATURES),
            plot_type="bar",
            show=False,
        )
        shap_path = output_path.parent / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(shap_path), artifact_path="explainability")

    return metrics
