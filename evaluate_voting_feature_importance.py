"""Evaluate feature importance and base-estimator agreement for best voting models."""

import argparse
import ast
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from classifier_configs import get_classifier

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sklearn.set_config(
    assume_finite=False,
    skip_parameter_validation=True,
)

SEXES = ("women")
BASE_ESTIMATORS = ("knn", "svm", "random_forest")
RESULTS_DIR = Path("results_interpretability")


def parse_hyperparameters(raw_params: str) -> dict:
    """Parse stored hyperparameters from the randomized-search CSV output."""
    sanitized = re.sub(r"np\.float64\(([^()]*)\)", r"\1", raw_params)
    return ast.literal_eval(sanitized)


def get_feature_names(config: dict) -> list[str]:
    """Reconstruct the dataset column order from the dataset config."""
    feature_names = ["sex", "age"]

    if config["diff_pitch"]:
        feature_names.append("diff_pitch")

    feature_names.append("mean_f0")

    if config["stdev_f0"]:
        feature_names.append("stdev_f0")

    feature_names.extend(["hnr", "jitter", "shimmer"])

    if config["spectral_centroid"]:
        feature_names.append("spectral_centroid")

    if config["spectral_contrast"]:
        feature_names.extend([f"spectral_contrast_{idx}" for idx in range(7)])

    if config["spectral_flatness"]:
        feature_names.append("spectral_flatness")

    if config["spectral_rolloff"]:
        feature_names.append("spectral_rolloff")

    if config["zero_crossing_rate"]:
        feature_names.append("zero_crossing_rate")

    if config["shannon_entropy"]:
        feature_names.append("shannon_entropy")

    if config["skewness"]:
        feature_names.append("skewness")

    return feature_names


def load_dataset(sex: str, dataset_id: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load a serialized dataset and reconstruct its feature names."""
    if sex == "w":
        sex = "women"
    dataset_dir = Path("training_data", sex, dataset_id)
    with dataset_dir.joinpath("dataset.pk").open("rb") as handle:
        dataset = pickle.load(handle)
    with dataset_dir.joinpath("config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    feature_names = get_feature_names(config)
    X = np.asarray(dataset["data"], dtype=np.float64)
    y = np.asarray(dataset["labels"], dtype=np.int64)

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name count mismatch for {sex}/{dataset_id}: "
            f"{X.shape[1]} columns in dataset vs {len(feature_names)} reconstructed names."
        )

    return X, y, feature_names, config


def load_best_voting_model(sex: str, selection_source: str) -> dict:
    """Get the best-scoring voting configuration for a given sex."""
    if sex == "w":
        sex = "women"

    if selection_source == "xvalidation":
        result_file = Path("results_xvalidation", f"voting_{sex}_top1000_100times.csv")
        table = pd.read_csv(result_file)
        best_row = table.sort_values("test_uar", kind="mergesort").iloc[-1]
        return {
            "sex": sex,
            "dataset_id": str(best_row["dataset"]),
            "selection_source": selection_source,
            "test_recall": float(best_row["test_recall"]),
            "test_recall_stdev": float(best_row["test_recall_stdev"]),
            "test_specificity": float(best_row["test_specificity"]),
            "test_specificity_stdev": float(best_row["test_specificity_stdev"]),
            "test_uar": float(best_row["test_uar"]),
            "test_uar_stdev": float(best_row["test_uar_stdev"]),
            "params": parse_hyperparameters(best_row["hyperparameters"]),
            "params_raw": best_row["hyperparameters"],
        }

    result_file = Path("results", "voting", sex, "0", "results.csv")
    table = pd.read_csv(result_file)
    best_row = table.sort_values("mean_test_uar", kind="mergesort").iloc[-1]

    return {
        "sex": sex,
        "dataset_id": "0",
        "selection_source": selection_source,
        "mean_test_recall": float(best_row["mean_test_recall"]),
        "mean_test_specificity": float(best_row["mean_test_specificity"]),
        "mean_test_uar": float(best_row["mean_test_uar"]),
        "params": parse_hyperparameters(best_row["params"]),
        "params_raw": best_row["params"],
    }


def summarize_predictions(predictions: pd.DataFrame) -> pd.Series:
    """Aggregate out-of-fold prediction agreement for base estimators."""
    pair_knn_svm = predictions["knn_pred"] == predictions["svm_pred"]
    pair_knn_rf = predictions["knn_pred"] == predictions["random_forest_pred"]
    pair_svm_rf = predictions["svm_pred"] == predictions["random_forest_pred"]
    unanimous = pair_knn_svm & pair_knn_rf
    disagreement = ~unanimous

    knn_dissent = (predictions["svm_pred"] == predictions["random_forest_pred"]) & (
        predictions["knn_pred"] != predictions["svm_pred"]
    )
    svm_dissent = (predictions["knn_pred"] == predictions["random_forest_pred"]) & (
        predictions["svm_pred"] != predictions["knn_pred"]
    )
    rf_dissent = (predictions["knn_pred"] == predictions["svm_pred"]) & (
        predictions["random_forest_pred"] != predictions["knn_pred"]
    )

    majority_correct = predictions["voting_pred"] == predictions["y_true"]

    return pd.Series(
        {
            "n_samples": len(predictions),
            "knn_svm_agreement": pair_knn_svm.mean(),
            "knn_svm_disagreement": 1.0 - pair_knn_svm.mean(),
            "knn_random_forest_agreement": pair_knn_rf.mean(),
            "knn_random_forest_disagreement": 1.0 - pair_knn_rf.mean(),
            "svm_random_forest_agreement": pair_svm_rf.mean(),
            "svm_random_forest_disagreement": 1.0 - pair_svm_rf.mean(),
            "unanimous_agreement": unanimous.mean(),
            "any_disagreement": disagreement.mean(),
            "knn_dissent_rate": knn_dissent.mean(),
            "svm_dissent_rate": svm_dissent.mean(),
            "random_forest_dissent_rate": rf_dissent.mean(),
            "unanimous_accuracy": majority_correct[unanimous].mean()
            if unanimous.any()
            else np.nan,
            "disagreement_accuracy": majority_correct[disagreement].mean()
            if disagreement.any()
            else np.nan,
            "voting_recall": recall_score(predictions["y_true"], predictions["voting_pred"]),
            "voting_specificity": recall_score(
                predictions["y_true"], predictions["voting_pred"], pos_label=0
            ),
            "voting_uar": balanced_accuracy_score(
                predictions["y_true"], predictions["voting_pred"]
            ),
            "knn_uar": balanced_accuracy_score(predictions["y_true"], predictions["knn_pred"]),
            "svm_uar": balanced_accuracy_score(predictions["y_true"], predictions["svm_pred"]),
            "random_forest_uar": balanced_accuracy_score(
                predictions["y_true"], predictions["random_forest_pred"]
            ),
        }
    )


def evaluate_best_voting_classifier(
    sex: str,
    selection_source: str,
    n_splits: int,
    n_repeats: int,
    permutation_repeats: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run cross-validated permutation importance and agreement analysis."""
    best_model = load_best_voting_model(sex, selection_source)
    X, y, feature_names, config = load_dataset(sex, best_model["dataset_id"])

    pipeline, _ = get_classifier(
        "voting",
        both_sexes=sex == "both",
        random_seed=RANDOM_SEED,
        hyperparameters=best_model["params"],
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
    )

    importance_rows: list[dict] = []
    prediction_rows: list[dict] = []
    fold_rows: list[dict] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        estimator = clone(pipeline)
        estimator.fit(X[train_idx], y[train_idx])

        fold_predictions = {"y_true": y[test_idx]}
        voting_predictions = estimator.predict(X[test_idx])
        fold_predictions["voting_pred"] = voting_predictions

        classifiers = estimator.named_steps["classifier"].named_estimators_
        for estimator_name in BASE_ESTIMATORS:
            fold_predictions[f"{estimator_name}_pred"] = classifiers[estimator_name].predict(
                estimator.named_steps["minmaxscaler"].transform(X[test_idx])
            )

        fold_frame = pd.DataFrame(fold_predictions)
        fold_frame["sex"] = sex
        fold_frame["dataset_id"] = best_model["dataset_id"]
        fold_frame["fold"] = fold_index
        fold_frame["sample_index"] = test_idx
        prediction_rows.extend(fold_frame.to_dict("records"))

        fold_summary = summarize_predictions(fold_frame)
        fold_rows.append(
            {
                "sex": sex,
                "dataset_id": best_model["dataset_id"],
                "fold": fold_index,
                **fold_summary.to_dict(),
            }
        )

        importances = permutation_importance(
            estimator,
            X[test_idx],
            y[test_idx],
            scoring="balanced_accuracy",
            n_repeats=permutation_repeats,
            random_state=RANDOM_SEED + fold_index,
            n_jobs=n_jobs,
        )
        for feature_index, feature_name in enumerate(feature_names):
            importance_rows.append(
                {
                    "sex": sex,
                    "dataset_id": best_model["dataset_id"],
                    "fold": fold_index,
                    "feature": feature_name,
                    "importance_mean": importances.importances_mean[feature_index],
                    "importance_std": importances.importances_std[feature_index],
                }
            )

    importance_df = pd.DataFrame(importance_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    fold_df = pd.DataFrame(fold_rows)
    importance_summary = (
        importance_df.groupby(["sex", "dataset_id", "feature"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            within_permutation_std_mean=("importance_std", "mean"),
        )
        .sort_values(["sex", "importance_mean"], ascending=[True, False], kind="mergesort")
    )

    best_model["feature_config"] = json.dumps(config, sort_keys=True)
    return importance_df, importance_summary, prediction_df, {
        "best_model": best_model,
        "fold_summary": fold_df,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cross-validated permutation feature importance for the best voting "
            "classifier for men, women, and both sexes, and measure agreement/disagreement "
            "among its KNN, SVM, and Random Forest base estimators."
        )
    )
    parser.add_argument(
        "--sexes",
        nargs="+",
        default=list(SEXES),
        choices=SEXES,
        help="Sex-specific datasets to evaluate.",
    )
    parser.add_argument(
        "--selection-source",
        choices=("xvalidation", "raw"),
        default="xvalidation",
        help="Source used to choose the best voting classifier for each sex.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of folds for repeated stratified cross-validation.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of cross-validation repeats.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
        help="Number of shuffles used by permutation importance in each test fold.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for permutation importance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where output CSV files will be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_importances = []
    all_importance_summaries = []
    all_predictions = []
    all_fold_summaries = []
    all_best_models = []

    for sex in args.sexes:
        (
            importances_df,
            importance_summary_df,
            predictions_df,
            extra_outputs,
        ) = evaluate_best_voting_classifier(
            sex=sex,
            selection_source=args.selection_source,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            permutation_repeats=args.permutation_repeats,
            n_jobs=args.n_jobs,
        )

        all_importances.append(importances_df)
        all_importance_summaries.append(importance_summary_df)
        all_predictions.append(predictions_df)

        fold_summary_df = extra_outputs["fold_summary"]
        fold_summary_df.to_csv(
            args.output_dir.joinpath(f"{sex}_agreement_by_fold.csv"),
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        all_fold_summaries.append(fold_summary_df)
        all_best_models.append(extra_outputs["best_model"])

    importance_df = pd.concat(all_importances, ignore_index=True)
    importance_summary_df = pd.concat(all_importance_summaries, ignore_index=True)
    prediction_df = pd.concat(all_predictions, ignore_index=True)
    fold_summary_df = pd.concat(all_fold_summaries, ignore_index=True)
    best_models_df = pd.DataFrame(all_best_models)

    agreement_summary_df = pd.DataFrame(
        [
            {"sex": sex, **summarize_predictions(group).to_dict()}
            for sex, group in prediction_df.groupby("sex")
        ]
    )

    best_models_df.to_csv(
        args.output_dir.joinpath("best_voting_models.csv"),
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    importance_df.to_csv(
        args.output_dir.joinpath("voting_permutation_importance_by_fold.csv"),
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    importance_summary_df.to_csv(
        args.output_dir.joinpath("voting_permutation_importance_summary.csv"),
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    prediction_df.to_csv(
        args.output_dir.joinpath("base_estimator_oof_predictions.csv"),
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    fold_summary_df.to_csv(
        args.output_dir.joinpath("agreement_by_fold_all_sexes.csv"),
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    agreement_summary_df.to_csv(
        args.output_dir.joinpath("base_estimator_agreement_summary.csv"),
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


if __name__ == "__main__":
    main()
