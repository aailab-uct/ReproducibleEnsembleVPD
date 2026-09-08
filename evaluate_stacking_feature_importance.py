"""Permutation importance and stacking-member agreement for dataset 0."""

import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from classifier_configs import get_classifier

SEED = 42
N_SPLITS = 10
N_REPEATS = 10
N_PERMUTATIONS = 20
N_JOBS = -1

COHORTS = ("women", "men", "both")
OUTPUT_DIR = Path("results_interpretability")

FEATURE_NAMES = (
    "sex",
    "age",
    "diff_pitch",
    "mean_f0",
    "stdev_f0",
    "hnr",
    "jitter",
    "shimmer",
    "spectral_centroid",
    "spectral_contrast_1",
    "spectral_contrast_2",
    "spectral_contrast_3",
    "spectral_contrast_4",
    "spectral_contrast_5",
    "spectral_contrast_6",
    "spectral_contrast_7",
    "spectral_flatness",
    "spectral_rolloff",
    "zero_crossing_rate",
    "shannon_entropy",
    "skewness",
)


def load_hyperparameters(cohort, stacking_classifier):
    results = pd.read_csv(Path("results_xvalidation", f"{stacking_classifier}_{cohort}_top1_100times.csv"))
    best_index = results["test_uar"].idxmax()
    hyperparameters: str = results.loc[best_index, "hyperparameters"]  # type: ignore
    return ast.literal_eval(hyperparameters)


def load_dataset(cohort):
    path = Path("training_data", cohort, "0", "dataset.pk")
    with path.open("rb") as dataset_file:
        dataset = pickle.load(dataset_file)
    features = np.asarray(dataset["data"], dtype=np.float64)
    labels = np.asarray(dataset["labels"])
    return features, labels


def main(stacking_classifier="stacking_rbf"):
    importance_rows = []
    diagnostic_rows = []

    for cohort in COHORTS:
        hyperparameters = load_hyperparameters(cohort, stacking_classifier)
        features, labels = load_dataset(cohort)
        cross_validation = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=SEED,
        )

        for split_index, (train_index, test_index) in tqdm(
            enumerate(cross_validation.split(features, labels)), total=N_SPLITS * N_REPEATS
        ):
            repeat = split_index // N_SPLITS + 1
            fold = split_index % N_SPLITS + 1
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            pipeline, _ = get_classifier(
                stacking_classifier,
                random_seed=SEED,
                hyperparameters=hyperparameters,
            )
            pipeline.set_params(classifier__n_jobs=N_JOBS)
            pipeline.fit(X_train, y_train)

            stacking_predictions = pipeline.predict(X_test)
            members = pipeline.named_steps["classifier"].named_estimators_
            knn = members["knn"].predict(X_test)
            svm = members["svm"].predict(X_test)
            random_forest = members["random_forest"].predict(X_test)

            importance = permutation_importance(
                pipeline,
                X_test,
                y_test,
                scoring="balanced_accuracy",
                n_repeats=N_PERMUTATIONS,
                random_state=SEED,
                n_jobs=N_JOBS,
            )
            for feature_index, feature in enumerate(FEATURE_NAMES):
                importance_rows.append(
                    {
                        "cohort": cohort,
                        "repeat": repeat,
                        "fold": fold,
                        "feature_index": feature_index,
                        "feature": feature,
                        "importance_mean": importance.importances_mean[feature_index],
                        "importance_permutation_std": importance.importances_std[feature_index],
                    }
                )

            knn_svm = knn == svm
            knn_rf = knn == random_forest
            svm_rf = svm == random_forest
            unanimous = knn_svm & knn_rf
            diagnostic_rows.append(
                {
                    "cohort": cohort,
                    "repeat": repeat,
                    "fold": fold,
                    "held_out_rows": len(test_index),
                    "stacking_sensitivity": recall_score(y_test, stacking_predictions, pos_label=1),
                    "stacking_specificity": recall_score(y_test, stacking_predictions, pos_label=0),
                    "stacking_balanced_accuracy": balanced_accuracy_score(y_test, stacking_predictions),
                    "knn_balanced_accuracy": balanced_accuracy_score(y_test, knn),
                    "svm_balanced_accuracy": balanced_accuracy_score(y_test, svm),
                    "random_forest_balanced_accuracy": balanced_accuracy_score(y_test, random_forest),
                    "knn_svm_agreement_rate": np.mean(knn_svm),
                    "knn_random_forest_agreement_rate": np.mean(knn_rf),
                    "svm_random_forest_agreement_rate": np.mean(svm_rf),
                    "unanimous_rate": np.mean(unanimous),
                    "knn_dissent_rate": np.mean(knn != stacking_predictions),
                    "svm_dissent_rate": np.mean(svm != stacking_predictions),
                    "random_forest_dissent_rate": np.mean(random_forest != stacking_predictions),
                }
            )

    importance_by_fold = pd.DataFrame(importance_rows)
    importance_summary = (
        importance_by_fold.groupby(["cohort", "feature_index", "feature"], sort=False)["importance_mean"]
        .agg(
            mean_importance="mean",
            between_fold_std=lambda values: values.std(ddof=0),
            fold_count="size",
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance_by_fold.to_csv(OUTPUT_DIR / f"{stacking_classifier}_importance_by_fold.csv", index=False)
    importance_summary.to_csv(OUTPUT_DIR / f"{stacking_classifier}_importance_summary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(OUTPUT_DIR / f"{stacking_classifier}_diagnostics_by_fold.csv", index=False)


if __name__ == "__main__":
    main()
