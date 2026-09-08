"""
Script that performs Optuna TPE search for classifiers across various datasets.
Datasets are evaluated via stratified 10-fold cross-validation.
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from classifier_configs import get_classifier

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# setup random seed to make code reproducible
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_OPTUNA_TRIALS = 1000


# speedup
sklearn.set_config(
    assume_finite=False,
    skip_parameter_validation=True,
)

# specification of evaluated metrics
scoring_dict = {
    "recall": make_scorer(recall_score),
    "specificity": make_scorer(recall_score, pos_label=0),
    "uar": make_scorer(balanced_accuracy_score, adjusted=False),
}


optuna.logging.set_verbosity(optuna.logging.WARNING)


def format_time(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def suggest_parameter(trial: optuna.Trial, name: str, distribution: list | dict):
    """Bridge the existing sklearn/scipy parameter space to Optuna suggestions."""
    if isinstance(distribution, list):
        return trial.suggest_categorical(name, distribution)
    else:
        number = trial.suggest_float(name, **distribution)
        if distribution.get("step", -1) == 1:
            number = int(number)
        return number


def suggest_hyperparameters(trial: optuna.Trial, param_grid: dict | list[dict]):
    """Sample one parameter set from a dict or a list of alternative dict spaces."""
    if isinstance(param_grid, list):
        branch = trial.suggest_categorical("search_space", list(range(len(param_grid))))
        active_grid = param_grid[branch]
        parameter_prefix = f"search_space_{branch}__"
    else:
        active_grid = param_grid
        parameter_prefix = ""

    return {
        name: suggest_parameter(trial, f"{parameter_prefix}{name}", distribution)
        for name, distribution in active_grid.items()
    }


def run_optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    classifier: str,
    param_grid: dict | list[dict],
    cross_validation: StratifiedKFold,
):
    """Run an Optuna TPE search and return rows compatible with the existing results.csv format."""
    progress_label = f"{classifier} TPE search"
    total_fits = N_OPTUNA_TRIALS * cross_validation.get_n_splits()
    validation_n_jobs = -1
    trial_rows = []
    started_at = time.time()

    def objective(trial: optuna.Trial):
        params = suggest_hyperparameters(trial, param_grid)
        print(params)
        pipeline.set_params(**params)
        scores = cross_validate(
            pipeline,
            X=X,
            y=y,
            scoring=scoring_dict,
            cv=cross_validation,
            n_jobs=validation_n_jobs,
        )
        result_row = {
            "params": params,
            "mean_test_recall": float(np.mean(scores["test_recall"])),
            "mean_test_specificity": float(np.mean(scores["test_specificity"])),
            "mean_test_uar": float(np.mean(scores["test_uar"])),
        }
        trial_rows.append(result_row)

        fits_completed = len(trial_rows) * cross_validation.get_n_splits()
        elapsed_seconds = time.time() - started_at
        seconds_per_fit = elapsed_seconds / fits_completed
        remaining_fits = total_fits - fits_completed
        eta_seconds = remaining_fits * seconds_per_fit
        sys.stdout.write(
            f"\r{progress_label}: {fits_completed}/{total_fits} fits completed"
            f" | elapsed {format_time(elapsed_seconds)}"
            f" | avg {seconds_per_fit:.2f}s/fit"
            f" | ETA {format_time(eta_seconds)}"
        )
        sys.stdout.flush()
        return result_row["mean_test_uar"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return pd.DataFrame(trial_rows)


def main(sex: str = "women", classifier_str="stacking_adaboost"):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "stacking_adaboost".
    """
    # setup paths to training data and results
    training_data = Path(".").joinpath("training_data", sex, "0")
    results_data = Path(".").joinpath("results", classifier_str, sex, "0")
    results_data.mkdir(exist_ok=True, parents=True)

    # load dataset
    with open(training_data.joinpath("dataset.pk"), "rb") as f:
        train_set = pickle.load(f)
    X = np.array(train_set["data"], dtype=np.float64)
    if sex != "both":
        assert np.all(X[:, 0] == X[0, 0]), "First row is not sex"
        X = X[:, 1:]
    dataset = {
        "X": X.copy(),
        "y": np.array(train_set["labels"]),
    }

    pipeline, param_grid = get_classifier(classifier_str, random_seed=RANDOM_SEED)
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    total_fits = N_OPTUNA_TRIALS * cross_validation.get_n_splits()
    print(f"Starting {classifier_str} TPE search for {sex}: {total_fits} fits total.")
    search_results = run_optuna_search(
        X=dataset["X"],
        y=dataset["y"],
        pipeline=pipeline,
        classifier=classifier_str,
        param_grid=param_grid,
        cross_validation=cross_validation,
    )

    # dump Optuna search results to results.csv
    search_results.round(6)[["params", "mean_test_recall", "mean_test_specificity", "mean_test_uar"]].to_csv(
        results_data.joinpath("results.csv"),
        index=False,
        mode="w",
        header=True,
        encoding="utf8",
        lineterminator="\n",
    )


# pylint: enable=too-many-locals


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        prog="classifier_pipeline.py",
        description="Perform Optuna TPE search for a classifier across various datasets. ",
    )
    parser.add_argument("classifier", type=str)
    parser.add_argument("sex", type=str)
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    main(sex_to_compute, used_classifier)
