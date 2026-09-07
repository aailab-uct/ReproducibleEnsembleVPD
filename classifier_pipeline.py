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
from sklearn.metrics import balanced_accuracy_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from classifier_configs import get_classifier

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# setup random seed to make code reproducible
RANDOM_SEED = 23 
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


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def suggest_parameter(trial: optuna.Trial, name: str, distribution):
    """Bridge the existing sklearn/scipy parameter space to Optuna suggestions."""
    if isinstance(distribution, list):
        return trial.suggest_categorical(name, distribution)

    if hasattr(distribution, "dist") and hasattr(distribution, "args"):
        if distribution.dist.name == "randint":
            low, high = distribution.args
            return trial.suggest_int(name, int(low), int(high) - 1)
        if distribution.dist.name in {"reciprocal", "loguniform"}:
            low, high = distribution.args
            return trial.suggest_float(name, float(low), float(high), log=True)

    raise TypeError(f"Unsupported parameter distribution for {name}: {distribution!r}")


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
        pipeline, _ = get_classifier(classifier, random_seed=RANDOM_SEED)
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
            f" | elapsed {format_seconds(elapsed_seconds)}"
            f" | avg {seconds_per_fit:.2f}s/fit"
            f" | ETA {format_seconds(eta_seconds)}"
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


def get_datasets_to_process(datasets_path: Path, results_data: Path, dataset_slice: tuple | int | None = None):
    """
    Get datasets that were not evaluated.
    :param datasets_path: Path, path to training datasets
    :param results_data: Path, path to results folder (used to check which datasets were already evaluated)
    :param dataset_slice: int, slice of datasets to evaluate. If None all datasets will be evaluated.
    If tuple, slice will be used, e.g. (10, 100). If int, first n datasets will be evaluated.
    :return: list of datasets to evaluate
    """
    # get all datasets in training_data
    td = sorted([str(x.name) for x in datasets_path.iterdir()])
    if isinstance(dataset_slice, tuple):
        td = list(td)[dataset_slice[0] : dataset_slice[1]]
    elif isinstance(dataset_slice, int):
        td = list(td)[:dataset_slice]

    # Only treat a dataset as processed once it actually has a non-empty results.csv.
    tr = []
    for result_dir in results_data.iterdir():
        result_file = result_dir / "results.csv"
        if result_file.is_file() and result_file.stat().st_size > 0:
            tr.append(str(result_dir.name))
    # only evaluate datasets that were not processed so far
    to_do = sorted(set(td) - set(tr))

    return to_do


# pylint: disable=too-many-locals


def main(sex: str = "women", classifier="stacking_rbf", dataset_slice=None):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "stacking_rbf".
    :param dataset_slice: The slice of the dataset to process. Default is None.
    """
    # setup paths to training data and results
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", classifier, sex)
    results_data.mkdir(exist_ok=True, parents=True)

    # get datasets to be evaluated
    dataset = get_datasets_to_process(training_data, results_data, dataset_slice)
    dataset = sorted(dataset)
    if not dataset:
        print(f"No pending datasets for {classifier} on {sex}.")

    for training_dataset_str in dataset:
        results_file = results_data.joinpath(str(training_dataset_str))
        # path to training dataset
        training_dataset = training_data.joinpath(training_dataset_str)
        # load dataset
        with open(training_data.joinpath(str(training_dataset.name), "dataset.pk"), "rb") as f:
            train_set = pickle.load(f)
        dataset = {
            "X": np.array(train_set["data"], dtype=np.float64),
            "y": np.array(train_set["labels"]),
        }
        # Stacking models encapsulate scaling inside the base learners that need it,
        # which keeps the stacker's internal CV from seeing globally scaled data.
        pipeline, param_grid = get_classifier(classifier, random_seed=RANDOM_SEED)
        cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        total_fits = N_OPTUNA_TRIALS * cross_validation.get_n_splits()
        print(f"Starting {classifier} TPE search for {training_dataset.name}: {total_fits} fits total.")
        search_results = run_optuna_search(
            X=dataset["X"],
            y=dataset["y"],
            classifier=classifier,
            param_grid=param_grid,
            cross_validation=cross_validation,
        )
        # create folder with the training_dataset name to store results
        results_file.mkdir(parents=True, exist_ok=True)
        # no need to write header again and again and again,...
        if results_file.joinpath("results.csv").exists():
            header = False
        else:
            header = True

        # dump Optuna search results to results.csv
        search_results.round(6)[
            ["params", "mean_test_recall", "mean_test_specificity", "mean_test_uar"]
        ].to_csv(
            results_file.joinpath("results.csv"),
            index=False,
            mode="a",
            header=header,
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
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    data_slice = None if not args.test else 50
    main(sex_to_compute, used_classifier, data_slice)
