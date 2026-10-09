"""
Evaluate classifiers with nested stratified cross-validation and Optuna TPE.
The outer loop uses 10 folds; inner searches use 10 repetitions of 10 folds.
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
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate

from classifier_configs import get_classifier

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# setup random seed to make code reproducible
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_OPTUNA_TRIALS = 500
N_OUTER_SPLITS = 10
N_INNER_SPLITS = 10
N_INNER_REPEATS = 1


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
    cross_validation: RepeatedStratifiedKFold,
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
        candidate = clone(pipeline).set_params(**params)
        scores = cross_validate(
            candidate,
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
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=20, multivariate=True),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return pd.DataFrame(trial_rows)


def run_nested_cross_validation(X, y, pipeline, classifier, param_grid, results_data):
    """Tune only on outer training folds and save independent held-out scores.

    results.csv retains the existing metric columns, with one row per outer
    fold in split order. Inner trial scores are saved separately for inspection.
    """
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    outer_rows = []
    for fold, (train_indices, test_indices) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X[train_indices], y[train_indices]
        inner_cv = RepeatedStratifiedKFold(
            n_splits=N_INNER_SPLITS,
            n_repeats=N_INNER_REPEATS,
            random_state=RANDOM_SEED,
        )
        search_results = run_optuna_search(
            X=X_train,
            y=y_train,
            pipeline=pipeline,
            classifier=f"{classifier} outer fold {fold}/{N_OUTER_SPLITS}",
            param_grid=param_grid,
            cross_validation=inner_cv,
        )
        search_results.to_csv(results_data / f"inner_results_fold_{fold}.csv", index=False)
        valid_results = search_results[np.isfinite(search_results["mean_test_uar"])]
        if valid_results.empty:
            raise ValueError(f"No valid Optuna trials in outer fold {fold}.")
        best_params = valid_results.loc[valid_results["mean_test_uar"].idxmax(), "params"]
        best_pipeline = clone(pipeline).set_params(**best_params)
        best_pipeline.fit(X_train, y_train)
        predictions = best_pipeline.predict(X[test_indices])
        y_test = y[test_indices]
        outer_rows.append({
            "params": best_params,
            "mean_test_recall": float(recall_score(y_test, predictions)),
            "mean_test_specificity": float(recall_score(y_test, predictions, pos_label=0)),
            "mean_test_uar": float(balanced_accuracy_score(y_test, predictions)),
        })
        print(f"Outer fold {fold}/{N_OUTER_SPLITS}: UAR={outer_rows[-1]['mean_test_uar']:.6f}")
    return pd.DataFrame(outer_rows)


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
    total_fits = N_OUTER_SPLITS * (N_OPTUNA_TRIALS * N_INNER_SPLITS * N_INNER_REPEATS + 1)
    print(f"Starting {classifier_str} nested CV for {sex}: {total_fits} fits total.")
    search_results = run_nested_cross_validation(
        X=dataset["X"],
        y=dataset["y"],
        pipeline=pipeline,
        classifier=classifier_str,
        param_grid=param_grid,
        results_data=results_data,
    )

    # Save held-out outer-fold scores; these estimate the tuning procedure's performance.
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
        description="Perform nested CV with Optuna and inner 10-times 10-fold CV.",
    )
    parser.add_argument("classifier", type=str)
    parser.add_argument("sex", type=str)
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    main(sex_to_compute, used_classifier)
