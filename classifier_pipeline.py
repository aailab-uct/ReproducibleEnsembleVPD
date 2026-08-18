"""
Script that performs randomized search for classifiers across various datasets.
Datasets are balanced via SMOTE and results are obtained via stratified
10-fold cross-validation.
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, make_scorer, recall_score
from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
)

from classifier_configs import get_classifier

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# setup random seed to make code reproducible
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_ITER_SEARCH = 50000


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


class ProgressRandomizedSearchCV(RandomizedSearchCV):
    """RandomizedSearchCV variant that prints a single-line progress counter."""

    def __init__(
        self,
        *args,
        progress_label: str = "Random search",
        progress_splits: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_label = progress_label
        self.progress_splits = progress_splits

    def _run_search(self, evaluate_candidates):
        def format_seconds(seconds: float) -> str:
            total_seconds = max(0, round(seconds))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            if hours:
                return f"{hours:d}h {minutes:02d}m {secs:02d}s"
            if minutes:
                return f"{minutes:d}m {secs:02d}s"
            return f"{secs:d}s"

        total_fits = self.n_iter * self.progress_splits
        started_at = time.time()
        for candidate_index, params in enumerate(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            ),
            start=1,
        ):
            evaluate_candidates([params])
            fits_completed = candidate_index * self.progress_splits
            elapsed_seconds = time.time() - started_at
            seconds_per_fit = elapsed_seconds / fits_completed
            remaining_fits = total_fits - fits_completed
            eta_seconds = remaining_fits * seconds_per_fit
            sys.stdout.write(
                f"\r{self.progress_label}: {fits_completed}/{total_fits} fits completed"
                f" | elapsed {format_seconds(elapsed_seconds)}"
                f" | avg {seconds_per_fit:.2f}s/fit"
                f" | ETA {format_seconds(eta_seconds)}"
            )
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()


def get_datasets_to_process(
    datasets_path: Path, results_data: Path, dataset_slice: tuple | int | None = None
):
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

    # get all datasets, that were already evaluated
    tr = sorted([str(x.name) for x in results_data.iterdir()])
    # only evaluate datasets that were not processed so far
    to_do = sorted(set(td) - set(tr))

    return to_do


# pylint: disable=too-many-locals


def main(sex: str = "women", classifier="voting", dataset_slice=None):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "voting".
    :param dataset_slice: The slice of the dataset to process. Default is None.
    """
    # setup paths to training data and results
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", classifier, sex)
    results_data.mkdir(exist_ok=True, parents=True)

    # get datasets to be evaluated
    dataset = get_datasets_to_process(training_data, results_data, dataset_slice)
    dataset = sorted(dataset)

    for training_dataset_str in dataset:
        results_file = results_data.joinpath(str(training_dataset_str))
        # path to training dataset
        training_dataset = training_data.joinpath(training_dataset_str)
        # print(f"evaluate {training_dataset}")
        results_data.joinpath(str(training_dataset.name)).mkdir(
            parents=True, exist_ok=True
        )
        # load dataset
        with open(
            training_data.joinpath(str(training_dataset.name), "dataset.pk"), "rb"
        ) as f:
            train_set = pickle.load(f)
        dataset = {
            "X": np.array(train_set["data"], dtype=np.float64),
            "y": np.array(train_set["labels"]),
        }
        # imblearn pipeline perform the resampling only with the training dataset
        # and scaling according to training dataset
        pipeline, param_grid = get_classifier(
            classifier, both_sexes=sex == "both", random_seed=RANDOM_SEED
        )
        cross_validation = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=RANDOM_SEED
        )
        # sklearn randomized search with cross validation
        search_n_jobs = -1  # if classifier == "voting" else -1

        total_fits = N_ITER_SEARCH * cross_validation.get_n_splits()
        print(
            f"Starting {classifier} random search for {training_dataset.name}: {total_fits} fits total."
        )
        random_search = ProgressRandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=N_ITER_SEARCH,
            cv=cross_validation,
            scoring=scoring_dict,
            n_jobs=search_n_jobs,
            refit=False,
            random_state=RANDOM_SEED,
            progress_label=f"{classifier} random search for {training_dataset.name}",
            progress_splits=cross_validation.get_n_splits(),
        )
        # else:
        #     random_search = RandomizedSearchCV(
        #         pipeline,
        #         param_distributions=param_grid,
        #         n_iter=N_ITER_SEARCH,
        #         cv=cross_validation,
        #         scoring=scoring_dict,
        #         n_jobs=search_n_jobs,
        #         refit=False,
        #         random_state=RANDOM_SEED,
        #     )
        random_search.fit(dataset["X"], dataset["y"])
        # create folder with the training_dataset name to store results
        results_file.mkdir(exist_ok=True)
        # no need to write header again and again and again,...
        if results_file.joinpath("results.csv").exists():
            header = False
        else:
            header = True

        # dump randomized search results to results.csv
        pd.DataFrame(random_search.cv_results_).round(6)[
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
        description="Perform randomized search for a classifier across various datasets. ",
    )
    parser.add_argument("classifier", type=str)
    parser.add_argument("sex", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    data_slice = None if not args.test else 50
    main(sex_to_compute, used_classifier, data_slice)
