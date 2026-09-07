"""
This script performs repeated stratified 10-fold cross validation 100 times for the
best 1000 searched stacking classifiers for a combination of dataset and hyperparameters.
"""

import csv
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tqdm
from sklearn.metrics import balanced_accuracy_score, make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from classifier_configs import get_classifier

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# number of classifiers that should be 100x 10-fold cross-validated
TOP_CLASSIFIERS = 1

# set random seed to allow reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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


# pylint: disable=too-many-locals
def repeated_cross_validation_fit(sex: str, dataset_id: str, clf: str, hyper_parameters: dict, n_repeats: int = 1000):
    """
    Function that performs n_repeats times (by default 100x) 10-fold stratified cross-validation for a classifier
    specified by clf with hyperparameters specified by hyper_parameters for a sex and dataset_id.
    The results are saved into the csv file for each combination of classifier and sex.
    :param sex: women/men
    :param dataset_id: ID of dataset that should be used for validation
    :param clf: classifier type
    :param hyper_parameters: classifier hyperparameters
    :param n_repeats: number of repeats for cross-validation
    :return:
    """
    # result and dataset paths specification
    results_path = Path(".", "results_xvalidation")
    dataset_path = Path(".", "training_data", sex, dataset_id)
    # load dataset
    with open(dataset_path.joinpath("dataset.pk"), "rb") as f:
        train_set = pickle.load(f)
    X = np.array(train_set["data"], dtype=np.float64)
    y = np.array(train_set["labels"])

    # get classifier and its pipeline specified in classifier_pipeline.py
    pipeline, _ = get_classifier(
        clf,
        random_seed=RANDOM_SEED,
        hyperparameters=hyper_parameters,
    )
    # reproducible stratified k-fold cross validation
    stratified_kfold = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=10, random_state=RANDOM_SEED)
    validation_n_jobs = -1
    results = cross_validate(
        pipeline,
        X=X,
        y=y,
        scoring=scoring_dict,
        cv=stratified_kfold,
        n_jobs=validation_n_jobs,
    )
    results_average = {
        "dataset": dataset_id,
        "classifier": clf,
        "hyperparameters": hyper_parameters,
    }
    # drop unused stats
    del results["fit_time"]
    del results["score_time"]
    # compute average metrics and corresponding standard deviations
    for key, values in results.items():
        results_average[key] = round(np.mean(values), 6)
        results_average[f"{key}_stdev"] = round(np.std(values), 6)

    # dump results
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path.joinpath(f"{clf}_{sex}_top{TOP_CLASSIFIERS}_{n_repeats}times.csv")
    headers = results_average.keys()
    if results_file.is_file():
        header = False
    else:
        header = True
    with open(results_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if header:
            writer.writeheader()
        writer.writerow(results_average)


# pylint: enable=too-many-locals


def main(classifiers: list[str], sexes: list[str]):
    """
    Function to find top 1000 results for each type of classifier and sex.
    Performance of those classifiers evaluated in 100 repeats of stratified 10-fold
    cross-validation, provided by repeated_crossvalidation_fit() function. The results
    are stored as csv files and their checksum is compared to the checksums of results
    that are stated in the article.
    :param classifiers: list of classifiers to evaluate
    :param sexes: list of sexes to evaluate
    """

    tables = {}
    for classifier in classifiers:
        # Load the results for each classifier
        for sex in sexes:
            # Load the results for each sex
            result_dir = Path(".", "results", classifier, sex)
            result_files = []
            for csv_path in result_dir.rglob("results.csv"):
                # iterate over all datasets
                df = pd.read_csv(csv_path)
                result_files.append(df)
                result_files[-1]["data"] = csv_path.parent.name  # store the dataset id in the dataframe
                result_files[-1]["clf"] = classifier  # store the classifier name in the dataframe
            tables[f"{classifier}_{sex}"] = (
                pd.concat(
                    result_files  #  store the results in the dictionary
                )
                .sort_values(
                    "mean_test_uar",
                    kind="mergesort",  # sort the results by uar
                )
                .reset_index(drop=True)
            )
    # for best performing classifiers perform the repeated cross validation
    for name, table in tables.items():
        sex = name.split("_")[-1]
        top_k = min(TOP_CLASSIFIERS, len(table))
        for _, row in tqdm.tqdm(table.tail(top_k).iterrows(), total=top_k):
            classifier = row["clf"]
            print(f"{row['mean_test_uar']} - uar ......... for {classifier}")

            hyper_parameters = json.loads(row["params"].replace("'", '"').replace("np.float64(", "").replace(")", ""))

            repeated_cross_validation_fit(
                sex=sex,
                dataset_id=row["data"],
                clf=classifier,
                hyper_parameters=hyper_parameters,
            )


if __name__ == "__main__":
    main(
        classifiers=["svm_rbf"],
        sexes=["women", ],
    )
