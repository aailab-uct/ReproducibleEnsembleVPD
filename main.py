"""
This is the main file of the project. It is used to run all the other files in the correct order
and to check everything is working as expected.
"""

import os
import warnings
from multiprocessing import freeze_support

from average_results import process_results
from classifier_pipeline import main as classifier_pipeline
from datasets_generator import main as datasets_generator
from feature_maker import main as feature_maker
from repeated_cross_validation_best import main as repeated_cross_validation_best

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# machine dependent settings (set this according to available threads on your computational infrastructure)
MAX_WORKERS = 24


def main(classifiers: list[str], sexes: list[str]):
    print("Make features")
    feature_maker()
    print("Generate datasets")
    datasets_generator(max_workers=MAX_WORKERS)

    # 4. run classifier_pipeline.py
    for sex in sexes:
        for classifier in classifiers:
            print(f"Running {classifier} on {sex} dataset.")
            classifier_pipeline(sex=sex, classifier=classifier)

    # 5. run average_results.py to get the best datasets according to average performance of classifiers
    print("-" * 79)
    print("Step 5: Computing average performance of classifiers for each dataset.")
    process_results(classifiers)

    # 6. run svm_repeated_xvalidation.py to evaluate the best classifiers on the best on 10 ten average datasets
    print("-" * 79)
    print("Step 6: Running repeated stratified 10 fold cross-validation for top classifiers.")
    repeated_cross_validation_best(classifiers=classifiers, sexes=sexes)


if __name__ == "__main__":
    freeze_support()
    main(
        classifiers=[
            "knn",
            "svm_rbf",
            "svm_poly",
            "random_forest",
            "adaboost",
            "voting_rbf",
            "voting_poly",
            "voting_adaboost",
        ],
        sexes=["women", "men", "both"],
    )
