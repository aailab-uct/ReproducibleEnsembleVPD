"""
This is the main file of the project. It is used to run all the other files in the correct order
and to check everything is working as expected.
"""
import sys
import os
from multiprocessing import freeze_support
from pathlib import Path
from typing import List


from src.checksum import test_file_list_from_file, update_checksums
from average_results import process_results
from classifier_pipeline import main as classifier_pipeline
from datasets_generator import main as datasets_generator
from repeated_cross_validation_best import main as repeated_cross_validation_best

# Disable warnings
import os, warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# machine dependent settings (set this according to available threads on your computational infrastructure)
MAX_WORKERS = 24

# dir and filenames
ORIGINAL_DATA_DIR = Path("svd_db")
PREPROCESSED_DATA_DIR = Path("dataset")
CREATED_DATASETS_DIR = Path("training_data")
FILE_INFORMATION_PATH = Path("misc").joinpath("svd_information.csv")
REQUIRED_INITIAL_CHECKSUMS = Path("misc").joinpath("data_used.sha256")
CHECKSUMS_AFTER_I = Path(".").joinpath("misc", "after_I.sha256")
CHECKSUMS_AFTER_II = Path(".").joinpath("misc", "after_II.sha256")
CHECKSUMS_AFTER_III = Path(".").joinpath("misc", "after_III.sha256")
CHECKSUMS_AFTER_IV = Path(".").joinpath("misc", "after_IV.sha256")
CHECKSUMS_AFTER_V = Path(".").joinpath("misc", "after_V.sha256")
CHECKSUMS_AFTER_VI = Path(".").joinpath("misc", "after_VI.sha256")


def data_preparation(check_checksums=False):
    """Function containing first three steps of the pipeline
    :param check_checksums: If True, check if the output of the first three steps is the same as ours."""


    # 3. run datasets_generator.py
    print("-"*79)
    print("Step 3: Datasets Generation")
    print("Running datasets generation.")
    datasets_generator(max_workers=MAX_WORKERS)

    # check if the created files are the same as ours
    if check_checksums:
        print("Checking if the output after datasets generation is the same as ours.")
        ok, _ = test_file_list_from_file(CHECKSUMS_AFTER_III)
        if not ok:
            print("The output data after datasets generation is not the same as ours!")
            print("Continuing anyway. Press Ctrl+C to stop.")

# pylint: disable=too-many-branches,too-many-statements


def main(classifiers: List[str], sexes: List[str]):
    """Function containing the rest of the pipeline
    :param classifiers: List of classifiers to evaluate
    :param sexes: List of sexes to evaluate"""
    print("-" * 79)
    print("Starting the pipeline.")
    print("-" * 79)

    # check if the folder svd_db exists and if the checskums are the same
    print("Checking if the input data is the same as ours.")
    check_futher = False
    # 1-3. run data_preprocessing.py, feature_extraction.py, datasets_generator.py if the checksums are not the same
    data_preparation(check_futher)


    # check if the script is running all the classifiers and sexes
    # if not, we will ignore missing files in checksums and add new checksums to the file
    # because we assume that the user (or more likely us) is computing new results
    if classifiers != ["voting"] or \
        sexes != ["women", "men", "both"]:
        print("The classifiers or sexes are not the same as ours! We will ignore missing files in checksums.")
        ignore_missing_files = True
    else:
        ignore_missing_files = False

    # 4. run classifier_pipeline.py
    print("-" * 79)
    print("Step 4: Evaluating classifiers")
    for sex in sexes:
        for classifier in classifiers:
            print(f"Running {classifier} on {sex} dataset.")
            classifier_pipeline(sex=sex, classifier=classifier)

    # 5. run average_results.py to get the best datasets according to average performance of classifiers
    print("-" * 79)
    print("Step 5: Computing average performance of classifiers for each dataset.")
    process_results(classifiers)
    # if check_futher:
    #     print("Checking if the average results are the same as ours.")
    #     ok, _ = test_file_list_from_file(CHECKSUMS_AFTER_V, ignore_missing_files=ignore_missing_files)
    #     if not ok:
    #         print("The average result over all datasets and classifiers is not the same as ours!")
    #         print("Continuing anyway. Press Ctrl+C to stop.")
    #     if ignore_missing_files:
    #         print("We assume that you are computing new results and we will update the checksums.")
    #         update_checksums(Path("results_tables"), CHECKSUMS_AFTER_V, quiet=False)

    # 6. run svm_repeated_xvalidation.py to evaluate the best classifiers on the best on 10 ten average datasets
    print("-" * 79)
    print("Step 6: Running repeated stratified 10 fold cross-validation for top classifiers.")
    repeated_cross_validation_best(classifiers=classifiers,
                                   sexes=sexes)
    # if check_futher:
    #     print("Checking if the average results obtained during repeated cross-validation are the same as ours.")
    #     ok, _ = test_file_list_from_file(CHECKSUMS_AFTER_VI)
    #     if not ok:
    #         print("The average results for repeated cross-validation step are not same as ours!")
    #         print("Continuing anyway. Press Ctrl+C to stop.")
    #     if ignore_missing_files:
    #         print("We assume that you are computing new results and we will update the checksums.")
    #         update_checksums(Path("results_xvalidation"), CHECKSUMS_AFTER_VI, quiet=False)
# pylint: enable=too-many-branches,too-many-statements


if __name__ == "__main__":
    freeze_support()
    main(classifiers=["knn", "svm_rbf", "svm_poly", "random_forest", "voting"],
         sexes=["women", "men", "both"])
