"""Module for loading pipelines with corresponding classifiers and hyperparameters for search."""

import os
import warnings

from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ADABOOST_N_EST = [50, 100, 150, 200, 250, 300, 350, 400]

# definition of classifier parameters for search
grids = {
    "svm_poly": {
        "classifier__C": loguniform(0.1, 10000),
        "classifier__kernel": ["poly"],
        "classifier__gamma": loguniform(0.001, 1),
        "classifier__degree": [2, 3, 4, 5, 6],
    },
    "svm_rbf": {
        "classifier__C": loguniform(0.1, 10000),
        "classifier__kernel": ["rbf"],
        "classifier__gamma": loguniform(0.001, 1),
    },
    "knn": {
        "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    },
    "random_forest": {
        "classifier__n_estimators": randint(20, 251),
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": [2, 3, 4, 5, 6],
        "classifier__max_features": ["sqrt"],
    },
    "adaboost": {
        "classifier__n_estimators": ADABOOST_N_EST,
        "classifier__learning_rate": [0.1, 1, 10],
    },
}

# voting classifiers
grids |= {
    "voting_rbf": {
        "classifier__voting": ["hard"],
    }
    | {k.replace("__", "__svm__"): v for k, v in grids["svm_rbf"].items()}
    | {k.replace("__", "__knn__"): v for k, v in grids["knn"].items()}
    | {k.replace("__", "__random_forest__"): v for k, v in grids["random_forest"].items()},
    "voting_poly": {
        "classifier__voting": ["hard"],
    }
    | {k.replace("__", "__svm__"): v for k, v in grids["svm_poly"].items()}
    | {k.replace("__", "__knn__"): v for k, v in grids["knn"].items()}
    | {k.replace("__", "__random_forest__"): v for k, v in grids["random_forest"].items()},
    "voting_adaboost": [
        {
            "classifier__voting": ["hard"],
        }
        | {k.replace("__", "__svm__"): v for k, v in grids["svm_poly"].items()}
        | {k.replace("__", "__adaboost__"): v for k, v in grids["adaboost"].items()}
        | {k.replace("__", "__random_forest__"): v for k, v in grids["random_forest"].items()},
        {
            "classifier__voting": ["hard"],
        }
        | {k.replace("__", "__svm__"): v for k, v in grids["svm_rbf"].items()}
        | {k.replace("__", "__adaboost__"): v for k, v in grids["adaboost"].items()}
        | {k.replace("__", "__random_forest__"): v for k, v in grids["random_forest"].items()},
    ],
}


def get_classifier(
    classifier_name: str,
    random_seed: int = 42,
    hyperparameters: dict | None = None,
):
    """
    Get classifier with the given name.
    :param classifier_name: str, name of the classifier
    currently supported: "svm_poly", "svm_rbf", "knn", "gauss_nb", "random_forest",
    "adaboost", "decisiontree", "voting"
    :param random_seed: int, random seed
    :param hyperparameters: dict, hyperparameters if already known (for repeated cross-validation)
    default: None
    :return: classifier
    :return: parameter space for search
    """
    if hyperparameters is None:
        processed_hyperparameters = {}
    else:
        processed_hyperparameters = {}
        for key, value in hyperparameters.items():
            key = key.replace("classifier__", "")
            processed_hyperparameters[key] = value
    match classifier_name:
        case "svm_poly":
            classifier = (
                "classifier",
                SVC(
                    max_iter=int(5e5),
                    random_state=random_seed,
                    class_weight="balanced",
                    **processed_hyperparameters,
                ),
            )
        case "svm_rbf":
            classifier = (
                "classifier",
                SVC(
                    max_iter=int(5e5),
                    random_state=random_seed,
                    class_weight="balanced",
                    **processed_hyperparameters,
                ),
            )
        case "knn":
            classifier = (
                "classifier",
                KNeighborsClassifier(n_jobs=-1, **processed_hyperparameters),
            )
        case "adaboost":
            classifier = (
                "classifier",
                AdaBoostClassifier(random_state=random_seed),
            )
        case "random_forest":
            classifier = (
                "classifier",
                RandomForestClassifier(
                    random_state=random_seed,
                    n_jobs=-1,
                    class_weight="balanced",
                    **processed_hyperparameters,
                ),
            )

        case "voting" | "voting_rbf" | "voting_poly":
            classifier = (
                "classifier",
                VotingClassifier(
                    estimators=[
                        (
                            "svm",
                            SVC(
                                max_iter=int(5e5),
                                random_state=random_seed,
                                class_weight="balanced",
                            ),
                        ),
                        ("knn", KNeighborsClassifier(n_jobs=1)),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                random_state=random_seed,
                                n_jobs=1,
                                class_weight="balanced",
                            ),
                        ),
                    ],
                    n_jobs=-1,
                    #                    **processed_hyperparameters
                ),
            )
        case "voting_adaboost":
            classifier = (
                "classifier",
                VotingClassifier(
                    estimators=[
                        (
                            "svm",
                            SVC(
                                max_iter=int(5e5),
                                random_state=random_seed,
                                class_weight="balanced",
                            ),
                        ),
                        (
                            "adaboost",
                            AdaBoostClassifier(random_state=random_seed),
                        ),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                random_state=random_seed,
                                n_jobs=1,
                                class_weight="balanced",
                            ),
                        ),
                    ],
                    n_jobs=-1,
                    #                    **processed_hyperparameters
                ),
            )
        case _:
            raise ValueError(f"Unknown classifier: {classifier_name}")

    pipe = Pipeline([("minmaxscaler", MinMaxScaler()), classifier])
    if hyperparameters is not None:
        pipe.set_params(**hyperparameters)
    return pipe, grids[classifier_name]
