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


# definition of classifier parameters for search
grids = {
    "svm_rbf": {
        "classifier__C": [0.1, 0.3, 1, 3, 10 , 31, 100, 316 , 1000, 3162, 10000, 31622], # logaritmic scale
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.001, 0.01, 0.1, 1, "auto"],
    },
    "knn": {
        "classifier__n_neighbors": list(range(1,25+1,2)),
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    },
    "random_forest": {
        "classifier__n_estimators": list(range(16, 512+1, 16)),
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": list(range(1,8+1,1)),
        "classifier__max_features": ["sqrt"],
    },
    "adaboost": {
        "classifier__n_estimators": list(range(25,500+1,25)),
        "classifier__learning_rate": [0.001, 0.01, 0.1, 1, 10],
    },
}

# voting classifiers
grids |= {
    "voting_knn": {
        "classifier__voting": ["hard"],
    }
    | {k.replace("__", "__svm__"): v for k, v in grids["svm_rbf"].items()}
    | {k.replace("__", "__knn__"): v for k, v in grids["knn"].items()}
    | {k.replace("__", "__random_forest__"): v for k, v in grids["random_forest"].items()},
    "voting_adaboost": {
            "classifier__voting": ["hard"],
        }
        | {k.replace("__", "__svm__"): v for k, v in grids["svm_rbf"].items()}
        | {k.replace("__", "__adaboost__"): v for k, v in grids["adaboost"].items()}
        | {k.replace("__", "__random_forest__"): v for k, v in grids["random_forest"].items()},
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

        case "voting_knn":
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
