"""Module for loading pipelines with corresponding classifiers and hyperparameters for search."""

import os
import warnings

from imblearn.pipeline import Pipeline
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Disable warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def _get_scaled_model(estimator):
    return SklearnPipeline(
        [
            ("minmaxscaler", MinMaxScaler()),
            ("model", estimator),
        ]
    )


# definition of classifier parameters for search
grids = {
    "svm_poly": {
        "classifier__C": {"low": 0.1, "high": 100_000, "log": True},
        "classifier__kernel": ["poly"],
        "classifier__gamma": {"low": 0.001, "high": 1, "log": True},
        "classifier__degree": {"low": 2, "high": 6, "step": 1},
    },
    "svm_rbf": {
        "classifier__C": {"low": 0.1, "high": 100_000, "log": True},
        "classifier__kernel": ["rbf"],
        "classifier__gamma": {"low": 0.001, "high": 1, "log": True},
    },
    "random_forest": {
        "classifier__n_estimators": {"low": 20, "high": 250, "step": 1},
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": {"low": 2, "high": 6, "step": 1},
        "classifier__max_features": ["sqrt"],
    },
    "adaboost": {
        "classifier__n_estimators": {"low": 40, "high": 400, "step": 1},
        "classifier__learning_rate": {"low": 0.01, "high": 100, "log": True},
    },
}

# stacking classifiers
grids |= {
    "stacking_adaboost": [
        {
            **{
                k.replace("classifier__", "classifier__svm__model__"): v
                for k, v in grids["svm_poly"].items()
            },
            **{
                k.replace("classifier__", "classifier__adaboost__model__"): v
                for k, v in grids["adaboost"].items()
            },
            **{
                k.replace("classifier__", "classifier__random_forest__model__"): v
                for k, v in grids["random_forest"].items()
            },
        },
        {
            **{
                k.replace("classifier__", "classifier__svm__model__"): v
                for k, v in grids["svm_rbf"].items()
            },
            **{
                k.replace("classifier__", "classifier__adaboost__model__"): v
                for k, v in grids["adaboost"].items()
            },
            **{
                k.replace("classifier__", "classifier__random_forest__model__"): v
                for k, v in grids["random_forest"].items()
            },
        },
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
    "adaboost", "decisiontree", "stacking"
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

        case "stacking_adaboost":
            classifier = (
                "classifier",
                StackingClassifier(
                    estimators=[
                        (
                            "svm",
                            _get_scaled_model(
                                SVC(
                                    max_iter=int(5e5),
                                    random_state=random_seed,
                                    class_weight="balanced",
                                )
                            ),
                        ),
                        (
                            "adaboost",
                            _get_scaled_model(
                                AdaBoostClassifier(random_state=random_seed)
                            ),
                        ),
                        (
                            "random_forest",
                            _get_scaled_model(
                                RandomForestClassifier(
                                    random_state=random_seed,
                                    n_jobs=1,
                                    class_weight="balanced",
                                )
                            ),
                        ),
                    ],
                    final_estimator=LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_seed,
                    ),
                    cv=StratifiedKFold(
                        n_splits=10, shuffle=True, random_state=random_seed
                    ),
                    n_jobs=-1,
                ),
            )
        case _:
            raise ValueError(f"Unknown classifier: {classifier_name}")

    pipe = Pipeline([("minmaxscaler", MinMaxScaler()), classifier])

    if hyperparameters is not None:
        pipe.set_params(**hyperparameters)
    return pipe, grids[classifier_name]
