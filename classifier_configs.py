"""Module for loading pipelines with corresponding classifiers and hyperparameters for search."""
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# Disable warnings
import os, warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def continuous_uniform_from_values(values):
    """Create a continuous uniform distribution from the numeric min/max in values."""
    numeric_values = sorted(value for value in values if isinstance(value, (int, float)))
    low = numeric_values[0]
    high = numeric_values[-1]
    return uniform(loc=low, scale=high - low)


def integer_uniform_from_values(values):
    """Create a discrete uniform distribution from the numeric min/max in values."""
    numeric_values = sorted(int(value) for value in values if isinstance(value, (int, float)))
    low = numeric_values[0]
    high = numeric_values[-1]
    return randint(low, high + 1)


# definition of classifier parameters for search
grids = {
    'svm_poly': {
        "classifier__C": loguniform(0.1, 10000),
        "classifier__kernel": ["poly"],
        "classifier__gamma": loguniform(0.001, 1),
        "classifier__degree": integer_uniform_from_values([2, 3, 4, 5, 6])
    },
    'svm_rbf': {
        "classifier__C": loguniform(0.1, 10000),
        "classifier__kernel": ["rbf"],
        "classifier__gamma": loguniform(0.001, 1),
    },
    'knn': {
        "classifier__n_neighbors": integer_uniform_from_values([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": integer_uniform_from_values([1, 2])},
    'random_forest': {
        "classifier__n_estimators": randint(20, 251),
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": integer_uniform_from_values([2, 3, 4, 5, 6]),
        "classifier__max_features": ["sqrt"]
    },
    'voting': [
        {
            "classifier__voting": ["hard"],
            "classifier__svm__kernel": ["rbf"],
            "classifier__svm__C": loguniform(0.1, 10000),
            "classifier__svm__gamma": loguniform(0.001, 1),
            "classifier__knn__n_neighbors": integer_uniform_from_values([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
            "classifier__knn__weights": ["uniform", "distance"],
            "classifier__knn__p": integer_uniform_from_values([1, 2]),
            "classifier__random_forest__n_estimators": randint(20, 251),
            "classifier__random_forest__criterion": ["gini"],
            "classifier__random_forest__min_samples_split": integer_uniform_from_values([2, 3, 4, 5, 6]),
            "classifier__random_forest__max_features": ["sqrt"]
        },
        {
            "classifier__voting": ["hard"],
            "classifier__svm__kernel": ["poly"],
            "classifier__svm__C": loguniform(0.1, 10000),
            "classifier__svm__gamma": loguniform(0.001, 1),
            "classifier__svm__degree": integer_uniform_from_values([2, 3, 4, 5, 6]),
            "classifier__knn__n_neighbors": integer_uniform_from_values([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
            "classifier__knn__weights": ["uniform", "distance"],
            "classifier__knn__p": integer_uniform_from_values([1, 2]),
            "classifier__random_forest__n_estimators": randint(20, 251),
            "classifier__random_forest__criterion": ["gini"],
            "classifier__random_forest__min_samples_split": integer_uniform_from_values([2, 3, 4, 5, 6]),
            "classifier__random_forest__max_features": ["sqrt"]
        }
    ]

}


def get_classifier(classifier_name: str, both_sexes: bool = False, random_seed: int = 42, hyperparameters: dict = None):
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
            classifier = ("classifier", SVC(max_iter=int(5e5),
                                            random_state=random_seed,
                                            **processed_hyperparameters))
        case "svm_rbf":
            classifier = ("classifier", SVC(max_iter=int(5e5),
                                            random_state=random_seed,
                                            **processed_hyperparameters))
        case "knn":
            classifier = ("classifier", KNeighborsClassifier(**processed_hyperparameters))

        case "random_forest":
            classifier = ("classifier", RandomForestClassifier(random_state=random_seed,
                                                               **processed_hyperparameters))

        case "voting":
            classifier = (
                "classifier",
                VotingClassifier(
                    estimators=[
                        ("svm", SVC(max_iter=int(5e5), random_state=random_seed)),
                        ("knn", KNeighborsClassifier()),
                        ("random_forest", RandomForestClassifier(random_state=random_seed)),
                    ],
#                    **processed_hyperparameters
                )
            )
        case _:
            raise ValueError(f"Unknown classifier: {classifier_name}")

    pipe = Pipeline([
        ("minmaxscaler", MinMaxScaler()),
        classifier
    ])
    if hyperparameters is not None:
        pipe.set_params(**hyperparameters)
    return pipe, grids[classifier_name]
