"""Module for loading pipelines with corresponding classifiers and hyperparameters for search."""
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# Disable warnings
import os, warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# definition of classifier parameters for search
grids = {
    'svm_poly': {
        "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
        "classifier__kernel": ["poly"],
        "classifier__gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],
        "classifier__degree": [2, 3, 4, 5, 6]
    },
    'svm_rbf': {
        "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],
    },
    'knn': {
        "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2]},
    'random_forest': {
        "classifier__n_estimators": [50, 75, 100, 125, 150, 175],
        "classifier__criterion": ["gini"],
        "classifier__min_samples_split": [2, 3, 4, 5, 6],
        "classifier__max_features": ["sqrt"]
    },
    'voting': [
        {
            "classifier__voting": ["hard"],
            "classifier__svm__kernel": ["rbf"],
            "classifier__svm__C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
            "classifier__svm__gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],
            "classifier__knn__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            "classifier__knn__weights": ["uniform", "distance"],
            "classifier__knn__p": [1, 2],
            "classifier__random_forest__n_estimators": [50, 75, 100, 125, 150, 175],
            "classifier__random_forest__criterion": ["gini"],
            "classifier__random_forest__min_samples_split": [2, 3, 4, 5, 6],
            "classifier__random_forest__max_features": ["sqrt"]
        },
        {
            "classifier__voting": ["hard"],
            "classifier__svm__kernel": ["poly"],
            "classifier__svm__C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
            "classifier__svm__gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],
            "classifier__svm__degree": [2, 3, 4, 5, 6],
            "classifier__knn__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
            "classifier__knn__weights": ["uniform", "distance"],
            "classifier__knn__p": [1, 2],
            "classifier__random_forest__n_estimators": [50, 75, 100, 125, 150, 175],
            "classifier__random_forest__criterion": ["gini"],
            "classifier__random_forest__min_samples_split": [2, 3, 4, 5, 6],
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
                    **processed_hyperparameters
                )
            )
        case _:
            raise ValueError(f"Unknown classifier: {classifier_name}")

    pipe = Pipeline([
        ("minmaxscaler", MinMaxScaler()),
        classifier
    ])
    return pipe, grids[classifier_name]
