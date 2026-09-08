import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

from classifier_configs import get_classifier

RANDOM_SEED = 42
N_SPLITS = 10
N_REPEATS = 100
N_PERMUTATIONS = 100
OUTPUT_FOLDER = Path("feature_importance")

FEATURE_NAMES = (
    "sex",
    "age",
    "diff_pitch",
    "mean_f0",
    "stdev_f0",
    "hnr",
    "jitter",
    "shimmer",
    "spectral_centroid",
    "spectral_contrast_1",
    "spectral_contrast_2",
    "spectral_contrast_3",
    "spectral_contrast_4",
    "spectral_contrast_5",
    "spectral_contrast_6",
    "spectral_contrast_7",
    "spectral_flatness",
    "spectral_rolloff",
    "zero_crossing_rate",
    "shannon_entropy",
    "skewness",
)
CLASSIFIER_NAMES = {
    "stacking_adaboost": "Stacking",
    "svm_rbf": "SVM (rbf)",
    "svm_poly": "SVM (polynomial)",
    "random_forest": "RF",
    "adaboost": "AdaBoost",
}


def evaluate_model(classifier, parameters, data, sex):
    """Fit the full pipeline on each training fold and permute held-out inputs."""
    features = np.asarray(data["data"], dtype=np.float64)
    labels = np.asarray(data["labels"])
    feat_names = FEATURE_NAMES

    if sex != "both":
        assert np.all(features[:, 0] == features[0, 0]), "First row is not sex"
        features = features[:, 1:]
        feat_names = feat_names[1:]

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    frames = []
    for split, (train, test) in enumerate(
        tqdm(
            cv.split(features, labels),
            total=N_SPLITS * N_REPEATS,
        )
    ):
        pipeline, _ = get_classifier(classifier, random_seed=RANDOM_SEED, hyperparameters=parameters)
        # Parallelize permutations, avoiding nested estimator process pools.
        pipeline.fit(features[train], labels[train])
        baseline = balanced_accuracy_score(labels[test], pipeline.predict(features[test]))
        importance = permutation_importance(
            pipeline,
            features[test],
            labels[test],
            scoring="balanced_accuracy",
            n_repeats=N_PERMUTATIONS,
            random_state=RANDOM_SEED + split,
            n_jobs=-1,
        )

        frames.append(
            pd.DataFrame(
                {
                    "classifier": classifier,
                    "sex": sex,
                    "repeat": split // N_SPLITS + 1,
                    "fold": split % N_SPLITS + 1,
                    "held_out_rows": len(test),
                    "baseline_uar": baseline,
                    "feature_name": feat_names,
                    "importances_mean": importance["importances_mean"],
                    "importances_std": importance["importances_std"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_importance(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean UAR decrease with between-fold standard deviation error bars."""
    ranked = summary.sort_values("mean_importance", ascending=False, kind="stable")
    sex = ranked["sex"].iloc[0]
    classifier = ranked["classifier"].iloc[0]
    figure = Figure(figsize=(9, 12), layout="constrained")
    axis = figure.subplots()
    means = ranked["mean_importance"].to_numpy() * 100
    axis.barh(
        ranked["feature_name"].str.replace("_", " "),
        means,
        xerr=ranked["between_fold_std"].to_numpy() * 100,
        color=np.where(means >= 0, "#3274A1", "#C76A42"),
        error_kw={"ecolor": "#444444", "capsize": 3, "elinewidth": 1},
    )
    axis.invert_yaxis()
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    population = "Men and women combined" if sex == "both" else sex.capitalize()
    axis.set_title(
        f"{population} — {CLASSIFIER_NAMES[classifier]}\nPermutation feature importance",
        pad=12,
    )
    axis.set_xlabel("Mean decrease in UAR (percentage points)\nError bars: ±1 standard deviation across fold means")

    figure.savefig(output_dir / f"{classifier}_{sex}_importance.pdf")


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    settings = pd.DataFrame()
    for filename in Path("results_xvalidation").glob("*.csv"):
        sex = str(filename.stem).split("_")[-3]
        current = pd.read_csv(filename)
        current["sex"] = sex
        current = current[["sex", "classifier", "hyperparameters", "test_uar"]]
        settings = pd.concat([settings, current])
    print(settings[["sex", "classifier", "test_uar"]].to_string(index=False), flush=True)

    settings = settings[
        settings["test_uar"] == settings.groupby("sex")["test_uar"].transform("max")
    ]

    all_folds_importances = pd.DataFrame()
    summary_importances = pd.DataFrame()

    for _, row in settings.iterrows():
        print(row)
        print(f"Computing {row['classifier']} for {row['sex']}.")

        with (Path("training_data") / row["sex"] / "0" / "dataset.pk").open("rb") as handle:
            data = pickle.load(handle)

        parameters = ast.literal_eval(row["hyperparameters"])
        by_fold = evaluate_model(row["classifier"], parameters, data, row["sex"])

        all_folds_importances = pd.concat([all_folds_importances, by_fold])
        print(all_folds_importances)

        summary = (
            by_fold.groupby(["sex", "classifier", "feature_name"], sort=False)["importances_mean"]
            .agg(
                mean_importance="mean",
                between_fold_std=lambda values: values.std(ddof=0),
                fold_count="size",
            )
            .reset_index()
            .sort_values("mean_importance", ascending=False, kind="stable")
        )
        summary_importances = pd.concat([summary_importances, summary])

        plot_importance(summary, OUTPUT_FOLDER)

    summary_importances.to_csv(OUTPUT_FOLDER / "summary_importances.csv")
    all_folds_importances.to_csv(OUTPUT_FOLDER / "all_folds_importances.csv")


if __name__ == "__main__":
    main()
