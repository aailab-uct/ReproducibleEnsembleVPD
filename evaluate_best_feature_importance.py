"""Evaluate held-out permutation importance of the highest-UAR model per group.

Run: python evaluate_best_feature_importance.py
Replot saved summaries: python evaluate_best_feature_importance.py --plots-only
Ranked plots are saved as PNG and PDF alongside the CSV summaries.
Importance is the decrease in UAR (balanced accuracy), in units of 0--1.
Negative values are retained. Summary standard deviations describe variation
between fold means, not confidence intervals. Selection uses existing CV results;
these analyses are conditional on that selection, not an independent assessment.
"""

import argparse
import ast
import json
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
from evaluate_stacking_feature_importance import FEATURE_NAMES
from generate_latex_tables import CLASSIFIER_NAMES, RESULT_FILENAME


GROUPS = ("men", "women", "both")


def select_best_results(input_dir: Path) -> pd.DataFrame:
    """Search every CSV row; ties retain the first sorted filename/row."""
    frames = []
    required = {"classifier", "dataset", "hyperparameters", "test_uar"}
    for path in sorted(input_dir.rglob("*.csv")):
        match = RESULT_FILENAME.fullmatch(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path, dtype={"dataset": str})
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
        frame["test_uar"] = pd.to_numeric(frame["test_uar"], errors="raise")
        if frame[list(required)].isna().any().any() or not frame["test_uar"].between(0, 1).all():
            raise ValueError(f"{path}: missing values or invalid UAR")
        frame["group"] = match.group(1)
        frame["source_file"] = str(path)
        frame["source_row"] = np.arange(len(frame)) + 1
        frames.append(frame)
    if not frames:
        raise ValueError(f"No cross-validation results found in {input_dir}")
    results = pd.concat(frames, ignore_index=True)
    missing = set(GROUPS) - set(results["group"])
    if missing:
        raise ValueError(f"Missing groups: {sorted(missing)}")
    indices = results.groupby("group")["test_uar"].idxmax()
    return results.loc[indices].set_index("group").loc[list(GROUPS)].reset_index()


def load_dataset(dataset_dir: Path):
    """Recover feature order from the configuration written by datasets_generator."""
    with (dataset_dir / "dataset.pk").open("rb") as handle:
        dataset = pickle.load(handle)
    config = json.loads((dataset_dir / "config.json").read_text())
    always_present = {"sex", "age", "mean_f0", "hnr", "jitter", "shimmer"}
    names = [
        name for name in FEATURE_NAMES
        if name in always_present or config[
            "spectral_contrast" if name.startswith("spectral_contrast_") else name
        ]
    ]
    features = np.asarray(dataset["data"], dtype=np.float64)
    labels = np.asarray(dataset["labels"])
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError(f"{dataset_dir}: feature count does not match config.json")
    if labels.ndim != 1 or len(labels) != len(features) or set(labels) != {0, 1}:
        raise ValueError(f"{dataset_dir}: expected one binary label (0/1) per row")
    if not np.isfinite(features).all():
        raise ValueError(f"{dataset_dir}: non-finite feature values")
    return features, labels, names


def evaluate_model(selected, features, labels, names, args):
    """Fit the full pipeline on each training fold and permute held-out inputs."""
    if np.bincount(labels.astype(int)).min() < args.n_splits:
        raise ValueError(f"{selected.group}: fewer samples per class than folds")
    parameters = ast.literal_eval(selected.hyperparameters)
    if not isinstance(parameters, dict):
        raise ValueError("Expected a hyperparameter dictionary")
    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits, n_repeats=args.cv_repeats, random_state=args.seed,
    )
    frames = []
    for split, (train, test) in enumerate(tqdm(
        cv.split(features, labels), total=args.n_splits * args.cv_repeats, desc=selected.group,
    )):
        pipeline, _ = get_classifier(selected.classifier, random_seed=args.seed, hyperparameters=parameters)
        # Parallelize permutations, avoiding nested estimator process pools.
        pipeline.set_params(**{
            name: 1 for name in pipeline.get_params() if name.endswith("n_jobs")
        })
        pipeline.fit(features[train], labels[train])
        baseline = balanced_accuracy_score(labels[test], pipeline.predict(features[test]))
        importance = permutation_importance(
            pipeline, features[test], labels[test], scoring="balanced_accuracy",
            n_repeats=args.permutations, random_state=args.seed + split, n_jobs=args.n_jobs,
        )
        frames.append(pd.DataFrame({
            "group": selected.group,
            "classifier": selected.classifier,
            "dataset": selected.dataset,
            "repeat": split // args.n_splits + 1,
            "fold": split % args.n_splits + 1,
            "held_out_rows": len(test),
            "baseline_uar": baseline,
            "feature_index": np.arange(len(names)),
            "feature": names,
            "importance_mean": importance.importances_mean,
            "importance_permutation_std": importance.importances_std,
        }))
    return pd.concat(frames, ignore_index=True)


def plot_importance(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean UAR decrease with between-fold standard deviation error bars."""
    ranked = summary.sort_values("mean_importance", ascending=False, kind="stable")
    group = ranked["group"].iloc[0]
    classifier = ranked["classifier"].iloc[0]
    figure = Figure(figsize=(9, max(4, 0.32 * len(ranked) + 2)), layout="constrained")
    axis = figure.subplots()
    means = ranked["mean_importance"].to_numpy() * 100
    axis.barh(
        ranked["feature"].str.replace("_", " "), means,
        xerr=ranked["between_fold_std"].to_numpy() * 100,
        color=np.where(means >= 0, "#3274A1", "#C76A42"),
        error_kw={"ecolor": "#444444", "capsize": 3, "elinewidth": 1},
    )
    axis.invert_yaxis()
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    population = "Men and women combined" if group == "both" else group.capitalize()
    axis.set_title(
        f"{population} — {CLASSIFIER_NAMES.get(classifier, classifier)}\n"
        "Permutation feature importance", pad=12,
    )
    axis.set_xlabel(
        "Mean decrease in UAR (percentage points)\n"
        "Error bars: ±1 standard deviation across fold means"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"{group}_importance.{extension}", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results_xvalidation"))
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_interpretability/best_classifiers"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots-only", action="store_true", help="Plot existing summaries without refitting models")
    args = parser.parse_args()
    if args.plots_only:
        for group in GROUPS:
            plot_importance(pd.read_csv(args.output_dir / f"{group}_importance_summary.csv"), args.output_dir)
        print(f"Saved PNG and PDF plots to {args.output_dir}")
        return
    if args.n_splits < 2 or args.cv_repeats < 1 or args.permutations < 1 or args.n_jobs == 0:
        parser.error("Require at least 2 folds, positive repeats/permutations, and nonzero n-jobs")
    selected = select_best_results(args.input_dir)
    print(selected[["group", "classifier", "dataset", "test_uar"]].to_string(index=False), flush=True)
    datasets = [load_dataset(args.data_dir / row.group / row.dataset) for row in selected.itertuples()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_classifiers.csv", index=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    for row, dataset in zip(selected.itertuples(), datasets):
        by_fold = evaluate_model(row, *dataset, args)
        summary = (
            by_fold.groupby(["group", "classifier", "dataset", "feature_index", "feature"], sort=False)
            ["importance_mean"].agg(
                mean_importance="mean", between_fold_std=lambda values: values.std(ddof=0), fold_count="size",
            ).reset_index().sort_values("mean_importance", ascending=False, kind="stable")
        )
        by_fold.to_csv(args.output_dir / f"{row.group}_importance_by_fold.csv", index=False)
        summary.to_csv(args.output_dir / f"{row.group}_importance_summary.csv", index=False)
        plot_importance(summary, args.output_dir)
        print(f"Saved {row.group} feature importance to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
