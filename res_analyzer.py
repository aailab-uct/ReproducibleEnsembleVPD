import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_uar(data, color=None, **kwargs):
    ax = plt.gca()
    numeric_values = pd.to_numeric(data["value"], errors="coerce")
    if numeric_values.notna().all():
        plot_data = data.assign(plot_value=numeric_values)
        if numeric_values.nunique() > 50:
            plot_data["group"] = pd.qcut(
                numeric_values, q=50, labels=False, duplicates="drop"
            )
        else:
            plot_data["group"] = numeric_values

        grouped = plot_data.groupby("group", sort=True, observed=True)
        x = grouped["plot_value"].median().to_numpy()
        uar = grouped["mean_test_uar"]
        quantiles = uar.quantile([0, 0.25, 0.5, 0.75, 1]).unstack().to_numpy()
        ax.fill_between(
            x, quantiles[:, 0], quantiles[:, 4], color=color, alpha=0.15,
            label="min–max"
        )
        ax.fill_between(
            x, quantiles[:, 1], quantiles[:, 3], color=color, alpha=0.35,
            label="Q1–Q3"
        )
        ax.plot(x, quantiles[:, 2], color=color, label="median")
        if x.min() > 0 and x.max() / x.min() >= 100:
            ax.set_xscale("log")
        return

    plot_data = data.assign(plot_value=data["value"].astype(str))
    order = sorted(plot_data["plot_value"].unique())
    sns.boxplot(
        data=plot_data,
        x="plot_value",
        y="mean_test_uar",
        order=order,
        whis=(0, 100),
        color=color,
        fliersize=0,
    )
    plt.xticks(rotation=45, ha="right")


for classif in [
            # "knn",
            # "svm_rbf",
            # "svm_poly",
            # "random_forest",
            "adaboost",
        ]:
    data_full = pd.DataFrame()
    print(classif, "started")
    for sex in ["women", "men", "both"]:
        data = pd.read_csv(f"results.old/{classif}/{sex}/0/results.csv")
        params = pd.DataFrame.from_records(
            eval(value, {"__builtins__": {}, "np": np}) for value in data.pop("params")
        )
        param_keys = params.columns.tolist()
        data[param_keys] = params
        data =data.melt(value_vars=param_keys, id_vars="mean_test_uar")
        data["sex"] = sex
        data = (
            data.sort_values("mean_test_uar", ascending=False)
            .groupby(["variable", "value"], sort=False, dropna=False)
            .head(50)
        )
        data_full = pd.concat([data_full,data])

    grid = sns.FacetGrid(
        data_full, col="variable", row="sex", sharey=True, sharex=False
    )
    grid.map_dataframe(plot_uar)
    grid.set_axis_labels("value", "mean_test_uar")
    grid.add_legend()
    grid.figure.suptitle(classif)
    grid.figure.tight_layout()
    print(classif, "finalized")
plt.show()
