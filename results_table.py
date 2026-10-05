"""Minimal LaTeX results table from the nested CV results in results/."""

import csv
import statistics
from pathlib import Path

RESULTS_DIR = Path("results")
CLASSIFIERS = {
    "adaboost": "AdaBoost",
    "random_forest": "Random forest",
    "stacking_adaboost": "Stacking",
    "svm_rbf": "SVM (RBF)",
}
SEXES = ["women", "men", "both"]
METRICS = ["mean_test_recall", "mean_test_specificity", "mean_test_uar"]

# (clf, sex) -> [(mean, std) per metric] in percent over the outer folds of all repetitions
data = {}
for clf in CLASSIFIERS:
    for sex in SEXES:
        rows = []
        for path in (RESULTS_DIR / clf / sex).glob("*/results.csv"):
            with open(path, newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        data[clf, sex] = []
        for m in METRICS:
            values = [100 * float(row[m]) for row in rows]
            data[clf, sex].append((statistics.fmean(values), statistics.stdev(values)))

# best mean per group and metric is printed in bold
best = {
    (sex, i): max(data[clf, sex][i][0] for clf in CLASSIFIERS)
    for sex in SEXES
    for i in range(len(METRICS))
}

lines = [
    r"\begin{table*}[tb]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{Classifier performance by group. Values are mean $\pm$ standard deviation"
    r" in percent across the outer folds of nested 10-fold cross-validation. The highest"
    r" mean for each metric within each group is bold. UAR denotes unweighted average recall.}",
    r"\label{tab:xvalidation_combined}",
    r"\begin{tabular}{L{3.0cm}L{1.4cm}C{3cm}C{3cm}C{3cm}}",
    r"\hline",
    r"Classifier & Group & Sensitivity (\%) & Specificity (\%) & UAR (\%) \\",
    r"\hline",
]
for clf, clf_label in CLASSIFIERS.items():
    for offset, sex in enumerate(SEXES):
        label = rf"\multirow{{{len(SEXES)}}}{{*}}{{{clf_label}}}" if offset == 0 else ""
        cells = []
        for i, (mean, std) in enumerate(data[clf, sex]):
            cell = f"${mean:.1f} \\pm {std:.1f}$"
            if mean == best[sex, i]:
                cell = rf"{{\boldmath {cell}}}"
            cells.append(cell)
        lines.append(f"{label} & {sex} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
lines += [r"\end{tabular}", r"\end{table*}"]

out = RESULTS_DIR / "results_table.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"written to {out}")
