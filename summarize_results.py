"""Summarise the per-fold results written by train.py.

    python summarize_results.py              # Markdown table
    python summarize_results.py --plot       # also save the confusion matrices
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import CLASSES

ORDER = ["frontal", "left", "right", "independent", "fusion"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--plot", action="store_true", help="save confusion matrices as PNG")
    args = parser.parse_args()

    runs = {}
    for path in Path(args.results_dir).glob("*.json"):
        runs[path.stem] = json.loads(path.read_text())
    if not runs:
        raise SystemExit("no results in {}".format(args.results_dir))

    def sort_key(tag):
        view = runs[tag][0]["view"]
        return (ORDER.index(view) if view in ORDER else len(ORDER), tag)

    print("| Configuration | " + " | ".join("Fold {}".format(f) for f in range(1, 6))
          + " | Accuracy | Macro F1 | AUROC |")
    print("|---" * 9 + "|")
    for tag in sorted(runs, key=sort_key):
        records = runs[tag]
        by_fold = {}
        for record in records:
            by_fold.setdefault(record["fold"], []).append(record["test_acc"])
        fold_cells = " | ".join("{:.3f}".format(np.mean(by_fold[f])) if f in by_fold else "-"
                                for f in range(1, 6))
        stats = []
        for metric in ("test_acc", "test_f1", "test_auroc"):
            values = [r[metric] for r in records]
            stats.append("{:.3f} ± {:.3f}".format(np.mean(values), np.std(values)))
        print("| {} | {} | {} |".format(tag, fold_cells, " | ".join(stats)))

    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sn

        for tag, records in runs.items():
            matrix = np.sum([r["confusion_matrix"] for r in records], axis=0).astype(float)
            matrix /= matrix.sum(axis=1, keepdims=True)
            plt.figure(figsize=(6, 5))
            sn.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                       xticklabels=CLASSES, yticklabels=CLASSES)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(tag)
            plt.tight_layout()
            plt.savefig(Path(args.results_dir) / "confusion_{}.png".format(tag), dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
