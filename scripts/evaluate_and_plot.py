"""Compare saved contact maps against a reference dataset.

Example:
    python scripts/evaluate_and_plot.py truth.h5 model_a.h5 model_b.h5 \
        --labels Model-A Model-B -o results/comparison

Reference groups contain output, name, family, and length datasets. Prediction
groups contain prediction (binary contact maps, as expected by the existing
evaluation helpers). Samples are taken from the first prediction file and
matched by group key. Extra samples in other files are ignored.
CSVs include standard and shifted metrics. Violin plots compare per-sample F1
overall and within each family, using the same style as plot_combined_evaluation.
"""

import argparse, os, h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from utils.evaluation import compute_sample_metrics, summarize_by_family, summarize_metrics
from utils.plots import violin_plot


def evaluate_files(dataset, predictions, labels):
    """Return sample, overall, and family dataframes with a method column."""
    if not predictions:
        raise ValueError("Provide at least one prediction file.")
    if len(labels) != len(predictions) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique label per prediction file.")
    rows = []
    with h5py.File(predictions[0], "r") as first:
        samples = list(first.keys())
    if not samples:
        raise ValueError(f"No samples in {predictions[0]}")

    with h5py.File(dataset, "r") as reference:
        for path, label in zip(predictions, labels):
            with h5py.File(path, "r") as predicted:
                for sample in tqdm(samples, desc=label, unit="sample"):
                    try:
                        data = reference[sample]
                        matrix = predicted[sample]["predicted"][()]
                        row = compute_sample_metrics(matrix, data)
                    except (KeyError, ValueError, AssertionError, TypeError) as exc:
                        raise ValueError(f"Cannot evaluate sample {sample!r} in {str(path)!r} against {str(dataset)!r}: {exc}") from exc
                    rows.append({"method": label, "prediction_file": str(path), "sample": sample, **row})

    per_sample = pd.DataFrame(rows)
    summaries, families = [], []
    for label, group in per_sample.groupby("method", sort=False):
        summary = summarize_metrics(group, postprocessing="", input_dim=None)
        summary = summary.drop(columns=["postprocessing", "input_dim"])
        summary.insert(0, "method", label)
        summaries.append(summary)
        family = summarize_by_family(group)
        family.insert(0, "method", label)
        families.append(family)
    return per_sample, pd.concat(summaries, ignore_index=True), pd.concat(families, ignore_index=True)


def plot_comparison(samples: pd.DataFrame, outputfile: str, by_family: bool = False) -> None:
    """Adapt long-form evaluation rows to the shared F1 violin plot helper."""
    scores = {}
    column_groups = {}
    groups = samples.groupby("family", sort=True) if by_family else [(None, samples)]
    methods = samples["method"].unique()
    for family, group in groups:
        for method in methods:
            values = group.loc[group["method"] == method, "f1"].reset_index(drop=True)
            label = f"{family} / {method}" if by_family else method
            # The shared helper removes the final underscore-separated component.
            scores[f"{label}_F1"] = values
            column_groups[f"{label}_F1"] = family 
    try:
        violin_plot(pd.DataFrame(scores), "Family / run" if by_family else "Runs", outputfile=outputfile, groups=column_groups if by_family else None)
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=str, help="Reference HDF5 with true structures and metadata.")
    parser.add_argument("predictions", type=str, nargs="+", help="One or more prediction HDF5 files.")
    parser.add_argument("--labels", nargs="+", help="Unique plot labels, in prediction-file order (default: file stems).")
    parser.add_argument("-o", "--output", type=str, default="comparison", help="Output prefix for CSVs and PNGs.")
    args = parser.parse_args()
    
    labels = args.labels or [os.path.basename(path) for path in args.predictions]
    if len(labels) != len(args.predictions) or len(set(labels)) != len(labels):
        parser.error("Provide one unique label per prediction file; use --labels for duplicate file stems.")

    try:
        samples, summary, families = evaluate_files(args.dataset, args.predictions, labels)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    figure_dir = 'figures'
    results_dir = 'results'
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for suffix, frame in [("per_sequence_metrics", samples), ("summary_metrics", summary), ("family_metrics", families)]:
        frame.to_csv(f"{results_dir}/{args.output}_{suffix}.csv", index=False)
    plot_comparison(samples, f"{figure_dir}/{args.output}_violin.png")
    plot_comparison(samples, f"{figure_dir}/{args.output}_family_violin.png", by_family=True)
    print(summary.to_string(index=False))
    print(f"Saved evaluation tables and violin plots with prefix {args.output}")


if __name__ == "__main__":
    main()
