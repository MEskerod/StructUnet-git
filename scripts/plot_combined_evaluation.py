"""Compare per-sequence F1 scores from multiple evaluation runs."""

import argparse
from pathlib import Path

import pandas as pd

from utils.plots import violin_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the F1 distributions from multiple evaluate_dataset.py runs.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("CSV_FILE", "NAME"), required=True, help="Evaluation per-sequence metrics CSV and its label (repeat as needed).")
    parser.add_argument("-o", "--output", type=Path, default=Path("figures/evaluation_runs_violin.png"), help="Output image (default: figures/evaluation_runs_violin.png).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = {}

    for filename, name in args.run:
        if name in scores:
            raise ValueError(f"Run name must be unique: {name!r}")

        path = Path(filename)
        data = pd.read_csv(path)
        if "f1" not in data.columns:
            raise ValueError(f"{path} has no 'f1' column")

        f1 = pd.to_numeric(data["f1"], errors="coerce").dropna().reset_index(drop=True)
        if len(f1) < 2:
            raise ValueError(f"{path} contains fewer than two numeric F1 values")

        # violin_plot removes the final underscore-separated component from labels.
        scores[f"{name}_F1"] = f1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    violin_plot(pd.DataFrame(scores), "Runs", outputfile=str(args.output))
    print(f"Saved violin plot to {args.output}")


if __name__ == "__main__":
    main()
