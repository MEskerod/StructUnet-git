import argparse, os, h5py, torch, multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.model import post_process, prepare_postprocessing
from utils.evaluation import compute_sample_metrics_multiple, ensure_2d_tensor, extract_pairs, score_pair_sets


STANDARD_PAIRS = {"AU", "UA", "CG", "GC", "GU", "UG"}

def initialize_worker() -> None:
	"""Prevent each worker from creating its own pool of Torch threads."""
	torch.set_num_threads(1)

def post_process_mode(task: tuple) -> tuple:
	"""Post-process one raw prediction using one mode in a worker process."""
	prediction, sequence, mode = task
	result = post_process(torch.from_numpy(prediction), sequence, mode, device="cpu")
	return mode, result.numpy()

def definite_upper_triangle(sequence: str, matrix: torch.Tensor) -> torch.Tensor:
	"""Return upper-triangle values, excluding pairs involving an unknown base."""
	matrix = ensure_2d_tensor(matrix)
	indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
	definite = torch.tensor(["N" not in (sequence[i] + sequence[j]).upper() for i, j in zip(indices[0].tolist(), indices[1].tolist())], dtype=torch.bool)
	return matrix[indices[0], indices[1]][definite]


def extract_noncanonical_pairs(sequence: str, matrix: torch.Tensor, threshold: float) -> set:
	"""Extract predicted/true non-canonical pairs, excluding pairs involving N."""
	matrix = ensure_2d_tensor(matrix)
	indices = torch.nonzero(torch.triu(matrix > threshold, diagonal=1), as_tuple=False)
	pairs = set()
	for i_tensor, j_tensor in indices:
		i, j = int(i_tensor), int(j_tensor)
		pair = (sequence[i] + sequence[j]).upper()
		if "N" not in pair and pair not in STANDARD_PAIRS:
			pairs.add((i, j))
	return pairs


def count_definite_pairs(sequence: str, matrix: torch.Tensor, threshold: float) -> int:
	"""Count base pairs whose nucleotide identities are both known."""
	matrix = ensure_2d_tensor(matrix)
	indices = torch.nonzero(torch.triu(matrix > threshold, diagonal=1), as_tuple=False)
	return sum("N" not in (sequence[int(i)] + sequence[int(j)]).upper() for i, j in indices)


def add_noncanonical_metrics(row: dict, predictions: dict, data: h5py.Group, sequence: str, threshold: float) -> None:
	target_pairs = extract_noncanonical_pairs(sequence, data["output"][()], threshold)
	true_pair_count = count_definite_pairs(sequence, data["output"][()], threshold)
	row["true_noncanonical_pair_count"] = len(target_pairs)
	row["true_definite_pair_count"] = true_pair_count
	row["true_noncanonical_fraction"] = len(target_pairs) / true_pair_count if true_pair_count else float("nan")

	for mode, prediction in predictions.items():
		predicted_pairs = extract_noncanonical_pairs(sequence, prediction, threshold)
		predicted_pair_count = count_definite_pairs(sequence, prediction, threshold)
		tp, fp, fn, precision, recall, f1 = score_pair_sets(predicted_pairs, target_pairs)
		row[f"{mode}_noncanonical_pair_count"] = len(predicted_pairs)
		row[f"{mode}_definite_pair_count"] = predicted_pair_count
		row[f"{mode}_noncanonical_fraction"] = len(predicted_pairs) / predicted_pair_count if predicted_pair_count else float("nan")
		row[f"{mode}_noncanonical_tp"] = tp
		row[f"{mode}_noncanonical_fp"] = fp
		row[f"{mode}_noncanonical_fn"] = fn
		row[f"{mode}_noncanonical_precision"] = precision
		row[f"{mode}_noncanonical_recall"] = recall
		row[f"{mode}_noncanonical_f1"] = f1


def update_histograms(hist_pos: np.ndarray, hist_neg: np.ndarray, scores: torch.Tensor, labels: torch.Tensor) -> None:
	scores = np.clip(scores.detach().cpu().numpy(), 0.0, 1.0)
	labels = labels.detach().cpu().numpy().astype(bool)
	if labels.any():
		hist_pos += np.histogram(scores[labels], bins=len(hist_pos), range=(0.0, 1.0))[0]
	if (~labels).any():
		hist_neg += np.histogram(scores[~labels], bins=len(hist_neg), range=(0.0, 1.0))[0]


def histogram_curve(hist_pos: np.ndarray, hist_neg: np.ndarray) -> dict:
	total_pos, total_neg = hist_pos.sum(), hist_neg.sum()
	if total_pos == 0 or total_neg == 0:
		raise ValueError("ROC requires at least one positive and one negative example.")

	tp = np.cumsum(hist_pos[::-1])
	fp = np.cumsum(hist_neg[::-1])
	tpr = np.concatenate(([0.0], tp / total_pos))
	fpr = np.concatenate(([0.0], fp / total_neg))
	thresholds = np.concatenate(([1.0], np.linspace(1.0, 0.0, len(hist_pos), endpoint=False)))
	return {"threshold": thresholds, "fpr": fpr, "tpr": tpr, "auc": np.trapz(tpr, fpr), "positives": int(total_pos), "negatives": int(total_neg)}


def aggregate_scores(tp, fp, fn) -> tuple:
	precision = tp / (tp + fp) if tp + fp else float("nan")
	recall = tp / (tp + fn) if tp + fn else float("nan")
	if tp == 0 and fp + fn:
		f1 = 0.0
	elif pd.isna(precision) or pd.isna(recall) or precision + recall == 0:
		f1 = float("nan")
	else:
		f1 = 2 * precision * recall / (precision + recall)
	return precision, recall, f1


def summarize_modes(df: pd.DataFrame, modes: list) -> pd.DataFrame:
	rows = []
	for mode in modes:
		true_has_pk = df["true_has_pk"].astype(bool)
		pred_has_pk = df[f"{mode}_has_pk"].astype(bool)
		pk_presence_tp = int((true_has_pk & pred_has_pk).sum())
		pk_presence_fp = int((~true_has_pk & pred_has_pk).sum())
		pk_presence_fn = int((true_has_pk & ~pred_has_pk).sum())

		pk_tp = pd.to_numeric(df[f"{mode}_pk_pair_tp"], errors="coerce").sum()
		pk_fp = pd.to_numeric(df[f"{mode}_pk_pair_fp"], errors="coerce").sum()
		pk_fn = pd.to_numeric(df[f"{mode}_pk_pair_fn"], errors="coerce").sum()
		nc_tp = pd.to_numeric(df[f"{mode}_noncanonical_tp"], errors="coerce").sum()
		nc_fp = pd.to_numeric(df[f"{mode}_noncanonical_fp"], errors="coerce").sum()
		nc_fn = pd.to_numeric(df[f"{mode}_noncanonical_fn"], errors="coerce").sum()

		pk_precision, pk_recall, pk_f1 = aggregate_scores(pk_tp, pk_fp, pk_fn)
		presence_precision, presence_recall, presence_f1 = aggregate_scores(pk_presence_tp, pk_presence_fp, pk_presence_fn)
		nc_precision, nc_recall, nc_f1 = aggregate_scores(nc_tp, nc_fp, nc_fn)
		non_pk = ~true_has_pk

		rows.append({
			"mode": mode,
			"evaluated_sequences": len(df),
			"mean_precision": pd.to_numeric(df[f"{mode}_precision"], errors="coerce").mean(),
			"mean_recall": pd.to_numeric(df[f"{mode}_recall"], errors="coerce").mean(),
			"mean_f1": pd.to_numeric(df[f"{mode}_f1"], errors="coerce").mean(),
			"mean_f1_shift": pd.to_numeric(df[f"{mode}_f1_shift"], errors="coerce").mean(),
			"pk_presence_tp": pk_presence_tp,
			"pk_presence_fp": pk_presence_fp,
			"pk_presence_fn": pk_presence_fn,
			"pk_presence_precision": presence_precision,
			"pk_presence_recall": presence_recall,
			"pk_presence_f1": presence_f1,
			"pk_pair_tp": pk_tp,
			"pk_pair_fp": pk_fp,
			"pk_pair_fn": pk_fn,
			"pk_pair_precision": pk_precision,
			"pk_pair_recall": pk_recall,
			"pk_pair_f1": pk_f1,
			"false_pk_rate_non_pk_sequences": pred_has_pk[non_pk].mean() if non_pk.any() else float("nan"),
			"noncanonical_tp": nc_tp,
			"noncanonical_fp": nc_fp,
			"noncanonical_fn": nc_fn,
			"noncanonical_precision": nc_precision,
			"noncanonical_recall": nc_recall,
			"noncanonical_f1": nc_f1,
		})
	return pd.DataFrame(rows)





def save_roc(curves: dict, output_prefix: str, figure_dir: str = 'figures', results_dir: str = 'results') -> None:
	rows = []
	fig, ax = plt.subplots(figsize=(6, 6))
	for mode, curve in curves.items():
		rows.append(pd.DataFrame({"mode": mode, "threshold": curve["threshold"], "fpr": curve["fpr"], "tpr": curve["tpr"]}))
		ax.plot(curve["fpr"], curve["tpr"], linewidth=2, label=f"{mode} (AUC = {curve['auc']:.4f})")
	ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.legend(frameon=False, loc="lower right")
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	fig.tight_layout()
	fig.savefig(f"{figure_dir}/roc.png", dpi=300, bbox_inches="tight")
	plt.close(fig)
	pd.concat(rows, ignore_index=True).to_csv(f"{results_dir}/roc.csv", index=False)


def plot_true_vs_predicted_fractions(df: pd.DataFrame, modes: list, metric: str, label: str, output_file: str) -> None:
	true_column = f"true_{metric}_fraction"
	fig, ax = plt.subplots(figsize=(6, 6))
	has_data = False
	for mode in modes:
		predicted_column = f"{mode}_{metric}_fraction"
		valid = df[[true_column, predicted_column]].apply(pd.to_numeric, errors="coerce").dropna()
		if valid.empty:
			continue
		has_data = True
		ax.scatter(valid[true_column], valid[predicted_column], alpha=0.45, s=18, label=mode)
	if not has_data:
		plt.close(fig)
		return
	ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel(f"True {label} fraction")
	ax.set_ylabel(f"Predicted {label} fraction")
	ax.legend(frameon=False)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_mode_boxplot(df: pd.DataFrame, modes: list, suffix: str, ylabel: str, output_file: str, true_pk_only: bool = False) -> None:
	plot_df = df[df["true_has_pk"].astype(bool)] if true_pk_only else df
	data, labels = [], []
	for mode in modes:
		values = pd.to_numeric(plot_df[f"{mode}_{suffix}"], errors="coerce").dropna()
		if not values.empty:
			data.append(values.tolist())
			labels.append(mode)
	if not data:
		return
	fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 5))
	ax.boxplot(data, labels=labels, showfliers=False)
	ax.set_ylabel(ylabel)
	ax.tick_params(axis="x", rotation=25)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_false_pk_rate(summary_df: pd.DataFrame, output_file: str) -> None:
	fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(summary_df)), 5))
	ax.bar(summary_df["mode"], summary_df["false_pk_rate_non_pk_sequences"])
	ax.set_ylabel("False pseudoknot rate on non-pseudoknotted sequences")
	ax.tick_params(axis="x", rotation=25)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	fig.tight_layout()
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)


def parse_args():
	parser = argparse.ArgumentParser(description="Compare the post-processing modes for a given model and dataset.")
	parser.add_argument("predictions", type=str, help="HDF5 file containing unprocessed predictions.")
	parser.add_argument("dataset", type=str, help="HDF5 file containing the dataset.")
	parser.add_argument("-o", "--output", type=str, default="postprocessing_comparison", help="Output prefix for the comparison results. CSV files will be saved in the 'results' directory and PNG files in the 'figures' directory.")
	parser.add_argument("--postprocessing_modes", nargs="+", choices=["restricted", "unrestricted", "restricted_noncrossing", "unrestricted_noncrossing"], default=["restricted", "unrestricted", "restricted_noncrossing", "unrestricted_noncrossing"], help="Post-processing modes to compare.")
	parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used to interpret contact matrices as binary structures.")
	parser.add_argument("--num_thresholds", type=int, default=1001, help="Number of thresholds used for the ROC approximation.")
	parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores to use for parallel processing (Default: 1).")
	return parser.parse_args()


def main():
	args = parse_args()
	if args.num_thresholds < 2:
		raise ValueError("--num_thresholds must be at least 2")
	if args.cores < 1:
		raise ValueError("--cores must be at least 1")

	figure_dir = f'figures/{args.output}'
	results_dir = f'results/{args.output}'
	os.makedirs(figure_dir, exist_ok=True)
	os.makedirs(results_dir, exist_ok=True)

	rows = []
	histograms = {
		"restricted": [np.zeros(args.num_thresholds - 1, dtype=np.int64), np.zeros(args.num_thresholds - 1, dtype=np.int64)],
		"unrestricted": [np.zeros(args.num_thresholds - 1, dtype=np.int64), np.zeros(args.num_thresholds - 1, dtype=np.int64)]}

	worker_count = min(args.cores, len(args.postprocessing_modes))
	pool = multiprocessing.get_context("spawn").Pool(processes=worker_count, initializer=initialize_worker) if worker_count > 1 else None

	try:
		with h5py.File(args.dataset, "r") as h5_data, h5py.File(args.predictions, "a") as h5_predictions:
			for prediction in tqdm(list(h5_predictions.keys()), desc="Comparing post-processing modes", unit="predictions"):
				group = h5_predictions[prediction]
				data = h5_data[prediction]
				sequence = data["sequence"][()].decode("utf-8")
				raw_prediction = ensure_2d_tensor(group["predicted"][()])
				predictions = {}

				missing_modes = [mode for mode in args.postprocessing_modes if mode not in group]
				tasks = [(raw_prediction.numpy(), sequence, mode) for mode in missing_modes]
				if pool is None:
					processed = [post_process_mode(task) for task in tasks]
				else: 
					processed = pool.map(post_process_mode, tasks)

				for mode, postprocessed_prediction in processed: 
					group.create_dataset(mode, data=postprocessed_prediction, compression="gzip", compression_opts=4)
				for mode in args.postprocessing_modes:
					predictions[mode] = group[mode][()]

				target_scores = definite_upper_triangle(sequence, data["output"][()])
				labels = target_scores > args.threshold
				unrestricted_scores = definite_upper_triangle(sequence, (raw_prediction + raw_prediction.T) / 2)
				restricted_scores = definite_upper_triangle(sequence, prepare_postprocessing(raw_prediction, sequence, device="cpu"))
				update_histograms(histograms["restricted"][0], histograms["restricted"][1], restricted_scores, labels)
				update_histograms(histograms["unrestricted"][0], histograms["unrestricted"][1], unrestricted_scores, labels)

				row = compute_sample_metrics_multiple(predictions, data, threshold=args.threshold)
				add_noncanonical_metrics(row, predictions, data, sequence, args.threshold)
				true_pair_count = len(extract_pairs(data["output"][()], threshold=args.threshold))
				row["true_pair_count"] = true_pair_count
				row["true_pk_pair_fraction"] = row["true_pk_pair_count"] / true_pair_count if true_pair_count else float("nan")
				for mode, mode_prediction in predictions.items():
					predicted_pair_count = len(extract_pairs(mode_prediction, threshold=args.threshold))
					row[f"{mode}_pair_count"] = predicted_pair_count
					row[f"{mode}_pk_pair_fraction"] = row[f"{mode}_pk_pair_count"] / predicted_pair_count if predicted_pair_count else float("nan")
				rows.append(row)

	finally: 
		if pool is not None:
			pool.close()
			pool.join()

	per_sequence_df = pd.DataFrame(rows)
	per_sequence_df.to_csv(f"{results_dir}/per_sequence.csv", index=False)
	summary_df = summarize_modes(per_sequence_df, args.postprocessing_modes)

	curves = {mode: histogram_curve(*histogram) for mode, histogram in histograms.items()}
	for mode, curve in curves.items():
		summary_df.loc[summary_df["mode"] == mode, "roc_auc"] = curve["auc"]
		summary_df.loc[summary_df["mode"] == mode, "roc_positives"] = curve["positives"]
		summary_df.loc[summary_df["mode"] == mode, "roc_negatives"] = curve["negatives"]

	summary_df.to_csv(f"{results_dir}/summary.csv", index=False)
	save_roc(curves, args.output, figure_dir=figure_dir, results_dir=results_dir)
	plot_true_vs_predicted_fractions(per_sequence_df, args.postprocessing_modes, "noncanonical", "non-canonical-pair", f"{figure_dir}/noncanonical_fraction.png")
	plot_true_vs_predicted_fractions(per_sequence_df, args.postprocessing_modes, "pk_pair", "pseudoknot-pair", f"{figure_dir}/pseudoknot_pair_fraction.png")
	plot_mode_boxplot(per_sequence_df, args.postprocessing_modes, "f1", "F1 score", f"{figure_dir}/f1_comparison.png")
	plot_mode_boxplot(per_sequence_df, args.postprocessing_modes, "noncanonical_f1", "Non-canonical-pair F1", f"{figure_dir}/noncanonical_f1_comparison.png")
	plot_mode_boxplot(per_sequence_df, args.postprocessing_modes, "pk_pair_recall", "Pseudoknot-pair recall", f"{figure_dir}/pk_pair_recall_comparison.png", true_pk_only=True)
	plot_mode_boxplot(per_sequence_df, args.postprocessing_modes, "pk_pair_f1", "Pseudoknot-pair F1", f"{figure_dir}/pk_pair_f1_comparison.png", true_pk_only=True)
	plot_false_pk_rate(summary_df, f"{figure_dir}/false_pk_rate.png")
	print(summary_df.to_string(index=False))


if __name__ == "__main__":
	main()
