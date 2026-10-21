"""Run the analyses and figures previously spread across the project notebooks."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import h5py as h5

from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")
import colorcet as cc



ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}
NUCLEOTIDES = set("AUGC")
COLORS = mpl.colormaps.get("cet_glasbey_dark", mpl.colormaps["tab20"]).colors
METHOD_COLORS = {"Random": COLORS[11],
                 "ViennaRNA": COLORS[5],
                 "CONTRAfold": COLORS[9],
                 "CNNfold": COLORS[0],
                 "UFold": COLORS[6],
                 "StructUnet8": COLORS[1],
                 "StructUnet": COLORS[4]}
FAMILY_COLORS = {"none": COLORS[0],
                 "16S_rRNA": COLORS[4],
                 "5S_rRNA": COLORS[3],
                 "23S_rRNA": COLORS[5],
                 "RNaseP": COLORS[6],
                 "SRP": COLORS[9],
                 "tRNA": COLORS[8],
                 "tmRNA": COLORS[7],
                 "group_I_intron": COLORS[0],
                 "group_II_intron": COLORS[1],
                 "Telomerase": COLORS[2]}


def path(relative_path: str) -> Path:
    return ROOT / relative_path


def output_path(relative_path: str) -> Path:
    destination = path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def available(relative_path: str, purpose: str) -> bool:
    if path(relative_path).is_file():
        return True
    print(f"Skipping {purpose}: missing {relative_path}")
    return False


def save_figure(fig, relative_path: str) -> None:
    destination = output_path(relative_path)
    fig.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {destination.relative_to(ROOT)}")


def read_sequence(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) == 0:
        value = value.item()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return "".join(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value)


def noncanonical_counts(structure: np.ndarray, sequence: str) -> tuple[int, Counter]:
    pair_i, pair_j = np.nonzero(np.triu(structure > 0.5, k=1))
    noncanonical = Counter()
    for i, j in zip(pair_i, pair_j):
        left, right = sequence[i], sequence[j]
        if left in NUCLEOTIDES and right in NUCLEOTIDES:
            pair = left + right
            if pair not in CANONICAL_PAIRS:
                noncanonical[pair] += 1
    return len(pair_i), noncanonical


def analyze_prediction_noncanonical_pairs() -> None:
    print("\n=== Predicted non-canonical pairs ===")
    data_path = "data/converted/RNAStrAlign8_dataA_test.h5"
    prediction_path = "results/StructUnet8_dataA_evaluation_predictions.h5"
    if not available(data_path, "prediction pair analysis") or not available(prediction_path, "prediction pair analysis"):
        return

    true_counts = Counter()
    predicted_counts = Counter()
    sequences_with_true = set()
    sequences_with_predicted = set()
    matched = 0
    with h5.File(path(data_path), "r") as data, h5.File(path(prediction_path), "r") as predictions:
        keys = sorted(set(data.keys()) & set(predictions.keys()))
        for index, key in enumerate(keys, start=1):
            sequence = read_sequence(data[key]["sequence"])
            _, true_pairs = noncanonical_counts(data[key]["output"][:], sequence)
            _, predicted_pairs = noncanonical_counts(predictions[key]["predicted"][:], sequence)
            true_counts.update(true_pairs)
            predicted_counts.update(predicted_pairs)
            if true_pairs:
                sequences_with_true.add(key)
            if predicted_pairs:
                sequences_with_predicted.add(key)
            matched += 1
            if matched % 100 == 0:
                print(f"Processed {matched}/{len(keys)} sequences")

    print(f"Sequences compared: {matched}")
    print(f"Sequences with true non-canonical pairs: {len(sequences_with_true)}")
    print(f"Sequences with predicted non-canonical pairs: {len(sequences_with_predicted)}")
    print(f"True non-canonical pair counts by type: {dict(sorted(true_counts.items()))}")
    print(f"Predicted non-canonical pair counts by type: {dict(sorted(predicted_counts.items()))}")
    print(f"Total true non-canonical pairs: {sum(true_counts.values())}")
    print(f"Total predicted non-canonical pairs: {sum(predicted_counts.values())}")


def analyze_dataset_noncanonical_pairs() -> None:
    print("\n=== RNAStrAlign data-set base pairs ===")
    datasets = {
        "training": "data/converted/RNAStrAlign8_dataA_train.h5",
        "test": "data/converted/RNAStrAlign8_dataA_test.h5",
        "validation": "data/converted/RNAStrAlign8_dataA_val.h5",
    }
    totals = Counter()
    for split, relative_path in datasets.items():
        if not available(relative_path, f"{split} base-pair analysis"):
            continue
        base_pairs = 0
        noncanonical_pairs = 0
        sequences_with_noncanonical = 0
        sequences = 0
        with h5.File(path(relative_path), "r") as data:
            for key in data.keys():
                sequence = read_sequence(data[key]["sequence"])
                pair_count, noncanonical = noncanonical_counts(data[key]["output"][:], sequence)
                base_pairs += pair_count
                noncanonical_pairs += sum(noncanonical.values())
                sequences_with_noncanonical += bool(noncanonical)
                sequences += 1
        totals.update(base_pairs=base_pairs, noncanonical_pairs=noncanonical_pairs, sequences=sequences, sequences_with_noncanonical=sequences_with_noncanonical)
        print(f"{split}: {sequences} sequences, {base_pairs} base pairs, {noncanonical_pairs} non-canonical pairs, {sequences_with_noncanonical} sequences containing non-canonical pairs")
    if totals["base_pairs"]:
        fraction = totals["noncanonical_pairs"] / totals["base_pairs"]
        print(f"Total base pairs: {totals['base_pairs']}")
        print(f"Total non-canonical base pairs: {totals['noncanonical_pairs']}")
        print(f"Fraction of base pairs that are non-canonical: {fraction:.6f}")
        print(f"Total sequences: {totals['sequences']}")
        print(f"Sequences with non-canonical base pairs: {totals['sequences_with_noncanonical']}")


def analyze_pseudoknots() -> pd.DataFrame | None:
    print("\n=== RNAStrAlign pseudoknots ===")
    relative_path = "data/RNAStrAlign.tsv"
    if not available(relative_path, "pseudoknot analysis"):
        return None
    data = pd.read_csv(path(relative_path), sep="\t")
    structures = data["secondary_structure"].fillna("").astype(str)
    data["has_pseudoknot"] = structures.str.contains(r"\[", regex=True)
    data["pseudoknotted_basepairs"] = structures.map(lambda value: value.count("[") + value.count("{") + value.count("<"))
    data["basepairs"] = structures.map(lambda value: value.count("(") + value.count("[") + value.count("{") + value.count("<"))
    total_pairs = int(data["basepairs"].sum())
    pseudoknotted_pairs = int(data["pseudoknotted_basepairs"].sum())
    print(f"Sequences containing '[' pseudoknot notation: {int(data['has_pseudoknot'].sum())}")
    print(f"Sequences in RNAStrAlign: {len(data)}")
    print(f"Total base pairs: {total_pairs}")
    print(f"Pseudoknotted base pairs: {pseudoknotted_pairs}")
    print(f"Fraction of base pairs that are pseudoknotted: {pseudoknotted_pairs / total_pairs if total_pairs else float('nan'):.6f}")
    return data


def analyze_postprocessing_metrics() -> pd.DataFrame | None:
    print("\n=== Post-processing non-canonical metrics ===")
    relative_path = "results/StructUnet8_dataA_postprocessing_comparison/per_sequence.csv"
    if not available(relative_path, "post-processing metrics"):
        return None
    data = pd.read_csv(path(relative_path))
    required = {
        "unrestricted_noncanonical_fraction",
        "family",
        "unrestricted_noncanonical_fp",
        "unrestricted_noncanonical_tp",
        "unrestricted_noncanonical_recall",
    }
    if not required.issubset(data.columns):
        print(f"Skipping post-processing metrics: missing columns {sorted(required - set(data.columns))}")
        return None
    noncanonical = data[data["unrestricted_noncanonical_fraction"] > 0]
    print("Families among sequences with predicted non-canonical pairs:")
    print(noncanonical["family"].value_counts().to_string())
    print(f"False-positive non-canonical pairs: {noncanonical['unrestricted_noncanonical_fp'].sum():.0f}")
    print(f"True-positive non-canonical pairs: {noncanonical['unrestricted_noncanonical_tp'].sum():.0f}")
    print(f"Mean non-canonical recall: {noncanonical['unrestricted_noncanonical_recall'].mean():.6f}")
    return data


def print_prediction_examples() -> None:
    print("\n=== Prediction examples (every 100th sequence) ===")
    data_path = "data/converted/RNAStrAlign8_dataA_test.h5"
    prediction_path = "results/StructUnet8_dataA_evaluation_predictions.h5"
    if not available(data_path, "prediction examples") or not available(prediction_path, "prediction examples"):
        return

    def dot_bracket(pairs: list[tuple[int, int]], length: int) -> str:
        bracket_types = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("A", "a"), ("B", "b")]
        layers: list[list[tuple[int, int]]] = []
        for pair in sorted(pairs):
            start, end = pair
            for layer in layers:
                if not any((i < start < j < end) or (start < i < end < j) for i, j in layer):
                    layer.append(pair)
                    break
            else:
                layers.append([pair])
        if len(layers) > len(bracket_types):
            raise ValueError("Structure exceeds the available dot-bracket alphabets")
        structure = ["."] * length
        for layer, (opening, closing) in zip(layers, bracket_types):
            for start, end in layer:
                if start < 0 or end >= length:
                    raise ValueError(f"Pair ({start}, {end}) is out of bounds for length {length}")
                structure[start] = opening
                structure[end] = closing
        return "".join(structure)

    with h5.File(path(data_path), "r") as data, h5.File(path(prediction_path), "r") as predictions:
        keys = sorted(set(data.keys()) & set(predictions.keys()))
        for index, key in enumerate(keys):
            if index % 100:
                continue
            sequence = read_sequence(data[key]["sequence"])
            true_matrix = data[key]["output"][:]
            predicted_matrix = predictions[key]["predicted"][:]
            true_pairs = list(zip(*np.nonzero(np.triu(true_matrix > 0.5, k=1))))
            predicted_pairs = list(zip(*np.nonzero(np.triu(predicted_matrix > 0.5, k=1))))
            print(f"Processed key: {key}")
            print(f">{key}_prediction\n{sequence}\n{dot_bracket(predicted_pairs, len(sequence))}")
            print(f">{key}_data\n{sequence}\n{dot_bracket(true_pairs, len(sequence))}")


def plot_pseudoknot_fractions(data: pd.DataFrame | None) -> None:
    if data is None:
        return
    fraction = data["has_pseudoknot"].groupby(data["family"]).mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(fraction.index, fraction.values)
    ax.set_xlabel("Family")
    ax.set_ylabel("Fraction of sequences with pseudoknots")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    save_figure(fig, "figures/paper/supp/fraction_by_family.svg")
    print("Fraction with pseudoknots by family:")
    print(fraction.to_string(float_format=lambda value: f"{value:.4f}"))


def plot_postprocessing(post: pd.DataFrame | None) -> None:
    if post is None:
        return
    metrics = [
        ("unrestricted_f1", "Unrestricted\nwith pseudoknots"),
        ("unrestricted_noncrossing_f1", "Unrestricted\nwithout pseudoknots"),
        ("restricted_f1", "Restricted\nwith pseudoknots"),
        ("restricted_noncrossing_f1", "Restricted\nwithout pseudoknots"),
    ]
    columns = [column for column, _ in metrics if column in post]
    if len(columns) != len(metrics):
        print("Skipping post-processing plots: one or more F1 columns are missing")
        return

    values = [post[column].dropna().to_numpy() for column in columns]
    labels = [label for _, label in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    violins = ax.violinplot(values, positions=np.arange(1, 5), vert=False, showextrema=False)
    for body, color in zip(violins["bodies"], [COLORS[8], COLORS[6], COLORS[3], COLORS[4]]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    for position, values_for_metric in enumerate(values, start=1):
        if not values_for_metric.size:
            continue
        q1, median, q3 = np.percentile(values_for_metric, [25, 50, 75])
        iqr = q3 - q1
        inliers = values_for_metric[(values_for_metric >= q1 - 1.5 * iqr) & (values_for_metric <= q3 + 1.5 * iqr)]
        ax.hlines(position, q1, q3, color="black", linewidth=5, zorder=3)
        ax.hlines(position, inliers.min(), inliers.max(), color="black", linewidth=1, zorder=3)
        ax.scatter(median, position, marker="|", color="white", s=50, zorder=4)
    ax.set_yticks(range(1, 5), labels)
    ax.set_xlabel("F1 score")
    ax.set_xlim(0.9, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "figures/paper/fig3/violin_plot.svg")

    scatter_specs = [
        ("true_noncanonical_pair_count", "unrestricted_noncanonical_pair_count", "True non-canonical pairs", "Predicted non-canonical pairs", "predicted_vs_true_noncanonical_pairs.svg", True),
        ("true_pk_pair_fraction", "unrestricted_pk_pair_fraction", "True pseudoknot pair fraction", "Predicted pseudoknot pair fraction", "predicted_vs_true_pseudoknot_pairs.svg", True),
    ]
    for x_column, y_column, xlabel, ylabel, filename, identity in scatter_specs:
        if x_column not in post or y_column not in post:
            print(f"Skipping {filename}: required metric columns are missing")
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(post[x_column], post[y_column], alpha=0.5, color=COLORS[4], s=16)
        if identity:
            ax.axline((0, 0), slope=1, color="black")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_figure(fig, f"figures/paper/fig3/{filename}")

    if {"true_noncanonical_fraction", "unrestricted_f1", "restricted_f1"}.issubset(post.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(post["true_noncanonical_fraction"], post["unrestricted_f1"] - post["restricted_f1"], alpha=0.5, color=COLORS[4], s=16)
        ax.set_xlabel("True non-canonical fraction", fontsize=14)
        ax.set_ylabel("Delta F1 (unrestricted - restricted)", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_figure(fig, "figures/paper/fig3/delta_f1_vs_true_noncanonical_fraction.svg")

    for column in ("unrestricted_f1", "restricted_f1"):
        if column in post:
            print(f"{column} mean: {post[column].mean():.6f}")


def plot_method_comparison() -> None:
    relative_path = "results/RNAStrAlign_method_comparison_per_sequence_metrics.csv"
    if not available(relative_path, "method comparison plots"):
        return
    data = pd.read_csv(path(relative_path))
    if not {"method", "f1"}.issubset(data.columns):
        print("Skipping method comparison plots: method or f1 column is missing")
        return
    method_order = data["method"].dropna().unique().tolist()
    method_values = [data.loc[data["method"] == method, "f1"].dropna().to_numpy() for method in method_order]
    valid = [(method, values) for method, values in zip(method_order, method_values) if values.size]
    if not valid:
        print("Skipping method comparison violin: no numeric F1 values")
        return
    method_order, method_values = zip(*valid)
    fig, ax = plt.subplots(figsize=(7, max(4, len(method_order) * 0.8)))
    violins = ax.violinplot(method_values, positions=np.arange(1, len(method_order) + 1), vert=False, widths=0.8, showextrema=False)
    for body, method in zip(violins["bodies"], method_order):
        body.set_facecolor(METHOD_COLORS.get(method, COLORS[0]))
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for position, method_values_for_group in enumerate(method_values, start=1):
        q1, median, q3 = np.percentile(method_values_for_group, [25, 50, 75])
        iqr = q3 - q1
        inliers = method_values_for_group[(method_values_for_group >= q1 - 1.5 * iqr) & (method_values_for_group <= q3 + 1.5 * iqr)]
        ax.hlines(position, q1, q3, color="black", linewidth=5, zorder=3)
        ax.hlines(position, inliers.min(), inliers.max(), color="black", linewidth=1.2, zorder=3)
        ax.scatter(median, position, marker="|", color="white", s=50, zorder=4)
    ax.set_yticks(np.arange(1, len(method_order) + 1), method_order, fontsize=12)
    ax.set_xlabel("F1 score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "figures/paper/supp/method_comparison.svg")
    print("Mean F1 by method:")
    print(data.groupby("method")["f1"].agg(["count", "mean", "median"]).to_string())

    if {"pk_pair_f1", "true_has_pk", "pred_has_pk"}.issubset(data.columns):
        methods = [method for method in ("StructUnet", "CNNfold") if method in set(data["method"])]
        if methods:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 6))
            pk_values = [data.loc[data["method"] == method, "pk_pair_f1"].dropna() for method in methods]
            boxes1 = ax1.boxplot(pk_values, labels=methods, patch_artist=True, showfliers=False, widths=0.5)
            for box, method in zip(boxes1["boxes"], methods):
                box.set_facecolor(METHOD_COLORS.get(method, COLORS[0]))
            ax1.set_ylabel("Pseudoknot pair F1 score", fontsize=14)
            ax1.set_ylim(0, 1)
            pseudoknot_data = data[data["true_has_pk"]]
            f1_values = [pseudoknot_data.loc[pseudoknot_data["method"] == method, "f1"].dropna() for method in methods]
            boxes2 = ax2.boxplot(f1_values, labels=methods, patch_artist=True, showfliers=False, widths=0.5)
            for box, method in zip(boxes2["boxes"], methods):
                box.set_facecolor(METHOD_COLORS.get(method, COLORS[0]))
            ax2.set_ylabel("F1 score for sequences with pseudoknots", fontsize=14)
            ax2.set_ylim(0.8, 1)
            ax2.set_yticks(np.arange(0.8, 1.01, 0.05))
            for axis in (ax1, ax2):
                axis.spines[["top", "right"]].set_visible(False)
                axis.tick_params(axis="x", labelsize=12)
                axis.tick_params(axis="y", labelsize=12)
            fig.tight_layout()
            save_figure(fig, "figures/paper/fig3/box_plot.svg")

        agreement = data.groupby("method").apply(lambda group: (group["true_has_pk"] == group["pred_has_pk"]).mean())
        print("Pseudoknot presence agreement by method:")
        print(agreement.to_string(float_format=lambda value: f"{value:.6f}"))


def normalize_family(value: str) -> str:
    lookup = {family.lower(): family for family in FAMILY_COLORS}
    return lookup.get(value.lower(), value)


def plot_dataset_information() -> None:
    rna_path = "data/RNAStrAlign.tsv"
    archive_path = "data/ArchiveII.tsv"
    if not available(rna_path, "dataset plots") or not available(archive_path, "dataset plots"):
        return
    rna = pd.read_csv(path(rna_path), sep="\t")
    archive = pd.read_csv(path(archive_path), sep="\t")
    rna["length"] = rna["sequence"].astype(str).str.len()
    archive["length"] = archive["sequence"].astype(str).str.len()

    print("ArchiveII sequence counts by family:")
    print(archive.groupby("family").size().to_string())

    bin_width = 50
    maximum = int(max(rna["length"].max(), archive["length"].max()))
    bins = np.arange(0, maximum + bin_width, bin_width)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    ax1.hist(rna["length"], weights=rna["length"] / len(rna), bins=bins, alpha=0.5, edgecolor="black", color=COLORS[3], label="RNAStrAlign")
    ax1.hist(archive["length"], weights=archive["length"] / len(archive), bins=bins, alpha=0.5, edgecolor="black", color=COLORS[4], label="ArchiveII")
    ax1.set_xlabel("Sequence length", size=16)
    ax1.set_ylabel("Length-weighted count")
    ax1.legend(frameon=False)

    ranges = archive.groupby("family")["length"].agg(["min", "max"]).sort_values("min")
    lane_ends: list[list[float]] = []
    for family, row in ranges.iterrows():
        lane = next((index for index, ends in enumerate(lane_ends) if row["min"] > ends[-1] + 50), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append([])
        lane_ends[lane].append(row["max"])
        color = FAMILY_COLORS.get(normalize_family(str(family)), COLORS[10])
        ax2.plot([row["min"], row["max"]], [lane, lane], color=color, linewidth=4)
        ax2.scatter([row["min"], row["max"]], [lane, lane], color=color, marker="^", s=45)
        ax2.text(row["max"], lane, f" {family}", va="center", fontsize=9)
    ax2.set_xlabel("Sequence length")
    ax2.set_yticks([])
    for axis in (ax1, ax2):
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "figures/paper/fig2/fig2_2.svg")

    rna_families = rna["family"].astype(str).map(normalize_family)
    archive_families = archive["family"].astype(str).map(normalize_family)
    ordered_families = list(dict.fromkeys([*FAMILY_COLORS, *sorted(set(rna_families) | set(archive_families))]))
    rna_counts = rna_families.value_counts().reindex(ordered_families, fill_value=0)
    archive_counts = archive_families.value_counts().reindex(ordered_families, fill_value=0)
    display_names = [family.replace("_", " ") for family in ordered_families]
    colors = [FAMILY_COLORS.get(family, COLORS[10]) for family in ordered_families]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, counts, title in ((ax1, rna_counts, "RNAStrAlign"), (ax2, archive_counts, "ArchiveII")):
        axis.bar(range(len(counts)), counts.values, color=colors, edgecolor="black")
        axis.set_title(title, fontsize=16)
        axis.set_xticks(range(len(display_names)), display_names, rotation=45, ha="right", fontsize=10)
        axis.set_xlabel("")
        axis.spines[["top", "right"]].set_visible(False)
    ax1.set_ylabel("Number of sequences", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "figures/paper/fig2/fig2_1.svg")
    print("Dataset sizes:")
    print(f"RNAStrAlign: {len(rna)} sequences")
    print(f"ArchiveII: {len(archive)} sequences")


def plot_performance_by_length(post_rna: pd.DataFrame | None, post_archive: pd.DataFrame | None) -> None:
    if post_rna is None or post_archive is None:
        return
    required = {"family", "length", "unrestricted_f1"}
    if not required.issubset(post_rna.columns) or not required.issubset(post_archive.columns):
        print("Skipping performance-by-length plot: required columns are missing")
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharey=True)
    for axis, data, title in zip(axes, (post_rna, post_archive), ("RNAStrAlign", "ArchiveII")):
        for family, group in data.groupby("family"):
            axis.scatter(group["length"], group["unrestricted_f1"], s=12, alpha=0.5, color=FAMILY_COLORS.get(normalize_family(str(family)), COLORS[10]), label=family)
        axis.set_xscale("log")
        axis.set_ylim(0, 1.01)
        axis.set_ylabel(f"{title}\nF1 score", fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Length", fontsize=14)
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="center right", frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    save_figure(fig, "figures/paper/fig4/performance_plot.svg")


def plot_input_dimension_comparison() -> None:
    files = (
        "results/StructUnet8_dataA_evaluation_per_sequence_metrics.csv",
        "results/StructUnet16_dataA_evaluation_per_sequence_metrics.csv")
    if not all(available(file, "input-dimension comparison") for file in files):
        return
    data8, data16 = (pd.read_csv(path(file)) for file in files)
    if "f1" not in data8 or "f1" not in data16:
        print("Skipping input-dimension comparison: f1 column is missing")
        return
    values = [data8["f1"].dropna().to_numpy(), data16["f1"].dropna().to_numpy()]
    fig, ax = plt.subplots(figsize=(6, 3))
    violins = ax.violinplot(values, positions=[1, 2], vert=False, widths=0.8, showextrema=False)
    for body, color in zip(violins["bodies"], [COLORS[3], COLORS[4]]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for position, group in enumerate(values, start=1):
        q1, median, q3 = np.percentile(group, [25, 50, 75])
        iqr = q3 - q1
        inliers = group[(group >= q1 - 1.5 * iqr) & (group <= q3 + 1.5 * iqr)]
        ax.hlines(position, q1, q3, color="black", linewidth=5, zorder=3)
        ax.hlines(position, inliers.min(), inliers.max(), color="black", linewidth=1.2, zorder=3)
        ax.scatter(median, position, marker="|", color="white", s=50, zorder=4)
    ax.set_yticks([1, 2], ["StructUnet8", "StructUnet16"], fontsize=12)
    ax.set_xlabel("F1 score", fontsize=12)
    ax.set_xlim(0.7, 1.02)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "figures/paper/supp/structunet_8_16_comparison.svg")
    print("Per-sequence F1 summary by input dimension:")
    print(pd.DataFrame({"StructUnet8": pd.Series(values[0]), "StructUnet16": pd.Series(values[1])}).describe().to_string())


def run(args: argparse.Namespace) -> None:
    pseudoknot_data = analyze_pseudoknots()
    post_rna = analyze_postprocessing_metrics()
    post_archive_path = "results/StructUnet8_dataA_ArchiveII_postprocessing_comparison/per_sequence.csv"
    post_archive = pd.read_csv(path(post_archive_path)) if available(post_archive_path, "ArchiveII performance plot") else None

    plot_pseudoknot_fractions(pseudoknot_data)
    plot_postprocessing(post_rna)
    plot_method_comparison()
    plot_dataset_information()
    plot_performance_by_length(post_rna, post_archive)
    plot_input_dimension_comparison()

    if not args.skip_hdf5_scan:
        analyze_prediction_noncanonical_pairs()
        analyze_dataset_noncanonical_pairs()
    if not args.skip_examples:
        print_prediction_examples()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hdf5-scan", action="store_true", help="Skip the full HDF5 base-pair scans, which can take a while.")
    parser.add_argument("--skip-examples", action="store_true", help="Do not print one prediction/reference structure for every 100 sequences.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()