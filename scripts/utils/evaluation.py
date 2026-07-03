import os, torch

import numpy as np
import pandas as pd
import torch.nn.functional as F


def ensure_2d_tensor(matrix) -> torch.Tensor:
    """
    Convert a loaded matrix to a 2D CPU float tensor.
    """
    if isinstance(matrix, torch.Tensor):
        tensor = matrix.detach().cpu().float()
    else:
        tensor = torch.as_tensor(matrix, dtype=torch.float32)

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.squeeze(0)

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(tensor.shape)}")

    return tensor


def has_pk(pairings) -> bool:
    """
    Check whether a pairing vector contains a pseudoknot.
    """
    pairings = torch.as_tensor(pairings)
    for idx in range(len(pairings)):
        i, j = idx, int(pairings[idx])
        start, end = min(i, j), max(i, j)
        if i == j:
            continue
        if torch.max(pairings[start:end]) > end or torch.min(pairings[start:end]) < start:
            return True
    return False


def extract_pairs(matrix: torch.Tensor, threshold: float = 0.5) -> list:
    """
    Extract paired positions from the strict upper triangle of a contact matrix.
    """
    matrix = ensure_2d_tensor(matrix)
    indices = torch.nonzero(torch.triu(matrix > threshold, diagonal=1), as_tuple=False)
    return [(int(i.item()), int(j.item())) for i, j in indices]


def find_pseudoknot_pairs(pairs: list) -> set:
    """
    Return the subset of pairs that participate in at least one crossing.
    """
    pk_pairs = set()
    sorted_pairs = sorted((min(i, j), max(i, j)) for i, j in pairs)

    for idx, (i, j) in enumerate(sorted_pairs):
        for k, l in sorted_pairs[idx + 1 :]:
            crosses = (i < k < j < l) or (k < i < l < j)
            if crosses:
                pk_pairs.add((i, j))
                pk_pairs.add((k, l))

    return pk_pairs


def score_pair_sets(predicted_pairs: set, true_pairs: set) -> tuple:
    """
    Compute TP, FP, FN, precision, recall, and F1 on two pair sets.
    """
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")

    if tp == 0 and (fp > 0 or fn > 0):
        f1 = 0.0
    elif tp + fp == 0 and tp + fn == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, f1


def evaluate(y_pred: torch.Tensor, y_true: torch.Tensor, device: str, epsilon: float = 1e-10, allow_shift: bool = False, include_unpaired: bool = False) -> tuple:
    """
    Evaluate a predicted contact map against the true contact map.
    """
    assert y_pred.shape == y_true.shape

    if include_unpaired:
        mask = torch.ones_like(y_true, device=device)
    else:
        mask = torch.ones_like(y_true, device=device) - torch.eye(y_true.shape[-1], device=device)

    y_pred = y_pred * mask
    y_true = y_true * mask

    if allow_shift:
        kernel = torch.tensor(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            device=device,
            dtype=torch.float32,
        )
        y_pred_filtered = F.conv2d(
            y_pred.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze(0).squeeze(0)
        y_pred_filtered = (y_pred_filtered > 0).float()
        fn = (y_true * (1 - y_pred_filtered)).sum()
    else:
        fn = (y_true * (1 - y_pred)).sum()

    pred_p = y_pred.sum()
    true_p = y_true.sum()

    tp = true_p - fn
    fp = pred_p - tp

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision.item(), recall.item(), f1.item()


def calculate_weighted_f1(lengths: list, f1s: list) -> float:
    """
    Calculate the length-weighted F1 score while ignoring missing values.
    """
    valid = [
        (length, f1)
        for length, f1 in zip(lengths, f1s)
        if pd.notna(length) and pd.notna(f1)
    ]
    total_length = sum(length for length, _ in valid)
    if total_length == 0:
        return float("nan")
    return sum(length * f1 for length, f1 in valid) / total_length


def compute_sample_metrics(predicted: torch.Tensor, data, device: str = "cpu", threshold: float = 0.5) -> dict:
    """
    Compute per-sequence standard and pseudoknot metrics.
    """
    predicted = ensure_2d_tensor(predicted)
    target = ensure_2d_tensor(data.output)

    precision, recall, f1 = evaluate(predicted, target, device)
    precision_shift, recall_shift, f1_shift = evaluate(predicted, target, device, allow_shift=True)

    true_pairs = extract_pairs(target, threshold=threshold)
    pred_pairs = extract_pairs(predicted, threshold=threshold)

    true_pk_pairs = find_pseudoknot_pairs(true_pairs)
    pred_pk_pairs = find_pseudoknot_pairs(pred_pairs)

    pk_tp, pk_fp, pk_fn, pk_precision, pk_recall, pk_f1 = score_pair_sets(pred_pk_pairs, true_pk_pairs)

    true_has_pk = bool(true_pk_pairs)
    pred_has_pk = bool(pred_pk_pairs)

    return {
        "name": os.path.basename(data.name),
        "family": data.family,
        "length": data.length,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_shift": precision_shift,
        "recall_shift": recall_shift,
        "f1_shift": f1_shift,
        "true_has_pk": true_has_pk,
        "pred_has_pk": pred_has_pk,
        "true_pk_pair_count": len(true_pk_pairs),
        "pred_pk_pair_count": len(pred_pk_pairs),
        "pk_pair_tp": pk_tp,
        "pk_pair_fp": pk_fp,
        "pk_pair_fn": pk_fn,
        "pk_pair_precision": pk_precision,
        "pk_pair_recall": pk_recall,
        "pk_pair_f1": pk_f1,
    }


def summarize_metrics(df: pd.DataFrame, postprocessing: str, input_dim: int, hidden_dim: int) -> pd.DataFrame:
    """
    Produce a one-row summary dataframe for the full evaluation run.
    """
    pk_sequence_tp = int(((df["pred_has_pk"]) & (df["true_has_pk"])).sum())
    pk_sequence_fp = int(((df["pred_has_pk"]) & (~df["true_has_pk"])).sum())
    pk_sequence_fn = int(((~df["pred_has_pk"]) & (df["true_has_pk"])).sum())

    pk_sequence_precision = pk_sequence_tp / (pk_sequence_tp + pk_sequence_fp) if (pk_sequence_tp + pk_sequence_fp) else float("nan")
    pk_sequence_recall = pk_sequence_tp / (pk_sequence_tp + pk_sequence_fn) if (pk_sequence_tp + pk_sequence_fn) else float("nan")
    if pk_sequence_tp == 0 and (pk_sequence_fp > 0 or pk_sequence_fn > 0):
        pk_sequence_f1 = 0.0
    elif pd.isna(pk_sequence_precision) or pd.isna(pk_sequence_recall) or (pk_sequence_precision + pk_sequence_recall) == 0:
        pk_sequence_f1 = float("nan")
    else:
        pk_sequence_f1 = 2 * pk_sequence_precision * pk_sequence_recall / (pk_sequence_precision + pk_sequence_recall)

    total_pk_tp = pd.to_numeric(df["pk_pair_tp"], errors="coerce").fillna(0).sum()
    total_pk_fp = pd.to_numeric(df["pk_pair_fp"], errors="coerce").fillna(0).sum()
    total_pk_fn = pd.to_numeric(df["pk_pair_fn"], errors="coerce").fillna(0).sum()

    pk_pair_precision = total_pk_tp / (total_pk_tp + total_pk_fp) if (total_pk_tp + total_pk_fp) else float("nan")
    pk_pair_recall = total_pk_tp / (total_pk_tp + total_pk_fn) if (total_pk_tp + total_pk_fn) else float("nan")
    if total_pk_tp == 0 and (total_pk_fp > 0 or total_pk_fn > 0):
        pk_pair_f1 = 0.0
    elif pd.isna(pk_pair_precision) or pd.isna(pk_pair_recall) or (pk_pair_precision + pk_pair_recall) == 0:
        pk_pair_f1 = float("nan")
    else:
        pk_pair_f1 = 2 * pk_pair_precision * pk_pair_recall / (pk_pair_precision + pk_pair_recall)

    summary = {
        "n_sequences": len(df),
        "postprocessing": postprocessing,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "mean_precision": pd.to_numeric(df["precision"], errors="coerce").mean(),
        "mean_recall": pd.to_numeric(df["recall"], errors="coerce").mean(),
        "mean_f1": pd.to_numeric(df["f1"], errors="coerce").mean(),
        "mean_precision_shift": pd.to_numeric(df["precision_shift"], errors="coerce").mean(),
        "mean_recall_shift": pd.to_numeric(df["recall_shift"], errors="coerce").mean(),
        "mean_f1_shift": pd.to_numeric(df["f1_shift"], errors="coerce").mean(),
        "weighted_f1": calculate_weighted_f1(df["length"].tolist(), pd.to_numeric(df["f1"], errors="coerce").tolist()),
        "pk_pair_precision": pk_pair_precision,
        "pk_pair_recall": pk_pair_recall,
        "pk_pair_f1": pk_pair_f1,
        "pk_sequence_precision": pk_sequence_precision,
        "pk_sequence_recall": pk_sequence_recall,
        "pk_sequence_f1": pk_sequence_f1,
        "false_pseudoknot_rate": (df["pred_has_pk"] != df["true_has_pk"]).mean(),
        "pk_sequence_tp": pk_sequence_tp,
        "pk_sequence_fp": pk_sequence_fp,
        "pk_sequence_fn": pk_sequence_fn,
    }

    return pd.DataFrame([summary])


def summarize_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one summary row per family.
    """
    family_rows = []
    for family, group in df.groupby("family", dropna=False):
        summary_df = summarize_metrics(group.reset_index(drop=True), postprocessing="", input_dim=np.nan, hidden_dim=np.nan)
        row = summary_df.iloc[0].to_dict()
        row["family"] = family
        row["n_sequences"] = len(group)
        family_rows.append(row)

    family_df = pd.DataFrame(family_rows)
    ordered_columns = [
        "family",
        "n_sequences",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_precision_shift",
        "mean_recall_shift",
        "mean_f1_shift",
        "weighted_f1",
        "pk_pair_precision",
        "pk_pair_recall",
        "pk_pair_f1",
        "pk_sequence_precision",
        "pk_sequence_recall",
        "pk_sequence_f1",
        "false_pseudoknot_rate",
    ]
    return family_df[ordered_columns]
