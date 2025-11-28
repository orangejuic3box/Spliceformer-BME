"""
evaluation_metrics.py

Helper functions for evaluating splice-site prediction models.

Contains:
  • print_topl_statistics:
        Computes top-k·L accuracy (for k = 0.5, 1, 2, 4) and AUPRC.
        Prints results in a fixed tab-separated format.
        Used during evaluation in eval.py.

  • topk_statistics:
        Same metric calculation but cleaner return format, optional printing.

  • cross_entropy_2d:
        Mean per-position cross-entropy between predicted and true 3-class labels.

  • kullback_leibler_divergence_2d:
        Mean per-position KL divergence between predicted and true distributions.

All functions assume:
  – y_true is a binary vector (for top-k metrics) or a 3-channel one-hot tensor.
  – y_pred is a continuous score vector (same length as y_true).
"""

import numpy as np
from sklearn.metrics import average_precision_score


def print_topl_statistics(y_true, y_pred):
    """
    Compute top-k·L metrics and AUPRC, then PRINT the results.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary vector where 1 marks true splice sites.
    y_pred : array-like of shape (N,)
        Predicted scores (higher = more likely positive).

    Returns
    -------
    topkl_accuracy : list of float
        Accuracy at k = [0.5L, 1L, 2L, 4L], where L = number of true positives.
    threshold : list of float
        Score thresholds for selecting the top k predictions.
    n_true : int
        Number of positive sites.
    auprc : float
        Area under PR curve.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Indices of true positive sites
    idx_true = np.nonzero(y_true == 1)[0]

    # Handle degenerate case: no positives
    if idx_true.size == 0:
        print(
            "No positive sites in this set – skipping top-kL statistics. "
            "AUPRC and top-kL will be reported as NaN."
        )
        topkl_accuracy = [float("nan")] * 4
        threshold = [float("nan")] * 4
        auprc = float("nan")

        # Print 12-tab output placeholder (matching expected format)
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            *["nan"] * 4,   # top-kL
            *["nan"] * 4,   # thresholds
            0,             # n_true
            "nan", "nan", "nan",   # AUPRC, correct_1, total_1
        ))
        return (topkl_accuracy, threshold, 0, auprc)

    # Sort predictions ascending
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    correct_1 = 0.0   # numerator for k=1
    total_1 = 1.0     # denominator for k=1

    # Evaluate for k = [0.5L, 1L, 2L, 4L]
    for top_length in [0.5, 1, 2, 4]:
        # k = rounded(top_length * L) but at least 1
        k = max(int(round(top_length * len(idx_true))), 1)
        k = min(k, len(y_pred))  # safety

        # Retrieve indices of top-k predicted scores
        idx_pred = argsorted_y_pred[-k:]

        # Count overlap with true positives
        correct = np.size(np.intersect1d(idx_true, idx_pred))
        total = float(min(len(idx_pred), len(idx_true)))

        # Save values for k = 1L so eval.py can compute top-1·L metrics
        if top_length == 1:
            correct_1 = correct
            total_1 = total

        # Accuracy = (# correct predictions) / min(k, L)
        acc = correct / total if total > 0 else float("nan")
        topkl_accuracy.append(acc)

        # Threshold = k-th largest predicted score
        threshold.append(sorted_y_pred[-k])

    # AUPRC from scikit-learn (micro-average)
    auprc = average_precision_score(y_true, y_pred)

    # Pretty-print the metrics in fixed 12-column format
    threshold_print = [
        f"{v:.4f}" if np.isfinite(v) else "nan" for v in threshold
    ]
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
        *[
            f"{v:.4f}" if np.isfinite(v) else "nan"
            for v in topkl_accuracy
        ],
        *threshold_print,
        len(idx_true),
        f"{auprc:.4f}" if np.isfinite(auprc) else "nan",
        f"{correct_1:.4f}" if total_1 > 0 else "nan",
        f"{total_1:.4f}" if total_1 > 0 else "nan",
    ))

    return (topkl_accuracy, threshold, len(idx_true), auprc)


def topk_statistics(y_true, y_pred, verbose=True):
    """
    Simpler, non-printing version of top-k·L metric computation.

    Parameters
    ----------
    y_true : array-like (binary)
    y_pred : array-like (scores)
    verbose : bool
        Whether to print the summary.

    Returns
    -------
    topkl_accuracy : list of float
    thresholds : list of float
    n_true : int
    auprc : float
    """
    auprc = average_precision_score(y_true, y_pred)
    idx_true = np.nonzero(y_true == 1)[0]

    if len(idx_true) == 0:
        if verbose:
            print("No positive labels present; top-k undefined.")
        return (
            [0.0] * 4,
            [0.0] * 4,
            0,
            auprc,
        )

    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:
        k = int(top_length * len(idx_true))
        k = max(k, 1)

        idx_pred = argsorted_y_pred[-k:]
        correct = np.size(np.intersect1d(idx_true, idx_pred))
        total = float(min(len(idx_pred), len(idx_true)))
        topkl_accuracy.append(correct / total)

        threshold.append(sorted_y_pred[-k])

    if verbose:
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.round(topkl_accuracy[0], 4),
            np.round(topkl_accuracy[1], 4),
            np.round(topkl_accuracy[2], 4),
            np.round(topkl_accuracy[3], 4),
            np.round(auprc, 4),
            np.round(threshold[0], 4),
            np.round(threshold[1], 4),
            np.round(threshold[2], 4),
            np.round(threshold[3], 4),
            len(idx_true),
        ))

    return (topkl_accuracy, threshold, len(idx_true), auprc)


def cross_entropy_2d(y_true, y_pred):
    """
    Compute mean cross-entropy for 3-class predictions over a 2D grid.

    Parameters
    ----------
    y_true : np.ndarray, shape (B, L, 3)
        One-hot true labels.
    y_pred : np.ndarray, shape (B, L, 3)
        Predicted probabilities for the same classes.

    Returns
    -------
    float
        Mean cross-entropy across all positions in all batches.
    """
    eps = np.finfo(np.float32).eps
    return -np.sum(
        y_true[:, :, 0] * np.log(y_pred[:, :, 0] + eps)
        + y_true[:, :, 1] * np.log(y_pred[:, :, 1] + eps)
        + y_true[:, :, 2] * np.log(y_pred[:, :, 2] + eps)
    ) / (y_true.shape[0] * y_true.shape[1])


def kullback_leibler_divergence_2d(y_true, y_pred):
    """
    Compute mean KL divergence KL(y_true || y_pred) over a 2D 3-class grid.

    Parameters
    ----------
    y_true : np.ndarray, shape (B, L, 3)
        Ground-truth probability distribution (usually one-hot).
    y_pred : np.ndarray, shape (B, L, 3)
        Predicted probability distribution.

    Returns
    -------
    float
        Mean KL divergence.
    """
    eps = np.finfo(np.float32).eps
    return -np.mean(
        y_true[:, :, 0] * np.log(y_pred[:, :, 0] / (y_true[:, :, 0] + eps) + eps)
        + y_true[:, :, 1] * np.log(y_pred[:, :, 1] / (y_true[:, :, 1] + eps) + eps)
        + y_true[:, :, 2] * np.log(y_pred[:, :, 2] / (y_true[:, :, 2] + eps) + eps)
    )
