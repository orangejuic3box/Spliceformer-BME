"""
gpu_metrics.py

GPU-accelerated evaluation utilities for splice-site models.

Provides:
  • torch_intersect:
        Intersection of two 1D torch tensors (like np.intersect1d, but on GPU).

  • topk_statistics_cuda:
        Compute top-k·L accuracy (k = 1·L) on GPU.

  • average_precision_score (torch-based):
        GPU implementation of average precision (AUPRC) given binary targets
        and continuous scores.

  • resample:
        Random resampling indices for bootstrap (with or without replacement).

  • calculate_topk / calculate_ap:
        Convenience wrappers to compute macro top-k and AUPRC over
        acceptor + donor classes.

  • run_bootstrap:
        Bootstrap procedure on GPU(s) to estimate confidence intervals for
        macro AUPRC and top-1·L (topk) score.

Notes
-----
This code assumes:
  • Multiple CUDA devices are available (cuda:0, cuda:1, cuda:2) for
    `run_bootstrap`. If you have fewer GPUs, you’ll need to adapt devices.
  • y_true_acceptor and y_true_donor are binary tensors (0/1).
  • y_pred_* are prediction scores in [0,1].
"""

import torch
import numpy as np
from tqdm import tqdm


def torch_intersect(a, b):
    """
    Compute intersection of two 1D CUDA tensors.

    Parameters
    ----------
    a, b : torch.Tensor (1D)
        Tensors containing indices / labels.

    Returns
    -------
    intersection : torch.Tensor
        1D tensor containing elements that appear in both `a` and `b`.
    """
    # Concatenate and count occurrences of each value
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)

    # Elements that appear more than once are in the intersection
    intersection = a_cat_b[torch.where(counts.gt(1))]
    return intersection


def topk_statistics_cuda(y_true, y_pred):
    """
    Compute top-k·L accuracy on GPU for k = 1.

    Parameters
    ----------
    y_true : torch.Tensor, shape (N,)
        Binary labels (0/1) on CUDA.
    y_pred : torch.Tensor, shape (N,)
        Prediction scores on CUDA.

    Returns
    -------
    topk_accuracy : float
        Accuracy at k = 1·L, where L = # of true positives.
    threshold : float
        Score threshold for the top-k predictions.
    """
    # Indices of true positive entries
    idx_true = torch.nonzero(y_true == 1)[:, 0]

    # Sort predicted scores ascending; argsorted_y_pred contains indices
    sorted_y_pred, argsorted_y_pred = torch.sort(y_pred)

    top_length = 1
    # Number of top predictions to select: k = 1·L
    k = int(top_length * idx_true.size()[0])
    idx_pred = argsorted_y_pred[-k:]

    # Intersection between true-positive indices and predicted top-k indices
    correct = torch_intersect(idx_true, idx_pred).size()[0]
    total = float(min(len(idx_pred), len(idx_true)))

    topk_accuracy = correct / total if total > 0 else 0.0

    # Threshold score for the kth largest prediction
    threshold = sorted_y_pred[-int(top_length * len(idx_true))]
    return (topk_accuracy, threshold)


def average_precision_score(targets, predictions, device):
    """
    GPU implementation of average precision (AUPRC).

    Parameters
    ----------
    targets : torch.Tensor, shape (N,)
        Binary labels (0/1).
    predictions : torch.Tensor, shape (N,)
        Predicted scores.
    device : torch.device
        Device used to allocate small helper tensors.

    Returns
    -------
    float
        Average precision score (area under precision-recall curve).
    """
    # Sort predictions in descending order
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_targets = targets[sorted_indices]

    # Cumulative true positives and false positives
    cum_true_positives = torch.cumsum(sorted_targets, dim=0)
    cum_false_positives = torch.cumsum(1 - sorted_targets, dim=0)

    precision = cum_true_positives / (cum_true_positives + cum_false_positives)
    # Recall is TP / total positives; last element of cum_true_positives is L
    recall = cum_true_positives / cum_true_positives[-1]

    # Prepend (0,0) to recall and precision to close the curve
    recall = torch.cat([torch.tensor([0.0]).to(device), recall])
    precision = torch.cat([torch.tensor([0.0]).to(device), precision])

    # Trapezoidal-like area under the PR curve: sum (Δrecall * precision)
    average_precision = torch.sum((recall[1:] - recall[:-1]) * precision[1:])

    return average_precision.item()


def resample(tensor, replace=True, n_samples=None, random_state=None, device=None):
    """
    Generate resampling indices for bootstrap.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor whose first dimension we will resample.
    replace : bool, default True
        Whether to sample with replacement (bootstrap) or without.
    n_samples : int or None
        Number of samples to draw. If None, equals tensor.size(0).
    random_state : int or None
        Random seed for reproducibility.
    device : torch.device or None
        Device on which to allocate index tensor.

    Returns
    -------
    indices : torch.Tensor of dtype torch.long
        Resampled indices into `tensor`.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples = n_samples or tensor.size(0)
    if replace:
        indices = torch.randint(
            0,
            tensor.size(0),
            (n_samples,),
            dtype=torch.long,
            device=device,
        )
    else:
        indices = torch.randperm(tensor.size(0), device=device)[:n_samples]

    return indices


def calculate_topk(y_true_acceptor, y_pred_acceptor, y_true_donor, y_pred_donor):
    """
    Compute macro top-1·L accuracy averaged over acceptor and donor labels.

    Parameters
    ----------
    y_true_acceptor, y_true_donor : torch.Tensor
        Binary labels for acceptor/donor (0/1).
    y_pred_acceptor, y_pred_donor : torch.Tensor
        Predicted scores.

    Returns
    -------
    float
        Mean of acceptor and donor top-1·L accuracies.
    """
    topk_acceptor, _ = topk_statistics_cuda(y_true_acceptor, y_pred_acceptor)
    topk_donor, _ = topk_statistics_cuda(y_true_donor, y_pred_donor)
    return (topk_acceptor + topk_donor) / 2


def calculate_ap(
    y_true_acceptor,
    y_pred_acceptor,
    y_true_donor,
    y_pred_donor,
    device0,
    device1,
):
    """
    Compute macro AUPRC (average precision) for acceptor + donor.

    Parameters
    ----------
    y_true_acceptor, y_true_donor : torch.Tensor
        Binary labels for acceptor/donor.
    y_pred_acceptor, y_pred_donor : torch.Tensor
        Predicted scores.
    device0, device1 : torch.device
        Devices used to allocate helper tensors in average_precision_score.

    Returns
    -------
    float
        Mean of acceptor and donor AUPRC.
    """
    ap_acceptor = average_precision_score(
        y_true_acceptor,
        y_pred_acceptor,
        device0,
    )
    ap_donor = average_precision_score(
        y_true_donor,
        y_pred_donor,
        device1,
    )
    return (ap_acceptor + ap_donor) / 2


def run_bootstrap(
    Y_true_acceptor,
    Y_pred_acceptor,
    Y_true_donor,
    Y_pred_donor,
    n_bootstraps=1000,
):
    """
    Bootstrap confidence intervals for macro AUPRC and top-1·L accuracy.

    Assumes access to at least 3 GPUs (cuda:0, cuda:1, cuda:2):
      • cuda:0 / cuda:1 hold acceptor/donor predictions.
      • cuda:2 is used to generate bootstrap indices.

    Parameters
    ----------
    Y_true_acceptor, Y_true_donor : array-like
        Binary ground-truth labels (0/1).
    Y_pred_acceptor, Y_pred_donor : array-like
        Model prediction scores for acceptor/donor.
    n_bootstraps : int, default 1000
        Number of bootstrap resamples.

    Prints
    ------
    Initial macro topk and AUPRC, then
    95% confidence intervals for both metrics.
    """
    # Devices: adjust depending on your hardware setup.
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")

    # Move data to GPUs
    Y_true_acceptor_cuda = torch.as_tensor(
        Y_true_acceptor, dtype=torch.int8
    ).to(device0)
    Y_pred_acceptor_cuda = torch.as_tensor(
        Y_pred_acceptor, dtype=torch.float32
    ).to(device0)

    Y_true_donor_cuda = torch.as_tensor(
        Y_true_donor, dtype=torch.int8
    ).to(device1)
    Y_pred_donor_cuda = torch.as_tensor(
        Y_pred_donor, dtype=torch.float32
    ).to(device1)

    # Compute baseline (non-bootstrapped) scores
    topk_score = calculate_topk(
        Y_true_acceptor_cuda,
        Y_pred_acceptor_cuda,
        Y_true_donor_cuda,
        Y_pred_donor_cuda,
    )
    ap_score = calculate_ap(
        Y_true_acceptor_cuda,
        Y_pred_acceptor_cuda,
        Y_true_donor_cuda,
        Y_pred_donor_cuda,
        device0,
        device1,
    )
    print(topk_score, ap_score)

    ap_scores = []
    topk_scores = []

    # Bootstrap loop
    for i in tqdm(range(n_bootstraps)):
        # Sample indices (with replacement) on a third device
        resampled_indices = resample(
            Y_true_acceptor_cuda,
            random_state=i,
            device=device2,
        )
        # Apply same indices to all four arrays (paired bootstrap)
        resampled_y_true_acceptor = Y_true_acceptor_cuda[resampled_indices]
        resampled_y_pred_acceptor = Y_pred_acceptor_cuda[resampled_indices]
        resampled_y_true_donor = Y_true_donor_cuda[resampled_indices]
        resampled_y_pred_donor = Y_pred_donor_cuda[resampled_indices]

        del resampled_indices
        torch.cuda.empty_cache()

        # Compute metrics on resampled data
        topk = calculate_topk(
            resampled_y_true_acceptor,
            resampled_y_pred_acceptor,
            resampled_y_true_donor,
            resampled_y_pred_donor,
        )
        ap = calculate_ap(
            resampled_y_true_acceptor,
            resampled_y_pred_acceptor,
            resampled_y_true_donor,
            resampled_y_pred_donor,
            device0,
            device1,
        )

        ap_scores.append(ap)
        topk_scores.append(topk)

    # 95% CI for AUPRC
    ci_lower = np.percentile(ap_scores, 2.5)
    ci_upper = np.percentile(ap_scores, 97.5)
    print(
        f"average precision score = {ap_score} "
        f"(95% confidence interval: [{ci_lower}, {ci_upper}])"
    )

    # 95% CI for top-1·L accuracy
    ci_lower = np.percentile(topk_scores, 2.5)
    ci_upper = np.percentile(topk_scores, 97.5)
    print(
        f"topk score = {topk_score} "
        f"(95% confidence interval: [{ci_lower}, {ci_upper}])"
    )
