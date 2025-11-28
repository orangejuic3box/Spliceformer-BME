"""
losses.py

Loss functions used for training splice-site models.

Contains:
  • categorical_crossentropy_2d:
        Class-weighted (optional) categorical cross-entropy over a 2D layout
        (channels × positions).

  • binary_crossentropy_2d:
        Standard binary cross-entropy over a 2D layout.

  • kl_div_2d:
        KL divergence between 3-class distributions, with optional
        temperature scaling (distillation-style).
"""

import torch
from typing import Optional, Sequence


class categorical_crossentropy_2d:
    """
    Categorical cross-entropy for 3-class predictions over a 2D grid.

    Expected input shapes
    ---------------------
    y_true : (B, 3, L)
        One-hot labels per class (3) and position (L).
    y_pred : (B, 3, L)
        Predicted probabilities per class and position.

    If `mask=True`, each class channel can have a different weight and the
    loss is normalized by the total (weighted) number of positive entries.
    If `mask=False`, the loss is normalized by sum(y_true) (i.e., number of
    labeled positions), treating all classes equally.
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        mask: bool = False,
    ):
        """
        Parameters
        ----------
        weights : sequence of float or None
            Per-class weights [w0, w1, w2] used when mask=True.
            If None and mask=True, all classes get weight 1.
        mask : bool
            Use class weights and mask normalization if True.
        """
        self.mask = mask
        # Small constant to avoid log(0)
        self.eps = torch.finfo(torch.float32).eps

        if self.mask:
            # Initialize per-class weights
            if weights is None:
                self.weights = torch.ones(3, dtype=torch.float32)
            else:
                self.weights = torch.as_tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor, shape (B, 3, L)
        y_true : torch.Tensor, shape (B, 3, L)

        Returns
        -------
        torch.Tensor (scalar)
        """
        if self.mask:
            # Unpack class weights
            w0, w1, w2 = self.weights[0], self.weights[1], self.weights[2]

            # Weighted sum of log-probabilities where y_true is 1
            loss_sum = torch.sum(
                w0 * y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + self.eps)
                + w1 * y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + self.eps)
                + w2 * y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + self.eps)
            )

            # Normalization by total (weighted) label mass
            weight_sum = (
                torch.sum(
                    w0 * y_true[:, 0, :]
                    + w1 * y_true[:, 1, :]
                    + w2 * y_true[:, 2, :]
                )
                + self.eps
            )

            # Return negative average log-likelihood
            return -loss_sum / weight_sum
        else:
            # Normalize by total label mass (across all classes)
            prob_sum = torch.sum(y_true) + self.eps
            ce = -torch.sum(
                y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + self.eps)
                + y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + self.eps)
                + y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + self.eps)
            ) / prob_sum
            return ce


class binary_crossentropy_2d:
    """
    Binary cross-entropy over a 2D layout.

    Expected shapes
    ---------------
    y_true : (B, L) or broadcastable to that
        Binary labels (0/1).
    y_pred : (B, L) or broadcastable
        Predicted probabilities in [0, 1].
    """

    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute mean binary cross-entropy.

        Parameters
        ----------
        y_pred : torch.Tensor
        y_true : torch.Tensor

        Returns
        -------
        torch.Tensor (scalar)
        """
        # BCE(y_true, y_pred) = -E[ y*log(p) + (1-y)*log(1-p) ]
        loss = torch.mean(
            y_true * torch.log(y_pred + self.eps)
            + (1 - y_true) * torch.log(1 - y_pred + self.eps)
        )
        return -loss


class kl_div_2d:
    """
    KL divergence for 3-class distributions over a 2D layout.

    Supports optional temperature scaling (`temp`) similar to
    knowledge distillation: if temp != 1, the target distribution
    is softened via softmax(log(y_true) / temp) before computing KL.

    Expected shapes
    ---------------
    y_true : (B, 3, L)
        Target distribution (often one-hot).
    y_pred : (B, 3, L)
        Predicted distribution (probabilities).
    """

    def __init__(self, temp: float = 1.0):
        """
        Parameters
        ----------
        temp : float
            Temperature factor. When != 1, y_true is re-normalized
            via a softmax at that temperature, and the final loss
            is scaled by temp**2 (standard distillation trick).
        """
        self.eps = torch.finfo(torch.float32).eps
        self.temp = temp

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute mean KL divergence KL(y_true || y_pred) over (B, L).

        Parameters
        ----------
        y_pred : torch.Tensor, shape (B, 3, L)
        y_true : torch.Tensor, shape (B, 3, L)

        Returns
        -------
        torch.Tensor (scalar)
        """
        if self.temp != 1:
            # Apply softmax to log(y_true) / temp along the class dimension
            y_true = torch.nn.Softmax(dim=1)(
                torch.log(y_true + self.eps) / self.temp
            )

        # Element-wise 3-class KL divergence; mean across batch and positions,
        # scaled by temp^2 in line with distillation formulation.
        return -torch.mean(
            (
                y_true[:, 0, :]
                * torch.log(
                    y_pred[:, 0, :] / (y_true[:, 0, :] + self.eps) + self.eps
                )
                + y_true[:, 1, :]
                * torch.log(
                    y_pred[:, 1, :] / (y_true[:, 1, :] + self.eps) + self.eps
                )
                + y_true[:, 2, :]
                * torch.log(
                    y_pred[:, 2, :] / (y_true[:, 2, :] + self.eps) + self.eps
                )
            )
            * self.temp**2
        )
