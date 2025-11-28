"""
train.py

Generic training loop for splice-site models (e.g., SpliceFormer).

Features
--------
- Supports:
    * mixed precision (AMP) on CUDA
    * gradient accumulation
    * exponential LR scheduler + warmup schedule
    * optional REINFORCE-style policy loss (for action-selection policy)
- Tracks:
    * running and EMA loss per phase (train/val)
    * top-k·L & AUPRC on validation via print_topl_statistics
- Saves:
    * best model weights based on validation loss
    * a DataFrame of train / val loss history
"""

import torch
import numpy as np
from tqdm import tqdm
from .evaluation_metrics import print_topl_statistics
import pandas as pd

from torch.cuda.amp import autocast, GradScaler


def trainModel(
    model,
    fileName,
    criterion,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    warmup,
    BATCH_SIZE,
    epochs,
    device,
    verbose=1,
    CL_max=40000,
    lowValidationGPUMem=False,  # kept for API compatibility; unused
    skipValidation=False,
    NUM_ACCUMULATION_STEPS=1,
    reinforce=True,
    continous_labels=False,     # kept for API compatibility; unused
    no_softmax=False,
):
    """
    Train a model with optional policy-gradient term and mixed precision.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train. If `reinforce=True`, forward must return either:
          - outputs
          - or (outputs, acceptor_actions, donor_actions,
                acceptor_log_probs, donor_log_probs)
    fileName : str
        Path to save the best model weights (.pt).
    criterion : callable
        Loss function taking (outputs, targets_trimmed) → scalar tensor.
    train_loader, val_loader : DataLoader
        PyTorch DataLoaders for training and validation.
    optimizer : torch.optim.Optimizer
        Optimizer (e.g., AdamW).
    scheduler : torch.optim.lr_scheduler
        Epoch-level LR scheduler (e.g., ExponentialLR). Can be None.
    warmup : transformers-style scheduler or None
        Step-level warmup scheduler (e.g. get_constant_schedule_with_warmup).
        Only used for the first few epochs.
    BATCH_SIZE : int
        Batch size (not directly used, but passed for compatibility).
    epochs : int
        Number of training epochs.
    device : torch.device
        Training device.
    verbose : int
        Unused here; kept for API compatibility.
    CL_max : int
        Total context length (used to crop targets_full → targets).
    lowValidationGPUMem : bool
        Unused; reserved for potential low-memory validation mode.
    skipValidation : bool
        If True, skip validation phase entirely and always save model.
    NUM_ACCUMULATION_STEPS : int
        Number of mini-batches to accumulate gradients over
        before stepping the optimizer.
    reinforce : bool
        If True and model returns actions + log_probs, add a small REINFORCE
        term to the loss to train the policy.
    continous_labels : bool
        Unused; reserved for potential regression-style labels.
    no_softmax : bool
        If True, apply softmax to outputs *inside* train loop
        before extracting metrics (for models that return logits).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - "loss" (train loss per epoch)
          - "val_loss" (if skipValidation=False)
    """
    # Track per-epoch losses
    losses = {"train": [], "val": []}

    # Multiplier for EMA smoothing
    multiplier = 0.01
    eps = torch.finfo(torch.float32).eps

    # Exponential moving averages of acceptor/donor policy accuracy
    acceptor_acc_avg = torch.tensor(0.0, device=device)
    donor_acc_avg = torch.tensor(0.0, device=device)

    # Best validation loss for checkpointing
    best_val_loss = float("inf")

    # Enable AMP if running on CUDA
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        for phase in ["train", "val"]:
            # Optionally skip validation
            if skipValidation and phase == "val":
                continue

            dataloader = train_loader if phase == "train" else val_loader
            loop = tqdm(dataloader, desc=f"{phase} epoch {epoch+1}/{epochs}")

            # Train or eval mode
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            ema_loss = 0.0
            # (ema_l1, ema_a_recall, ema_d_recall are kept for extension/debug)
            ema_l1 = 0.0
            ema_a_recall = 0.0
            ema_d_recall = 0.0
            n_steps_completed = 0

            # For validation metrics (top-k·L & AUPRC)
            Y_true_acceptor, Y_true_donor = [], []
            Y_pred_acceptor, Y_pred_donor = [], []

            # Counter for gradient accumulation
            n_accum = 0

            for i, (batch_features, targets) in enumerate(loop):
                batch_features = batch_features.to(device)
                targets_full = targets.to(device)

                # Trim context from labels: keep only the core segment of length SL
                targets = targets_full[:, :, CL_max // 2 : -CL_max // 2]

                # Zero grad every NUM_ACCUMULATION_STEPS
                if i % NUM_ACCUMULATION_STEPS == 0:
                    optimizer.zero_grad()

                # Enable or disable grad depending on phase
                if phase == "train":
                    grad_context = torch.enable_grad()
                else:
                    grad_context = torch.no_grad()

                with grad_context:
                    # Mixed precision region (if enabled)
                    with autocast(enabled=use_amp):
                        outputs = model(batch_features)

                        # Defaults for policy-related outputs
                        acceptor_actions = donor_actions = None
                        acceptor_log_probs = donor_log_probs = None

                        # If model returns (outputs, actions, log_probs, ...)
                        if isinstance(outputs, tuple):
                            if reinforce:
                                acceptor_actions = outputs[1]
                                donor_actions = outputs[2]
                                acceptor_log_probs = outputs[3]
                                donor_log_probs = outputs[4]
                            # Primary outputs are first element
                            outputs = outputs[0]

                        # Supervised loss (criterion handles logits/probs)
                        train_loss = (
                            criterion(outputs, targets) / NUM_ACCUMULATION_STEPS
                        )

                    # Optionally apply softmax if model returned raw logits
                    if no_softmax:
                        outputs = torch.nn.Softmax(dim=1)(outputs)

                    # Optional REINFORCE-style term for the policy
                    if (
                        phase == "train"
                        and reinforce
                        and acceptor_actions is not None
                        and donor_actions is not None
                    ):
                        # Rewards based on mismatch between two channels:
                        #   acceptors reward: y_acceptor - y_donor
                        #   donors reward:    y_donor - y_acceptor
                        acceptor_reward = torch.gather(
                            targets_full[:, 1, :] - targets_full[:, 2, :],
                            1,
                            acceptor_actions,
                        )
                        donor_reward = torch.gather(
                            targets_full[:, 2, :] - targets_full[:, 1, :],
                            1,
                            donor_actions,
                        )

                        # Track moving average of policy "accuracy" for logging
                        if phase == "train":
                            acceptor_acc = torch.nanmean(
                                torch.sum(acceptor_reward > 0, dim=1)
                                / (torch.sum(targets_full[:, 1, :] > 0, dim=1) + eps)
                            )
                            acceptor_acc_avg = (
                                acceptor_acc * multiplier
                                + acceptor_acc_avg * (1 - multiplier)
                            )

                            donor_acc = torch.nanmean(
                                torch.sum(donor_reward > 0, dim=1)
                                / (torch.sum(targets_full[:, 2, :] > 0, dim=1) + eps)
                            )
                            donor_acc_avg = (
                                donor_acc * multiplier
                                + donor_acc_avg * (1 - multiplier)
                            )

                        # REINFORCE loss: negative expected reward under log_probs
                        acceptor_loss = -torch.mean(
                            torch.sum(acceptor_log_probs * acceptor_reward, dim=1)
                        ) / NUM_ACCUMULATION_STEPS
                        donor_loss = -torch.mean(
                            torch.sum(donor_log_probs * donor_reward, dim=1)
                        ) / NUM_ACCUMULATION_STEPS
                        reinforce_loss = acceptor_loss + donor_loss

                        # Tiny coefficient (1e-6) so this term doesn't dominate
                        train_loss = train_loss + 1e-6 * reinforce_loss

                # ------------------------------------------------------------------
                # Backward + optimizer step (with gradient accumulation and AMP)
                # ------------------------------------------------------------------
                if phase == "train":
                    if use_amp:
                        # Scale loss for mixed precision, then backprop
                        scaler.scale(train_loss).backward()
                        n_accum += 1

                        # Step optimizer every NUM_ACCUMULATION_STEPS or at end
                        if (n_accum == NUM_ACCUMULATION_STEPS) or (
                            i + 1 == len(loop)
                        ):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                max_norm=1.0,
                                norm_type=2.0,
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            n_accum = 0

                            # LR warmup only in early epochs
                            if epoch < 5 and warmup is not None:
                                warmup.step()
                    else:
                        # Standard (non-AMP) backward
                        train_loss.backward()
                        n_accum += 1

                        if (n_accum == NUM_ACCUMULATION_STEPS) or (
                            i + 1 == len(loop)
                        ):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                max_norm=1.0,
                                norm_type=2.0,
                            )
                            optimizer.step()
                            n_accum = 0

                            if epoch < 5 and warmup is not None:
                                warmup.step()

                # ------------------------------------------------------------------
                # Bookkeeping: track loss & (for val) prediction statistics
                # ------------------------------------------------------------------
                loss_val = float(train_loss.detach().cpu().item())
                running_loss += loss_val
                n_steps_completed += 1

                # EMA of loss
                if n_steps_completed == 1:
                    ema_loss = loss_val
                else:
                    ema_loss = ema_loss * (1 - multiplier) + multiplier * loss_val

                # During validation, accumulate scores for top-k / AUPRC
                if phase == "val":
                    with torch.no_grad():
                        if outputs.ndim == 3 and outputs.size(1) >= 3:
                            # Acceptors = channel 1, donors = channel 2
                            acc_scores = (
                                outputs[:, 1, :].detach().cpu().numpy().ravel()
                            )
                            don_scores = (
                                outputs[:, 2, :].detach().cpu().numpy().ravel()
                            )
                            acc_true = targets[:, 1, :].detach().cpu().numpy().ravel()
                            don_true = targets[:, 2, :].detach().cpu().numpy().ravel()

                            Y_true_acceptor.append(acc_true)
                            Y_true_donor.append(don_true)
                            Y_pred_acceptor.append(acc_scores)
                            Y_pred_donor.append(don_scores)

                # Show current loss and EMA in tqdm bar
                loop.set_postfix(
                    {
                        "loss": f"{loss_val:.4f}",
                        "ema_loss": f"{ema_loss:.4f}",
                    }
                )

            # ----------------------------------------------------------------------
            # End of epoch: aggregate losses and metrics
            # ----------------------------------------------------------------------
            epoch_loss = running_loss / max(n_steps_completed, 1)
            if phase == "train":
                losses["train"].append(epoch_loss)
            else:
                losses["val"].append(epoch_loss)

            if phase == "val":
                # If we collected any predictions, compute top-k·L + AUPRC
                if Y_true_acceptor:
                    yta = np.concatenate(Y_true_acceptor)
                    ytd = np.concatenate(Y_true_donor)
                    ypa = np.concatenate(Y_pred_acceptor)
                    ypd = np.concatenate(Y_pred_donor)

                    print("Validation acceptor stats:")
                    print_topl_statistics(yta, ypa)
                    print("Validation donor stats:")
                    print_topl_statistics(ytd, ypd)

                # Save best model by validation loss
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), fileName)
            elif skipValidation:
                # If validation is skipped, always save the latest model
                torch.save(model.state_dict(), fileName)

        # Step epoch-level scheduler (e.g. ExponentialLR)
        if scheduler is not None:
            scheduler.step()

    # ----------------------------------------------------------------------
    # Return training history as a DataFrame
    # ----------------------------------------------------------------------
    if skipValidation:
        return pd.DataFrame({"loss": losses["train"]})
    else:
        return pd.DataFrame(
            {"loss": losses["train"], "val_loss": losses["val"]}
        )
