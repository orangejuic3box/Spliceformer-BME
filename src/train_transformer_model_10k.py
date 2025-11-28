"""
train_transformer_model_10k.py

Train a 10k-context SpliceFormer model on multi-species splice-site data.

High-level pipeline
-------------------
1. Parse CLI arguments (paths, hyperparameters, species list).
2. Load train/val annotations + labels + sparse sequence matrices
   using getData_multispecies.
3. Tile transcripts into fixed-length windows (tiles) using
   getDataPointListFull.
4. Subsample negative-only tiles per species to control class imbalance.
5. Wrap tiles into spliceDataset objects and torch DataLoaders.
6. Build the SpliceFormer model (SpliceAI front-end + policy + transformer).
7. Configure optimizer (AdamW), LR schedule (ExpLR + warmup).
8. Train with trainModel, optionally including a REINFORCE term for the policy.
9. Save best model weights and log training history.
"""

import os
import argparse
import logging
import time
import random
from collections import Counter

import numpy as np
import torch
from torch import backends

from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup

from src.dataloader import (
    getData_multispecies,
    spliceDataset,
    getDataPointListFull,
)
from src.model import SpliceFormer
from src.train import trainModel
from src.weight_init import keras_init

# cuDNN optimization flags:
#   benchmark=True   → autotune for best conv algorithms given shapes.
#   deterministic=False → allow non-deterministic ops for speed.
backends.cudnn.benchmark = True
backends.cudnn.deterministic = False


def setup_logger():
    """
    Configure a simple logger for training output.

    Returns
    -------
    logger : logging.Logger
        Logger named "spliceformer_10k".
    """
    logger = logging.getLogger("spliceformer_10k")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def parse_args():
    """
    Parse command-line arguments for training SpliceFormer 10k.

    Important flags
    ---------------
    --data-dir   : root directory for processed_data (annotations, labels, seq).
    --species    : one or more species to train on.
    --sl         : segment length used for tiling transcripts.
    --epochs     : number of training epochs.
    --lr         : base learning rate for AdamW.
    --weight-decay : L2 regularization for AdamW.
    --warmup-ratio : fraction of total steps for LR warmup.
    --n-transformer-blocks / --depth / --heads :
                    model size for SpliceFormer.
    --no-reinforce : disable policy-gradient term in trainModel.
    """
    parser = argparse.ArgumentParser(
        description="Train SpliceFormer 10k-context model on multi-species data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_data",
        help="Path to processed_data directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (in epochs). (Used inside trainModel).",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.95,
        help="Exponential LR decay factor per epoch.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="spliceformer_10k_fir",
        help="Base name for saved model (.pt will be appended).",
    )
    parser.add_argument(
        "--sl",
        type=int,
        default=600,
        help="Segment length (SL) used for tiling transcripts.",
    )
    parser.add_argument(
        "--n-transformer-blocks",
        type=int,
        default=4,
        help="Number of transformer blocks.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Internal depth parameter (Transformer depth).",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total training steps used for LR warmup.",
    )
    parser.add_argument(
        "--no-reinforce",
        action="store_true",
        help="Disable policy-gradient (reinforce) term even if supported.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=["homo_sapiens"],
        help=(
            "Species to train on. Must match the names used in the "
            "processed data directory (e.g. homo_sapiens mus_musculus "
            "pan_troglodytes danio_rerio)."
        ),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = setup_logger()

    # Select device (prefer GPU if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Segment length and context length
    SL = args.sl
    CL = args.sl
    CL_max = CL
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    logger.info(
        f"Config: L={args.depth}, CL={CL}, CL_max={CL_max}, SL={SL}, "
        f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={args.epochs}, NUM_WORKERS={NUM_WORKERS}"
    )

    species_list = args.species
    logger.info("Training on species: %s", ", ".join(species_list))

    # Normalize data_dir (drop trailing /)
    data_dir = args.data_dir.rstrip("/")

    # ----------------------------------------------------------------------
    # 1. Load train/validation annotations and label dictionaries
    # ----------------------------------------------------------------------
    logger.info("Loading train annotations and sequence data...")
    annotation_train, transcriptToLabel_train, seqData = getData_multispecies(
        data_dir=data_dir,
        setType="train",
        species_list=species_list,
    )

    logger.info("Loading validation annotations...")
    annotation_val, transcriptToLabel_val, _ = getData_multispecies(
        data_dir=data_dir,
        setType="val",
        species_list=species_list,
    )

    logger.info("annotation_train rows: %d", len(annotation_train))
    logger.info("annotation_val rows: %d", len(annotation_val))
    logger.info("train label keys: %d", len(transcriptToLabel_train))
    logger.info("val label keys: %d", len(transcriptToLabel_val))

    # ----------------------------------------------------------------------
    # 2. Build tile lists (DataPointFull objects) for train and val
    # ----------------------------------------------------------------------
    logger.info("Building train/val point lists...")
    train_points = getDataPointListFull(
        annotation_train,
        transcriptToLabel_train,
        SL,
        CL_max,
        shift=SL,  # non-overlapping tiles
    )
    val_points = getDataPointListFull(
        annotation_val,
        transcriptToLabel_val,
        SL,
        CL_max,
        shift=SL,
    )

    logger.info("Train points (tiles): %d", len(train_points))
    logger.info("Val points (tiles): %d", len(val_points))

    # ----------------------------------------------------------------------
    # 3. Subsample negative-only tiles per species (class balancing)
    # ----------------------------------------------------------------------
    # Fraction of negative-only tiles to KEEP per species.
    species_neg_keep_fraction = {
        "homo_sapiens": 0.02,
        "mus_musculus": 0.02,
        "pan_troglodytes": 0.02,
        "danio_rerio": 0.02,
    }
    default_neg_keep_fraction = 0.02

    # Split train tiles into positive vs negative-only by species
    species_pos = {}
    species_neg = {}
    for dp in train_points:
        sp = dp.transcript.split('---')[0]
        if len(dp.splice_type) > 0:
            species_pos.setdefault(sp, []).append(dp)
        else:
            species_neg.setdefault(sp, []).append(dp)

    before_total = len(train_points)
    logger.info(
        "Before subsampling: %d train tiles across %d species",
        before_total,
        len(set(list(species_pos.keys()) + list(species_neg.keys()))),
    )
    for sp in sorted(set(list(species_pos.keys()) + list(species_neg.keys()))):
        n_pos = len(species_pos.get(sp, []))
        n_neg = len(species_neg.get(sp, []))
        logger.info("  %s: %d positive, %d negative-only", sp, n_pos, n_neg)

    # Build a new train_points list with all positives + subsampled negatives
    new_train_points = []
    for sp in sorted(set(list(species_pos.keys()) + list(species_neg.keys()))):
        pos_list = species_pos.get(sp, [])
        neg_list = species_neg.get(sp, [])

        # Always keep all positive tiles
        new_train_points.extend(pos_list)

        if neg_list:
            frac = species_neg_keep_fraction.get(sp, default_neg_keep_fraction)
            if frac >= 1.0:
                k = len(neg_list)
            else:
                k = max(int(round(frac * len(neg_list))), 1)
                k = min(k, len(neg_list))

            random.shuffle(neg_list)
            kept_neg = neg_list[:k]
            new_train_points.extend(kept_neg)
        else:
            kept_neg = []

        logger.info(
            "After TRAIN subsampling for %s: keep %d positives and %d negatives",
            sp,
            len(pos_list),
            len(kept_neg),
        )

    train_points = new_train_points
    random.shuffle(train_points)
    logger.info("After TRAIN subsampling: %d train tiles total", len(train_points))

    # ----------------------------------------------------------------------
    # 4. Subsample negative-only tiles for VAL
    # ----------------------------------------------------------------------
    species_pos_val = {}
    species_neg_val = {}

    for dp in val_points:
        sp = dp.transcript.split('---')[0]
        if len(dp.splice_type) > 0:
            species_pos_val.setdefault(sp, []).append(dp)
        else:
            species_neg_val.setdefault(sp, []).append(dp)

    logger.info("VAL (before subsampling) tiles per species:")
    for sp in sorted(set(list(species_pos_val.keys()) + list(species_neg_val.keys()))):
        n_pos = len(species_pos_val.get(sp, []))
        n_neg = len(species_neg_val.get(sp, []))
        logger.info("  %s: %d positive, %d negative-only", sp, n_pos, n_neg)

    new_val_points = []
    for sp in sorted(set(list(species_pos_val.keys()) + list(species_neg_val.keys()))):
        pos_list = species_pos_val.get(sp, [])
        neg_list = species_neg_val.get(sp, [])

        new_val_points.extend(pos_list)

        if neg_list:
            frac = species_neg_keep_fraction.get(sp, default_neg_keep_fraction)
            if frac >= 1.0:
                k = len(neg_list)
            else:
                k = max(int(round(frac * len(neg_list))), 1)
                k = min(k, len(neg_list))
            random.shuffle(neg_list)
            kept_neg = neg_list[:k]
            new_val_points.extend(kept_neg)
            logger.info(
                "After VAL subsampling for %s: keep %d positives and %d negatives",
                sp,
                len(pos_list),
                len(kept_neg),
            )

    val_points = new_val_points
    random.shuffle(val_points)
    logger.info("After VAL subsampling: %d val tiles total", len(val_points))

    # Helper to log per-species tile counts
    def count_species(points, label):
        counts = Counter(dp.transcript.split('---')[0] for dp in points)
        logger.info("%s tiles per species: %s", label, dict(counts))

    count_species(train_points, "TRAIN (after subsampling)")
    count_species(val_points, "VAL (after subsampling)")

    logger.info(
        "Train points: %d, Val points: %d",
        len(train_points),
        len(val_points),
    )

    # ----------------------------------------------------------------------
    # 5. Wrap tiles as Datasets and DataLoaders
    # ----------------------------------------------------------------------
    train_dataset = spliceDataset(train_points)
    val_dataset = spliceDataset(val_points)

    # Attach shared seqData (sparse sequence matrices) so DataPointFull.getData
    # can access them.
    train_dataset.seqData = seqData
    val_dataset.seqData = seqData

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ----------------------------------------------------------------------
    # 6. Build model, optimizer, and schedulers
    # ----------------------------------------------------------------------
    logger.info("Building SpliceFormer model...")
    model = SpliceFormer(
        CL_max,
        bn_momentum=0.01 / max(args.grad_accum_steps, 1),
        depth=args.depth,
        heads=args.heads,
        n_transformer_blocks=args.n_transformer_blocks,
    )
    # Keras-like initialization for conv and linear layers
    model.apply(keras_init)
    model = model.to(device)

    # Optional DataParallel across multiple GPUs
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel over {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Epoch-level exponential decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_gamma
    )

    # Step-level warmup scheduler: constant LR after warmup
    steps_per_epoch = max(len(train_loader) // max(args.grad_accum_steps, 1), 1)
    total_steps = steps_per_epoch * args.epochs
    num_warmup_steps = max(int(args.warmup_ratio * total_steps), 1)

    warmup_sched = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps
    )

    from src.losses import categorical_crossentropy_2d

    # 3-class categorical cross-entropy loss over (B, 3, L)
    criterion = categorical_crossentropy_2d().loss

    save_name = f"{args.save_name}.pt"

    # ----------------------------------------------------------------------
    # 7. Train the model
    # ----------------------------------------------------------------------
    logger.info("Starting training...")
    start = time.time()

    history = trainModel(
        model=model,
        fileName=save_name,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup=warmup_sched,
        BATCH_SIZE=BATCH_SIZE,
        epochs=args.epochs,
        device=device,
        verbose=1,
        CL_max=CL_max,
        lowValidationGPUMem=False,
        skipValidation=False,
        NUM_ACCUMULATION_STEPS=args.grad_accum_steps,
        reinforce=not args.no_reinforce,
        continous_labels=False,
        no_softmax=False,
    )

    elapsed = (time.time() - start) / 60.0
    logger.info(f"Training + validation completed in {elapsed:.1f} minutes.")
    logger.info(f"Model weights saved as {save_name}")
    logger.info(f"History keys: {list(history.keys())}")


if __name__ == "__main__":
    main()
