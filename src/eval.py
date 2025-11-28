"""
eval.py

Evaluate a trained 10k-context SpliceFormer model on the multi-species TEST set.

High-level steps:
  1. Parse CLI arguments (paths, species, model hyperparams, checkpoint).
  2. Load test annotations, labels, and sparse sequence matrices
     using getData_multispecies (from dataloader).
  3. Tile transcripts into SL-length windows with getDataPointListFull.
  4. Wrap tiles in a PyTorch DataLoader via spliceDataset.
  5. Load the SpliceFormer model + checkpoint.
  6. Run inference over all test tiles.
  7. Compute top-k·L and PR-AUC metrics separately for acceptor/donor.
  8. Save metrics to JSON and raw scores to NPZ.
"""

import argparse
import logging
import time
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import backends

# cuDNN configuration:
#   benchmark=True  → enable auto-tuning for better performance
#   deterministic=False → allow non-deterministic algorithms for speed
backends.cudnn.benchmark = True
backends.cudnn.deterministic = False

from src.dataloader import (
    getData_multispecies,
    spliceDataset,
    getDataPointListFull,
)
from src.model import SpliceFormer
from src.evaluation_metrics import print_topl_statistics


def setup_logger():
    """
    Configure a simple console logger for evaluation.

    Returns
    -------
    logger : logging.Logger
        Logger instance named "spliceformer_10k_eval".
    """
    logger = logging.getLogger("spliceformer_10k_eval")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if setup_logger() is called more than once
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
    Parse command-line arguments for evaluation.

    Key arguments:
      --data-dir   : processed_data root directory.
      --species    : one or more species to evaluate on.
      --sl         : segment length (must match training).
      --depth      : transformer depth (SpliceFormer hyperparam).
      --heads      : number of attention heads.
      --n-transformer-blocks : number of blocks (must match training).
      --checkpoint : model checkpoint (.pt) to load.
      --out-prefix : prefix for saving metrics JSON + NPZ scores.
    """
    p = argparse.ArgumentParser(
        description=(
            "Evaluate 10k-context SpliceFormer model on multi-species TEST set "
            "using top-kL + PR-AUC metrics."
        )
    )

    p.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_data",
        help="Path to processed_data directory (where *_train/val/test files live).",
    )
    p.add_argument(
        "--species",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Species to evaluate on, e.g. homo_sapiens, mus_musculus, "
            "pan_troglodytes, danio_rerio. "
            "You can pass one or more: "
            "--species homo_sapiens or "
            "--species homo_sapiens mus_musculus"
        ),
    )
    p.add_argument(
        "--sl",
        type=int,
        default=10000,
        help="Segment length (SL) used for training (must match).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Batch size for evaluation.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Transformer depth (must match training).",
    )
    p.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of transformer heads (must match training).",
    )
    p.add_argument(
        "--n-transformer-blocks",
        type=int,
        default=4,
        help="Number of transformer blocks (must match training).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt weights file (the --save-name from training, with .pt).",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output files (JSON + NPZ), e.g. eval_human_only.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    # Choose device: prefer GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    SL = args.sl
    # CL_max is the context length, set equal to SL (i.e., full context size)
    CL_max = SL

    logger.info(
        f"Eval config: SL={SL}, CL_max={CL_max}, batch={args.batch_size}, "
        f"species={args.species}"
    )

    # -------------------------------------------------------------------------
    # 1. Load TEST annotations + sequence data
    # -------------------------------------------------------------------------
    logger.info("Loading TEST annotations and sequence data...")
    annotation_test, transcriptToLabel_test, seqData = getData_multispecies(
        data_dir=args.data_dir,
        setType="test",
        species_list=args.species,
    )

    logger.info("annotation_test rows: %d", len(annotation_test))
    logger.info("test label keys: %d", len(transcriptToLabel_test))

    # -------------------------------------------------------------------------
    # 2. Build test tile list
    # -------------------------------------------------------------------------
    logger.info("Building test tile list...")
    test_points = getDataPointListFull(
        annotation_test,
        transcriptToLabel_test,
        SL,
        CL_max,
        shift=SL,  # non-overlapping tiles: shift equals SL
    )

    logger.info("Total TEST tiles: %d", len(test_points))

    # Count tiles per species from transcript names, which look like
    #   "species---transcript_id"
    counts = Counter(dp.transcript.split('---')[0] for dp in test_points)
    logger.info("TEST tiles per species: %s", dict(counts))

    # Wrap into a Dataset and attach the seqData dict (sparse matrices)
    test_dataset = spliceDataset(test_points)
    test_dataset.seqData = seqData

    # PyTorch DataLoader for batched evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # -------------------------------------------------------------------------
    # 3. Build model and load checkpoint
    # -------------------------------------------------------------------------
    logger.info("Building SpliceFormer model...")
    model = SpliceFormer(
        CL_max,
        bn_momentum=0.01,
        depth=args.depth,
        heads=args.heads,
        n_transformer_blocks=args.n_transformer_blocks,
    )
    model = model.to(device)

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)

    # If checkpoint was saved using DataParallel / DDP, strip 'module.' prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        logger.info("Stripping 'module.' prefix from state_dict keys")
        new_sd = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict = new_sd

    model.load_state_dict(state_dict)
    model.eval()

    # -------------------------------------------------------------------------
    # 4. Run TEST inference
    # -------------------------------------------------------------------------
    logger.info("Starting TEST inference...")
    start = time.time()

    # We track acceptor and donor labels/scores separately
    Y_true_acceptor = []
    Y_true_donor = []
    Y_pred_acceptor = []
    Y_pred_donor = []

    with torch.no_grad():
        for batch_idx, (batch_features, targets_full) in enumerate(test_loader):
            batch_features = batch_features.to(device)
            targets_full = targets_full.to(device)

            # targets_full has shape (B, C, SL + CL_max).
            # Remove CL_max/2 context from each side to get the core SL region.
            targets = targets_full[:, :, CL_max // 2 : -CL_max // 2]

            # Forward pass through the SpliceFormer model
            outputs = model(batch_features)
            # If the model returns multiple outputs (e.g. auxiliary), take first
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Assuming model output dims:
            #   outputs: (B, num_classes, SL)
            # class index 1: acceptor, class index 2: donor
            acc_scores = outputs[:, 1, :].detach().cpu().numpy().ravel()
            don_scores = outputs[:, 2, :].detach().cpu().numpy().ravel()

            # Ground truth labels for acceptor/donor (one-hot channel indices)
            acc_true = targets[:, 1, :].detach().cpu().numpy().ravel()
            don_true = targets[:, 2, :].detach().cpu().numpy().ravel()

            Y_true_acceptor.append(acc_true)
            Y_true_donor.append(don_true)
            Y_pred_acceptor.append(acc_scores)
            Y_pred_donor.append(don_scores)

            # Periodic progress log
            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    "Processed %d / %d batches",
                    batch_idx + 1,
                    len(test_loader),
                )

    elapsed = (time.time() - start) / 60.0
    logger.info(f"Finished TEST inference in {elapsed:.1f} minutes.")

    # Concatenate over all batches to get flat arrays
    yta = np.concatenate(Y_true_acceptor)
    ytd = np.concatenate(Y_true_donor)
    ypa = np.concatenate(Y_pred_acceptor)
    ypd = np.concatenate(Y_pred_donor)

    # -------------------------------------------------------------------------
    # 5. Compute evaluation metrics
    # -------------------------------------------------------------------------
    logger.info("==== TEST metrics: ACCEPTOR ====")
    topk_acc_a, thr_a, n_true_a, auprc_a = print_topl_statistics(yta, ypa)

    logger.info("==== TEST metrics: DONOR ====")
    topk_acc_d, thr_d, n_true_d, auprc_d = print_topl_statistics(ytd, ypd)

    # Macro-average AUPRC and top-1·L across acceptor/donor
    macro_auprc = float((auprc_a + auprc_d) / 2.0)
    macro_top1L = float((topk_acc_a[1] + topk_acc_d[1]) / 2.0)

    logger.info("Macro AUPRC: %.4f", macro_auprc)
    logger.info("Macro top-1L accuracy: %.4f", macro_top1L)

    # -------------------------------------------------------------------------
    # 6. Save metrics and raw scores
    # -------------------------------------------------------------------------
    out_json = args.out_prefix + ".json"
    out_npz = args.out_prefix + ".npz"

    metrics = {
        "species": args.species,
        "SL": SL,
        "checkpoint": args.checkpoint,
        "acceptor": {
            "topk_acc": list(map(float, topk_acc_a)),
            "thresholds": list(map(float, thr_a)),
            "n_true": int(n_true_a),
            "auprc": float(auprc_a),
        },
        "donor": {
            "topk_acc": list(map(float, topk_acc_d)),
            "thresholds": list(map(float, thr_d)),
            "n_true": int(n_true_d),
            "auprc": float(auprc_d),
        },
        "macro": {
            "auprc": macro_auprc,
            "top1L": macro_top1L,
        },
    }

    # Save metrics as nicely formatted JSON
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw label/score arrays for further analysis / plotting
    np.savez_compressed(
        out_npz,
        y_true_acceptor=yta,
        y_true_donor=ytd,
        y_pred_acceptor=ypa,
        y_pred_donor=ypd,
    )

    logger.info(f"Wrote metrics JSON to {out_json}")
    logger.info(f"Wrote raw scores to {out_npz}")


if __name__ == "__main__":
    main()
