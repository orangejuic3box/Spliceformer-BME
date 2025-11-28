#!/usr/bin/env python3
"""
Full Spliceformer eval analysis with logging, readable labels,
downsampled bootstrap, and selective bootstrapping.

- Reads all *.json (summary metrics) and *.npz (raw scores)
- Builds paper-style metrics tables
- Adds bootstrap CIs for macro AUPRC & macro top-1L
- Groups models by training species and test species category
- Produces heatmaps, barplots, and PR curves
"""

import json
import glob
import os
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("full_eval")

# -----------------------------
# Config
# -----------------------------
OUTDIR = "eval_analysis_full_clean"
os.makedirs(OUTDIR, exist_ok=True)

N_BOOT = 100
RANDOM_SEED = 123

# How many positions to keep at most for bootstrapping
MAX_BOOT_BASES = 100_000_000

# You can edit this set to control which test sets get CIs.
BOOTSTRAP_TEST_SUFFIXES = {"human", "pig", "eleg", "humanFish"}

# Map model name prefix -> training species description (internal)
MODEL_TRAINING = {
    "humanModel": "human",
    "humanChimpModel": "human+chimp",
    "humanMouseModel": "human+mouse",
    "humanFishModel": "human+zebrafish",
    "all4Model": "human+chimp+mouse+zebrafish"
}

MODEL_LABELS = {
    "humanModel": "Human only",
    "humanChimpModel": "Human + Chimp",
    "humanMouseModel": "Human + Mouse",
    "humanFishModel": "Human + Zebrafish",
    "all4Model": "Human + Chimp + Mouse + Zebrafish"
}

TRAINING_SPECIES_LABELS = {
    "human": "Human only",
    "human+chimp": "Human + Chimp",
    "human+mouse": "Human + Mouse",
    "human+zebrafish": "Human + Zebrafish",
    "human+chimp+mouse+zebrafish": "Human + Chimp + Mouse + Zebrafish",
}

SPECIES_CATEGORY = {
    "human": "mammal",
    "humanOwn": "mammal",
    "humanChimp": "multi_mammal",
    "humanMouse": "multi_mammal",
    "humanFish": "multi_mammal+fish",
    "pig": "mammal",
    "sus_scrofa": "mammal",
    "eleg": "nematode",
    "caenorhabditis_elegans": "nematode",
    "fish": "fish",
    "danio_rerio": "fish",
}

TEST_SUFFIX_LABELS = {
    "human": "Human",
    "humanOwn": "Human (own test)",
    "humanChimp": "Human + Chimp",
    "humanMouse": "Human + Mouse",
    "humanFish": "Human + Zebrafish",
    "pig": "Pig",
    "eleg": "C. elegans",
    "sus_scrofa": "Pig",
    "fish": "Fish",
    "danio_rerio": "Zebrafish",
}

# -----------------------------
# Helpers
# -----------------------------

def parse_run_name(run_name: str) -> Tuple[str, str]:
    """
    Split run_name like 'humanModel_on_human' into (model, test_suffix).
    """
    if "_on_" in run_name:
        model, test = run_name.split("_on_", 1)
    else:
        model, test = run_name, ""
    return model, test


def load_json(path: str) -> Dict:
    with open(path) as f:
        m = json.load(f)

    acceptor = m["acceptor"]
    donor = m["donor"]
    macro = m["macro"]

    run_name = os.path.basename(path).replace(".json", "")
    model, test_suffix = parse_run_name(run_name)

    return {
        "file": os.path.basename(path),
        "run_name": run_name,
        "model": model,
        "test_suffix": test_suffix,
        "species_eval_raw": ",".join(m.get("species", [])),
        "checkpoint": os.path.basename(m.get("checkpoint", "")),

        # macro metrics
        "macro_AUPRC": float(macro.get("auprc")),
        "macro_top1L": float(macro.get("top1L")),

        # class-specific metrics (top-kL order: [0.5L, 1L, 2L, 4L])
        "acc_top1L_acceptor": float(acceptor["topk_acc"][1]),
        "acc_top1L_donor": float(donor["topk_acc"][1]),
        "AUPRC_acceptor": float(acceptor["auprc"]),
        "AUPRC_donor": float(donor["auprc"]),
    }


def list_eval_files() -> Tuple[List[str], List[str]]:
    json_files = sorted(glob.glob("*.json"))
    npz_files = sorted(glob.glob("*.npz"))
    return json_files, npz_files


# ---------- Bootstrap ----------

def topkL_global(y_true: np.ndarray, y_pred: np.ndarray, factor: float = 1.0) -> float:
    """
    Spliceformer-style top-kL accuracy:
    - Let L = #true splice sites (y_true == 1)
    - k = factor * L
    - Pick top-k predictions; return fraction of true sites captured.
    """
    idx_true = np.where(y_true == 1)[0]
    n_true = len(idx_true)
    if n_true == 0:
        return np.nan

    k = int(round(factor * n_true))
    k = max(1, min(k, len(y_pred)))

    order = np.argsort(y_pred)[::-1]  # descending
    top_idx = order[:k]
    hits = (y_true[top_idx] == 1).sum()
    return hits / n_true


@dataclass
class BootstrapResult:
    macro_auprc_mean: float
    macro_auprc_ci_low: float
    macro_auprc_ci_high: float
    macro_top1L_mean: float
    macro_top1L_ci_low: float
    macro_top1L_ci_high: float


def bootstrap_metrics(
    yta,
    ypa,
    ytd,
    ypd,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
    run_label: str = "",
) -> BootstrapResult:
    """
    Bootstrap macro AUPRC and macro top-1L across flattened positions,
    using an optional downsampled subset of positions to make this tractable.

    yta, ytd: true labels for acceptor/donor
    ypa, ypd: predictions for acceptor/donor
    """

    rng = np.random.default_rng(seed)
    N = len(yta)
    assert len(yta) == len(ypa) == len(ytd) == len(ypd)

    # ---- Downsample once if N is huge ----
    if N > MAX_BOOT_BASES:
        idx_ds = rng.choice(N, size=MAX_BOOT_BASES, replace=False)
        logger.info(
            f"[{run_label}] Downsampling from N={N} to "
            f"N_eff={len(idx_ds)} for bootstrap."
        )
        yta = yta[idx_ds]
        ypa = ypa[idx_ds]
        ytd = ytd[idx_ds]
        ypd = ypd[idx_ds]
        N_eff = len(yta)
    else:
        N_eff = N
        logger.info(f"[{run_label}] Using full N={N_eff} positions for bootstrap.")

    logger.info(
        f"[{run_label}] Computing full metrics (no bootstrap) on N_eff={N_eff}..."
    )
    ap_a_full = average_precision_score(yta, ypa)
    ap_d_full = average_precision_score(ytd, ypd)
    macro_auprc_full = 0.5 * (ap_a_full + ap_d_full)

    top1L_a_full = topkL_global(yta, ypa, factor=1.0)
    top1L_d_full = topkL_global(ytd, ypd, factor=1.0)
    macro_top1L_full = 0.5 * (top1L_a_full + top1L_d_full)

    ap_macro_samples = []
    top1L_macro_samples = []

    logger.info(
        f"[{run_label}] Starting bootstrap with n_boot={n_boot}, "
        f"N_eff={N_eff} (downsampled from N={N})..."
    )
    t0 = time.time()
    report_every = max(1, n_boot // 10)

    for b in range(n_boot):
        idx = rng.integers(0, N_eff, size=N_eff)

        yta_b = yta[idx]
        ypa_b = ypa[idx]
        ytd_b = ytd[idx]
        ypd_b = ypd[idx]

        ap_a = average_precision_score(yta_b, ypa_b)
        ap_d = average_precision_score(ytd_b, ypd_b)
        ap_macro = 0.5 * (ap_a + ap_d)

        t1_a = topkL_global(yta_b, ypa_b, factor=1.0)
        t1_d = topkL_global(ytd_b, ypd_b, factor=1.0)
        t1_macro = 0.5 * (t1_a + t1_d)

        ap_macro_samples.append(ap_macro)
        top1L_macro_samples.append(t1_macro)

        if (b + 1) % report_every == 0 or (b + 1) == n_boot:
            elapsed = time.time() - t0
            logger.info(
                f"[{run_label}] Bootstrap {b+1}/{n_boot} "
                f"({(b+1)/n_boot*100:.1f}%), elapsed={elapsed/60:.1f} min"
            )

    ap_macro_samples = np.array(ap_macro_samples)
    top1L_macro_samples = np.array(top1L_macro_samples)

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    ap_ci_low, ap_ci_high = ci(ap_macro_samples)
    t1_ci_low, t1_ci_high = ci(top1L_macro_samples)

    logger.info(
        f"[{run_label}] Done bootstrap. "
        f"Macro AUPRC={macro_auprc_full:.4f} "
        f"[{ap_ci_low:.4f}, {ap_ci_high:.4f}]; "
        f"Macro top-1L={macro_top1L_full:.4f} "
        f"[{t1_ci_low:.4f}, {t1_ci_high:.4f}]"
    )

    return BootstrapResult(
        macro_auprc_mean=float(macro_auprc_full),
        macro_auprc_ci_low=ap_ci_low,
        macro_auprc_ci_high=ap_ci_high,
        macro_top1L_mean=float(macro_top1L_full),
        macro_top1L_ci_low=t1_ci_low,
        macro_top1L_ci_high=t1_ci_high,
    )


# -----------------------------
# Main analysis
# -----------------------------

def main():
    logger.info("=== Starting full evaluation analysis ===")
    json_files, npz_files = list_eval_files()
    if not json_files:
        logger.error("No .json files found – aborting.")
        return

    logger.info(f"Found {len(json_files)} JSON and {len(npz_files)} NPZ files.")
    start_time = time.time()

    # ---- Load JSON metrics into DataFrame ----
    logger.info("Loading JSON metric files...")
    rows = []
    for i, path in enumerate(json_files, 1):
        logger.info(f"[JSON {i}/{len(json_files)}] {path}")
        rows.append(load_json(path))
    df = pd.DataFrame(rows)

    # Add training species info
    df["training_species"] = df["model"].map(MODEL_TRAINING).fillna("unknown")

    # Add readable labels
    df["model_readable"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["training_species_readable"] = df["training_species"].map(
        TRAINING_SPECIES_LABELS
    ).fillna(df["training_species"])
    df["test_species_category"] = df["test_suffix"].map(
        SPECIES_CATEGORY
    ).fillna("other")
    df["test_suffix_readable"] = df["test_suffix"].map(
        TEST_SUFFIX_LABELS
    ).fillna(df["test_suffix"])

    # ---- Attach bootstrap CIs using NPZ ----
    logger.info("Preparing NPZ lookup for bootstrap...")
    npz_lookup = {os.path.basename(p).replace(".npz", ""): p for p in npz_files}

    boot_cols = {
        "macro_AUPRC_ci_low": [],
        "macro_AUPRC_ci_high": [],
        "macro_top1L_ci_low": [],
        "macro_top1L_ci_high": [],
    }

    logger.info("Starting per-run bootstrapping (selective) over all eval runs...")
    for idx, row in df.iterrows():
        run_name = row["run_name"]
        test_suffix = row["test_suffix"]
        label = f"{run_name} ({idx+1}/{len(df)})"
        npz_path = npz_lookup.get(run_name)

        # If no NPZ → no bootstrap
        if npz_path is None:
            logger.warning(f"[{label}] No NPZ file found – skipping bootstrap.")
            boot_cols["macro_AUPRC_ci_low"].append(np.nan)
            boot_cols["macro_AUPRC_ci_high"].append(np.nan)
            boot_cols["macro_top1L_ci_low"].append(np.nan)
            boot_cols["macro_top1L_ci_high"].append(np.nan)
            continue

        # If this test_suffix is not in the whitelist → skip bootstrap
        if test_suffix not in BOOTSTRAP_TEST_SUFFIXES:
            logger.info(
                f"[{label}] test_suffix='{test_suffix}' not in "
                f"BOOTSTRAP_TEST_SUFFIXES – skipping bootstrap."
            )
            boot_cols["macro_AUPRC_ci_low"].append(np.nan)
            boot_cols["macro_AUPRC_ci_high"].append(np.nan)
            boot_cols["macro_top1L_ci_low"].append(np.nan)
            boot_cols["macro_top1L_ci_high"].append(np.nan)
            continue

        logger.info(f"[{label}] Loading NPZ: {npz_path}")
        data = np.load(npz_path)
        yta = data["y_true_acceptor"].astype(np.int8)
        ytd = data["y_true_donor"].astype(np.int8)
        ypa = data["y_pred_acceptor"].astype(np.float32)
        ypd = data["y_pred_donor"].astype(np.float32)

        br = bootstrap_metrics(
            yta,
            ypa,
            ytd,
            ypd,
            n_boot=N_BOOT,
            seed=RANDOM_SEED,
            run_label=label,
        )

        boot_cols["macro_AUPRC_ci_low"].append(br.macro_auprc_ci_low)
        boot_cols["macro_AUPRC_ci_high"].append(br.macro_auprc_ci_high)
        boot_cols["macro_top1L_ci_low"].append(br.macro_top1L_ci_low)
        boot_cols["macro_top1L_ci_high"].append(br.macro_top1L_ci_high)

    for k, vals in boot_cols.items():
        df[k] = vals

    summary_csv = os.path.join(OUTDIR, "eval_summary_with_bootstrap.csv")
    df.to_csv(summary_csv, index=False)
    logger.info(f"Wrote summary with bootstrap to: {summary_csv}")

    # ---- Pivot: model × test_suffix matrices (using readable labels) ----
    logger.info("Building model × test_suffix matrices...")
    pivot_auprc = df.pivot(
        index="model_readable", columns="test_suffix_readable", values="macro_AUPRC"
    )
    pivot_top1L = df.pivot(
        index="model_readable", columns="test_suffix_readable", values="macro_top1L"
    )

    pivot_auprc_path = os.path.join(OUTDIR, "macro_auprc_matrix.csv")
    pivot_top1L_path = os.path.join(OUTDIR, "macro_top1L_matrix.csv")
    pivot_auprc.to_csv(pivot_auprc_path)
    pivot_top1L.to_csv(pivot_top1L_path)
    logger.info(f"Wrote AUPRC matrix to: {pivot_auprc_path}")
    logger.info(f"Wrote top1L matrix to: {pivot_top1L_path}")

    # -----------------------------
    # Heatmaps: macro AUPRC + top1L
    # -----------------------------
    logger.info("Plotting heatmaps...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_auprc, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Macro AUPRC (Model × Test set)")
    plt.tight_layout()
    heatmap_auprc_png = os.path.join(OUTDIR, "heatmap_macro_AUPRC.png")
    plt.savefig(heatmap_auprc_png, dpi=200)
    plt.close()
    logger.info(f"Wrote heatmap: {heatmap_auprc_png}")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_top1L, annot=True, fmt=".3f", cmap="magma")
    plt.title("Macro top-1L (Model × Test set)")
    plt.tight_layout()
    heatmap_top1L_png = os.path.join(OUTDIR, "heatmap_macro_top1L.png")
    plt.savefig(heatmap_top1L_png, dpi=200)
    plt.close()
    logger.info(f"Wrote heatmap: {heatmap_top1L_png}")

    # -----------------------------
    # Barplots: each model across test species
    # -----------------------------
    logger.info("Plotting per-model bar charts...")
    models_readable = sorted(df["model_readable"].unique())
    for model_name in models_readable:
        sub = df[df["model_readable"] == model_name].sort_values("test_suffix_readable")
        plt.figure(figsize=(10, 4))
        bars = plt.bar(sub["test_suffix_readable"], sub["macro_AUPRC"])
        plt.ylabel("Macro AUPRC")
        plt.title(f"Macro AUPRC — {model_name}")
        plt.xticks(rotation=45)

        # -------- ADD BAR LABELS --------
        for bar, value in zip(bars, sub["macro_AUPRC"]):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

        plt.tight_layout()

        out_path = os.path.join(
            OUTDIR, f"bar_macroAUPRC_{model_name.replace(' ', '_')}.png"
        )
        plt.savefig(out_path, dpi=200)
        plt.close()
        logger.info(f"[{model_name}] Wrote bar chart: {out_path}")

    # -----------------------------
    # Grouped by training vs test category
    # -----------------------------
    logger.info("Aggregating by training species vs test category...")
    grouped = (
        df.groupby(["training_species", "test_species_category"])
        .agg(
            macro_AUPRC_mean=("macro_AUPRC", "mean"),
            macro_AUPRC_std=("macro_AUPRC", "std"),
            macro_top1L_mean=("macro_top1L", "mean"),
            macro_top1L_std=("macro_top1L", "std"),
        )
        .reset_index()
    )
    # add readable training labels
    grouped["training_species_readable"] = grouped["training_species"].map(
        TRAINING_SPECIES_LABELS
    ).fillna(grouped["training_species"])

    grouped_csv = os.path.join(OUTDIR, "grouped_training_vs_test_category.csv")
    grouped.to_csv(grouped_csv, index=False)
    logger.info(f"Wrote grouped summary to: {grouped_csv}")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=grouped,
        x="training_species_readable",
        y="macro_AUPRC_mean",
        hue="test_species_category",
        errorbar=None,
    )

    plt.ylabel("Macro AUPRC (mean across runs)")
    plt.title("Macro AUPRC by training species vs test category")
    plt.xticks(rotation=45)

    # ------- ADD BAR LABELS -------
    for container in ax.containers:
        # container = bars for each hue category
        ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.tight_layout()

    grouped_bar_png = os.path.join(
        OUTDIR, "bar_training_vs_test_category_AUPRC.png"
    )
    plt.savefig(grouped_bar_png, dpi=200)
    plt.close()
    logger.info(f"Wrote grouped bar plot: {grouped_bar_png}")

    # -----------------------------
    # PR curves for all runs (acceptor & donor)
    # -----------------------------
    logger.info("Plotting PR curves for all runs (acceptor & donor)...")

    def plot_pr_all(class_true_key, class_pred_key, title, out_png):
        plt.figure(figsize=(8, 6))
        for run_name, npz_path in npz_lookup.items():
            data = np.load(npz_path)
            y_true = data[class_true_key]
            y_pred = data[class_pred_key]
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            label = run_name
            plt.plot(recall, precision, label=label)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        logger.info(f"Wrote PR plot: {out_png}")

    pr_acc_png = os.path.join(OUTDIR, "PR_acceptor_all.png")
    pr_don_png = os.path.join(OUTDIR, "PR_donor_all.png")
    plot_pr_all(
        "y_true_acceptor",
        "y_pred_acceptor",
        "PR curves (acceptor, all runs)",
        pr_acc_png,
    )
    plot_pr_all(
        "y_true_donor",
        "y_pred_donor",
        "PR curves (donor, all runs)",
        pr_don_png,
    )

    total_min = (time.time() - start_time) / 60.0
    logger.info(f"=== ALL ANALYSIS COMPLETE in {total_min:.1f} minutes ===")
    logger.info(f"Outputs written under: {OUTDIR}/")


if __name__ == "__main__":
    main()
