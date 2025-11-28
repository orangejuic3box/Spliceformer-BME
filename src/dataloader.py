"""
dataloader.py

Utilities for loading multi-species splice site datasets created by
create_dataset_multispecies.py, and wrapping them in a PyTorch Dataset.

Main pieces:
  • getData_multispecies: load annotations, labels, and sparse sequence matrices
    for one or more species and a given split (train/val/test).
  • DataPointFull: represents one sliding window over a transcript.
  • getDataPointListFull: convert transcript-level labels into a list of
    DataPointFull objects (windows).
  • spliceDataset: lightweight torch.utils.data.Dataset wrapper that calls
    DataPointFull.getData to produce (X, Y) for a model.
"""

import numpy as np
from torch.utils.data import Dataset
import pickle
import pandas as pd
import os
from scipy.sparse import load_npz
from glob import glob
from math import ceil
from typing import Any, Dict

# Map integer splice label → one-hot vector:
#   0 → donor, 1 → acceptor, 2 → other splice label, 3 → background
OUT_MAP = np.asarray(
    [
        [1, 0, 0],  # 0: donor
        [0, 1, 0],  # 1: acceptor
        [0, 0, 1],  # 2: other
        [0, 0, 0],  # 3: background / masked
    ]
)


def ceil_div(x, y):
    """Integer ceiling of x / y."""
    return int(ceil(float(x) / y))


def getData_multispecies(data_dir, setType, species_list):
    """
    Load annotation, labels, and sparse sequence matrices for multiple species.

    Parameters
    ----------
    data_dir : str
        Root directory containing processed data from create_dataset_multispecies.py.
    setType : {"train", "val", "test"}
        Which split to load.
    species_list : list of str
        Species names to load (e.g. ["homo_sapiens", "mus_musculus"]).

    Returns
    -------
    annotation_all : pd.DataFrame
        Combined annotation table. Columns:
            transcript  : str – key used to look up labels
            gene        : str or None – gene ID (if present in name)
            chrom       : str – species|chromosome (e.g. "homo_sapiens|1")
            strand      : {"+" or "-"}
            tx_start    : int – transcript genomic start
            tx_end      : int – transcript genomic end
    transcriptToLabel_all : dict
        Mapping "species---transcript_id" → (Y_type, Y_idx),
        where Y_type and Y_idx are numpy arrays.
    seqData_all : dict
        Mapping "species|chrom" → CSR sparse matrix of shape (L, 5),
        encoding sequence for that chromosome.
    """
    all_annotations = []
    transcriptToLabel_all = {}
    seqData_all = {}

    def _parse_name(name: str):
        """
        Parse transcript name from annotation / label files.

        Expected formats:
          • "species---gene---transcript"
          • "species---transcript"

        Returns
        -------
        species : str
        gene_id : str or None
        tx_key  : str – composite key "species---transcript_id"
                  used consistently across tables.
        """
        parts = name.split('---')

        if len(parts) == 3:
            species, gene_id, tx_id = parts
        elif len(parts) == 2:
            species, tx_id = parts
            gene_id = None
        else:
            raise ValueError(
                f"Unexpected name format (expected 'species---gene---transcript' "
                f"or 'species---transcript'): {name}"
            )

        tx_key = f"{species}---{tx_id}"
        return species, gene_id, tx_key

    # Iterate over all requested species and aggregate their data
    for species_name in species_list:
        # 1) Load transcript-level labels for this species + split
        label_path = f"{data_dir}/sparse_discrete_label_data_{species_name}_{setType}.pickle"
        with open(label_path, "rb") as handle:
            tx2lab = pickle.load(handle)

        # Remap keys so that labels are indexed by "species---transcript"
        remapped = {}
        for name, value in tx2lab.items():
            species, gene_id, tx_key = _parse_name(name)
            remapped[tx_key] = value

        transcriptToLabel_all.update(remapped)

        # 2) Load annotation table for this species + split
        ann_path = f"{data_dir}/annotations/annotation_{species_name}_{setType}.txt"
        ann = pd.read_csv(
            ann_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3, 4],
        )
        ann.columns = ["name", "chrom", "strand", "tx_start", "tx_end"]

        # Split composite names into species, gene, and transcript keys
        species_col, gene_col, transcript_col = zip(
            *[_parse_name(n) for n in ann["name"].values]
        )
        ann["species"] = species_col
        ann["gene"] = gene_col
        ann["transcript"] = transcript_col

        # Rename chromosome as "<species>|<chrom>" to avoid collisions
        ann["chrom"] = ann["chrom"].apply(
            lambda c, s=species_name: f"{s}|{c}"
        )

        # Keep only relevant columns in the combined annotation
        ann = ann[["transcript", "gene", "chrom", "strand", "tx_start", "tx_end"]]
        all_annotations.append(ann)

        # 3) Load sparse sequence matrices (.npz) for this species + split
        pattern = f"{data_dir}/sparse_sequence_data/{species_name}_*_{setType}.npz"
        for path in glob(pattern):
            fname = os.path.basename(path)
            base = fname[:-4] if fname.endswith(".npz") else fname

            # Check suffix (must end with "_<setType>")
            suffix = f"_{setType}"
            if not base.endswith(suffix):
                raise ValueError(f"Unexpected npz filename (suffix): {fname}")
            mid = base[: -len(suffix)]

            # Check prefix (must start with "<species_name>_")
            prefix = f"{species_name}_"
            if not mid.startswith(prefix):
                raise ValueError(f"Unexpected npz filename (prefix): {fname}")

            # Extract chromosome name from "<species_name>_<chrom>"
            chrom_name = mid[len(prefix):]
            chrom_key = f"{species_name}|{chrom_name}"

            # Store as CSR for efficient row slicing in getData()
            seqData_all[chrom_key] = load_npz(path).tocsr()

    # Concatenate annotations from all species into a single DataFrame
    annotation_all = pd.concat(all_annotations, axis=0, ignore_index=True)
    return annotation_all, transcriptToLabel_all, seqData_all


class DataPointFull:
    """
    Represents a single sliding window over a transcript, with:

      • genomic window [start, end]
      • transcript bounds [tx_start, tx_end]
      • splice site locations and types within that window
      • masking to avoid reading outside the transcript body

    getData(seqData) builds:
      X : np.ndarray, shape (4, SL + CL_max)
          One-hot sequence window (A/C/G/T channels).
      Y : np.ndarray, shape (3, SL + CL_max)
          Splice-site labels per position (donor, acceptor, other/background).
    """

    def __init__(
        self,
        transcript,
        gene,
        chrom,
        strand,
        start,
        end,
        tx_start,
        tx_end,
        splice_loc,
        splice_type,
        SL,
        CL_max,
        shift,
        mask_l,
        mask_r,
        include_pos=False,
    ):
        # Transcript ID / key
        self.transcript = transcript
        self.gene = gene

        # Chromosome key used to index seqData (e.g. "species|1")
        self.chrom = chrom
        self.strand = strand

        # Genomic window over which we will build X/Y
        self.start = start
        self.end = end

        # Full transcript genomic bounds
        self.tx_start = tx_start
        self.tx_end = tx_end

        # Positions (relative to transcript) and types of splice sites
        self.splice_loc = splice_loc
        self.splice_type = splice_type

        # Window length (SL) and context length (CL_max) around window
        self.SL = SL
        self.CL_max = CL_max

        # Shift offset used to align splice locations inside the window
        self.shift = shift

        # Left / right masks: number of positions at the window edges
        # that do not correspond to actual transcript sequence.
        self.mask_l = mask_l
        self.mask_r = mask_r

        # Optionally include absolute genomic positions and IDs for downstream use
        self.include_pos = include_pos

    def getData(self, seqData):
        """
        Construct input X and label Y for this datapoint.

        Parameters
        ----------
        seqData : dict[str, scipy.sparse.csr_matrix]
            Mapping chrom key → sparse sequence matrix:
              rows : genomic positions (0-based)
              cols : 0-3 = one-hot A/C/G/T, 4 = mask (non-ACGT)

        Returns
        -------
        X : np.ndarray, shape (4, SL + CL_max)
        Y : np.ndarray, shape (3, SL + CL_max)
        (optionally) pos, chrm, transcript if include_pos=True:
            pos       : np.ndarray of genomic positions
            chrm      : np.ndarray of chrom keys (same length as pos)
            transcript: np.ndarray of transcript IDs (same length as pos)
        """
        # X: 4 channels (A,C,G,T) across the extended window
        X = np.zeros((4, self.SL + self.CL_max), dtype=np.float32)
        # r: mask channel (from column 4 of seqData); 1 where masked/non-ACGT
        r = np.zeros((self.SL + self.CL_max), dtype=np.uint8)
        # Y: 3-class output (donor, acceptor, other) per position
        Y = np.zeros((3, self.SL + self.CL_max), dtype=np.float32)

        if self.include_pos:
            # Absolute genomic positions corresponding to the window
            pos = np.arange(self.start - 1, self.end)
            chrm = np.repeat(self.chrom, self.SL)
            transcript = np.repeat(self.transcript, self.SL)

        # Extract sequence slice from sparse matrix, then align it
        # into the window depending on strand and masks.
        if self.strand == '+':
            X[:, self.mask_l:self.SL + self.CL_max - self.mask_r] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_l:
                    self.end + self.CL_max // 2 - self.mask_r,
                    :4,
                ]
                .toarray()
                .T
            )
            r[self.mask_l:self.SL + self.CL_max - self.mask_r] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_l:
                    self.end + self.CL_max // 2 - self.mask_r,
                    4,
                ]
                .toarray()[:, 0]
            )
        else:
            # On the minus strand, we extract the sequence window in genomic
            # coordinates, then reverse-complement the orientation for X and r.
            X[:, self.mask_r:self.SL + self.CL_max - self.mask_l] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_r:
                    self.end + self.CL_max // 2 - self.mask_l,
                    :4,
                ]
                .toarray()
                .T
            )
            r[self.mask_r:self.SL + self.CL_max - self.mask_l] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_r:
                    self.end + self.CL_max // 2 - self.mask_l,
                    4,
                ]
                .toarray()[:, 0]
            )
            # Reverse the genomic direction for the minus strand
            X = X[:, ::-1]
            r = r[::-1]
            # Also flip base-channel ordering (A,C,G,T) → (T,G,C,A)
            X = X[::-1, :]

        # Set base label: every valid sequence position in the core window
        # (excluding masked edges) is initially considered as background (class 2)
        # before we overwrite splice sites and masked regions.
        Y[0, self.mask_l:(self.SL + self.CL_max - self.mask_r)] = np.ones(
            self.SL + self.CL_max - self.mask_r - self.mask_l
        )
        # Place explicit splice site labels from splice_loc/splice_type.
        # Index inside the extended window is offset by shift and CL_max//2.
        Y[:, self.splice_loc - self.shift + self.CL_max // 2] = OUT_MAP[
            np.array(self.splice_type, dtype=np.int8)
        ].T

        # Any position with r == 1 is considered masked (e.g., non-ACGT),
        # and we overwrite its label with the background vector.
        r_sum = np.sum(r)
        if r_sum > 0:
            Y[:, r == 1] = OUT_MAP[3 * np.ones(int(r_sum), dtype=np.int8)].T

        if self.include_pos:
            return X.copy(), Y.copy(), pos, chrm, transcript
        else:
            return X.copy(), Y.copy()


def getDataPointListFull(
    annotation,
    transcriptToLabel,
    SL,
    CL_max,
    shift,
    include_pos=False,
):
    """
    Convert a transcript-level annotation table into a list of DataPointFull.

    Strategy
    --------
    For each transcript:
      • Determine its genomic length.
      • Slide a window of length SL across the transcript in steps of `shift`.
      • For each window, compute which labeled splice sites fall into the
        extended context region (SL + CL_max), taking strand into account.
      • Compute masks and offsets so DataPointFull.getData can later build
        the correct X/Y arrays.

    Parameters
    ----------
    annotation : pd.DataFrame
        Output from getData_multispecies (per-split).
    transcriptToLabel : dict
        Mapping "species---transcript" → (Y_type, Y_idx).
    SL : int
        Core sequence length of each window.
    CL_max : int
        Total context length (extra bases) around the core window.
    shift : int
        Step size between consecutive windows along the transcript.
    include_pos : bool, optional
        If True, DataPointFull.getData will also return absolute positions.

    Returns
    -------
    data : list[DataPointFull]
    """
    data = []
    for idx in range(annotation.shape[0]):
        transcript = annotation['transcript'].values[idx]
        gene = annotation['gene'].values[idx]
        chrom = annotation['chrom'].values[idx]
        strand = annotation['strand'].values[idx]
        tx_start = annotation['tx_start'].values[idx]
        tx_end = annotation['tx_end'].values[idx]

        length = tx_end - tx_start + 1
        # Number of windows needed to cover the full transcript
        num_points = ceil_div(length, shift)

        # If we don't have labels for this transcript, skip it
        if transcript not in transcriptToLabel:
            continue

        Y_type, Y_idx = transcriptToLabel[transcript]
        label = [np.array(Y_type), np.array(Y_idx)]

        for i in range(num_points):
            if strand == '+':
                # On the plus strand, windows slide from tx_start to tx_end
                start, end = tx_start + shift * i, tx_start + SL + shift * i - 1
                if i == 0:
                    # Reference start for this transcript's first window
                    start_point = start

                # Determine which labeled positions fall into the extended
                # context region for this window
                inRange = [
                    l >= start - start_point - CL_max // 2
                    and l <= end - start_point + CL_max // 2
                    for l in label[1]
                ]

                # mask_l: how many positions on the left are outside the transcript
                mask_l = tx_start - np.min([start - CL_max // 2, tx_start])
                # mask_r: how many positions on the right are outside the transcript
                mask_r = np.max([end + CL_max // 2, tx_end]) - tx_end

                data.append(
                    DataPointFull(
                        transcript,
                        gene,
                        chrom,
                        strand,
                        start,
                        end,
                        tx_start,
                        tx_end,
                        label[1][inRange],
                        label[0][inRange],
                        SL,
                        CL_max,
                        start - start_point,
                        mask_l,
                        mask_r,
                        include_pos,
                    )
                )
            else:
                # On the minus strand, windows slide from tx_end backwards
                start, end = tx_end - SL - shift * i + 1, tx_end - shift * i
                if i == 0:
                    # Reference point is the end of the first window
                    start_point = end

                inRange = [
                    l >= start_point - end - CL_max // 2
                    and l <= start_point - start + CL_max // 2
                    for l in label[1]
                ]

                # mask_l/mask_r mirrored relative to plus strand
                mask_l = np.max([end + CL_max // 2, tx_end]) - tx_end
                mask_r = tx_start - np.min([start - CL_max // 2, tx_start])

                data.append(
                    DataPointFull(
                        transcript,
                        gene,
                        chrom,
                        strand,
                        start,
                        end,
                        tx_start,
                        tx_end,
                        label[1][inRange],
                        label[0][inRange],
                        SL,
                        CL_max,
                        start_point - end,
                        mask_l,
                        mask_r,
                        include_pos,
                    )
                )
    return data


class spliceDataset(Dataset):
    """
    PyTorch Dataset wrapper around a list of DataPointFull objects.

    Each __getitem__ call:
      • Calls DataPointFull.getData(seqData) to obtain (X, Y)
      • Optionally applies transform / target_transform
    """

    def __init__(self, annotation, transform=None, target_transform=None):
        """
        Parameters
        ----------
        annotation : list[DataPointFull]
            List produced by getDataPointListFull.
        transform : callable or None
            Optional transform applied to X.
        target_transform : callable or None
            Optional transform applied to Y.
        """
        self.annotation = annotation
        self.transform = transform
        self.target_transform = target_transform
        # seqData must be set externally after construction:
        #   dataset.seqData = seqData_all
        self.seqData: Dict[str, Any] | None = None

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        # Extract raw X, Y from the underlying DataPointFull
        X, Y = self.annotation[idx].getData(self.seqData)
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            Y = self.target_transform(Y)
        return X, Y
