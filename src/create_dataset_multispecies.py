"""create_dataset_multispecies.py

This script builds multi-species splice-site prediction datasets.
It:
  • Parses GTF annotations via gffutils
  • Loads genome sequences via pyfastx
  • Defines a human-first orthology-based train/val/test split
  • Extracts intron junctions per transcript and encodes them as sparse labels
  • Builds per-chromosome sparse, one-hot-like sequence matrices
"""

import os
import random
from collections import defaultdict
import numpy as np
from math import ceil
import pickle
from tqdm import tqdm
import re
import pyfastx
import gffutils
from scipy.sparse import dok_matrix, save_npz
import argparse

# Root directories containing genome + annotation files for each species.
# Each directory is expected to contain exactly one .gtf and exactly one
# FASTA (.fa or .fasta) file.
SPECIES_DIRS = {
    "danio_rerio": "data/danio_rerio",
    "mus_musculus": "data/mus_musculus",
    "pan_troglodytes": "data/pan_troglodytes",
    "homo_sapiens": "data/homo_sapiens",
    "sus_scrofa": "data/sus_scrofa",
    "c_elegans": "data/c_elegans",
}

# Unused in the rest of the script, but conceptually maps canonical
# chromosome names like "1" → "chr1".
CHROM_GROUP = {
    "1": "chr1", "2": "chr2", "3": "chr3", "4": "chr4",
    "5": "chr5", "6": "chr6", "7": "chr7", "8": "chr8",
    "9": "chr9", "10": "chr10", "11": "chr11", "12": "chr12",
    "13": "chr13", "14": "chr14", "15": "chr15", "16": "chr16",
    "17": "chr17", "18": "chr18", "19": "chr19", "20": "chr20",
    "21": "chr21", "22": "chr22", "X": "chrX", "Y": "chrY",
}

# Orthology table used to define cross-species gene groups
ORTHOLOGY_FILE = "data/orthology.tsv"

# Root directory for all processed outputs
OUTPUT_DIR = "data/processed_data"

# Train/val/test split parameters for orthogroups and human singletons
SEED = 13
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Restrict to specific transcript/gene biotypes
ALLOWED_BIOTYPES = {"protein_coding"}

# Integer encoding → one-hot for nucleotides (A,C,G,T,N/other)
# These are not used directly in the final sparse sequence matrices,
# but document the intended encoding: 0 = unknown; 1–4 = A,C,G,T.
IN_MAP = np.asarray([
    [0, 0, 0, 0, 0],  # padding / N
    [1, 0, 0, 0, 0],  # A
    [0, 1, 0, 0, 0],  # C
    [0, 0, 1, 0, 0],  # G
    [0, 0, 0, 1, 0],  # T
    [0, 0, 0, 0, 1],  # extra / unused
])

# Mapping from label index → 3-class target vector.
# (Used elsewhere in the codebase; here it documents that
#  we model donors, acceptors, and non-splice positions.)
OUT_MAP = np.asarray([
    [1, 0, 0],  # donor
    [0, 1, 0],  # acceptor
    [0, 0, 1],  # other splice label (if present)
    [0, 0, 0],  # background
])


def parse_args():
    """Parse command-line arguments.

    --species: one or more species labels (keys of SPECIES_DIRS), or
               the special value "all" (default) to process every species.
    """

    parser = argparse.ArgumentParser(
        description="Create splice site datasets for one or more species."
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            "Species to process. Use labels from SPECIES_DIRS "
            "(e.g. homo_sapiens mus_musculus pan_troglodytes danio_rerio), "
            "or 'all' (default) to process every species."
        ),
    )
    return parser.parse_args()


def build_human_first_gene_split_map(
    species_cfg,
    orthology,
    gene_to_group,
    train_frac,
    val_frac,
    test_frac,
    seed=13,
    human_species="homo_sapiens",
):
    """Create a mapping (species, gene_id) → split (train/val/test).

    Strategy: "human-first" split.
    --------------------------------
    1. Load the human GTF database and enumerate all human gene IDs.
    2. For each orthology group that contains at least one human gene,
       create a group entry ("orthogroup", group_id).
    3. For any human gene that is *not* in the orthology table, create a
       singleton group ("human_singleton", gene_id).
    4. Shuffle all groups with a fixed seed and assign them to
       train/val/test according to the requested fractions.
    5. Every gene in an orthogroup inherits the split of its group.
       Human singletons inherit the split of their singleton entry.

    Non-human genes that lack a human ortholog will *not* appear in the
    resulting mapping; downstream code treats them as test-only (if at all).
    """

    if human_species not in species_cfg:
        raise ValueError(f"Expected {human_species} in species_cfg")

    human_gtf_path = species_cfg[human_species]["gtf"]
    human_db_path = human_gtf_path + ".db"

    # The human GTF DB (created with gffutils.create_db) must exist beforehand
    if not os.path.exists(human_db_path):
        raise FileNotFoundError(
            f"Human GTF DB not found at {human_db_path}. "
            f"Please prebuild it with gffutils.create_db before running "
            f"create_dataset_multispecies.py."
        )

    # Open the human annotation database
    human_db = gffutils.FeatureDB(human_db_path, keep_order=True)

    # Collect all human gene IDs present in the annotation
    human_gene_ids = set()
    for gene in human_db.features_of_type("gene"):
        gid = gene.attributes.get("gene_id", ["NA"])[0]
        human_gene_ids.add(gid)

    # Identify orthology groups that contain at least one of these genes
    groups_with_human = set()
    for (species, gene_id), group_id in gene_to_group.items():
        if species == human_species and gene_id in human_gene_ids:
            groups_with_human.add(group_id)

    # Collect all groups: human-containing orthogroups + human singletons
    group_list = []

    # 1) orthogroups
    for group_id in groups_with_human:
        group_list.append(("orthogroup", group_id))

    # 2) human-only genes
    human_genes_in_orthology = {
        gene_id for (sp, gene_id) in gene_to_group.keys() if sp == human_species
    }
    human_only_genes = sorted(human_gene_ids - human_genes_in_orthology)
    for gid in human_only_genes:
        group_list.append(("human_singleton", gid))

    # Reproducibly shuffle groups and assign splits
    rng = random.Random(seed)
    rng.shuffle(group_list)

    total_groups = len(group_list)
    n_train = int(round(total_groups * train_frac))
    n_val = int(round(total_groups * val_frac))
    n_test = total_groups - n_train - n_val

    group_to_split = {}
    idx = 0

    for _ in range(n_train):
        group_to_split[group_list[idx]] = "train"
        idx += 1
    for _ in range(n_val):
        group_to_split[group_list[idx]] = "val"
        idx += 1
    for _ in range(n_test):
        group_to_split[group_list[idx]] = "test"
        idx += 1

    # Build final mapping (species, gene_id) → split
    gene_split_map = {}

    # Assign orthogroup members
    for (species, gene_id), group_id in gene_to_group.items():
        if group_id not in groups_with_human:
            continue
        split = group_to_split.get(("orthogroup", group_id))
        if split is None:
            continue
        gene_split_map[(species, gene_id)] = split

    # Assign human-only genes
    for gid in human_only_genes:
        split = group_to_split.get(("human_singleton", gid))
        if split is None:
            continue
        gene_split_map[(human_species, gid)] = split

    return gene_split_map


def build_species_config():
    """Locate each species' GTF and FASTA files.

    For every species directory in SPECIES_DIRS, enforce that there is
    exactly one GTF and exactly one FASTA file, then store their full
    paths in a dictionary:

        {species: {"gtf": <path>, "fasta": <path>}}
    """

    species_cfg = {}
    for species, folder in SPECIES_DIRS.items():
        # Find GTF file
        gtf_files = [f for f in os.listdir(folder) if f.endswith(".gtf")]
        if len(gtf_files) != 1:
            raise ValueError(
                f"Expected exactly one .gtf in {folder} for species {species}, "
                f"found {len(gtf_files)}: {gtf_files}"
            )
        gtf_path = os.path.join(folder, gtf_files[0])

        # Find FASTA file
        fasta_files = [
            f for f in os.listdir(folder)
            if f.endswith(".fa") or f.endswith(".fasta")
        ]
        if len(fasta_files) != 1:
            raise ValueError(
                f"Expected exactly one FASTA in {folder} for species {species}, "
                f"found {len(fasta_files)}: {fasta_files}"
            )
        fasta_path = os.path.join(folder, fasta_files[0])

        species_cfg[species] = {"gtf": gtf_path, "fasta": fasta_path}

    return species_cfg


def load_orthology_table(orthology_file):
    """Load a tab-separated orthology table.

    The expected format is at least three columns per line:
        species \t gene_id \t group_id

    Returns
    -------
    orthology : list of (species, gene_id, group_id)
        Raw list of entries.
    gene_to_group : dict
        Mapping (species, gene_id) → group_id for quick lookup.
    """

    orthology = []
    gene_to_group = {}

    with open(orthology_file) as f:
        header = f.readline()  # ignore header line
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 3:
                continue
            species, gene_id, group_id = fields[0], fields[1], fields[2]
            orthology.append((species, gene_id, group_id))
            gene_to_group[(species, gene_id)] = group_id

    return orthology, gene_to_group


def onehot_seq(seq):
    """Convert a nucleotide string into integer codes.

    Mapping:
        A/a → 1, C/c → 2, G/g → 3, T/t → 4, anything else → 0.

    The returned list can be further transformed into one-hot encodings
    using IN_MAP, or used directly as integer-coded sequence.
    """

    mapping = {
        "A": 1, "C": 2, "G": 3, "T": 4,
        "a": 1, "c": 2, "g": 3, "t": 4,
    }
    return [mapping.get(n, 0) for n in seq]

def createDataset_for_species(
    setType,
    species_name,
    gtf_db,
    fasta,
    gene_split_map,
    data_dir,
    allowed_biotypes=None,
):
    """Build dataset for one species and one split (train/val/test).

    For all genes in the GTF DB that:
      • belong to the requested split (based on gene_split_map), and
      • lie on an allowed chromosome, and
      • pass species-specific subsampling,
    we then:

      1. Collect transcripts for that gene.
      2. Apply transcript-level filters:
           - biotype in `allowed_biotypes` (if provided)
           - transcript_support_level (TSL) rules by species
           - at least one intron (≥ 2 exons)
      3. For each passing transcript, compute splice junction coordinates
         and convert them into label arrays (Y_type, Y_idx) in transcript
         coordinate space.
      4. Select species-specific top-K transcripts (by number of junctions).
      5. Write transcript metadata to an annotation file and store labels in
         a dictionary that will be pickled.
      6. After iterating all genes, build sparse sequence matrices for each
         chromosome used in this split, directly from the FASTA.

    Parameters
    ----------
    setType : {"train", "val", "test"}
        Which split to build.
    species_name : str
        Species key (e.g. "homo_sapiens").
    gtf_db : gffutils.FeatureDB
        Annotation database for this species.
    fasta : pyfastx.Fasta
        FASTA reader for genome sequences.
    gene_split_map : dict
        Mapping (species, gene_id) → split label.
    data_dir : str
        Root output directory (will contain annotations/, sparse_sequence_data/, etc.).
    allowed_biotypes : set of str or None
        Restrict transcripts to these biotypes if not None.
    """

    print(f"Building dataset for {species_name} -> split {setType}")

    # Normalize FASTA chromosome keys to a set for quick membership tests.
    try:
        fasta_chroms = set(fasta.keys())
    except TypeError:
        # Some pyfastx versions require explicit materialization
        fasta_chroms = set(list(fasta.keys()))

    # How many transcripts per gene to keep (top-K by number of junctions).
    # `None` means keep all transcripts that pass filters.
    species_top_k = {
        "homo_sapiens": None,
        "mus_musculus": None,
        "pan_troglodytes": None,
        "danio_rerio": None,
        "sus_scrofa": 1,
        "c_elegans": 1,
    }

    # Fraction of genes to keep per species (subsampling to control size).
    # Human is fully kept; non-human species are downsampled.
    gene_keep_frac = {
        "homo_sapiens": 1.0,
        "mus_musculus": 0.5,
        "pan_troglodytes": 0.5,
        "danio_rerio": 0.5,
        "sus_scrofa": 0.5,
        "c_elegans": 0.5,
    }

    # Prepare output directories (shared across species/splits)
    os.makedirs(data_dir, exist_ok=True)
    out_ann_dir = os.path.join(data_dir, "annotations")
    os.makedirs(out_ann_dir, exist_ok=True)
    seq_dir = os.path.join(data_dir, "sparse_sequence_data")
    os.makedirs(seq_dir, exist_ok=True)

    # Annotation file for this species + split.
    # Each line: name\tchromosome\tstrand\ttx_start\ttx_end
    ann_path = os.path.join(
        out_ann_dir, f"annotation_{species_name}_{setType}.txt"
    )
    # Append mode allows successive calls to extend the same file if needed.
    ann_f = open(ann_path, "a")

    # Map from transcript name → (Y_type, Y_idx) label arrays.
    # "name" is a composite key: "species---gene---transcript".
    transcriptToLabel = {}

    # Counters for logging and sanity checks
    total_genes_seen = 0
    genes_in_split = 0
    genes_with_transcripts = 0
    genes_with_good_transcript = 0

    total_transcripts_seen = 0
    transcripts_passing_biotype_tsl = 0
    transcripts_with_introns = 0

    # Track chromosomes that actually appear in this split so we can
    # build sparse sequence matrices only where needed.
    chroms_in_split = set()

    # Allowed chromosomes (human-like) and C. elegans-style.
    allowed_chroms = {
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
        "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "X", "Y",
    }
    allowed_chroms_ce = {"I", "II", "III", "IV", "V", "X"}
    allowed_chroms_all = allowed_chroms | allowed_chroms_ce

    # Count total genes for the progress bar.
    total_gene_count = 0
    for _ in gtf_db.features_of_type("gene"):
        total_gene_count += 1

    # Main gene loop
    for gene in tqdm(
        gtf_db.features_of_type("gene"),
        total=total_gene_count,
        desc=f"{species_name} – {setType}",
    ):
        total_genes_seen += 1

        raw_chrom = str(gene.chrom)

        # Harmonize gene.chrom with FASTA chromosome keys. This handles
        # cases where one uses "1" vs "chr1", etc.
        if raw_chrom in fasta_chroms:
            chrom_key = raw_chrom
        elif raw_chrom.startswith("chr") and raw_chrom[3:] in fasta_chroms:
            chrom_key = raw_chrom[3:]
        elif not raw_chrom.startswith("chr") and ("chr" + raw_chrom) in fasta_chroms:
            chrom_key = "chr" + raw_chrom
        else:
            # No matching chromosome in FASTA → skip this gene entirely.
            continue

        # Strip optional "chr" prefix before checking against allowed
        # chromosome sets (e.g., "chr1" → "1").
        clean = chrom_key[3:] if chrom_key.startswith("chr") else chrom_key
        if clean not in allowed_chroms_all:
            continue

        gene_id = gene.attributes.get("gene_id", ["NA"])[0]

        # Determine if this species has explicit split info.
        # If yes, we enforce split membership for each gene.
        has_split_info_for_species = any(
            s == species_name for (s, _gid) in gene_split_map.keys()
        )

        if has_split_info_for_species:
            # Genes must appear in gene_split_map and match the requested split.
            grp = gene_split_map.get((species_name, gene_id), None)
            if grp is None or grp != setType:
                continue
        else:
            # For species with no explicit split data (e.g., not present in
            # the orthology table), genes are only used in the test split.
            if setType != "test":
                continue

        # Species-specific random gene downsampling.
        keep_frac = gene_keep_frac.get(species_name, 1.0)
        if keep_frac < 1.0 and random.random() > keep_frac:
            continue

        genes_in_split += 1

        # Record that this chromosome participates in the split
        chroms_in_split.add(chrom_key)

        # Strand of the gene: '+' or '-'.
        strand = gene[6]

        # Retrieve all transcripts associated with this gene.
        transcripts = list(gtf_db.children(gene, featuretype="transcript"))
        if len(transcripts) == 0:
            # Gene with no transcript features is not useful
            continue

        genes_with_transcripts += 1

        good_transcripts = []  # transcripts that pass all filters

        # Transcript-level filters and label creation
        for transcript in transcripts:
            total_transcripts_seen += 1

            transcript_id = transcript.attributes.get(
                "transcript_id", ["NA"]
            )[0]
            transcript_biotype = transcript.attributes.get(
                "transcript_biotype", ["NA"]
            )[0]

            # Optional filter on transcript biotype
            if allowed_biotypes and transcript_biotype not in allowed_biotypes:
                continue

            # Transcript support level (TSL): filter low-confidence
            # transcripts differently per species.
            tsl = transcript.attributes.get(
                "transcript_support_level", [""]
            )[0].strip()

            if species_name == "homo_sapiens":
                # Require TSL and only accept the best-supported level (1)
                if not tsl:
                    continue
                if tsl.split()[0] != "1":
                    continue

            elif species_name in ("mus_musculus", "pan_troglodytes"):
                # If TSL is annotated, require it to be 1; otherwise allow.
                if tsl and tsl.split()[0] != "1":
                    continue

            elif species_name == "danio_rerio":
                # No explicit TSL filtering here, just pass.
                pass

            else:
                # For other species, if TSL exists and is not 1, drop.
                if tsl and tsl.split()[0] != "1":
                    continue

            transcripts_passing_biotype_tsl += 1

            # Get exons, ordered by genomic start coordinate.
            exons = list(
                gtf_db.children(transcript, featuretype="exon", order_by="start")
            )
            if len(exons) < 2:
                # Need at least two exons to define an intron junction.
                continue

            # We'll compute donor and acceptor genomic coordinates
            # from adjacent exon pairs.
            jn_start = []  # donors
            jn_end = []    # acceptors

            for e1, e2 in zip(exons[:-1], exons[1:]):
                donor = int(e1.end)       # 5' splice site (end of upstream exon)
                acceptor = int(e2.start)  # 3' splice site (start of downstream exon)

                jn_start.append(donor)
                jn_end.append(acceptor)

            transcripts_with_introns += 1

            # Transcript genomic bounds
            tx_start = int(transcript[3])
            tx_end = int(transcript[4])

            # Convert genomic junction coordinates into transcript-relative
            # positions (0-based) depending on strand.
            jn_start_arr = np.array(jn_start, dtype=np.int32)
            jn_end_arr = np.array(jn_end, dtype=np.int32)

            if strand == "+":
                donor_pos = jn_start_arr - tx_start      # donors (5' sites)
                acceptor_pos = jn_end_arr - tx_start     # acceptors (3' sites)
            else:
                # On the minus strand, direction is reversed.
                donor_pos = tx_end - jn_end_arr          # donors
                acceptor_pos = tx_end - jn_start_arr     # acceptors

            # Build label arrays:
            #   Y_type: type of event (2=donor, 1=acceptor)
            #   Y_idx : position in transcript coordinates
            Y_type = np.concatenate([
                2 * np.ones_like(donor_pos, dtype=np.int8),     # donor labels
                1 * np.ones_like(acceptor_pos, dtype=np.int8),  # acceptor labels
            ])
            Y_idx = np.concatenate([donor_pos, acceptor_pos]).astype(np.int32)

            # Package transcript info for later ranking and writing
            good_transcripts.append(
                {
                    "transcript": transcript,
                    "transcript_id": transcript_id,
                    "jn_start": jn_start,
                    "jn_end": jn_end,
                    "Y_type": Y_type,
                    "Y_idx": Y_idx,
                    "n_junctions": len(jn_start),
                }
            )

        if not good_transcripts:
            # No transcript passed filtering for this gene.
            continue

        genes_with_good_transcript += 1

        # Select top-K transcripts per gene based on number of junctions.
        top_k_transcripts = species_top_k.get(species_name, None)

        good_transcripts.sort(key=lambda d: d["n_junctions"], reverse=True)

        if top_k_transcripts is None:
            selected_transcripts = good_transcripts
        else:
            selected_transcripts = good_transcripts[:top_k_transcripts]

        # Write out annotations + labels for each selected transcript.
        for best in selected_transcripts:
            transcript = best["transcript"]
            transcript_id = best["transcript_id"]
            Y_type = best["Y_type"]
            Y_idx = best["Y_idx"]

            tx_start = int(transcript[3])
            tx_end = int(transcript[4])

            # Unique name tying species, gene, and transcript together.
            name = f"{species_name}---{gene_id}---{transcript_id}"
            transcriptToLabel[name] = (Y_type, Y_idx)

            # Annotation line: transcript name, chromosome, strand, start, end.
            ann_f.write(
                f"{name}\t{chrom_key}\t{strand}\t{tx_start}\t{tx_end}\n"
            )

    # At this point, we've processed all genes. Next, build sparse sequence
    # matrices for each chromosome that appeared in this split.
    from scipy.sparse import csr_matrix

    for chrom_key in sorted(chroms_in_split):
        print(
            f"[INFO] Building sequence matrix for {species_name} {setType} chrom {chrom_key} ...",
            flush=True,
        )

        # Extract full chromosome sequence as an uppercase ASCII string.
        seq = fasta[chrom_key].seq.upper()
        chrom_len = len(seq)

        # Vectorized mapping of nucleotides to integer codes.
        # We avoid Python-level loops over bases for speed.
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype="S1")

        codes = np.zeros(chrom_len, dtype=np.uint8)
        codes[seq_bytes == b"A"] = 1
        codes[seq_bytes == b"C"] = 2
        codes[seq_bytes == b"G"] = 3
        codes[seq_bytes == b"T"] = 4

        # We only keep positions that are real bases (A/C/G/T).
        mask = codes != 0
        if not np.any(mask):
            # No valid bases to store for this chromosome.
            continue

        # Row indices are genomic positions; column indices are base codes
        # (1..4), and the data is just 1. This effectively encodes
        # a very sparse (chrom_len × 5) one-hot-like matrix.
        rows = np.nonzero(mask)[0].astype(np.int32)
        cols = codes[mask].astype(np.int32)
        data = np.ones_like(rows, dtype=np.uint8)

        mat = csr_matrix(
            (data, (rows, cols)),
            shape=(chrom_len, 5),
            dtype=np.uint8,
        )

        out_path = os.path.join(
            seq_dir, f"{species_name}_{chrom_key}_{setType}.npz"
        )
        save_npz(out_path, mat)
        print(
            f"[INFO] Saved {out_path} (shape={mat.shape}, nnz={mat.nnz})",
            flush=True,
        )

    # Close annotation file now that we're done with all transcripts.
    ann_f.close()

    # Persist transcript labels as a pickle for fast loading at training time.
    with open(
        os.path.join(
            data_dir, f"sparse_discrete_label_data_{species_name}_{setType}.pickle"
        ),
        "wb",
    ) as handle:
        pickle.dump(transcriptToLabel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary stats to stdout (via tqdm.write for compatibility with progress bar)
    tqdm.write(
        f"[{species_name} – {setType}] "
        f"genes_seen={total_genes_seen}, "
        f"genes_in_split={genes_in_split}, "
        f"genes_with_transcripts={genes_with_transcripts}, "
        f"genes_with_good_transcript={genes_with_good_transcript}"
    )
    tqdm.write(
        f"[{species_name} – {setType}] "
        f"transcripts_seen={total_transcripts_seen}, "
        f"transcripts_passing_biotype_tsl={transcripts_passing_biotype_tsl}, "
        f"transcripts_with_introns={transcripts_with_introns}"
    )

    print(
        f"Finished building {species_name} {setType}: "
        f"{len(transcriptToLabel)} transcripts"
    )

    # Also write a plain-text summary log for later inspection.
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{species_name}_{setType}_summary.txt")

    try:
        with open(log_path, "w") as lf:
            lf.write(f"species: {species_name}\n")
            lf.write(f"split: {setType}\n")
            lf.write(f"genes_seen: {total_genes_seen}\n")
            lf.write(f"genes_in_split: {genes_in_split}\n")
            lf.write(f"genes_with_transcripts: {genes_with_transcripts}\n")
            lf.write(
                f"genes_with_good_transcript: {genes_with_good_transcript}\n"
            )
            lf.write(f"transcripts_seen: {total_transcripts_seen}\n")
            lf.write(
                f"transcripts_passing_biotype_tsl: "
                f"{transcripts_passing_biotype_tsl}\n"
            )
            lf.write(
                f"transcripts_with_introns: {transcripts_with_introns}\n"
            )
            lf.write(
                f"num_transcripts_written: {len(transcriptToLabel)}\n"
            )
    except Exception as e:
        tqdm.write(
            f"Warning: failed to write summary log for "
            f"{species_name} {setType} -> {e}"
        )


def main():
    """Entry point for the CLI script.

    High-level flow:
      1. Parse arguments (which species to process).
      2. Build species configuration (paths to GTF + FASTA per species).
      3. Load orthology relationships from ORTHOLOGY_FILE.
      4. Construct `gene_split_map` using the human-first strategy.
      5. For each species and each split, avoid recomputing work if
         all required output files already exist; otherwise call
         `createDataset_for_species`.
    """

    args = parse_args()
    species_cfg = build_species_config()

    # Load orthology table (species, gene_id → group_id)
    orthology, gene_to_group = load_orthology_table(ORTHOLOGY_FILE)

    # Build a gene-level split map such that:
    #   • All human genes are in exactly one of train/val/test.
    #   • Only non-human genes with human orthologs are included.
    gene_split_map = build_human_first_gene_split_map(
        species_cfg,
        orthology,
        gene_to_group,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=SEED,
        human_species="homo_sapiens",
    )

    # Determine which species to process.
    requested = args.species
    if "all" in requested:
        target_species = list(species_cfg.keys())
    else:
        unknown = [s for s in requested if s not in species_cfg]
        if unknown:
            raise ValueError(
                f"Unknown species requested: {unknown}. "
                f"Valid options are: {list(species_cfg.keys())}"
            )
        target_species = requested

    # We always build these three splits.
    splits = ["train", "val", "test"]

    # Process each species in turn.
    for species_name in target_species:
        paths = species_cfg[species_name]
        gtf_path = paths["gtf"]
        fasta_path = paths["fasta"]

        print(f"Preparing {species_name}: gtf={gtf_path} fasta={fasta_path}")

        # GTF DBs for all species must be prebuilt with gffutils.create_db.
        gtf_db_path = gtf_path + ".db"
        if not os.path.exists(gtf_db_path):
            raise FileNotFoundError(
                f"GTF DB for {species_name} not found at {gtf_db_path}. "
                f"Please prebuild it with gffutils.create_db before running "
                f"create_dataset_multispecies.py."
            )

        # Open the gffutils database and the FASTA for this species.
        gtf_db = gffutils.FeatureDB(gtf_db_path, keep_order=True)
        fasta = pyfastx.Fasta(fasta_path)

        for split in splits:
            out_ann_dir = os.path.join(OUTPUT_DIR, "annotations")
            ann_file = os.path.join(
                out_ann_dir, f"annotation_{species_name}_{split}.txt"
            )
            label_file = os.path.join(
                OUTPUT_DIR,
                f"sparse_discrete_label_data_{species_name}_{split}.pickle",
            )
            seq_dir = os.path.join(OUTPUT_DIR, "sparse_sequence_data")

            # Detect existing per-chromosome sequence npz files for this
            # species and split.
            chrom_npzs = []
            if os.path.isdir(seq_dir):
                chrom_npzs = [
                    f
                    for f in os.listdir(seq_dir)
                    if f.startswith(f"{species_name}_")
                    and f.endswith(f"_{split}.npz")
                ]

            # If annotation file, label pickle, and at least one chromosome
            # npz already exist, we consider this (species, split) done.
            if (
                os.path.exists(ann_file)
                and os.path.exists(label_file)
                and len(chrom_npzs) > 0
            ):
                print(f"[SKIP] {species_name} {split} already completed.")
                continue

            print(f"[RUN]  Building {species_name} {split} ...")

            createDataset_for_species(
                split,
                species_name,
                gtf_db,
                fasta,
                gene_split_map,
                OUTPUT_DIR,
                allowed_biotypes=ALLOWED_BIOTYPES,
            )

    print("All species processed.")


if __name__ == "__main__":
    main()