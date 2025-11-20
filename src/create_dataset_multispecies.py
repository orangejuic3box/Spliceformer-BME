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

SPECIES_DIRS = {
    "danio_rerio": "data/danio_rerio",
    #"homo_sapiens": "data/homo_sapiens",
    #"mus_musculus": "data/mus_musculus",
    #"pan_troglodyes": "data/pan_troglodytes",
}

ORTHOLOGY_FILE = "data/orthology.tsv"
OUTPUT_DIR = "data/processed_data"

# Train/val/test split parameters
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

ALLOWED_BIOTYPES = {"protein_coding"}

IN_MAP = np.asarray([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

OUT_MAP = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


def build_species_config():
    species_cfg = {}
    for species_name, folder in SPECIES_DIRS.items():
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Species folder not found: {folder}")

        files = os.listdir(folder)
        gtf_files = [f for f in files if f.endswith(".gtf")]
        fa_files = [f for f in files if f.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz"))]

        if len(gtf_files) != 1:
            raise RuntimeError(
                f"Expected exactly one .gtf in {folder}, found {len(gtf_files)}: {gtf_files}"
            )
        if len(fa_files) != 1:
            raise RuntimeError(
                f"Expected exactly one .fa/.fasta in {folder}, found {len(fa_files)}: {fa_files}"
            )

        species_cfg[species_name] = {
            "gtf": os.path.join(folder, gtf_files[0]),
            "fasta": os.path.join(folder, fa_files[0]),
        }

    return species_cfg


def one_hot_encode(Xd):
    return IN_MAP[Xd.astype('int8')]


def create_datapoints(seq, strand, tx_start, tx_end):
    seq = seq.upper()
    seq = re.sub(r'[^AGTC]', '5', seq)
    seq = seq.replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4')

    X0 = np.asarray([int(x) for x in seq])
    X = one_hot_encode(X0)
    return X


def ceil_div(x, y):
    return int(ceil(float(x) / y))


def getJunctions(gtf, transcript_id):
    transcript = gtf[transcript_id]
    strand = transcript[6]
    exon_junctions = []
    tx_start = int(transcript[3])
    tx_end = int(transcript[4])
    exons = list(gtf.children(transcript, featuretype="exon"))
    if len(exons) == 0:
        # try alternative exon feature names
        exons = list(gtf.children(transcript, featuretype="CDS"))
    for exon in exons:
        exon_start = int(exon[3])
        exon_end = int(exon[4])
        exon_junctions.append((exon_start, exon_end))

    intron_junctions = []

    if strand == '+':
        if len(exon_junctions) > 0:
            intron_start = exon_junctions[0][1]
            for i, exon_junction in enumerate(exon_junctions[1:]):
                intron_end = exon_junction[0]
                intron_junctions.append((intron_start, intron_end))
                if i + 1 != len(exon_junctions[1:]):
                    intron_start = exon_junction[1]

    elif strand == '-':
        exon_junctions.reverse()
        if len(exon_junctions) > 0:
            intron_start = exon_junctions[0][1]
            for i, exon_junction in enumerate(exon_junctions[1:]):
                intron_end = exon_junction[0]
                intron_junctions.append((intron_start, intron_end))
                if i + 1 != len(exon_junctions[1:]):
                    intron_start = exon_junction[1]

    jn_start = [x[0] for x in intron_junctions]
    jn_end = [x[1] for x in intron_junctions]
    Y_type, Y_idx = [], []
    if strand == '+':
        Y0 = -np.ones(tx_end - tx_start + 1)
        if len(jn_start) > 0:
            Y0 = np.zeros(tx_end - tx_start + 1)
            for c in jn_start:
                if tx_start <= c <= tx_end:
                    Y_type.append(2)
                    Y_idx.append(c - tx_start)
            for c in jn_end:
                if tx_start <= c <= tx_end:
                    Y_type.append(1)
                    Y_idx.append(c - tx_start)

    elif strand == '-':
        Y0 = -np.ones(tx_end - tx_start + 1)

        if len(jn_start) > 0:
            Y0 = np.zeros(tx_end - tx_start + 1)
            for c in jn_end:
                if tx_start <= c <= tx_end:
                    Y_type.append(2)
                    Y_idx.append(tx_end - c)
            for c in jn_start:
                if tx_start <= c <= tx_end:
                    Y_type.append(1)
                    Y_idx.append(tx_end - c)

    return jn_start, jn_end, Y_type, Y_idx


def load_orthology_table(path):
    # Expecting tab-separated: species \t gene_id \t orthogroup
    orthology = defaultdict(list)
    gene_to_group = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('species') or line.lower().startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Orthology file line malformed: {line}")
            species, gene_id, group = parts[0], parts[1], parts[2]
            orthology[group].append((species, gene_id))
            gene_to_group[(species, gene_id)] = group
    return orthology, gene_to_group


def split_groups(orthology_groups, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    groups = list(orthology_groups)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = set(groups[:n_train])
    val = set(groups[n_train:n_train + n_val])
    test = set(groups[n_train + n_val:])
    return train, val, test


def createDataset_for_species(
    setType,
    species_name,
    gtf_db,
    fasta,
    gene_split_map,
    data_dir,
    allowed_biotypes=None,
):
    print(f"Building dataset for {species_name} -> split {setType}")

    genes = list(gtf_db.features_of_type("gene"))

    CHROM_GROUP = list(fasta.keys())

    CANONICAL_CHROMS = {
        "homo_sapiens": {str(i) for i in range(1, 23)} | {"X", "Y"},
        "mus_musculus": {str(i) for i in range(1, 20)} | {"X", "Y"},
        "danio_rerio": {str(i) for i in range(1, 26)},  # 1–25
        "pan_troglodyes": {str(i) for i in range(1, 23)} | {"X", "Y"},
    }
    allowed_chroms = CANONICAL_CHROMS.get(species_name, None)

    out_seq_dir = os.path.join(data_dir, "sparse_sequence_data")
    os.makedirs(out_seq_dir, exist_ok=True)

    total_genes_seen = 0
    genes_in_split = 0
    genes_with_transcripts = 0
    genes_with_good_transcript = 0

    total_transcripts_seen = 0
    transcripts_passing_biotype_tsl = 0
    transcripts_with_introns = 0

    seqData = {}
    transcriptToLabel = {}

    prev_chrom = None

    for gene in tqdm(genes):
        total_genes_seen += 1

        chrom = gene[0]

        if chrom in CHROM_GROUP:
            chrom_key = chrom
        elif chrom.startswith("chr") and chrom[3:] in CHROM_GROUP:
            chrom_key = chrom[3:]
        elif ("chr" + chrom) in CHROM_GROUP:
            chrom_key = "chr" + chrom
        else:
            # skip contigs we don't have sequence for
            continue

        if allowed_chroms is not None:
            clean = chrom_key[3:] if chrom_key.startswith("chr") else chrom_key
            if clean not in allowed_chroms:
                continue

        gene_id = gene.attributes.get("gene_id", ["NA"])[0]

        grp = gene_split_map.get((species_name, gene_id), None)
        if grp is None:
            continue
        if grp != setType:
            continue

        genes_in_split += 1

        strand = gene[6]
        gene_start = int(gene[3])
        gene_end = int(gene[4])

        transcripts = list(gtf_db.children(gene, featuretype="transcript"))
        if len(transcripts) == 0:
            # try alternative transcript feature names
            transcripts = list(
                gtf_db.children(gene, featuretype=("transcript", "mRNA"))
            )
        if len(transcripts) == 0:
            continue

        genes_with_transcripts += 1

        good_transcripts = []

        for transcript in transcripts:
            total_transcripts_seen += 1

            transcript_id = transcript.attributes.get("transcript_id", ["NA"])[0]
            transcript_biotype = transcript.attributes.get(
                "transcript_biotype", ["NA"]
            )[0]

            # Biotype filter
            if allowed_biotypes and transcript_biotype not in allowed_biotypes:
                continue

            tsl = transcript.attributes.get("transcript_support_level", [""])[0].strip()
            if tsl and tsl.split()[0] != "1":
                continue
            transcripts_passing_biotype_tsl += 1

            # Junctions and intron filter
            jn_start, jn_end, Y_type, Y_idx = getJunctions(gtf_db, transcript_id)
            if len(jn_start) == 0:
                continue
            transcripts_with_introns += 1

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
            # No transcript under this gene passed filters
            continue

        genes_with_good_transcript += 1

        good_transcripts.sort(key=lambda d: d["n_junctions"], reverse=True)
        best = good_transcripts[0]

        transcript = best["transcript"]
        transcript_id = best["transcript_id"]
        jn_start = best["jn_start"]
        jn_end = best["jn_end"]
        Y_type = best["Y_type"]
        Y_idx = best["Y_idx"]

        tx_start = int(transcript[3])
        tx_end = int(transcript[4])

        if prev_chrom is None:
            prev_chrom = chrom_key
        elif chrom_key != prev_chrom:
            mat = seqData.get(prev_chrom, None)
            if mat is not None and mat.nnz > 0:
                try:
                    save_npz(
                        os.path.join(
                            out_seq_dir, f"{species_name}_{prev_chrom}_{setType}.npz"
                        ),
                        mat.tocoo(),
                    )
                except Exception as e:
                    tqdm.write(
                        f"Failed saving npz for {species_name} {prev_chrom}: {e}"
                    )
            if prev_chrom in seqData:
                del seqData[prev_chrom]
            prev_chrom = chrom_key

        if chrom_key not in seqData:
            try:
                L = len(fasta[chrom_key])
            except Exception:
                # cannot access sequence for this contig
                continue
            seqData[chrom_key] = dok_matrix((L, 5), dtype=np.int8)

        try:
            seq = fasta[chrom_key][gene_start - 1 : gene_end].seq
        except Exception as e:
            tqdm.write(
                f"Failed reading fasta for {species_name}:{chrom_key}:{gene_start}-{gene_end} -> {e}"
            )
            continue

        X = create_datapoints(seq, strand, gene_start, gene_end)
        seqData[chrom_key][gene_start - 1 : gene_end, :] = X


        # Map from "{species}---{transcript_id}" -> (Y_type, Y_idx)
        tx_key = f"{species_name}---{transcript_id}"
        transcriptToLabel[tx_key] = (Y_type, Y_idx)

        # Annotation line: "species---gene---transcript"
        name = f"{species_name}---{gene_id}---{transcript_id}"

        out_ann_dir = os.path.join(data_dir, "annotations")
        os.makedirs(out_ann_dir, exist_ok=True)
        ann_file = os.path.join(
            out_ann_dir, f"annotation_{species_name}_{setType}.txt"
        )

        with open(ann_file, "a") as the_file:
            the_file.write(
                f"{name}\t{chrom_key}\t{strand}\t{tx_start}\t{tx_end}\n"
            )

    for chrom, mat in list(seqData.items()):
        if mat is None:
            continue
        try:
            if hasattr(mat, "tocoo") and mat.nnz > 0:
                save_npz(
                    os.path.join(out_seq_dir, f"{species_name}_{chrom}_{setType}.npz"),
                    mat.tocoo(),
                )
        except Exception as e:
            tqdm.write(f"Failed saving npz for {species_name} {chrom}: {e}")

    with open(
        os.path.join(
            data_dir, f"sparse_discrete_label_data_{species_name}_{setType}.pickle"
        ),
        "wb",
    ) as handle:
        pickle.dump(transcriptToLabel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Sanity logs
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


def main():
    species_cfg = build_species_config()

    # Load orthology table
    orthology, gene_to_group = load_orthology_table(ORTHOLOGY_FILE)

    # Split orthology groups into train/val/test sets
    train_groups, val_groups, test_groups = split_groups(
        list(orthology.keys()),
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=SEED
    )

    # Map each (species, gene_id) -> which split it belongs to ("train"/"val"/"test")
    gene_split_map = {}
    for grp, members in orthology.items():
        if grp in train_groups:
            split = 'train'
        elif grp in val_groups:
            split = 'val'
        else:
            split = 'test'
        for species, gene_id in members:
            gene_split_map[(species, gene_id)] = split

    # Build datasets per species and per split
    splits = ['train', 'val', 'test']
    for species_name, paths in species_cfg.items():
        gtf_path = paths['gtf']
        fasta_path = paths['fasta']

        print(f"Preparing {species_name}: gtf={gtf_path} fasta={fasta_path}")

        # Build gffutils DB if needed
        db_path = gtf_path + '.db'
        if not os.path.exists(db_path):
            print(f"Creating gffutils DB for {gtf_path} -> {db_path}")
            gffutils.create_db(
                gtf_path,
                dbfn=db_path,
                force=True,
                merge_strategy='merge',
                verbose=False,
                disable_infer_genes=True,
                disable_infer_transcripts=True,
            )

        gtf_db = gffutils.FeatureDB(db_path)

        # Load FASTA via pyfastx
        fasta = pyfastx.Fasta(fasta_path)

        # Call createDataset_for_species once per split
        for split in splits:
            createDataset_for_species(
                split,
                species_name,
                paths,
                gtf_db,
                fasta,
                {k: v for k, v in gene_split_map.items()},
                OUTPUT_DIR,
                allowed_biotypes=ALLOWED_BIOTYPES
            )

    print('All species processed.')

if __name__ == '__main__':
    main()
