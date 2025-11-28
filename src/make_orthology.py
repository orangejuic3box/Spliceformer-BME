"""
make_orthology.py

Build an orthology table (orthology.tsv) for a set of human genes using
Ensembl Compara via the REST API.

Pipeline
--------
1. Find the human GTF (homo_sapiens/*.gtf or *.gtf.gz).
2. Select human genes with at least one "good" transcript:
     - protein_coding
     - transcript_support_level (TSL) == 1
     - at least 2 exons (≥1 splice junction)
3. Cache selected human genes + transcripts in text files so the
   selection step does not need to be repeated.
4. For each selected human gene, query Ensembl REST:
     /homology/id/homo_sapiens/<gene_id>?type=orthologues
5. Extract orthologs in TARGET_SPECIES and assign them to a new
   orthogroup ID (OG0000001, OG0000002, ...).
6. Append these mappings to orthology.tsv:
     species    gene_id    orthogroup
7. Log progress and periodically checkpoint state so that a long
   run can be resumed if interrupted.

This script is meant to be run once to prepare the orthology table used
by create_dataset_multispecies.py.
"""

import os
import gzip
import asyncio
from collections import defaultdict
import ssl
import certifi
import json
from datetime import datetime

import aiohttp

# Root data directory
BASE_DIR = "data"

# Human GTF directory and orthology output file
HUMAN_DIR = os.path.join(BASE_DIR, "homo_sapiens")
ORTHOLOGY_OUT = os.path.join(BASE_DIR, "orthology.tsv")

# Caches for the "good" human genes / transcripts used for orthology
GENE_CACHE = os.path.join(BASE_DIR, "selected_genes.txt")
TX_CACHE = os.path.join(BASE_DIR, "selected_transcripts.txt")

# Checkpoint files for resuming a partially completed orthology run
CHECKPOINT_FILE = os.path.join(BASE_DIR, "orthology_checkpoint.json")
PROCESSED_GENES_FILE = os.path.join(BASE_DIR, "processed_genes.txt")
LOG_FILE = os.path.join(BASE_DIR, "orthology_log.txt")

# Ensembl REST API endpoint
ENSEMBL_SERVER = "https://rest.ensembl.org"

# Non-human species for which we want orthologs
TARGET_SPECIES = [
    "mus_musculus",
    "danio_rerio",
    "pan_troglodytes",
]

# Concurrency / batching configuration for REST requests
MAX_CONCURRENT_REQUESTS = 16
BATCH_SIZE = 256

# Logging + checkpoint intervals (in number of processed genes)
LOG_EVERY = 100
CHECKPOINT_EVERY = 200


def find_gtf_file(folder: str) -> str:
    """
    Find the single GTF or GTF.GZ file in a folder.

    Parameters
    ----------
    folder : str
        Directory containing a single *.gtf or *.gtf.gz file.

    Returns
    -------
    str
        Full path to the GTF file.

    Raises
    ------
    RuntimeError
        If there is not exactly one candidate GTF file.
    """
    files = os.listdir(folder)
    gtf_candidates = [f for f in files if f.endswith(".gtf") or f.endswith(".gtf.gz")]
    if len(gtf_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one .gtf(.gz) in {folder}, found {len(gtf_candidates)}: {gtf_candidates}"
        )
    return os.path.join(folder, gtf_candidates[0])


def parse_gtf_attributes(attr_str: str) -> dict:
    """
    Parse a GTF attributes column into a dictionary.

    Example GTF attributes string:
        'gene_id "ENSG000001"; gene_biotype "protein_coding"; ...'

    Returns
    -------
    dict
        Mapping from attribute key → value.
    """
    attrs = {}
    for field in attr_str.strip().split(";"):
        field = field.strip()
        if not field:
            continue
        if " " not in field:
            # Malformed attribute; skip
            continue
        key, val = field.split(" ", 1)
        val = val.strip().strip('"')
        attrs[key] = val
    return attrs


def select_human_genes_with_valid_transcripts(gtf_path):
    """
    Scan the human GTF and select genes with "good" transcripts.

    Criteria for "valid" transcript:
      - protein_coding (transcript_biotype / transcript_type / gene_biotype)
      - TSL (transcript_support_level) is present and equals 1
      - At least 2 exons (>=1 splice junction)

    Parameters
    ----------
    gtf_path : str
        Path to human GTF or GTF.gz file.

    Returns
    -------
    genes_of_interest : list of str
        Sorted list of human gene IDs that have at least one valid transcript.
    valid_transcripts : set of str
        Set of human transcript IDs that satisfy the criteria.
    """
    opener = gzip.open if gtf_path.endswith(".gz") else open

    gene_biotype = {}
    tx_to_gene = {}
    tx_biotype = {}
    tx_tsl = {}
    tx_exon_count = defaultdict(int)

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            feature = fields[2]
            attr_str = fields[8]
            attrs = parse_gtf_attributes(attr_str)

            if feature == "gene":
                gene_id = attrs.get("gene_id")
                if not gene_id:
                    continue
                # gene_biotype or gene_type depending on annotation
                gb = attrs.get("gene_biotype") or attrs.get("gene_type")
                gene_biotype[gene_id] = gb

            elif feature == "transcript":
                gene_id = attrs.get("gene_id")
                tx_id = attrs.get("transcript_id")
                if not gene_id or not tx_id:
                    continue

                tx_to_gene[tx_id] = gene_id

                # transcript-level biotype, fallback to gene biotype
                tb = (
                    attrs.get("transcript_biotype")
                    or attrs.get("transcript_type")
                    or gene_biotype.get(gene_id)
                )
                tx_biotype[tx_id] = tb

                # transcript support level
                tsl = attrs.get("transcript_support_level")
                tx_tsl[tx_id] = tsl

            elif feature == "exon":
                # Count exons per transcript
                tx_id = attrs.get("transcript_id")
                if tx_id:
                    tx_exon_count[tx_id] += 1

    # Filter transcripts by biotype, TSL, and exon count
    valid_transcripts = set()
    for tx_id, gene_id in tx_to_gene.items():
        biotype = tx_biotype.get(tx_id) or gene_biotype.get(gene_id)
        if biotype != "protein_coding":
            continue

        exon_count = tx_exon_count.get(tx_id, 0)
        if exon_count < 2:
            continue

        tsl = tx_tsl.get(tx_id, "")
        if not tsl:
            continue
        if not tsl.split()[0] == "1":
            continue

        valid_transcripts.add(tx_id)

    # All genes that have at least one valid transcript
    genes_of_interest = {tx_to_gene[tx] for tx in valid_transcripts}
    genes_of_interest = sorted(genes_of_interest)

    print(
        f"Selected {len(valid_transcripts)} transcripts from {len(genes_of_interest)} genes that are:"
    )
    print("  - protein_coding")
    print("  - TSL=1")
    print("  - with >=2 exons (>=1 splice junction)")

    return genes_of_interest, valid_transcripts


def load_list_if_exists(path):
    """Load a list of strings from a text file (one per line), or return None."""
    if os.path.exists(path):
        with open(path) as f:
            items = [line.strip() for line in f if line.strip()]
        return items
    return None


def save_list(path, items):
    """Save a list of strings to a text file, one per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(str(it) + "\n")


def load_checkpoint():
    """
    Load checkpoint state if it exists, otherwise return default.

    Returns
    -------
    dict
        Keys:
          group_counter   : next orthogroup integer ID to use
          n_with_any_orth : how many human genes had at least one ortholog
          processed_genes : number of processed genes so far
          total_genes     : total number of target human genes (or None)
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {
        "group_counter": 1,
        "n_with_any_orth": 0,
        "processed_genes": 0,
        "total_genes": None,
    }


def save_checkpoint(group_counter, n_with_any_orth, processed_genes_count, total_genes):
    """
    Save checkpoint state to JSON for resuming later.
    """
    checkpoint = {
        "group_counter": group_counter,
        "n_with_any_orth": n_with_any_orth,
        "processed_genes": processed_genes_count,
        "total_genes": total_genes,
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def log_message(msg):
    """
    Log a message both to stdout and to LOG_FILE with a timestamp.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


async def fetch_homology_for_gene(session, gene_id, semaphore):
    """
    Fetch orthology information for a single human gene via Ensembl REST.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Shared HTTP session.
    gene_id : str
        Human gene ID (e.g. ENSG...).
    semaphore : asyncio.Semaphore
        Limits number of concurrent requests.

    Returns
    -------
    dict or None
        Parsed JSON (dict) on success, or None on error / unavailable.
    """
    params = {
        "type": "orthologues",
    }
    url = f"{ENSEMBL_SERVER}/homology/id/homo_sapiens/{gene_id}"

    async with semaphore:
        try:
            async with session.get(
                url,
                params=params,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status == 400:
                    # gene not found in this release, etc.
                    return None
                if resp.status == 429:
                    # Rate-limited; respect Retry-After header
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    await asyncio.sleep(retry_after)
                    return await fetch_homology_for_gene(session, gene_id, semaphore)
                if resp.status >= 500:
                    # Server-side error: skip this gene for now
                    return None

                data = await resp.json()
                return data
        except Exception as e:
            log_message(f"Warning: request failed for {gene_id}: {e}")
            return None


def extract_orthologs_from_response(resp_json):
    """
    Given a response JSON from Ensembl, extract triplets:
        (human_gene_id, target_species, target_gene_id)

    Parameters
    ----------
    resp_json : dict or None

    Returns
    -------
    list of (str, str, str)
        (human_id, species_name, target_gene_id).
    """
    if not resp_json or "data" not in resp_json:
        return []

    out = []
    for entry in resp_json["data"]:
        human_id = entry.get("id")
        for h in entry.get("homologies", []):
            target = h.get("target", {})
            target_id = target.get("id")
            target_species = target.get("species")
            if not target_id or not target_species:
                continue
            out.append((human_id, target_species, target_id))
    return out


async def build_orthology_table(human_gene_ids, out_path):
    """
    Build / append to the orthology.tsv file.

    For each human gene in `human_gene_ids`, query Ensembl for orthologs,
    filter to TARGET_SPECIES, and assign them to orthogroups OG0000001, etc.

    This function is robust to partial runs:
      - already processed genes are loaded from PROCESSED_GENES_FILE
      - checkpoint state is loaded/saved in CHECKPOINT_FILE
    """
    total_genes = len(human_gene_ids)

    ckpt = load_checkpoint()
    group_counter = ckpt.get("group_counter", 1)
    n_with_any_orth = ckpt.get("n_with_any_orth", 0)

    processed_genes_list = load_list_if_exists(PROCESSED_GENES_FILE) or []
    processed_genes = set(processed_genes_list)

    # Genes still to process in this run
    remaining_genes = [g for g in human_gene_ids if g not in processed_genes]
    already_processed = len(processed_genes)

    log_message(
        f"Total genes: {total_genes} | Already processed: {already_processed} | Remaining: {len(remaining_genes)}"
    )
    print(f"[INFO] Starting orthology run with {len(remaining_genes)} genes remaining...")

    if not remaining_genes:
        log_message("No genes left to process. Nothing to do.")
        print("[INFO] Nothing to do — all genes processed.")
        return

    # Decide whether to append to existing orthology.tsv or start fresh
    if os.path.exists(out_path) and group_counter > 1:
        out_fh = open(out_path, "a")
        print(f"[INFO] Appending to existing orthology file: {out_path}")
    else:
        out_fh = open(out_path, "w")
        out_fh.write("species\tgene_id\torthogroup\n")
        print(f"[INFO] Creating new orthology file: {out_path}")
        group_counter = 1
        n_with_any_orth = 0
        processed_genes = set()
        processed_genes_list = []

    # Track which human genes we've already processed across runs
    processed_fh = open(PROCESSED_GENES_FILE, "a")

    # Semaphore to cap number of concurrent HTTP requests
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Configure SSL context for aiohttp using certifi bundle
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=60)

    processed_this_run = 0

    # Main asynchronous HTTP loop
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in range(0, len(remaining_genes), BATCH_SIZE):
            batch_genes = remaining_genes[start:start + BATCH_SIZE]
            print(f"[INFO] Starting batch {start//BATCH_SIZE + 1}, size {len(batch_genes)}")

            # Fire off all requests in the batch
            coros = [fetch_homology_for_gene(session, gid, sem) for gid in batch_genes]
            responses = await asyncio.gather(*coros)

            print(f"[INFO] Batch completed, processing results...")

            # Process responses gene-by-gene
            for gene_id, resp in zip(batch_genes, responses):
                processed_this_run += 1
                processed_genes.add(gene_id)
                processed_fh.write(gene_id + "\n")

                if resp is None:
                    print(f"[WARN] No response for {gene_id}")
                else:
                    homs = extract_orthologs_from_response(resp)
                    print(f"[DEBUG] {gene_id}: found {len(homs)} raw homologies")

                    if homs:
                        # Organize target gene IDs by species
                        species_to_genes = defaultdict(set)
                        for human_id, species_name, target_id in homs:
                            if species_name not in TARGET_SPECIES:
                                continue
                            species_to_genes[species_name].add(target_id)

                        if species_to_genes:
                            # Assign a new orthogroup ID for this human gene
                            group_id = f"OG{group_counter:07d}"
                            print(f"[INFO] Writing group {group_id} for {gene_id}")

                            group_counter += 1
                            n_with_any_orth += 1

                            # First write the human gene
                            out_fh.write(f"homo_sapiens\t{gene_id}\t{group_id}\n")

                            # Then write each ortholog in target species
                            for species_name, gene_ids in species_to_genes.items():
                                for target_gene_id in gene_ids:
                                    out_fh.write(
                                        f"{species_name}\t{target_gene_id}\t{group_id}\n"
                                    )

                # Periodic logging and flushing
                if processed_this_run % LOG_EVERY == 0:
                    out_fh.flush()
                    processed_fh.flush()
                    msg = (
                        f"[PROGRESS] {processed_this_run} processed this run | "
                        f"{len(processed_genes)}/{total_genes} total | "
                        f"{n_with_any_orth} orthogroups so far"
                    )
                    print(msg)
                    log_message(msg)

                # Periodic checkpointing for safe resume
                if processed_this_run % CHECKPOINT_EVERY == 0:
                    save_checkpoint(
                        group_counter,
                        n_with_any_orth,
                        len(processed_genes),
                        total_genes,
                    )
                    print("[INFO] Checkpoint saved.")
                    log_message("Checkpoint saved.")

            print("[INFO] Finished processing batch.")

    out_fh.close()
    processed_fh.close()

    # Final checkpoint
    save_checkpoint(group_counter, n_with_any_orth, len(processed_genes), total_genes)
    print("[INFO] Run finished.")
    print(f"[INFO] Orthogroups written: {n_with_any_orth}")
    log_message(
        f"Run finished. Total processed: {len(processed_genes)} / {total_genes} | "
        f"Orthogroups written: {n_with_any_orth} | Output: {out_path}"
    )


def main():
    """
    Top-level driver:
      • Find human GTF
      • Select genes + transcripts (or load from cache)
      • Kick off async orthology build
    """
    human_gtf = find_gtf_file(HUMAN_DIR)
    log_message(f"Using human GTF: {human_gtf}")

    # Try loading cached selection of genes/transcripts
    genes_cached = load_list_if_exists(GENE_CACHE)
    tx_cached = load_list_if_exists(TX_CACHE)

    if genes_cached is not None and tx_cached is not None:
        genes_of_interest = genes_cached
        valid_transcripts = set(tx_cached)
        log_message(
            f"Loaded cached selection: {len(valid_transcripts)} transcripts from "
            f"{len(genes_of_interest)} genes."
        )
    else:
        # No cache found: recompute selection from scratch
        log_message("Selecting protein-coding, TSL=1 transcripts with >=1 splice junction...")
        genes_of_interest, valid_transcripts = select_human_genes_with_valid_transcripts(
            human_gtf
        )
        save_list(GENE_CACHE, genes_of_interest)
        save_list(TX_CACHE, sorted(valid_transcripts))
        log_message(f"Saved selected genes to {GENE_CACHE}")
        log_message(f"Saved selected transcripts to {TX_CACHE}")

    log_message(f"Human genes to query in Compara (total): {len(genes_of_interest)}")
    asyncio.run(build_orthology_table(genes_of_interest, ORTHOLOGY_OUT))


if __name__ == "__main__":
    main()
