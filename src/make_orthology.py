#!/usr/bin/env python3
import os
import gzip
import asyncio
from collections import defaultdict
import ssl
import certifi
import json
from datetime import datetime

import aiohttp

BASE_DIR = "data"

HUMAN_DIR = os.path.join(BASE_DIR, "homo_sapiens")
ORTHOLOGY_OUT = os.path.join(BASE_DIR, "orthology.tsv")

GENE_CACHE = os.path.join(BASE_DIR, "selected_genes.txt")
TX_CACHE = os.path.join(BASE_DIR, "selected_transcripts.txt")

CHECKPOINT_FILE = os.path.join(BASE_DIR, "orthology_checkpoint.json")
PROCESSED_GENES_FILE = os.path.join(BASE_DIR, "processed_genes.txt")
LOG_FILE = os.path.join(BASE_DIR, "orthology_log.txt")

ENSEMBL_SERVER = "https://rest.ensembl.org"

TARGET_SPECIES = [
    "mus_musculus",
    "danio_rerio",
    "pan_troglodytes",
]

MAX_CONCURRENT_REQUESTS = 16

BATCH_SIZE = 256

LOG_EVERY = 100
CHECKPOINT_EVERY = 200


def find_gtf_file(folder: str) -> str:
    files = os.listdir(folder)
    gtf_candidates = [f for f in files if f.endswith(".gtf") or f.endswith(".gtf.gz")]
    if len(gtf_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one .gtf(.gz) in {folder}, found {len(gtf_candidates)}: {gtf_candidates}"
        )
    return os.path.join(folder, gtf_candidates[0])


def parse_gtf_attributes(attr_str: str) -> dict:
    attrs = {}
    for field in attr_str.strip().split(";"):
        field = field.strip()
        if not field:
            continue
        if " " not in field:
            continue
        key, val = field.split(" ", 1)
        val = val.strip().strip('"')
        attrs[key] = val
    return attrs


def select_human_genes_with_valid_transcripts(gtf_path):
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
                gb = attrs.get("gene_biotype") or attrs.get("gene_type")
                gene_biotype[gene_id] = gb

            elif feature == "transcript":
                gene_id = attrs.get("gene_id")
                tx_id = attrs.get("transcript_id")
                if not gene_id or not tx_id:
                    continue

                tx_to_gene[tx_id] = gene_id

                tb = (
                    attrs.get("transcript_biotype")
                    or attrs.get("transcript_type")
                    or gene_biotype.get(gene_id)
                )
                tx_biotype[tx_id] = tb

                tsl = attrs.get("transcript_support_level")
                tx_tsl[tx_id] = tsl

            elif feature == "exon":
                tx_id = attrs.get("transcript_id")
                if tx_id:
                    tx_exon_count[tx_id] += 1

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

    genes_of_interest = {tx_to_gene[tx] for tx in valid_transcripts}
    genes_of_interest = sorted(genes_of_interest)

    print(f"Selected {len(valid_transcripts)} transcripts from {len(genes_of_interest)} genes that are:")
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
    checkpoint = {
        "group_counter": group_counter,
        "n_with_any_orth": n_with_any_orth,
        "processed_genes": processed_genes_count,
        "total_genes": total_genes,
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def log_message(msg):
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


async def fetch_homology_for_gene(session, gene_id, semaphore):
    """
    Query Ensembl REST homology endpoint for one human gene.
    Returns parsed JSON or None on hard error.
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
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    await asyncio.sleep(retry_after)
                    return await fetch_homology_for_gene(session, gene_id, semaphore)
                if resp.status >= 500:
                    return None

                data = await resp.json()
                return data
        except Exception as e:
            log_message(f"Warning: request failed for {gene_id}: {e}")
            return None


def extract_orthologs_from_response(resp_json):
    """
    Given the JSON from /homology/id, yield tuples:
      (human_gene_id, target_species, target_gene_id)
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
    Build orthology table with debugging print statements for progress visibility.
    """

    total_genes = len(human_gene_ids)

    ckpt = load_checkpoint()
    group_counter = ckpt.get("group_counter", 1)
    n_with_any_orth = ckpt.get("n_with_any_orth", 0)

    processed_genes_list = load_list_if_exists(PROCESSED_GENES_FILE) or []
    processed_genes = set(processed_genes_list)

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

    processed_fh = open(PROCESSED_GENES_FILE, "a")

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=60)

    processed_this_run = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in range(0, len(remaining_genes), BATCH_SIZE):
            batch_genes = remaining_genes[start:start + BATCH_SIZE]
            print(f"[INFO] Starting batch {start//BATCH_SIZE + 1}, size {len(batch_genes)}")

            coros = [fetch_homology_for_gene(session, gid, sem) for gid in batch_genes]
            responses = await asyncio.gather(*coros)

            print(f"[INFO] Batch completed, processing results...")

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
                        species_to_genes = defaultdict(set)
                        for human_id, species_name, target_id in homs:
                            if species_name not in TARGET_SPECIES:
                                continue
                            species_to_genes[species_name].add(target_id)

                        if species_to_genes:
                            group_id = f"OG{group_counter:07d}"
                            print(f"[INFO] Writing group {group_id} for {gene_id}")

                            group_counter += 1
                            n_with_any_orth += 1

                            out_fh.write(f"homo_sapiens\t{gene_id}\t{group_id}\n")

                            for species_name, gene_ids in species_to_genes.items():
                                for target_gene_id in gene_ids:
                                    out_fh.write(f"{species_name}\t{target_gene_id}\t{group_id}\n")
                
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

                if processed_this_run % CHECKPOINT_EVERY == 0:
                    save_checkpoint(group_counter, n_with_any_orth, len(processed_genes), total_genes)
                    print("[INFO] Checkpoint saved.")
                    log_message("Checkpoint saved.")

            print("[INFO] Finished processing batch.")

    out_fh.close()
    processed_fh.close()

    save_checkpoint(group_counter, n_with_any_orth, len(processed_genes), total_genes)
    print("[INFO] Run finished.")
    print(f"[INFO] Orthogroups written: {n_with_any_orth}")
    log_message(
        f"Run finished. Total processed: {len(processed_genes)} / {total_genes} | "
        f"Orthogroups written: {n_with_any_orth} | Output: {out_path}"
    )



def main():
    human_gtf = find_gtf_file(HUMAN_DIR)
    log_message(f"Using human GTF: {human_gtf}")

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
        log_message("Selecting protein-coding, TSL=1 transcripts with >=1 splice junction...")
        genes_of_interest, valid_transcripts = select_human_genes_with_valid_transcripts(human_gtf)
        save_list(GENE_CACHE, genes_of_interest)
        save_list(TX_CACHE, sorted(valid_transcripts))
        log_message(f"Saved selected genes to {GENE_CACHE}")
        log_message(f"Saved selected transcripts to {TX_CACHE}")

    log_message(f"Human genes to query in Compara (total): {len(genes_of_interest)}")
    asyncio.run(build_orthology_table(genes_of_interest, ORTHOLOGY_OUT))


if __name__ == "__main__":
    main()
