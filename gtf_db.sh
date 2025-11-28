#!/bin/bash
#SBATCH --job-name=build_gtf_dbs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=build_gtf_dbs_%j.out
#SBATCH --error=build_gtf_dbs_%j.err

module load python/3.11.5
source /scratch/spliceformer/venv/bin/activate

cd /scratch/spliceformer

echo "Starting GTF DB creation job..."
echo "Working directory: $(pwd)"

python << 'EOF'
import os
import gffutils

SPECIES_DIRS = {
    "sus_scrofa": "data/sus_scrofa",
    "c_elegans": "data/c_elegans",
}

for species, folder in SPECIES_DIRS.items():
    print(f"\n=== {species} ===")
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist! Skipping.")
        continue

    gtf_files = [f for f in os.listdir(folder) if f.endswith(".gtf")]
    if len(gtf_files) != 1:
        print(f"Expected exactly one .gtf file in {folder}, found: {gtf_files}")
        continue

    gtf_path = os.path.join(folder, gtf_files[0])
    db_path = gtf_path + ".db"

    if os.path.exists(db_path):
        print(f"DB already exists at {db_path}, skipping.")
        continue

    print(f"Creating DB at: {db_path}")
    print(f"From GTF: {gtf_path}")

    gffutils.create_db(
        gtf_path,
        dbfn=db_path,
        disable_infer_transcripts=True,
        disable_infer_genes=True,
    )

    print(f"Done building {species} database.")

print("\nAll DB creation tasks finished.")
EOF