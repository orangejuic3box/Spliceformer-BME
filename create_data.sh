#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --output=create_dataset_multispecies-%j.out
#SBATCH --error=create_dataset_multispecies-%j.err

module load python/3.11.5
source /scratch/spliceformer/venv/bin/activate

cd /scratch/spliceformer

if [ "$#" -eq 0 ]; then
  echo "No species passed, running for ALL species."
  python -m src.create_dataset_multispecies
else
  echo "Running for species: $@"
  python -m src.create_dataset_multispecies --species "$@"
fi