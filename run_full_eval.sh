#!/bin/bash
#SBATCH --job-name=splice_full_eval
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/full_eval-%j.out
#SBATCH --error=logs/full_eval-%j.err

module load python/3.11
source /scratch/spliceformer/venv/bin/activate
cd /scratch/spliceformer
python -u full_eval.py