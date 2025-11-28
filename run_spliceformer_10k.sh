#!/bin/bash
#SBATCH --job-name=spliceformer10k_fish
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/spliceformer10k_fish-%j.out
#SBATCH --error=logs/spliceformer10k_fish-%j.err

module load python/3.11
module load cuda

source /scratch/spliceformer/venv/bin/activate

python -m src.train_transformer_model_10k \
  --data-dir data/processed_data \
  --batch-size 64 \
  --epochs 10 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --save-name spliceformer10k_fish \
  --species homo_sapiens danio_rerio \
  --sl 10000 \
  --n-transformer-blocks 4 \
  --depth 2 \
  --heads 4 \
  --warmup-ratio 0.1 \
  --no-reinforce \
  --grad-accum-steps 1 \
  --num-workers 0
