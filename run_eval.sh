#!/bin/bash
#SBATCH --job-name=spliceformer10k_eval
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/spliceformer10k_eval-%j.out
#SBATCH --error=logs/spliceformer10k_eval-%j.err

module load python/3.11
module load cuda
source /scratch/spliceformer/venv/bin/activate

DATA_DIR="data/processed_data"
SL=10000
BATCH_SIZE=64
NUM_WORKERS=4
DEPTH=2
HEADS=4
N_BLOCKS=4

CKPT_HUMAN="spliceformer10k_human.pt"
CKPT_HUMAN_CHIMP="spliceformer10k_chimp.pt"
CKPT_HUMAN_MOUSE="spliceformer10k_mouse.pt"
CKPT_HUMAN_FISH="spliceformer10k_fish.pt"
CKPT_ALL4="spliceformer10k_all.pt"

run_eval () {
  local CKPT=$1       # checkpoint file
  local PREFIX=$2     # out-prefix
  shift 2             # remaining args are species

  echo "=== Running eval: checkpoint=${CKPT}, prefix=${PREFIX}, species=$@ ==="

  python -u -m src.eval \
    --data-dir "${DATA_DIR}" \
    --species "$@" \
    --sl "${SL}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --depth "${DEPTH}" \
    --heads "${HEADS}" \
    --n-transformer-blocks "${N_BLOCKS}" \
    --checkpoint "${CKPT}" \
    --out-prefix "${PREFIX}"
}

# ==========================================================
# 1) Each model on HUMAN-ONLY test data
# ==========================================================
run_eval "${CKPT_HUMAN}"        "humanModel_on_human"        homo_sapiens
run_eval "${CKPT_HUMAN_CHIMP}"  "humanChimpModel_on_human"   homo_sapiens
run_eval "${CKPT_HUMAN_MOUSE}"  "humanMouseModel_on_human"   homo_sapiens
run_eval "${CKPT_HUMAN_FISH}"   "humanFishModel_on_human"    homo_sapiens
run_eval "${CKPT_ALL4}"         "all4Model_on_human"         homo_sapiens

# ==========================================================
# 2) Each model on its OWN training species combo
# ==========================================================

# human-only model
run_eval "${CKPT_HUMAN}"        "humanModel_on_humanOwn"     homo_sapiens

# human + chimp
run_eval "${CKPT_HUMAN_CHIMP}"  "humanChimpModel_on_humanChimp" \
                                homo_sapiens pan_troglodytes

# human + mouse
run_eval "${CKPT_HUMAN_MOUSE}"  "humanMouseModel_on_humanMouse" \
                                homo_sapiens mus_musculus

# human + zebrafish
run_eval "${CKPT_HUMAN_FISH}"   "humanFishModel_on_humanFish" \
                                homo_sapiens danio_rerio

# all four species
run_eval "${CKPT_ALL4}"         "all4Model_on_all4" \
                                homo_sapiens pan_troglodytes mus_musculus danio_rerio

# ==========================================================
# 3) Cross-species examples (one “separate organism” each)
# ==========================================================


run_eval "${CKPT_HUMAN}"        "humanModel_on_pig"        sus_scrofa
run_eval "${CKPT_HUMAN_CHIMP}"  "humanChimpModel_on_pig"   sus_scrofa
run_eval "${CKPT_HUMAN_MOUSE}"  "humanMouseModel_on_pig"   sus_scrofa
run_eval "${CKPT_HUMAN_FISH}"   "humanFishModel_on_pig"    sus_scrofa
run_eval "${CKPT_ALL4}"         "all4Model_on_pig"         sus_scrofa
run_eval "${CKPT_HUMAN}"        "humanModel_on_eleg"       c_elegans
run_eval "${CKPT_HUMAN_CHIMP}"  "humanChimpModel_on_eleg"  c_elegans
run_eval "${CKPT_HUMAN_MOUSE}"  "humanMouseModel_on_eleg"  c_elegans
run_eval "${CKPT_HUMAN_FISH}"   "humanFishModel_on_eleg"   c_elegans
run_eval "${CKPT_ALL4}"         "all4Model_on_eleg"        c_elegans

echo "All evaluations finished."
