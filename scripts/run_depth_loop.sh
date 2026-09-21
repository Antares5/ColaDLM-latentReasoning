#!/usr/bin/env bash
# Training-free depth-loop experiment driver.
#
# Runs python -m cola_dlm.inference on a subset of tasks with the DiT
# block stack looped COLA_DIT_DEPTH_LOOPS times per forward (see
# modeling_cola_dit.py: ColaDiTModel.depth_loops).
#
# Usage:
#   RUN_NAME=r2_t8 LOOPS=2 TIMESTEPS=8 MAX_SAMPLES=200 \
#       TASKS="lambada mmlu" bash scripts/run_depth_loop.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. r2_t8}"
export COLA_DIT_DEPTH_LOOPS="${LOOPS:-1}"
export COLA_INFER_PER_SAMPLE_NOISE_SEED="${COLA_INFER_PER_SAMPLE_NOISE_SEED:-66}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TIMESTEPS="${TIMESTEPS:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
TASKS="${TASKS:-lambada mmlu}"
BATCH_SIZE="${BATCH_SIZE:-20}"
GPU="${GPU:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/eval_output/depth_loop}"
OUTPUT_DIR="${OUTPUT_ROOT}/tasks_${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "=== depth-loop run: ${RUN_NAME} (loops=${COLA_DIT_DEPTH_LOOPS}, timesteps=${TIMESTEPS}, samples=${MAX_SAMPLES}) ==="
START=$(date +%s)
for TASK in $TASKS; do
    CUDA_VISIBLE_DEVICES=$GPU python -m cola_dlm.inference \
        --dit_path "${DIT_PATH:-hf_models/cola_dlm/cola_dit}" \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --input_jsonl "generate_task_data/${TASK}.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --task_name "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --max_samples "$MAX_SAMPLES" \
        --max_new_tokens 32 \
        --timestep_num "$TIMESTEPS" \
        --guidance_scale 7.0 \
        --temperature 0.0 \
        --eos_token_id 100257 \
        --im_end_token_id 100265 \
        2>&1 | grep -v -E "Loading checkpoint|batch [0-9]+ \("
done
END=$(date +%s)
echo "=== ${RUN_NAME} done in $((END - START))s -> ${OUTPUT_DIR} ==="
