#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Configuration: edit this section to select the model and tasks.
# Comment out task entries that you do not want to run. Leave the TASKS
# array empty to skip evaluation.
# =============================================================================
MODEL_NAME="neuro-vl-8b-thinking"
MODEL_DIR="/path/to/neuro-vl-8b-thinking"
LANGUAGE_MODEL_DIR_NAME="Qwen3-8B"
RESULTS_ROOT="/path/to/results/"

ENABLE_THINKING=true
ENCODE_STUDIES_INDEPENDENTLY=true
SYSTEM_MESSAGE="You are Neuro-VL, created by MLiNS @ University of Michigan. You are a helpful assistant."

RDZV_ENDPOINT="localhost:29500"
NPROC_PER_NODE=8
VISUAL_RESOLUTION=(8 7 7)
VISUAL_ENCODER_NAME="vit_large_slice_scan_study_dualdinotxt1568"
DATALOADER_NUM_WORKERS=8
PER_DEVICE_EVAL_BATCH_SIZE=4
NUM_SAMPLES=1
TEMPERATURE=0.6

# MR Rate
MR_RATE_DATASET="mr_rate"
MR_RATE_DATA_ROOT="/path/to/neurobench_data/data/tensor/"
MR_RATE_CSV_ROOT="/path/to/neurobench_data/visual_question_answering"
MR_RATE_OUTPUT_ROOT="${RESULTS_ROOT%/}/mr_rate"
MR_RATE_TASKS=(
  describe
  diagnose
  localize
  report
)
# =============================================================================
# End configuration.
# =============================================================================

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

THINKING_ARGS=()
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  THINKING_ARGS+=(--enable_thinking)
fi

STUDY_ENCODING_ARGS=()
if [[ "${ENCODE_STUDIES_INDEPENDENTLY}" == "true" ]]; then
  STUDY_ENCODING_ARGS+=(--encode_studies_independently)
fi

run_evaluation() {
  local data_root="$1"
  local csv_path="$2"
  local output_dir="$3"

  if [[ ! -f "${csv_path}" ]]; then
    echo "Missing CSV: ${csv_path}" >&2
    exit 1
  fi

  torchrun \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    test_single_study.py \
    --valid_data "${data_root}=\"${csv_path}\"" \
    --visual_resolution "${VISUAL_RESOLUTION[@]}" \
    --visual_encoder_name "${VISUAL_ENCODER_NAME}" \
    --visual_encoder_path "${MODEL_DIR}/${VISUAL_ENCODER_NAME}.pt" \
    --connector_path "${MODEL_DIR}/connector_checkpoint.pt" \
    --language_model_name_or_path "${MODEL_DIR}/${LANGUAGE_MODEL_DIR_NAME}" \
    "${THINKING_ARGS[@]}" \
    "${STUDY_ENCODING_ARGS[@]}" \
    --system_message "${SYSTEM_MESSAGE}" \
    --output_dir "${output_dir}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --num_samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}"
}

for task in "${MR_RATE_TASKS[@]}"; do
  csv_family="closed_ended"
  if [[ "${task}" == "report" ]]; then
    csv_family="report_generation"
  fi
  run_evaluation \
    "${MR_RATE_DATA_ROOT}" \
    "${MR_RATE_CSV_ROOT}/${csv_family}/${MR_RATE_DATASET}/${task}.csv" \
    "${MR_RATE_OUTPUT_ROOT}/${task}/${MODEL_NAME}/"
done
