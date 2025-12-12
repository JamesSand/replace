#!/usr/bin/env bash
set -euo pipefail

# kill all process with script name
# pkill -f svd_decompose.py
# rm -rf logs qwenvl-svd

SCRIPT="svd_decompose.py"
# SCRIPT="svd_decompose_debug.py"

# # 三个模型 ID
# MODEL_IDS=(
#   "Qwen/Qwen2.5-VL-3B-Instruct"
#   "IffYuan/Embodied-R1-3B-Stage1"
#   "IffYuan/Embodied-R1-3B-v1"
# )

MODEL_IDS=(
  "agentica-org/DeepCoder-1.5B-Preview"
)

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

for MODEL_ID in "${MODEL_IDS[@]}"; do
  # 取最后一段作为 base name，比如 Qwen/Qwen2.5-VL-3B-Instruct -> Qwen2.5-VL-3B-Instruct
  # BASE_NAME="${MODEL_ID##*/}"
  BASE_NAME=$(basename "${MODEL_ID}")
  LOG_FILE="${LOG_DIR}/${BASE_NAME}.log"

  echo "Launching SVD for ${MODEL_ID}, log -> ${LOG_FILE}"

  # 后台并行跑，每个模型一个 log
  python "${SCRIPT}" --model_id "${MODEL_ID}" \
    > "${LOG_FILE}" 2>&1 &
done

echo "All SVD jobs launched. Waiting for them to finish..."
wait
echo "All SVD jobs completed."
