#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

model_before="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_after="agentica-org/DeepScaleR-1.5B-Preview"

cache_dir="./deepseek-r1-distill-qwen-1.5b_to_DeepScaleR-1.5B-Preview-cache"
output_dir="./deepseek-r1-distill-qwen-1.5b_to_DeepScaleR-1.5B-Preview-output"
alphas="0,0.2,0.4,0.6,0.8,1.0"

mkdir -p "$cache_dir" "$output_dir"

python blend.py \
  --model_before "$model_before" \
  --model_after "$model_after" \
  --cache_dir "$cache_dir" \
  --output_dir "$output_dir" \
  --stage both \
  --alphas "$alphas" \
  --low_cpu_mem
