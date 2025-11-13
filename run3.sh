

# before RL: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# after RL: agentica-org/DeepScaleR-1.5B-Preview

export CUDA_VISIBLE_DEVICES=3

# model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
model=agentica-org/DeepScaleR-1.5B-Preview

# Extract model name for log file
model_name=$(basename $model)
mkdir -p logs

max_tokens=$((6*1024))

python eval_v1.py \
  --model $model \
  --hf_dataset HuggingFaceH4/math-500 \
  --limit 500 \
  --n_samples 4 \
  --max_tokens $max_tokens \
  --out_json math500_results.json \
  2>&1 | tee logs/${model_name}.log
