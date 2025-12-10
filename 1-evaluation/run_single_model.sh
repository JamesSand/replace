#!/bin/bash

set -e

model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# alphas=(0.8 1.0)
gpus=(0 1 2 3)
# gpus=(6 7) 
num_gpus=${#gpus[@]}
max_tokens=$((16*1024))
n_samples=16
tokenizer=$model_path  # 用原始模型的 tokenizer
# tokenizer_basename=$(basename $tokenizer)

dataset_list=("HuggingFaceH4/aime_2024" "math-ai/amc23" "HuggingFaceH4/math-500")

mkdir -p logs
mkdir -p gen_results

# Run evaluations on GPUs in parallel
for i in "${!dataset_list[@]}"; do
    data_path=${dataset_list[$i]}
    gpu=${gpus[$((i % num_gpus))]}
    
    model_name="$(basename $model_path)_$(basename $data_path)"

        (
            export CUDA_VISIBLE_DEVICES=$gpu
            timestamp=$(date +%Y%m%d_%H%M%S)
            echo "Starting evaluation on $data_path on GPU ${gpu}"
            python eval_v1.py \
                --model $model_path \
                --tokenizer $tokenizer \
                --data_path $data_path \
                --gpu_mem_util 0.7 \
                --n_samples $n_samples \
                --max_tokens $max_tokens \
                --out_json gen_results/${model_name}_results.json \
                2>&1 | tee logs/${model_name}.log
        ) &
done

wait
echo "All evaluations completed!"
