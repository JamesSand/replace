#!/bin/bash

set -e

# this will lead to OOM

# base_dir="DeepScaleR-1.5B-Preview_sigma-blend_fp32"
# base_dir="Nemotron-Research-Reasoning-Qwen-1.5B_sigma-blend_fp32"
base_dir="DeepSeek-R1-Distill-Qwen-1.5B_sigma-blend_fp32"

alphas=(0.0 0.2 0.4 0.6 0.8 1.0)
# alphas=(0.8 1.0)
gpus=(0 1 2 3)
# gpus=(6 7) 
num_gpus=${#gpus[@]}
max_tokens=$((16*1024))
n_samples=16

tokenizer="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 

# tokenizer="nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"  # 用原始模型的 tokenizer

tokenizer_basename=$(basename $tokenizer)

dataset_list=("HuggingFaceH4/aime_2024" "math-ai/amc23" "HuggingFaceH4/math-500")

mkdir -p logs

# Run evaluations on GPUs in parallel
for data_path in "${dataset_list[@]}"; do
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    gpu=${gpus[$((i % num_gpus))]}
    model_path="${base_dir}/alpha_${alpha}"
    model_name="${tokenizer_basename}_alpha_${alpha}_$(basename $data_path)"

        (
            export CUDA_VISIBLE_DEVICES=$gpu
            timestamp=$(date +%Y%m%d_%H%M%S)
            echo "Starting alpha_${alpha} on $data_path on GPU ${gpu}"
            python eval_v1.py \
                --model $model_path \
                --tokenizer $tokenizer \
                --data_path $data_path \
                --gpu_mem_util 0.7 \
                --n_samples $n_samples \
                --max_tokens $max_tokens \
                --out_json ${model_name}_results.json \
                2>&1 | tee logs/${model_name}.log
        ) &
        
        # Wait if all GPUs are busy
        if [ $(((i + 1) % num_gpus)) -eq 0 ]; then
            wait
        fi
    done
    
    
done

wait
echo "All evaluations completed!"
