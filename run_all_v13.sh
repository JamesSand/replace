#!/bin/bash

set -e

base_dir="DeepScaleR-1.5B-Preview_sigma-blend_fp32"
alphas=(0.0 0.2 0.4 0.6 0.8 1.0)
# alphas=(0.8 1.0)
# gpus=(0 1)
gpus=(0 1 2 3 4 5 6 7) 
num_gpus=${#gpus[@]}
max_tokens=$((6*1024))
n_samples=16
tokenizer="agentica-org/DeepScaleR-1.5B-Preview"  # 使用原始模型的 tokenizer

mkdir -p logs

# Run evaluations on GPUs in parallel
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    gpu=${gpus[$((i % num_gpus))]}
    model_path="${base_dir}/alpha_${alpha}"
    model_name="alpha_${alpha}"
    
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        timestamp=$(date +%Y%m%d_%H%M%S)
        echo "Starting alpha_${alpha} on GPU ${gpu}"
        python eval_v1.py \
            --model $model_path \
            --tokenizer $tokenizer \
            --hf_dataset HuggingFaceH4/math-500 \
            --limit 500 \
            --n_samples $n_samples \
            --max_tokens $max_tokens \
            --out_json ${model_name}_results.json \
            2>&1 | tee logs/${model_name}_nsamples${n_samples}_maxto${max_tokens}.log
    ) &
    
    # Wait if all GPUs are busy
    if [ $(((i + 1) % num_gpus)) -eq 0 ]; then
        wait
    fi
done

wait
echo "All evaluations completed!"
