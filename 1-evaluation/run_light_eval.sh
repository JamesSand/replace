

#!/bin/bash

# conda create -n lighteval python=3.12 -y
# conda activate lighteval
# pip install lighteval vllm

set -e

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL
LOG_DIR=logs

gpus=(0 1 2 3)

echo "Model: $MODEL"
echo "Starting evaluations in 5 seconds..."
sleep 5

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# AIME 2024
(
    export CUDA_VISIBLE_DEVICES=${gpus[0]}
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "Starting evaluation on aime24 on GPU ${gpus[0]}"
    lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        2>&1 | tee $LOG_DIR/aime24_${timestamp}.log
) &

# MATH-500
(
    export CUDA_VISIBLE_DEVICES=${gpus[1]}
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "Starting evaluation on math_500 on GPU ${gpus[1]}"
    lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        2>&1 | tee $LOG_DIR/math_500_${timestamp}.log
) &

# GPQA Diamond
(
    export CUDA_VISIBLE_DEVICES=${gpus[2]}
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "Starting evaluation on gpqa:diamond on GPU ${gpus[2]}"
    lighteval vllm $MODEL_ARGS "lighteval|gpqa:diamond|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        2>&1 | tee $LOG_DIR/gpqa_diamond_${timestamp}.log
) &

# LiveCodeBench
(
    export CUDA_VISIBLE_DEVICES=${gpus[3]}
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "Starting evaluation on lcb:codegeneration on GPU ${gpus[3]}"
    lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        2>&1 | tee $LOG_DIR/lcb_codegeneration_${timestamp}.log
) &

wait
echo "All evaluations completed!" 

