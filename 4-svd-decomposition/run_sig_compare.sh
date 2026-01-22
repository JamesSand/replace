
set -ex

model_before_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base"
model_after_path="/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/szz_dion_debug/checkpoints/mini_dion_hf_step50"
output_name="msign-read-fp64-check"

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ${fsdp_input_path} \
#     --target_dir ${fsdp_output_path}

python svd_decompose.py \
    --model_before ${model_before_path} \
    --model_after ${model_after_path} \
    --layer 10 \
    --topk 0 \
    --out ./figs/${output_name}_layer10_q_gate_3rows.png




