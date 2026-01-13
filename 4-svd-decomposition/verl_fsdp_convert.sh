
set -ex

fsdp_merged_dir="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted"

# compare_input_model_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct"

compare_input_model_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base"



# 20 step checkpoints

fsdp_input_path="/lustre/fast/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-adamw-lr1e-5-kl-losscoef0.001-20260112_232037-g147/global_step_20/actor"

output_name="qwen3-1.7b-base-adamw-lr1e-5-kl-losscoef0.001-20260112_232037-g147-global_step_20"
fsdp_output_path="${fsdp_merged_dir}/${output_name}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${fsdp_input_path} \
    --target_dir ${fsdp_output_path}



python svd_decompose.py \
    --model_before ${compare_input_model_path} \
    --model_after ${fsdp_output_path} \
    --layer 10 \
    --topk 0 \
    --out ${output_name}_layer10_q_gate_3rows.png

# 60 step checkpoints

fsdp_input_path="/lustre/fast/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-adamw-lr1e-5-kl-losscoef0.001-20260112_232037-g147/global_step_60/actor"

output_name="qwen3-1.7b-base-adamw-lr1e-5-kl-losscoef0.001-20260112_232037-g147-global_step_60"
fsdp_output_path="${fsdp_merged_dir}/${output_name}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${fsdp_input_path} \
    --target_dir ${fsdp_output_path}



python svd_decompose.py \
    --model_before ${compare_input_model_path} \
    --model_after ${fsdp_output_path} \
    --layer 10 \
    --topk 0 \
    --out ${output_name}_layer10_q_gate_3rows.png


