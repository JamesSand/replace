
set -ex

# no change here
fsdp_merged_dir="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted"

# to process
# /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_60/actor
# /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_100/actor

# /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-svd-muon-lr-1e-2-fp64/global_step_20/actor

# /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-svd-muon-lr-1e-2-fp64/global_step_60/actor

fsdp_input_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_20/actor"
output_name="qwen1.7b-sgd-reset-muon-lr-1e-2-fp64-global_step_20"

fsdp_output_path="${fsdp_merged_dir}/${output_name}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${fsdp_input_path} \
    --target_dir ${fsdp_output_path}

# model_before_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct"
model_before_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base"
# model_before_path=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/DeepSeek-R1-Distill-Qwen-1.5B
model_after_path="${fsdp_output_path}"

python svd_decompose.py \
    --model_before ${model_before_path} \
    --model_after ${model_after_path} \
    --layer 10 \
    --topk 0 \
    --out ./figs/${output_name}_layer10_q_gate_3rows.png




# # 50 step checkpoints
# fsdp_input_path="/lustre/fast/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/ds-adamw-lr5e-6-nresp12/global_step_50/actor"
# output_name="ds-adamw-lr5e-6-nresp12-global_step_50"

# fsdp_input_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-svd-muon-lr-1e-7/global_step_50/actor"
# output_name="qwen1.7b-svd-muon-lr-1e-7-global_step_50"

# fsdp_output_path="${fsdp_merged_dir}/${output_name}"

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ${fsdp_input_path} \
#     --target_dir ${fsdp_output_path}

# python svd_decompose.py \
#     --model_before ${compare_input_model_path} \
#     --model_after ${fsdp_output_path} \
#     --layer 10 \
#     --topk 0 \
#     --out ./figs/${output_name}_layer10_q_gate_3rows.png



# fsdp_input_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-svd-muon-lr-1e-7/global_step_200/actor"
# output_name="qwen1.7b-svd-muon-lr-1e-7-global_step_200"

# fsdp_output_path="${fsdp_merged_dir}/${output_name}"

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ${fsdp_input_path} \
#     --target_dir ${fsdp_output_path}

# python svd_decompose.py \
#     --model_before ${compare_input_model_path} \
#     --model_after ${fsdp_output_path} \
#     --layer 10 \
#     --topk 0 \
#     --out ./figs/${output_name}_layer10_q_gate_3rows.png





# # # 200 step checkpoints
# # fsdp_input_path="/lustre/fast/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/ds-adamw-lr5e-6-nresp12/global_step_200/actor"
# # output_name="ds-adamw-lr5e-6-nresp12-global_step_200"

# fsdp_input_path="/lustre/fast/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-20260111_084404-g138/global_step_200/actor"
# output_name="qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_200"

# fsdp_output_path="${fsdp_merged_dir}/${output_name}"

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ${fsdp_input_path} \
#     --target_dir ${fsdp_output_path}



# python svd_decompose.py \
#     --model_before ${compare_input_model_path} \
#     --model_after ${fsdp_output_path} \
#     --layer 10 \
#     --topk 0 \
#     --out ${output_name}_layer10_q_gate_3rows.png


