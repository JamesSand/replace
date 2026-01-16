
set -ex

# # no change here
# fsdp_merged_dir="/ssd2/zhizhou/workspace/rotation-project/Lucky_RL/szz_dion_debug/models/Qwen/Qwen3-1.7B-Base"

# compare_input_model_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct"
compare_input_model_path="/ssd2/zhizhou/workspace/rotation-project/Lucky_RL/szz_dion_debug/models/Qwen/Qwen3-1.7B-Base"
# compare_input_model_path=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/DeepSeek-R1-Distill-Qwen-1.5B


# fsdp_input_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base"


# fsdp_output_path="${fsdp_merged_dir}/${output_name}"

fsdp_output_path="/ssd2/zhizhou/workspace/rotation-project/Lucky_RL/szz_dion_debug/ckpts_hf/qwen3_1p7b_fsdp2_hf_20260115_185837"

output_name="dion-test-run-20260115-185837"

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ${fsdp_input_path} \
#     --target_dir ${fsdp_output_path}

python svd_decompose.py \
    --model_before ${compare_input_model_path} \
    --model_after ${fsdp_output_path} \
    --layer 10 \
    --topk 0 \
    --out ./figs-debug-dion/${output_name}_layer10_q_gate_3rows.png




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


