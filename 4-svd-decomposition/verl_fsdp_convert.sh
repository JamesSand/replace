


fsdp_merged_dir="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted"

fsdp_input_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0109/llama-sgd-lr1e-2-20260110_020449/global_step_200/actor"

output_name="llama-sgd-lr1e-2-20260110_020449-global_step_200"
fsdp_output_path="${fsdp_merged_dir}/${output_name}"


# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ${fsdp_input_path} \
#     --target_dir ${fsdp_output_path}

compare_input_model_path="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct"

python svd_decompose.py \
    --model_before ${compare_input_model_path} \
    --model_after ${fsdp_output_path} \
    --layer 10 \
    --topk 0 \
    --out ${output_name}_layer10_q_gate_3rows.png
