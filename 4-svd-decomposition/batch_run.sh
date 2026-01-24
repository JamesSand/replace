
set -ex

# no change here
fsdp_merged_dir="/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted"
model_before_path="/fast/sliu/zhizhou/workspace/rotation-project/Lucky_RL/hf_models/Qwen3-1.7B-Base"

# batch to process (edit this list)
inputs=(
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_20/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-2-fp64/global_step_60/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-reset-muon-lr-1e-4-fp64/global_step_100/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-svd-muon-lr-1e-2-fp64/global_step_20/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-sgd-svd-muon-lr-1e-2-fp64/global_step_60/actor"
#   "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_50/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_100/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_150/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-sgd-lr1e-2-kl-losscoef0.001-20260111_033800-g134/global_step_200/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-adam-reset-muon-lr-1e-6-fp64/global_step_200/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen1.7b-adam-reset-muon-lr-1e-6-fp64/global_step_20/actor"
  # "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-adamw-lr1e-6-kl-losscoef0.001-20260111_033800-g137/global_step_200/actor"
  "/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/ckpts_verl/debug0110/qwen3-1.7b-base-adamw-lr1e-5-kl-losscoef0.001-20260112_232037-g147/global_step_200/actor"
)

for fsdp_input_path in "${inputs[@]}"; do
  # derive a clean output_name from .../<exp>/global_step_xxx/actor
  exp="$(basename "$(dirname "$(dirname "$fsdp_input_path")")")"
  step="$(basename "$(dirname "$fsdp_input_path")")"
  output_name="${exp}-${step}"

  fsdp_output_path="${fsdp_merged_dir}/${output_name}"

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${fsdp_input_path}" \
    --target_dir "${fsdp_output_path}"

  python svd_decompose.py \
    --model_before "${model_before_path}" \
    --model_after "${fsdp_output_path}" \
    --layer 10 \
    --topk 0 \
    --out "./figs/${output_name}_layer10_q_gate_3rows.png"
done



