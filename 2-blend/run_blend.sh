
model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct
model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-muon-muonlr1e-4-spectral_norm-muonadamlr1e-6-20260110_005142-global_step_200
cache_dir=./llama-muon-cache
output_dir=./llama-muon-output
# /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-adamw-lr1e-6-20260110_015014-global_step_200
mkdir -p $cache_dir
mkdir -p $output_dir

python blend.py \
  --model_before $model_before \
  --model_after $model_after \
  --cache_dir $cache_dir \
  --output_dir $output_dir \
  --stage both \
  --alphas 0,0.2,0.4,0.6,0.8,1.0 \
  --low_cpu_mem \
  --store_full_before



model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct
model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-adamw-lr1e-6-20260110_015014-global_step_200
cache_dir=./llama-adam-cache
output_dir=./llama-adam-output
# /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-adamw-lr1e-6-20260110_015014-global_step_200
mkdir -p $cache_dir
mkdir -p $output_dir

python blend.py \
  --model_before $model_before \
  --model_after $model_after \
  --cache_dir $cache_dir \
  --output_dir $output_dir \
  --stage both \
  --alphas 0,0.2,0.4,0.6,0.8,1.0 \
  --low_cpu_mem \
  --store_full_before





