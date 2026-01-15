

# Zhizhou note: what we do here:
# suppose we already have the fsdp conversted checkpoints
# Step1 we do svd decomposition
# step2 we do blenlding to get the singular value reset matrix

# model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base
# model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/qwen3-1.7b-base-adamw-lr1e-5-kl-losscoef0.001-20260112_232037-g147-global_step_60
# cache_dir=./qwen3-adam-1e-5-step-60-cache
# output_dir=./qwen3-adam-1e-5-step-60-output

# Step1 verl model merge

# Step2 vis sig value

# step3 generate blended model

# model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/DeepSeek-R1-Distill-Qwen-1.5B
# model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/ds-adamw-lr5e-6-nresp12-global_step_50
# cache_dir=./ds-adam-5e-6-step-50-cache
# output_dir=./ds-adam-5e-6-step-50-output


model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base
model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_50
cache_dir=./qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_50-cache
output_dir=./qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_50-output


mkdir -p $cache_dir
mkdir -p $output_dir

python blend.py \
  --model_before $model_before \
  --model_after $model_after \
  --cache_dir $cache_dir \
  --output_dir $output_dir \
  --stage both \
  --alphas 0,1.0 \
  --low_cpu_mem \
  --store_full_before




model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base
model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_200
cache_dir=./qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_200-cache
output_dir=./qwen3-1.7b-base-adamw-lr5e-6-kl-losscoef0.001-g138-global_step_200-output


mkdir -p $cache_dir
mkdir -p $output_dir

python blend.py \
  --model_before $model_before \
  --model_after $model_after \
  --cache_dir $cache_dir \
  --output_dir $output_dir \
  --stage both \
  --alphas 0,1.0 \
  --low_cpu_mem \
  --store_full_before


  




# model_before=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Llama-3.2-3B-Instruct
# model_after=/fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-adamw-lr1e-6-20260110_015014-global_step_200
# cache_dir=./llama-adam-cache
# output_dir=./llama-adam-output
# # /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/fsdp_converted/llama-adamw-lr1e-6-20260110_015014-global_step_200
# mkdir -p $cache_dir
# mkdir -p $output_dir

# python blend.py \
#   --model_before $model_before \
#   --model_after $model_after \
#   --cache_dir $cache_dir \
#   --output_dir $output_dir \
#   --stage both \
#   --alphas 0,0.2,0.4,0.6,0.8,1.0 \
#   --low_cpu_mem \
#   --store_full_before





