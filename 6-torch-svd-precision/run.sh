# python 001-cpu-gpu-test.py --n 2048 --cond-exp 10 --topk -1 --disable-tf32

python 002-calculate-cond-distribution.py --device cuda --dtype fp64 --disable-tf32 --plot-log10 --out ./cond_hist.png

python 003-qwen-weight-test.py \
  --model-dir /fast/sliu/zhizhou/workspace/rotation-project/shared_folder/hf_models/Qwen3-1.7B-Base \
  --tensor-name model.layers.10.self_attn.q_proj.weight \
  --disable-tf32 \
  --topk -1 \
  --out-dir ./svd_real_weight_out



