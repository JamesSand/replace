#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Reuse the HF cache populated in 001's reproduce — saves ~20min of downloads
export HF_HOME=/ssd1/zhizhou/workspace/rotation-project/replace/plot_reproduce/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME

# Cap BLAS threads (same as 001)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

source /home/zs7752/miniconda3/etc/profile.d/conda.sh
conda activate best176

echo "=========================================================="
echo "Running hq_scan_all.py (with new e_src / e_L / e_R / e_tgt)"
echo "=========================================================="
python hq_scan_all.py 2>&1 | tee scan.log

echo "=========================================================="
echo "Done. CSV at: $(pwd)/stiefel_analysis_metrics.csv"
echo "=========================================================="
