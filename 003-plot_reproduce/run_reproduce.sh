#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Redirect HF cache to /ssd1 so we don't fill the / partition (only 20G free)
export HF_HOME=/ssd1/zhizhou/workspace/rotation-project/replace/plot_reproduce/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME

# Cap BLAS threads — scipy's SVD uses all 128 cores by default which can starve
# the system under contention. 16 is plenty for matrices this size.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

source /home/zs7752/miniconda3/etc/profile.d/conda.sh
conda activate best176

echo "==========================================================="
echo "[1/2] Running hq_scan_all.py"
echo "      (downloads ~18G of models on first run)"
echo "==========================================================="
python hq_scan_all.py 2>&1 | tee scan.log

echo "==========================================================="
echo "[2/2] Running hq_plot_scan_all.py"
echo "==========================================================="
python hq_plot_scan_all.py 2>&1 | tee plot.log

echo "==========================================================="
echo "Done. Plots are in: $(pwd)/plots/"
echo "==========================================================="
ls -lh plots/
