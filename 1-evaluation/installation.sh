
set -e

conda init
conda create -n openr1 python=3.11 -y
conda activate openr1
pip install uv
uv pip install vllm==0.8.5.post1
uv pip install setuptools 

# && uv pip install flash-attn --no-build-isolation

# if not exist, clone
if [ ! -d "open-r1" ]; then
  git clone git@github.com:huggingface/open-r1.git
fi

cd open-r1
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
cd ..

wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install --no-cache-dir --force-reinstall flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 现在想在我的环境里边吧这个 bash run_light_test.sh 跑起来，但是总是遇到这个报错 /home/zs7752/miniconda3/en      
# vs/openr1/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.s  o: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE    
#  你想办法吧这个代码给我跑起来


wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl


pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


