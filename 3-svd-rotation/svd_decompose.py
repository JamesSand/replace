
#!/usr/bin/env python
# svd_decompose_hf.py
#
# 对单个 HuggingFace 模型的每一层权重做完整 SVD 分解，并把结果保存到磁盘。
#
# 用法示例：
#   python svd_decompose_hf.py --model_id Qwen/Qwen2.5-VL-3B-Instruct
#   python svd_decompose_hf.py --model_id IffYuan/Embodied-R1-3B-Stage1 --output_dir ./svd_r1_stage1
#
# 不指定 --output_dir 时，默认用 model_id 作为输出目录名。

import os
import argparse
from typing import Dict, Any, Tuple

import torch
# from transformers import AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


def is_matrix_param(t: torch.Tensor) -> bool:
    """只对维度 >= 2 的参数做 SVD（向量、标量就跳过）。"""
    return t.ndim >= 2


def run_svd_on_param(weight: torch.Tensor) -> Dict[str, Any]:
    """
    对一个权重张量做完整 SVD 分解。

    - weight: 任意形状 (d1, d2, ..., dk), k >= 2
    - reshape 成 (d1, d2*...*dk) 再做 SVD
    - 返回 U, S, Vh 以及原始 shape
    """
    orig_shape: Tuple[int, ...] = tuple(weight.shape)
    # reshape 成二维矩阵： [out_dim, in_dim]
    mat = weight.view(orig_shape[0], -1)

    # 转为 float32 再做 SVD，避免 float16 精度和算子问题
    mat = mat.to(torch.float32)

    # full_matrices=False 对大矩阵更省内存 / 时间
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

    return {
        "U": U,
        "S": S,
        "Vh": Vh,
        "orig_shape": orig_shape,
    }


def process_model(model_id: str, output_dir: str):
    print(f"\n=== Processing model: {model_id} ===")
    os.makedirs(output_dir, exist_ok=True)

    # 默认加载到 CPU，避免显存爆炸；权重用 fp16 存，SVD 时再转 fp32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    model.eval()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_matrix_param(param):
                continue

            print(f"  [SVD] {name} | shape={tuple(param.shape)}")

            weight_cpu = param.detach().cpu()
            svd_res = run_svd_on_param(weight_cpu)

            # 把参数名里的 '.' 替换掉，避免生成多级子目录；都放在同一层目录里
            filename = name.replace(".", "__") + ".pt"
            save_path = os.path.join(output_dir, filename)

            torch.save(svd_res, save_path)

    del model
    torch.cuda.empty_cache()
    print(f"=== Done model: {model_id}, saved to {output_dir} ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Do full SVD decomposition on all 2D+ weight tensors of a HuggingFace model."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model id, e.g. 'Qwen/Qwen2.5-VL-3B-Instruct'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save SVD results. "
            "If not set, use `model_id` as the output directory."
        ),
    )

    args = parser.parse_args()

    # 如果没指定 output_dir，就直接用 model_id 的 basename
    output_dir = args.output_dir if args.output_dir is not None else os.path.basename(args.model_id)
    
    # add qwenvl-svd-prefix
    output_dir = os.path.join("qwenvl-svd", output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    process_model(model_id=args.model_id, output_dir=output_dir)
    print("All done.")


if __name__ == "__main__":
    main()

