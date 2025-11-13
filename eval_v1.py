# -*- coding: utf-8 -*-
"""
Evaluate a single model on MATH500 using vLLM and your compute_score:
  from math_dapo import compute_score
  score = compute_score(solution_str, ground_truth,
                        is_longcot=False, is_use_math_verify=True)

- 每题生成 n 个答案（默认 4），逐个 compute_score；
- 该题分数 = n 次得分的平均；
- 最终准确率 = 全部题目的平均；
- 仅保存 JSON（不写 CSV）。

Install:
  pip install vllm datasets accelerate
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from datasets import load_dataset

# 直接使用你的判分函数（要求 math_dapo 在 PYTHONPATH 可导入）
from math_dapo import compute_score

# === 使用你给的 prompt（要求 boxed 输出）===
# PROMPT_TEMPLATE = "{problem} Let's think step by step and output the final answer within \\boxed{{}}."

instruction = "Let's think step by step and output the final answer within \\boxed{}."

def load_math500(hf_dataset: Optional[str],
                 data_path: Optional[str],
                 limit: Optional[int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if data_path:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "problem" in obj and "answer" in obj:
                    items.append({"problem": obj["problem"], "answer": obj["answer"]})
    else:
        if not hf_dataset:
            hf_dataset = "HuggingFaceH4/math-500"
        ds = load_dataset(hf_dataset, split="test")
        for ex in ds:
            p = ex.get("problem")
            a = ex.get("answer")
            if p is not None and a is not None:
                items.append({"problem": p, "answer": a})
            else:
                raise RuntimeError("Dataset examples must have 'problem' and 'answer' fields.")

    if limit is not None and limit > 0:
        items = items[:limit]

    if not items:
        raise RuntimeError("No data loaded. Check --hf_dataset/--data_path.")
    return items

def build_prompts(items: List[Dict[str, Any]]) -> List[str]:
    # return [PROMPT_TEMPLATE.format(problem=ex["problem"]) for ex in items]
    return [f"{ex['problem']} {instruction}" for ex in items]

def normalize_score(res: Any) -> float:
    """把 compute_score 的返回统一到 [0,1]."""
    if isinstance(res, bool):
        return 1.0 if res else 0.0
    if isinstance(res, (int, float)):
        return float(res)
    if isinstance(res, dict):
        if "score" in res:
            v = res["score"]
            if isinstance(v, bool): return 1.0 if v else 0.0
            return float(v)
        if "correct" in res:
            v = res["correct"]
            if isinstance(v, bool): return 1.0 if v else 0.0
            return float(v)
    return 0.0

def run_eval_for_model(model_name: str,
                       items: List[Dict[str, Any]],
                       n: int = 4,
                       max_tokens: int = 1024,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       tensor_parallel_size: int = 1,
                       dtype: Optional[str] = None,
                       trust_remote_code: bool = True,
                       gpu_memory_utilization: float = 0.95,
                       tokenizer: Optional[str] = None) -> Dict[str, Any]:
    """
    - vLLM 生成 n 个答案/题；
    - 直接 compute_score(原始输出字符串, gold, is_longcot=False, is_use_math_verify=True) 判分；
    - 返回 summary 与明细。
    """
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if dtype:
        llm_kwargs["dtype"] = dtype
    if tokenizer:
        llm_kwargs["tokenizer"] = tokenizer

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    prompts = build_prompts(items)
    outputs = llm.generate(prompts, sp)
    assert len(outputs) == len(items)

    per_item_scores: List[float] = []
    details: List[Dict[str, Any]] = []

    for i, (out, ex) in enumerate(zip(outputs, items)):
        generations = [o.text for o in out.outputs]  # 不做 boxed 提取，直接用原始字符串
        gold = ex["answer"]

        sample_scores: List[float] = []
        # raw_scores: List[Any] = []
        for sol in generations:
            res = compute_score(sol, gold, is_longcot=False, is_use_math_verify=True)
            sample_scores.append(res)
            # raw_scores.append(res)
            # sample_scores.append(normalize_score(res))

        item_score = sum(sample_scores) / max(len(sample_scores), 1)
        per_item_scores.append(item_score)

        details.append({
            "idx": i,
            "problem": ex["problem"],
            "gold": gold,
            "generations": generations,
            "sample_scores": sample_scores,  # 归一化到 [0,1]
            # "raw_scores": raw_scores,        # 原始返回（便于调试）
            "per_problem_score": item_score,
        })

    avg_score = sum(per_item_scores) / len(per_item_scores)
    return {
        "model": model_name,
        "avg_accuracy": avg_score,
        "n_samples_per_problem": n,
        "num_items": len(items),
        "details": details,
    }

def save_json(results: List[Dict[str, Any]], path: str):
    """写入最终 JSON。"""
    summary = [
        {
            "model": r["model"],
            "avg_accuracy": r["avg_accuracy"],
            "n_samples_per_problem": r["n_samples_per_problem"],
            "num_items": r["num_items"],
        }
        for r in results
    ]
    payload = {"summary": summary, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id/local path")
    ap.add_argument("--hf_dataset", type=str, default="HuggingFaceH4/math-500",
                    help="HF dataset with fields: problem, answer")
    ap.add_argument("--data_path", type=str, default=None,
                    help="Optional local JSONL (problem/answer). If provided, overrides --hf_dataset.")
    ap.add_argument("--limit", type=int, default=500, help="Number of problems to evaluate (<=500)")
    ap.add_argument("--n_samples", type=int, default=4, help="Samples per problem")
    ap.add_argument("--max_tokens", type=int, default=int(4 * 1024))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default=None, help="bfloat16/float16/float32, etc.")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--gpu_mem_util", type=float, default=0.95)
    ap.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path if different from model")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()
    
    # setup output json path
    if args.out_json is None:
        output_folder = "outputs"
        os.makedirs(output_folder, exist_ok=True)
        args.out_json = os.path.join(output_folder, f"{args.model.replace('/', '_')}_math500_results.json")

    items = load_math500(None if not args.data_path else args.hf_dataset,
                         args.data_path,
                         args.limit)

    common = dict(
        items=items,
        n=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_mem_util,
        tokenizer=args.tokenizer,
    )

    print(f"Running evaluation for model: {args.model}")
    result = run_eval_for_model(args.model, **common)
    print(f"Avg Accuracy: {result['avg_accuracy']:.4f} over {result['num_items']} problems.")

    save_json([result], args.out_json)
    print(f"Saved JSON to {args.out_json}")
    
    # print all args to terminal
    print("=== Evaluation Arguments ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")

    print("\n=== Summary ===")
    print(f"Model: {args.model}")
    print(f"  Mean Accuracy: {result['avg_accuracy']:.4f}")
    print(f"  Total Problems: {result['num_items']}")
    print(f"  Samples per Problem: {result['n_samples_per_problem']}")

if __name__ == "__main__":
    main()
