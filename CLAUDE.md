# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements SVD-based model weight blending for language models, specifically focused on analyzing and blending models before and after reinforcement learning (RL). The primary use case is creating hybrid models by linearly interpolating singular values between a base model and an RL-trained variant.

## Core Architecture

### SVD Weight Blending Pipeline

The repository implements a two-stage workflow:

1. **SVD Decomposition** (`svd_decompose.py`): Decomposes each 2D weight matrix of a model into U, S, Vh components
2. **Sigma Blending** (`replace.py`): Blends singular values (S) between two models while preserving the rotation matrices (U, Vh) from the target model

**Key insight**: The blending process uses `S_blend = alpha * S_after + (1-alpha) * S_before` while keeping U and Vh from the "after" model. This creates a spectrum of models where alpha=0.0 recovers the "before" model's behavior and alpha=1.0 keeps the "after" model unchanged.

### Model Comparison Framework

- **Base model**: Pre-RL trained model (e.g., `Qwen/Qwen2.5-Math-1.5B`)
- **After RL models**: Post-RL variants (e.g., `DeepSeek-R1-Distill-Qwen-1.5B`, `DeepScaleR-1.5B-Preview`)
- **Alpha sweep**: Generates models at alpha values [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

## Key Scripts and Usage

### SVD Decomposition

```bash
# Decompose a single model
python svd_decompose.py --model_id "Qwen/Qwen2.5-VL-3B-Instruct" --output_dir ./svd_output

# Decompose multiple models in parallel (uses run_svd_decompose.sh)
bash run_svd_decompose.sh
```

- Outputs: `.pt` files containing U, S, Vh, orig_shape for each layer
- Default output location: `qwenvl-svd/{model_basename}/`
- Parameter names are sanitized: dots → `__`, slashes → `_slash_`

### Sigma Blending

```bash
# Generate blended models for all alpha values
python replace.py
```

Configuration is done by editing constants at the top of `replace.py`:
- `MODEL_BEFORE`: Base model path/ID
- `MODEL_AFTER`: Target model path/ID
- `BASE_OUTPUT_DIR`: Output directory for blended models
- `ALPHAS`: List of alpha values to generate

Outputs:
- Models saved to: `{BASE_OUTPUT_DIR}/alpha_{alpha:.1f}/`
- Metadata: `svd_blend_meta.json` in each alpha directory
- Sigma logs: `{BASE_LOG_DIR}/alpha_{alpha:.1f}/{layer_name}.pt`

### Evaluation

```bash
# Evaluate a single model on math datasets
python eval_v1.py \
  --model <model_path> \
  --tokenizer <tokenizer_path> \
  --data_path "HuggingFaceH4/math-500" \
  --n_samples 16 \
  --max_tokens 16384 \
  --out_json results.json

# Evaluate all alpha variants in parallel
bash run_all_eval.sh

# Evaluate before-RL model only
bash run_before_rl.sh
```

Supported datasets:
- `HuggingFaceH4/math-500`
- `HuggingFaceH4/aime_2024`
- `math-ai/amc23`
- `math-ai/aime25`

The evaluation uses vLLM for inference and `math_dapo.compute_score()` for grading, which supports both Minerva-style answer extraction and algebraic equivalence checking via `math-verify`.

### Rotation Analysis

```bash
# Compare rotation matrices across three models
python svd_compare.py
```

This script analyzes how the U and Vh rotation matrices change across model variants:
- Computes rotation matrices: R1 (base→rl1), R2 (rl1→rl2), R3 (base→rl2)
- Validates R4 = R1 @ R2 ≈ R3 (compositional property)
- Outputs MSE curves showing rotation stability across layers
- Generates visualization: `rotation_mse_curves.png`

Configuration: Edit `MODEL_BASE`, `MODEL_RL1`, `MODEL_RL2` at the top of the script.

## Important Implementation Details

### SVD Processing

- All SVD operations are performed in **float32** on CPU to ensure numerical stability
- Only 2D weight tensors are processed (attention projections, MLP layers)
- Shape validation: Skips layers with mismatched shapes between before/after models
- Memory management: Intermediate tensors are explicitly deleted after each layer

### Model Saving

- Models are saved using `save_pretrained(safe_serialization=True)`
- Metadata includes: blended keys, skipped keys, layer shapes, alpha value
- Tokenizer should be loaded from the original models (not blended)

### Evaluation Configuration

The evaluation scripts use parallel GPU execution:
- `run_all_eval.sh`: Cycles through alphas on multiple GPUs
- GPU assignment: `gpu = ${gpus[$((i % num_gpus))]}`
- Results saved to: `gen_results/{model_name}_results.json`
- Logs saved to: `logs/{model_name}.log`

### Math Scoring

The `math_dapo.py` module provides the scoring function:
```python
compute_score(solution_str, ground_truth, is_longcot=False, is_use_math_verify=True)
```

Returns: 1.0 for correct, 0.0 for incorrect, negative values for format errors

Key features:
- Extracts last `\boxed{}` expression from solution
- Supports both string matching and semantic algebraic equivalence
- Handles multiple ground truth formats

## Directory Structure

```
.
├── replace.py              # Main sigma blending script
├── svd_decompose.py        # SVD decomposition for single model
├── svd_compare.py          # Rotation matrix analysis
├── eval_v1.py             # Evaluation on math benchmarks
├── math_dapo.py           # Math answer verification
├── run_svd_decompose.sh   # Parallel SVD for multiple models
├── run_all_eval.sh        # Parallel evaluation across alphas
├── run_before_rl.sh       # Evaluate base model only
├── qwenvl-svd/            # SVD decomposition outputs
├── svd_logs/              # Sigma blending logs
├── gen_results/           # Evaluation results (JSON)
├── logs/                  # Evaluation logs
└── outputs/               # Default eval output directory
```

## Development Notes

### Modifying Alpha Values

Edit the `ALPHAS` list in `replace.py` or `run_all_eval.sh` to control which interpolation points are generated/evaluated.

### Adding New Model Pairs

1. Update `MODEL_BEFORE` and `MODEL_AFTER` in `replace.py`
2. Update `BASE_OUTPUT_DIR` to a descriptive name
3. Update tokenizer path in evaluation scripts if models use different tokenizers

### GPU Memory Management

- `eval_v1.py` uses `--gpu_mem_util` (default 0.95) to control vLLM memory usage
- For OOM issues, reduce this value or decrease `n_samples`/`max_tokens`
- The evaluation scripts use `CUDA_VISIBLE_DEVICES` for GPU assignment

### Debugging SVD Issues

Common issues tracked in `svd_blend_meta.json`:
- `missing_in_before`: Layer exists in after but not before model
- `ndim!=2`: Non-matrix parameters (scalars, vectors)
- `shape_mismatch`: Different dimensions between models
- `sigma_mismatch`: Rank mismatch in SVD decomposition

Check the `skipped_keys` field in metadata to diagnose blending failures.
