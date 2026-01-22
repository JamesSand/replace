import os
import math
from datetime import datetime

import wandb
import pandas as pd


PROJECT_PATH = "zhizhousha/debug0110"
OUTPUT_DIR = "wandb_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 你要统计的指标名（W&B history 里的 key）
METRICS = [
    "timing_s/update_actor",
    "timing_s/step",
    "timing_s/gen",
]

# 方式1：按 run.name 选择（推荐：最直观）
TARGET_RUN_NAMES = [
    "qwen1.7b-adam-reset-muon-lr-1e-6-fp64",
    "qwen1.7b-adam-reset-muon-lr-5e-6-fp64",
    "qwen1.7b-adam-reset-muon-lr-1e-5-fp64",
]

# 方式2：按 run.id 选择（更稳定，不怕重名）
TARGET_RUN_IDS = [
    # "abc123def456",  # 如果你想用 id，就填这里，并把下面 use_ids=True
]

use_ids = False  # True=用 TARGET_RUN_IDS; False=用 TARGET_RUN_NAMES


def safe_mean(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and not math.isnan(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def fetch_metric_means(run, metrics):
    """
    用 scan_history 按 key 流式拉取，避免 history(samples=...) 截断或太慢。
    """
    buf = {k: [] for k in metrics}
    for row in run.scan_history(keys=metrics):
        for k in metrics:
            if k in row:
                buf[k].append(row[k])
    return {k: safe_mean(vs) for k, vs in buf.items()}, {k: buf[k] for k in metrics}


def main():
    api = wandb.Api()
    runs = api.runs(PROJECT_PATH)

    # 选择 runs
    selected = []
    if use_ids:
        wanted = set(TARGET_RUN_IDS)
        for r in runs:
            if r.id in wanted:
                selected.append(r)
    else:
        wanted = set(TARGET_RUN_NAMES)
        for r in runs:
            if r.name in wanted:
                selected.append(r)

    if not selected:
        raise RuntimeError("没选中任何 run。请检查 TARGET_RUN_NAMES / TARGET_RUN_IDS 是否写对。")

    # 逐个 run 统计
    per_run_rows = []
    overall_values = {k: [] for k in METRICS}

    for r in selected:
        means, raw = fetch_metric_means(r, METRICS)

        # overall 合并
        for k in METRICS:
            overall_values[k].extend(raw[k])

        per_run_rows.append({
            "run_name": r.name,
            "run_id": r.id,
            **{f"mean:{k}": means[k] for k in METRICS},
        })

    df = pd.DataFrame(per_run_rows)

    overall_means = {k: safe_mean(vs) for k, vs in overall_values.items()}

    # 输出到命令行 + txt
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(OUTPUT_DIR, f"timing_means_{ts}.txt")

    lines = []
    lines.append(f"Project: {PROJECT_PATH}")
    lines.append(f"Selected runs: {len(selected)}")
    lines.append("")

    lines.append("Per-run means:")
    lines.append(df.to_string(index=False))
    lines.append("")

    lines.append("Overall (all selected runs concatenated) means:")
    for k in METRICS:
        lines.append(f"  {k}: {overall_means[k]}")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nSaved to: {out_txt}")


if __name__ == "__main__":
    main()


