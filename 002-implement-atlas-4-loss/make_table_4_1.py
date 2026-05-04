"""
Generate Table 4.1 (Checkpoint-level Model Class Distances) in markdown,
following the exact (Layer, Module, Domain) row layout the user specified
in task.md.
"""
import pandas as pd

CSV_PATH = "/ssd1/zhizhou/workspace/rotation-project/replace/002-implement-atlas-4-loss/stiefel_analysis_metrics.csv"
OUT_MD   = "/ssd1/zhizhou/workspace/rotation-project/replace/002-implement-atlas-4-loss/table_4_1.md"

# Row layout per transition: (Layer, Module, Domain) — copied verbatim from task.md.
ROWS_PER_TRANSITION = {
    "Base->Stage1": [
        (0,  "self_attn.q_proj", "Lang"),
        (10, "self_attn.q_proj", "Lang"),
        (20, "self_attn.q_proj", "Lang"),
        (35, "self_attn.q_proj", "Lang"),
        (0,  "attn.qkv",         "Vis"),
        (15, "attn.qkv",         "Vis"),
        (31, "attn.qkv",         "Vis"),
    ],
    "Stage1->Stage2": [
        (0,  "self_attn.q_proj", "Lang"),
        (10, "mlp.gate_proj",    "Lang"),
        (20, "mlp.up_proj",      "Lang"),
        (35, "mlp.down_proj",    "Lang"),
        (0,  "attn.proj",        "Vis"),
        (15, "mlp.gate_proj",    "Vis"),
        (31, "mlp.up_proj",      "Vis"),
    ],
    "Base->Stage2": [
        (0,  "self_attn.q_proj", "Lang"),
        (10, "self_attn.k_proj", "Lang"),
        (20, "self_attn.v_proj", "Lang"),
        (35, "self_attn.o_proj", "Lang"),
        (0,  "mlp.gate_proj",    "Vis"),
        (15, "mlp.up_proj",      "Vis"),
        (31, "mlp.down_proj",    "Vis"),
    ],
}

DISPLAY_TITLE = {
    "Base->Stage1":   "Base → Stage1",
    "Stage1->Stage2": "Stage1 → Stage2",
    "Base->Stage2":   "Base → Stage2",
}


def fmt(x: float) -> str:
    return f"{x:.3e}"


def main():
    df = pd.read_csv(CSV_PATH)
    # Build a lookup keyed by (Module, Transition).
    df["key"] = list(zip(df["Module"], df["Transition"]))
    lookup = df.set_index("key")

    lines = ["# Table 4.1: Checkpoint-level Model Class Distances", ""]

    for transition, rows in ROWS_PER_TRANSITION.items():
        lines.append(f"## {DISPLAY_TITLE[transition]}")
        lines.append("")
        lines.append("| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |")
        lines.append("|-------|--------|--------|-------|-----|-----|-------|")
        for layer, module, domain in rows:
            module_name = f"{domain}_L{layer}.{module}"
            try:
                r = lookup.loc[(module_name, transition)]
            except KeyError:
                lines.append(f"| L{layer} | {module} | {domain} | MISSING | MISSING | MISSING | MISSING |")
                continue
            lines.append(
                f"| L{layer} | {module} | {domain} | "
                f"{fmt(r['e_src'])} | {fmt(r['e_L'])} | {fmt(r['e_R'])} | {fmt(r['e_tgt'])} |"
            )
        lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
