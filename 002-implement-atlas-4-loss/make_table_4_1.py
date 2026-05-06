"""
Generate Table 4.1 (Checkpoint-level Model Class Distances) in markdown.
v4 multi-r: emits one set of 3 transition sub-tables per truncation rank r.
The (Layer, Module, Domain) row layout per transition is fixed (from task.md).
"""
import os
import sys
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


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x)))}"
    except Exception:
        return str(x)


def fmt_pct(x) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return str(x)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Detect r values present in the CSV (v4 multi-r writes one row per (Module, Transition, r))
    if "r" not in df.columns:
        print("CSV has no 'r' column — old format. Bailing.", file=sys.stderr)
        sys.exit(1)

    r_values = sorted(df["r"].dropna().unique().astype(int).tolist())
    print(f"Found r values: {r_values}", file=sys.stderr)

    # MultiIndex keyed by (Module, Transition, r) for fast lookup
    lookup = df.set_index(["Module", "Transition", "r"])

    lines = [
        "# Table 4.1: Checkpoint-level Model Class Distances",
        "",
        f"_Source CSV: `{os.path.basename(CSV_PATH)}` ({len(df)} rows, r ∈ {r_values})_",
        "",
        "_Each block below uses one fixed truncation rank r. Within a block the same_",
        "_(Layer, Module, Domain) layout is reused so you can compare across r values._",
        "",
    ]

    for r in r_values:
        lines.append(f"---")
        lines.append("")
        lines.append(f"# r = {r}")
        lines.append("")

        for transition, rows in ROWS_PER_TRANSITION.items():
            lines.append(f"## {DISPLAY_TITLE[transition]} (r={r})")
            lines.append("")
            lines.append("| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |")
            lines.append("|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|")
            for layer, module, domain in rows:
                module_name = f"{domain}_L{layer}.{module}"
                try:
                    row = lookup.loc[(module_name, transition, r)]
                except KeyError:
                    lines.append(f"| L{layer} | {module} | {domain} | — | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| L{layer} | {module} | {domain} | "
                    f"{fmt_pct(row['retained_energy_src'])} | "
                    f"{float(row['gap_rel']):.2f} | "
                    f"{fmt(row['e_src'])} | {fmt(row['e_L'])} | {fmt(row['e_R'])} | {fmt(row['e_tgt'])} |"
                )
            lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
