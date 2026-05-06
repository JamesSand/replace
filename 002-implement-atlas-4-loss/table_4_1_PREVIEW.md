# Table 4.1: Checkpoint-level Model Class Distances

_Source CSV: `preview.csv` (279 rows, r ∈ [128, 256, 512, 1024])_

_Each block below uses one fixed truncation rank r. Within a block the same_
_(Layer, Module, Domain) layout is reused so you can compare across r values._

---

# r = 128

## Base → Stage1 (r=128)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 38.3% | 1.00 | 5.681e-02 | 4.047e-02 | 3.987e-02 | 3.120e-04 |
| L10 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L0 | attn.qkv | Vis | — | — | — | — | — | — |
| L15 | attn.qkv | Vis | — | — | — | — | — | — |
| L31 | attn.qkv | Vis | — | — | — | — | — | — |

## Stage1 → Stage2 (r=128)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 38.3% | 1.00 | 1.572e-01 | 1.124e-01 | 1.099e-01 | 1.455e-03 |
| L10 | mlp.gate_proj | Lang | — | — | — | — | — | — |
| L20 | mlp.up_proj | Lang | — | — | — | — | — | — |
| L35 | mlp.down_proj | Lang | — | — | — | — | — | — |
| L0 | attn.proj | Vis | — | — | — | — | — | — |
| L15 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.up_proj | Vis | — | — | — | — | — | — |

## Base → Stage2 (r=128)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 38.3% | 1.00 | 1.636e-01 | 1.172e-01 | 1.143e-01 | 1.483e-03 |
| L10 | self_attn.k_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.v_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.o_proj | Lang | — | — | — | — | — | — |
| L0 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L15 | mlp.up_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.down_proj | Vis | — | — | — | — | — | — |

---

# r = 256

## Base → Stage1 (r=256)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 55.0% | 1.00 | 5.255e-02 | 3.751e-02 | 3.680e-02 | 4.141e-04 |
| L10 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L0 | attn.qkv | Vis | — | — | — | — | — | — |
| L15 | attn.qkv | Vis | — | — | — | — | — | — |
| L31 | attn.qkv | Vis | — | — | — | — | — | — |

## Stage1 → Stage2 (r=256)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 55.0% | 1.00 | 1.847e-01 | 1.317e-01 | 1.296e-01 | 1.745e-03 |
| L10 | mlp.gate_proj | Lang | — | — | — | — | — | — |
| L20 | mlp.up_proj | Lang | — | — | — | — | — | — |
| L35 | mlp.down_proj | Lang | — | — | — | — | — | — |
| L0 | attn.proj | Vis | — | — | — | — | — | — |
| L15 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.up_proj | Vis | — | — | — | — | — | — |

## Base → Stage2 (r=256)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 55.0% | 1.00 | 1.869e-01 | 1.333e-01 | 1.310e-01 | 1.783e-03 |
| L10 | self_attn.k_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.v_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.o_proj | Lang | — | — | — | — | — | — |
| L0 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L15 | mlp.up_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.down_proj | Vis | — | — | — | — | — | — |

---

# r = 512

## Base → Stage1 (r=512)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 76.2% | 1.00 | 6.088e-02 | 4.343e-02 | 4.267e-02 | 5.710e-04 |
| L10 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L0 | attn.qkv | Vis | — | — | — | — | — | — |
| L15 | attn.qkv | Vis | — | — | — | — | — | — |
| L31 | attn.qkv | Vis | — | — | — | — | — | — |

## Stage1 → Stage2 (r=512)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 76.2% | 1.00 | 2.084e-01 | 1.482e-01 | 1.465e-01 | 2.242e-03 |
| L10 | mlp.gate_proj | Lang | — | — | — | — | — | — |
| L20 | mlp.up_proj | Lang | — | — | — | — | — | — |
| L35 | mlp.down_proj | Lang | — | — | — | — | — | — |
| L0 | attn.proj | Vis | — | — | — | — | — | — |
| L15 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.up_proj | Vis | — | — | — | — | — | — |

## Base → Stage2 (r=512)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 76.2% | 1.00 | 2.129e-01 | 1.516e-01 | 1.496e-01 | 2.299e-03 |
| L10 | self_attn.k_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.v_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.o_proj | Lang | — | — | — | — | — | — |
| L0 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L15 | mlp.up_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.down_proj | Vis | — | — | — | — | — | — |

---

# r = 1024

## Base → Stage1 (r=1024)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 94.5% | 1.00 | 8.683e-02 | 6.174e-02 | 6.109e-02 | 7.640e-04 |
| L10 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.q_proj | Lang | — | — | — | — | — | — |
| L0 | attn.qkv | Vis | — | — | — | — | — | — |
| L15 | attn.qkv | Vis | — | — | — | — | — | — |
| L31 | attn.qkv | Vis | — | — | — | — | — | — |

## Stage1 → Stage2 (r=1024)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 94.5% | 1.00 | 1.991e-01 | 1.416e-01 | 1.401e-01 | 2.822e-03 |
| L10 | mlp.gate_proj | Lang | — | — | — | — | — | — |
| L20 | mlp.up_proj | Lang | — | — | — | — | — | — |
| L35 | mlp.down_proj | Lang | — | — | — | — | — | — |
| L0 | attn.proj | Vis | — | — | — | — | — | — |
| L15 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.up_proj | Vis | — | — | — | — | — | — |

## Base → Stage2 (r=1024)

| Layer | Module | Domain | retained_E | gap_rel | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-----------:|--------:|------:|----:|----:|------:|
| L0 | self_attn.q_proj | Lang | 94.5% | 1.00 | 2.062e-01 | 1.467e-01 | 1.449e-01 | 2.892e-03 |
| L10 | self_attn.k_proj | Lang | — | — | — | — | — | — |
| L20 | self_attn.v_proj | Lang | — | — | — | — | — | — |
| L35 | self_attn.o_proj | Lang | — | — | — | — | — | — |
| L0 | mlp.gate_proj | Vis | — | — | — | — | — | — |
| L15 | mlp.up_proj | Vis | — | — | — | — | — | — |
| L31 | mlp.down_proj | Vis | — | — | — | — | — | — |
