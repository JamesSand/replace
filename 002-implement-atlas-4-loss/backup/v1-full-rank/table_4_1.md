# Table 4.1: Checkpoint-level Model Class Distances

## Base → Stage1

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | 4.513e-13 | 2.483e-13 | 2.495e-13 | 1.054e-03 |
| L10 | self_attn.q_proj | Lang | 3.092e-13 | 1.711e-13 | 1.733e-13 | 1.460e-03 |
| L20 | self_attn.q_proj | Lang | 2.814e-13 | 1.578e-13 | 1.597e-13 | 1.717e-03 |
| L35 | self_attn.q_proj | Lang | 2.962e-13 | 1.621e-13 | 1.639e-13 | 1.479e-03 |
| L0 | attn.qkv | Vis | 4.251e-02 | 4.251e-02 | 1.035e-13 | 2.067e-03 |
| L15 | attn.qkv | Vis | 5.913e-02 | 5.913e-02 | 1.221e-13 | 1.392e-03 |
| L31 | attn.qkv | Vis | 7.482e-02 | 7.482e-02 | 1.735e-13 | 1.619e-03 |

## Stage1 → Stage2

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | 4.341e-13 | 2.396e-13 | 2.419e-13 | 3.635e-03 |
| L10 | mlp.gate_proj | Lang | 3.836e-01 | 3.836e-01 | 5.536e-13 | 4.048e-03 |
| L20 | mlp.up_proj | Lang | 3.935e-01 | 3.935e-01 | 5.177e-13 | 4.673e-03 |
| L35 | mlp.down_proj | Lang | 3.027e-01 | 5.114e-13 | 3.027e-01 | 4.109e-03 |
| L0 | attn.proj | Vis | 7.680e-14 | 4.403e-14 | 4.425e-14 | 3.725e-03 |
| L15 | mlp.gate_proj | Vis | 1.606e-01 | 1.606e-01 | 1.134e-13 | 3.798e-03 |
| L31 | mlp.up_proj | Vis | 1.822e-01 | 1.822e-01 | 1.780e-13 | 4.131e-03 |

## Base → Stage2

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | 4.513e-13 | 2.483e-13 | 2.495e-13 | 3.723e-03 |
| L10 | self_attn.k_proj | Lang | 5.917e-02 | 4.654e-14 | 5.917e-02 | 1.781e-03 |
| L20 | self_attn.v_proj | Lang | 6.042e-02 | 3.853e-14 | 6.042e-02 | 1.899e-03 |
| L35 | self_attn.o_proj | Lang | 3.527e-13 | 1.935e-13 | 1.958e-13 | 1.177e-02 |
| L0 | mlp.gate_proj | Vis | 1.376e-01 | 1.376e-01 | 9.904e-14 | 5.215e-03 |
| L15 | mlp.up_proj | Vis | 1.667e-01 | 1.667e-01 | 1.268e-13 | 3.813e-03 |
| L31 | mlp.down_proj | Vis | 1.808e-01 | 1.690e-13 | 1.808e-01 | 4.175e-03 |
