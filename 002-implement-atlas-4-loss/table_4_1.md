# Table 4.1: Checkpoint-level Model Class Distances

## Base → Stage1

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | 7.531e+00 | 7.531e+00 | 7.531e+00 | 1.054e-03 |
| L10 | self_attn.q_proj | Lang | 5.478e+00 | 5.478e+00 | 5.478e+00 | 1.460e-03 |
| L20 | self_attn.q_proj | Lang | 5.444e+00 | 5.444e+00 | 5.444e+00 | 1.717e-03 |
| L35 | self_attn.q_proj | Lang | 4.763e+00 | 4.763e+00 | 4.763e+00 | 1.479e-03 |
| L0 | attn.qkv | Vis | 4.394e+00 | 4.394e+00 | 4.394e+00 | 2.067e-03 |
| L15 | attn.qkv | Vis | 4.156e+00 | 4.156e+00 | 4.156e+00 | 1.392e-03 |
| L31 | attn.qkv | Vis | 5.611e+00 | 5.611e+00 | 5.610e+00 | 1.619e-03 |

## Stage1 → Stage2

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | 7.532e+00 | 7.531e+00 | 7.531e+00 | 3.635e-03 |
| L10 | mlp.gate_proj | Lang | 1.431e+01 | 1.431e+01 | 1.430e+01 | 4.048e-03 |
| L20 | mlp.up_proj | Lang | 1.270e+01 | 1.270e+01 | 1.269e+01 | 4.673e-03 |
| L35 | mlp.down_proj | Lang | 1.211e+01 | 1.211e+01 | 1.211e+01 | 4.109e-03 |
| L0 | attn.proj | Vis | 1.652e+00 | 1.652e+00 | 1.652e+00 | 3.725e-03 |
| L15 | mlp.gate_proj | Vis | 3.585e+00 | 3.585e+00 | 3.582e+00 | 3.798e-03 |
| L31 | mlp.up_proj | Vis | 5.349e+00 | 5.349e+00 | 5.346e+00 | 4.131e-03 |

## Base → Stage2

| Layer | Module | Domain | e_src | e_L | e_R | e_tgt |
|-------|--------|--------|-------|-----|-----|-------|
| L0 | self_attn.q_proj | Lang | 7.532e+00 | 7.532e+00 | 7.531e+00 | 3.723e-03 |
| L10 | self_attn.k_proj | Lang | 1.999e+00 | 1.998e+00 | 1.999e+00 | 1.781e-03 |
| L20 | self_attn.v_proj | Lang | 1.684e+00 | 1.683e+00 | 1.684e+00 | 1.899e-03 |
| L35 | self_attn.o_proj | Lang | 5.748e+00 | 5.747e+00 | 5.747e+00 | 1.177e-02 |
| L0 | mlp.gate_proj | Vis | 3.687e+00 | 3.687e+00 | 3.684e+00 | 5.215e-03 |
| L15 | mlp.up_proj | Vis | 3.843e+00 | 3.843e+00 | 3.840e+00 | 3.813e-03 |
| L31 | mlp.down_proj | Vis | 5.277e+00 | 5.274e+00 | 5.277e+00 | 4.175e-03 |
