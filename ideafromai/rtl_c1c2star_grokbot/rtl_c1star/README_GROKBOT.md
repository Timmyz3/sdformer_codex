# rtl_c1star (Grok Bot)

Isolated C1* RTL. Implements OP-STW / PRRC / OGEC per microarch sketch.  
Never edit Codex C1 sources; read them only for interface inspiration.

## c1s_op_stw_predictor (Card A — OP-STW)

Optical-flow predictive spike-tile wake. When `valid_i` is asserted, each tile
computes:

```text
wake[i] = (abs(flow_cur[i] - flow_prev[i]) > TH_W) || (event_cnt[i] > TH_E)
```

`wake_bitmap` and `wake_valid` are registered: `wake_valid` is high one cycle
after `valid_i`. Reset (`rst_n=0`) clears both.

### Ports

| Port | Dir | Width / type | Description |
|---|---|---|---|
| `clk` | in | 1 | clock |
| `rst_n` | in | 1 | async active-low reset |
| `valid_i` | in | 1 | sample / compute strobe |
| `flow_cur[N_TILE]` | in | signed `FLOW_W` | current optical-flow per tile |
| `flow_prev[N_TILE]` | in | signed `FLOW_W` | previous optical-flow per tile |
| `event_cnt[N_TILE]` | in | `EVT_W` | event count per tile |
| `wake_bitmap` | out | `N_TILE` | registered wake bits |
| `wake_valid` | out | 1 | registered, 1 cycle after `valid_i` |

### Parameters

| Param | Default | Notes |
|---|---|---|
| `N_TILE` | 64 | number of tiles |
| `FLOW_W` | 8 | signed flow width |
| `EVT_W` | 8 | event-count width |
| `TH_W` | `8'sd2` | abs flow-delta threshold (strict `>`) |
| `TH_E` | `8'd1` | event-count threshold (strict `>`) |

TB: `tb_c1star/tb_c1s_op_stw_predictor.sv` (uses `N_TILE=8`).

## c1s_ecp_qkv_predictor (Card C — ECP-QKV)

Eager correlation prediction before QKV projection (idea pack 09 / FACT-style).
`proj_en[i] = (corr_score[i] > TH_S) || (use_wake && wake_bitmap[i])`.
Registered one cycle after `valid_i`. `skip_bitmap = ~proj_en_bitmap`.

TB: `tb_c1star/tb_c1s_ecp_qkv_predictor.sv` (N_TILE=8).

## c1s_mw_delta_buf (MW-ΔBuf — motion-warped residual)

Per-tile residual buffer (idea packs 08/09):

```text
delta[i]    = cur_samp[i] - ref_samp[i]     // signed SAMP_W+1
delta_nz[i] = (|delta[i]| > TH_D)           // sparse mask
```

`delta` / `delta_nz` / `delta_valid` are registered: `delta_valid` high one
cycle after `valid_i`. Reset clears residuals and mask.

### Ports

| Port | Dir | Width / type | Description |
|---|---|---|---|
| `clk` | in | 1 | clock |
| `rst_n` | in | 1 | async active-low reset |
| `valid_i` | in | 1 | sample strobe |
| `ref_samp[N_TILE]` | in | signed `SAMP_W` | reference / motion-comp sample |
| `cur_samp[N_TILE]` | in | signed `SAMP_W` | warped / current sample |
| `delta` | out | signed `N_TILE*(SAMP_W+1)` packed | registered residual (tile-major) |
| `delta_nz` | out | `N_TILE` | sparse nonzero mask |
| `delta_valid` | out | 1 | registered, 1 cycle after `valid_i` |

Packed layout: tile `i` residual in `delta[i*(SAMP_W+1) +: (SAMP_W+1)]` (iverilog-safe).

### Parameters

| Param | Default | Notes |
|---|---|---|
| `N_TILE` | 64 | number of tiles |
| `SAMP_W` | 8 | sample width |
| `TH_D` | `8'd0` | abs residual threshold (strict `>`) |

TB: `tb_c1star/tb_c1s_mw_delta_buf.sv` (N_TILE=8).
Synth flat: `flows/oss/wrappers/c1s_mw_delta_synth.sv`.
Wired in `c1s_top` alongside OP-STW / ECP-QKV.

## c1s_ogec_gate (OGEC — Occlusion-Gated Exact Capture)

Matched tiles take the exact-product path; unmatched take propagate/fill.

```text
exact_en[i] = match_ok[i]
prop_en[i]  = ~match_ok[i]   // when valid path active
ogec_valid  // registered, 1 cycle after valid_i
```

Ablation off: drive `match_ok = all-1s` → all ExactMatch.

TB: `tb_c1star/tb_c1s_ogec_gate.sv` (N_TILE=8).
Synth: `flows/oss/wrappers/c1s_ogec_synth.sv` (instantiates RTL, N_TILE=8).
Wired in `c1s_top`.

## c1s_front_pipe (thin C1* front-end)

Serial concept: **OP-STW → ECP-QKV (use_wake=1)** with **MW-ΔBuf** parallel to OP-STW.
`delta_nz` ORs into wake for ECP; combined output:

```text
tile_active = wake | proj_en | delta_nz   // valid with proj_valid (2 cycles after valid_i)
```

Default / TB / synth: **N_TILE=8**. Instantiates existing predictors (does not modify them).

TB: `tb_c1star/tb_c1s_front_pipe.sv`
Synth flat: `flows/oss/wrappers/c1s_front_pipe_synth.sv`


## c1s_prrc_ledger (PRRC — Pyramid Residual Budget / ledger)

Tracks remaining **exact-capture budget** per pyramid level. Coarse levels
seed budget; fine levels only take the exact path while `allow_exact=1`.

```text
reset:           budget[level] = INIT_BUDGET   (all levels)
valid && spend:  if budget[sel] > 0 → budget[sel] -= 1
valid && refill: budget[sel] = INIT_BUDGET     (refill wins over spend)
allow_exact:     budget[sel] > 0               (combinational)
ledger_valid:    registered valid_i (+1 cycle)
sel:             level_idx, or one-hot level_sel when |level_sel| != 0
```

### Port refinements vs sketch

| Sketch | Refined | Why |
|---|---|---|
| `output logic [BUDGET_W-1:0] budget [N_LEVEL]` unpacked | packed `budget [N_LEVEL*BUDGET_W-1:0]` | iverilog cannot drive unpacked array *outputs* |
| — | `clear`-style `refill_i` on selected level | as specified |
| — | added `clear_i` N/A | — |

Level `i` residual budget: `budget[i*BUDGET_W +: BUDGET_W]` (level0 = LSBs).
Default / TB / synth: **N_LEVEL=3**, **INIT_BUDGET=16** (TB uses 4 for spend-to-zero).
Explicit b0/b1/b2 banks (N_LEVEL=3 iverilog-friendly).

TB: `tb_c1star/tb_c1s_prrc_ledger.sv`  
Sim: `flows/oss/sim_prrc.sh`  
Synth: `flows/oss/wrappers/c1s_prrc_synth.sv`  
Wired thinly in `c1s_top` (`prrc_*` parallel side ports).


## c1s_exact_capture_wrap (Exact-product capture)

Gated by **OGEC `exact_en` × PRRC `allow_exact`**. When `capture_en` & `valid_i`,
latch OR-hit bitmap and accumulate popcount into `capture_cnt`. `busy` while
armed; `done` at CAPACITY or when `capture_en` drops (must drop to re-arm).

TB: `tb_c1star/tb_c1s_exact_capture_wrap.sv`  
Sim: `flows/oss/sim_exact_capture.sh`  
Side-wired in `c1s_top`.

## c1s_stats

Window counters: `wake_pop_cnt` / `proj_skip_cnt` / `delta_nz_cnt` under `window_en`.
`clear_i` or `!window_en` resets.

TB: `tb_c1star/tb_c1s_stats.sv`  
Sim: `flows/oss/sim_c1s_stats.sh`  
Side-wired in `c1s_top`.
