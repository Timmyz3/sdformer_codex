# rtl_c2star (Grok Bot)

Isolated C2* RTL. Implements HBG-RP / SMAM-RP / Motion-TTB / STH-Gate / ADP-MAC / ARM-Acc / MFBD / SP-Gate.  
Never edit Codex C2/TSBG sources; read-only.

## c2s_hbg_rp_packetizer (Card B — HBG-RP)

Hybrid binary-gate + real payload packetizer (ATLIF contract r1 defaults:
int8 amp, `EPS=1`, payloads not absorbable into W).

```text
amp_abs  = abs(amp)          // wider path handles most-negative (-128)
g        = amp_valid && (amp_abs > EPS)   // strict >
p        = g ? amp : 0
pe_clk_en = g                // 0 when !amp_valid
gp_valid = registered amp_valid (1 cycle delayed)
```

### Ports

| Port | Dir | Width / type | Description |
|---|---|---|---|
| `clk` | in | 1 | clock |
| `rst_n` | in | 1 | async active-low reset |
| `amp_valid` | in | 1 | amplitude valid strobe |
| `amp` | in | signed `AMP_W` | ATLIF amplitude |
| `g` | out | 1 | binary gate (combinational) |
| `p` | out | signed `AMP_W` | real payload (combinational) |
| `gp_valid` | out | 1 | registered handshake (`amp_valid` + 1) |
| `pe_clk_en` | out | 1 | same as `g` |

### Parameters

| Param | Default | Notes |
|---|---|---|
| `AMP_W` | 8 | amplitude width (int8) |
| `EPS` | `8'sd1` | absolute threshold; gate iff `amp_abs > EPS` |

TB: `tb_c2star/tb_c2s_hbg_rp_packetizer.sv`.

## c2s_smam_rp (SMAM-RP — dual-rail Mask-Add × ATLIF payload)

Real dual-rail unit (replaces stub guts):

```text
mac_en / mask_add_en = valid && spike_gate
mac_payload          = gate_fire ? payload : 0
mask_add_result      += 1 when gate_fire   // registered accumulate
out_valid            = registered valid
```

### Ports

| Port | Dir | Width / type | Description |
|---|---|---|---|
| `clk` / `rst_n` | in | 1 | clock / async reset |
| `valid` | in | 1 | beat valid |
| `spike_gate` | in | 1 | binary gate |
| `payload` | in | signed `AMP_W` | ATLIF payload |
| `mac_en` | out | 1 | MAC enable (gate rail) |
| `mask_add_en` | out | 1 | Mask-Add enable |
| `mac_payload` | out | signed `AMP_W` | payload when gated |
| `mask_add_result` | out | signed `ACC_W` | accumulate count |
| `out_valid` | out | 1 | registered valid |

TB: `tb_c2star/tb_c2s_smam_rp.sv`. Wired in `c2s_top`.

## c2s_smam_rp_stub

Thin wrapper instantiating `c2s_smam_rp` with legacy ports
(`gp_valid`/`g`/`p` → `valid`/`spike_gate`/`payload`; `payload_q` ← `mac_payload`).
Keeps older instantiations working; prefer `c2s_smam_rp` directly.

## c2s_motion_ttb_packer (Motion-TTB — midend packer)

Pack wake tiles into motion time-bundles `(tile_id, dt_bin, hyp_id)` for
MFBD-style delivery (Bishop TTB / Motion-TTB+OF-ECP idea packs).

```text
when valid_i:
  scan wake_bitmap[0..N_TILE-1] ascending
  pack up to MAX_BUNDLES entries {tile, dt_bin[tile], hyp_id[tile]}
bundle_count / bundle_* / bundle_valid  // registered 1-cycle
```

### Parameters

| Param | Default | Notes |
|---|---|---|
| `N_TILE` | 16 | 8 or 16 for TB |
| `DT_W` | 3 | Δt bin width |
| `HYP_W` | 2 | hyp id width |
| `MAX_BUNDLES` | 8 | pack cap |

### Ports

| Port | Dir | Description |
|---|---|---|
| `wake_bitmap` | in | per-tile wake |
| `dt_bin` / `hyp_id` | in | per-tile unpacked arrays (sim) |
| `bundle_tile/dt/hyp` | out | packed buses (entry i at `[i*W +: W]`) |
| `bundle_count` | out | number packed |
| `bundle_valid` | out | registered `valid_i` |

TB: `tb_c2star/tb_c2s_motion_ttb_packer.sv`.  
Synth flat: `flows/oss/wrappers/c2s_motion_ttb_synth.sv`.  
Wired thinly in `c2s_top` (`ttb_*` ports).

## c2s_sth_gate (STH-Gate — Spatial–Temporal Head Gate)

Idea 08 Sparse VideoGen remake: classify each head/token lane as spatial
vs temporal and emit enable masks for SDSA scheduling.

```text
when valid_i (registered +1 cycle):
  spat_en[i] = spat_score[i] > TH_SP
  temp_en[i] = temp_score[i] > TH_TP
  dual_en[i] = spat_en[i] & temp_en[i]   // both strong — rare
  skip_en[i] = ~(spat_en[i] | temp_en[i])
  gate_valid = registered valid_i
```

### Parameters

| Param | Default | Notes |
|---|---|---|
| `N_HEAD` | 8 | head/token lanes |
| `SCORE_W` | 8 | score bitwidth |
| `TH_SP` | `8'd2` | spatial threshold (strict `>`) |
| `TH_TP` | `8'd2` | temporal threshold (strict `>`) |

### Ports

| Port | Dir | Description |
|---|---|---|
| `spat_score` / `temp_score` | in | per-head unpacked scores |
| `spat_en` / `temp_en` | out | packed `[N_HEAD-1:0]` typed enables |
| `dual_en` | out | packed both-strong |
| `skip_en` | out | packed neither |
| `gate_valid` | out | registered `valid_i` |

TB: `tb_c2star/tb_c2s_sth_gate.sv`.  
Synth flat: `flows/oss/wrappers/c2s_sth_gate_synth.sv`.  
Wired thinly in `c2s_top` (`sth_*` ports).


## c2s_back_pipe (thin C2* mid/backend)

HBG-RP → held g/p → SMAM-RP; STH-Gate parallel (separate valid OK).  
Motion-TTB stays as side ports on `c2s_top` only.

TB: `tb_c2star/tb_c2s_back_pipe.sv`  
Sim: `flows/oss/sim_back_pipe.sh`  
Synth flat: `flows/oss/wrappers/c2s_back_pipe_synth.sv` (@ N_HEAD=8)


## c2s_adp_mac (ADP-MAC — bilateral bit-sparse MAC)

Preserves ATLIF amplitude on side `a` (no multiply-order swap). Dual-side
skip + SMAM/HBG `mac_en` gate the accumulate.

```text
do_mac   = mac_en && !skip_a && !skip_b
skipped  = !do_mac                         (combinational)
when valid_i && do_mac:  acc <= acc + (a * b)   // signed, width-safe
when clear_i:            acc <= 0               // optional clear (added)
out_valid                = registered valid_i
```

Skip / `!mac_en` holds `acc`. `clear_i` added beyond sketch (useful for TB / PE reset).

### Ports

| Port | Dir | Description |
|---|---|---|
| `a` | in | signed ATLIF payload (`W_A`) |
| `b` | in | signed weight (`W_B`) |
| `skip_a` / `skip_b` | in | bilateral bit-skip sides |
| `mac_en` | in | gate from SMAM/HBG |
| `clear_i` | in | clear accumulator (added) |
| `acc` | out | signed `W_ACC` accumulate |
| `skipped` | out | 1 if skip or !mac_en |
| `out_valid` | out | registered `valid_i` |

TB: `tb_c2star/tb_c2s_adp_mac.sv`  
Sim: `flows/oss/sim_adp_mac.sh`  
Synth: native RTL (no flat wrapper needed)  
Wired thinly in `c2s_top` (`adp_*` **parallel side ports** — not inside `back_pipe`).


## c2s_arm_acc (ARM-Acc — multi-hypothesis accumulator)

hARMS-style aperture contexts: N_HYP accumulators; `hyp_sel` steers signed add.

```text
when clear_i:            all acc <= 0
when valid_i && add_en:  acc[hyp_sel] <= acc[hyp_sel] + sext(data_i)
out_valid                = registered valid_i
acc_bus                  = packed [N_HYP*ACC_W-1:0]  // entry i at [i*ACC_W +: ACC_W]
```

| Param | Default | Notes |
|---|---|---|
| `N_HYP` | 4 | hypothesis contexts |
| `ACC_W` | 16 | accumulator width |
| `DATA_W` | 8 | signed input |

TB: `tb_c2star/tb_c2s_arm_acc.sv`  
Sim: `flows/oss/sim_arm_acc.sh`  
Synth: native RTL  
Wired thinly in `c2s_top` (`arm_*` parallel side ports — not inside `back_pipe`).

## c2s_mfbd (MFBD — motion-bundle delivery)

Steers a payload to a hyp destination lane from Motion-TTB-style packed
bundle descriptors. Simple beat: use **bundle[0]** `hyp_id` as one-hot lane.

```text
when valid_i && bundle_count!=0:
  lane_valid   = 1 << bundle_hyp[0]
  lane_payload = payload
deliver_valid  = registered valid_i
```

| Param | Default | Notes |
|---|---|---|
| `MAX_B` | 4 | max bundle slots |
| `TILE_W` / `DT_W` / `HYP_W` | 4 / 3 / 2 | descriptor fields |
| `PAY_W` | 8 | signed payload |
| `N_HYP` | `1<<HYP_W` | destination lanes |

TB: `tb_c2star/tb_c2s_mfbd.sv` (≥2 cases)  
Sim: `flows/oss/sim_mfbd.sh`  
Synth: native RTL (packed I/O)  
Wired thinly in `c2s_top` (`mfbd_*` side ports).

## c2s_sp_gate (SP-Gate — attention-mass schedule gate)

SpAtten-style mass threshold gate (not multiply reorder).

```text
run_en[i]  = (mass[i] > TH_M) | force_bitmap[i]
drop_en[i] = ~run_en[i]
gate_valid = registered valid_i   // +1 cycle
```

| Param | Default | Notes |
|---|---|---|
| `N` | 8 | lanes |
| `MASS_W` | 8 | mass width |
| `TH_M` | `8'd2` | strict `>` |

TB: `tb_c2star/tb_c2s_sp_gate.sv`  
Sim: `flows/oss/sim_sp_gate.sh`  
Synth flat: `flows/oss/wrappers/c2s_sp_gate_synth.sv`  
Wired thinly in `c2s_top` (`sp_*` side ports).


## c2s_stats

Window counters: `mac_en_cnt` / `skip_cnt` / `gate_fire_cnt` under `window_en`.
Measures dual-rail gate+payload activity — not multiply-reorder bookkeeping.

TB: `tb_c2star/tb_c2s_stats.sv`  
Sim: `flows/oss/sim_c2s_stats.sh`  
Side-wired in `c2s_top`.
