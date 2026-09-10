# C1* Ablation Ladder (Grok Bot)

**Generated (Asia/Shanghai):** 2026-09-06T02:21:46+0800  
**Tree:** `sdformer_c1c2star_grokbot`  
**Harness:** `flows/oss/ablation_c1_ladder.sh` + `tb_c1star/tb_c1s_ablation_ladder.sv`  
**Stimulus:** 8 deterministic frames, N_TILE=8 (same vectors all modes)  
**Counters:** `c1s_stats` (wake_pop / proj_skip / delta_nz) + `exact_capture` (exact_hit)

## Ladder

| Rung | Mode | wake_pop | proj_skip | delta_nz | exact_hit |
|---:|---|---:|---:|---:|---:|
| 1 | always-ish (`ALWAYS`) | 64 | 0 | 0 | 64 |
| 2 | OP-STW only (`OPSTW`) | 32 | 32 | 0 | 32 |
| 3 | OP-STW + ECP (`ECP`) | 32 | 28 | 0 | 36 |
| 4 | + MW-ΔBuf (`MW`) | 49 | 15 | 22 | 49 |
| 5 | + OGEC×PRRC (`EXACT`) | 49 | 15 | 22 | 19 |

## Interpretation (letter hygiene)

- **ALWAYS**: force all-tile wake (+high corr) — upper bound on PE wake / exact enqueue.
- **OPSTW**: optical-flow/event wake only; corr forced 0 → proj tracks wake; no residual OR.
- **ECP**: real corr_score gates proj with OP-STW wake (no MW).
- **MW**: residual `delta_nz` ORs into wake for ECP (full front-pipe style); `allow_exact=1`.
- **EXACT**: same wake as MW, but PRRC `INIT_BUDGET=3` spends 1 per OGEC beat → `allow_exact` drops → **exact_hit capped** vs MW.
- `exact_hit` = OGEC×capture; CAPACITY=128.
- Not AEE / not silicon power — RTL counter ablation only.

## Logs

- `out/sim/ablation_ALWAYS.log`
- `out/sim/ablation_OPSTW.log`
- `out/sim/ablation_ECP.log`
- `out/sim/ablation_MW.log`
- `out/sim/ablation_EXACT.log`
