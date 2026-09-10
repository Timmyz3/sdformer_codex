# C2* Ablation Ladder (Grok Bot)

**Generated (Asia/Shanghai):** 2026-09-06T13:07:49+0800  
**Tree:** `sdformer_c1c2star_grokbot`  
**Harness:** `flows/oss/ablation_c2_ladder.sh` + `tb_c2star/tb_c2s_ablation_ladder.sv`  
**Stimulus:** 16 deterministic amp beats (EPS=1), same vectors all modes  
**Counters:** `c2s_stats` (mac_en / skipped / gate_fire)

## Ladder

| Rung | Mode | mac_en | skipped | gate_fire |
|---:|---|---:|---:|---:|
| 1 | always mac (`ALWAYS`) | 16 | 0 | 16 |
| 2 | HBG-only EPS (`HBG`) | 8 | 8 | 8 |
| 3 | HBG+SMAM (`HBGSMAM`) | 8 | 8 | 8 |
| 4 | HBG+SMAM+ADP skip sides (`FULL`) | 4 | 12 | 8 |

## Interpretation (letter hygiene)

- **ALWAYS**: force mac_en+gate every beat — upper bound on MAC activity.
- **HBG**: EPS gate only (`|amp|>1`); mac_en tracks gate; skipped=!gate.
- **HBGSMAM**: HBG→SMAM dual-rail; mac_en=SMAM gate_fire (aligned).
- **FULL**: +ADP bilateral skip_a/skip_b; mac_en=do_mac; skipped=ADP skipped; gate_fire=HBG g.
- Not silicon power / not AEE — RTL counter ablation only.

## Logs

- `out/sim/ablation_c2_ALWAYS.log`
- `out/sim/ablation_c2_HBG.log`
- `out/sim/ablation_c2_HBGSMAM.log`
- `out/sim/ablation_c2_FULL.log`
