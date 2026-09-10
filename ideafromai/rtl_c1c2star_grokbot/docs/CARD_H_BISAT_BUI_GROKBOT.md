# Card H — BiSAT-Agg + BUI-GuardSDSA (Grok Bot)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot` only  
**Tag:** `// GROKBOT NEW FILE -- iscas_ssh`  
**Scope this turn:** H1 + H4 (**CFP/SCI already landed**)

---

## 中文摘要

Card H 后两把 RED 刀分别增强 **时间织物** 与 **HBG MAC 门控**，不替换 TMA / HBG-RP：

1. **BiSAT-Agg（`c2s_bisat_agg`）**：在 `c2s_tma_agg` 前向种子上加后向一致性融合；`bwd_en=0` / `ABLATE_FWD_ONLY` 退化为 TMA-compat；agree→`fuse_hit`，边缘 disagree→`occ_exit_hint`。
2. **BUI-GuardSDSA（`c2s_bui_guard_sdsa`）**：廉价数字 bit/幅度护栏（ub/lb 宽 + score MSB），输出 `token_keep` / `payload_en`（`& hbg_g`）进 HBG 路径；`ABLATE_KEEP_ALL` 对照。**无**独立稀疏预测器；**不**吸收 int8 payload。

**勿称：** first temporal OF HW；first sparse-attn；BAT/PADE GPU AEE；CIM。

---

## EN — What / Wiring / Prove-by / Do-not-claim

| | BiSAT-Agg (H1) | BUI-GuardSDSA (H4) |
|---|---|---|
| **What** | Fwd TMA seed + bwd feat consistency fuse | Bit-bound / magnitude guard → token_keep |
| **Wired** | ← tma_agg_flow / dir_code; → hyp / occ side-channel | ← SDSA score/ub/lb + hbg.g; → payload_en into HBG |
| **Policy** | agree hit++; disagree suppress; edge conflict → occ | ub&lt;TH drop; tight/ mag keep; payload&=hbg_g |
| **Ablation** | FWD_ONLY vs FULL (`fuse_hit` ↑ under FULL) | KEEP_ALL vs BUI (`payload_fire` ↓ under BUI) |

### Prove-by (example TB)

**BiSAT** (`sim_bisat_agg.sh`):
- Case1 bwd_off: fuse=tma, fuse_hit=0
- Case2 agree×4: fuse_hit=all, hyp=1; ablate hit=0
- Case3 edge disagree: occ_exit=1, fuse=tma>>1
- Counters (example): `cnt_fwd_only=1 cnt_bwd_used=9 cnt_fuse_hit=24 cnt_occ_exit=1`

**BUI** (`sim_bui_guard_sdsa.sh`):
- Case1 ub&lt;TH: drop-all; KEEP_ALL=all; ooo=1
- Case2 hbg_g=0: keep but payload_en=0
- Case3 split: keep=`11110000`
- Ablation (example): BUI_payload=6 vs KEEP_ALL=24; drops prove skip

### Do-not-claim
- Enhancers of knife-1 temporal fabric / knife-2 HBG MAC — **still 3-knife letter**.
- Not first event-OF HW / sparse-attn accelerator / BAT-on-ASIC AEE.
- No µ²/mW; no CIM; Fork A int8 proposal untouched.

## Files

| Path | Role |
|---|---|
| `rtl_c2star/c2s_bisat_agg.sv` | H1 RTL |
| `tb_c2star/tb_c2s_bisat_agg.sv` | H1 TB |
| `flows/oss/sim_bisat_agg.sh` | H1 sim |
| `rtl_c2star/c2s_bui_guard_sdsa.sv` | H4 RTL |
| `tb_c2star/tb_c2s_bui_guard_sdsa.sv` | H4 TB |
| `flows/oss/sim_bui_guard_sdsa.sh` | H4 sim |
