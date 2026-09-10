# Card G — TDE3-Prior + TMA-Agg (Grok Bot)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot`  
**Tag:** GROKBOT NEW FILE -- iscas_ssh

---

## 中文摘要

在独立评审新颖度 ~3.2（Borderline）之后，Card G 用**两个纯数字模块**加深刀刃，**不**声称模拟 CIM / Loihi 硅片 / ICCV GPU 精度：

1. **TDE3-Prior（C1\*）**：每 tile 维护事件年龄；用邻域时差代理给出 `dir_code` / `tde_conf` / `tde_wake`；冲突邻域降低置信（纹理抑制）。经 `c1s_wake_merge` 与 OP-STW / Δ 做 OR，**增强 OP-STW 唤醒先验，不是跳零**。
2. **TMA-Agg（C2\*）**：把 Tw 窗切成 `N_SLICE` 子片；按 `dir_code` **lookup 对齐**上一片特征；饱和一致性计数 → `pattern_hit` / `early_exit` / `hyp_hint`，供 MFBD / Motion-TTB。**时间运动聚合的首版 HW 草图**。

信件三刀：Knife1=OP-STW(+TDE3)、Knife2=HBG-RP（提案 vs ep35 二值）、Knife3=OGEC×PRRC→exact；**TMA 为支撑织物**。

---

## EN — What / Why / Wired / NOT claimed

| | TDE3-Prior | TMA-Agg |
|---|---|---|
| **What** | Digital TDE-3 bank: age + polarity/time-diff proxy → dir/conf/wake | Split Tw → lookup-align → aggregate consistency |
| **Why novelty** | Bio time-diff **prior** seeds OF wake (enhances OP-STW) | First-HW sketch of TMA-style split+lookup+agg under spike/OF schedule |
| **Wired** | `tde_wake` → `c1s_wake_merge` OR with `op_stw` \| `delta_nz` | Outputs `agg_valid` / `hyp_hint` / `early_exit` for MFBD / Motion-TTB |
| **NOT claimed** | Loihi / analog TDE / DualRail-CIM arrays | ICCV GPU TMA accuracy / physical CIM |

---

## Prove-by counters (TB)

### TDE3 (`flows/oss/sim_tde3_prior.sh`)
- Case1: facilitator tile2 → event tile3 → `dir=2`, `conf=7`, wake bit3
- Case2: facilitator tile5 → event tile4 → `dir=1`, `conf=7`
- Case3: idle → wake=0
- Case4: `wake_merge` ORs op_stw|tde|delta
- Example counters: `tde_wake_pop_sum=7`, `conf_sum=43`

### TMA (`flows/oss/sim_tma_agg.sh`)
- Case1: 3 consistent slices → `pattern_hit=all1`, `early_exit=1`
- Case2: dir=2 lookup-align builds hits + `hyp=2`
- Case3: mismatch → no early_exit
- Example counters: `hit_pop_sum=26`, `early_cnt=3`

### C1 EXACT ablation
| MW exact_hit | EXACT (PRRC budget=3) exact_hit |
|---:|---:|
| 49 | **19** (capped) |

---

## Files

| Path | Role |
|---|---|
| `rtl_c1star/c1s_tde3_prior.sv` | TDE3-Prior |
| `rtl_c1star/c1s_wake_merge.sv` | OR merge glue |
| `rtl_c2star/c2s_tma_agg.sv` | TMA-Agg |
| `tb_c1star/tb_c1s_tde3_prior.sv` | TB |
| `tb_c2star/tb_c2s_tma_agg.sv` | TB |
| `flows/oss/sim_tde3_prior.sh` | sim |
| `flows/oss/sim_tma_agg.sh` | sim |
