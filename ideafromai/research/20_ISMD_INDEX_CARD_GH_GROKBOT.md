<!-- GROKBOT NEW FILE -- iscas_ssh -->
# 20 — ISMD 索引卡：Card G/H + research 15–19c（Grok Bot）

**日期：** 2026-09-06（Asia/Shanghai）  
**用途：** 给 Codex / 人类在 **ismd** 上快速定位 ideafromai 调研 + HW 树 Card G/H 文档与 RTL。  
**硬边界：** 只谈本镜像树；**勿**碰 `nts07`；本地 git OK，**禁止 push**；父代理负责 scp 同步。

**HW tip（本轮起点）：** `sdformer_c1c2star_grokbot` @ **`73a995b`**  
`tcasii: Card H BiSAT-Agg + BUI-GuardSDSA RTL+TB`

---

## 一、三刀 + 增强器（信件口径）

### 三刀（信件主刀）
| # | 刀 | 角色 |
|---|---|---|
| 1 | **OP-STW**（+ **TDE3-Prior** 先验车道） | C1* 光流 / 稀疏唤醒；TDE3 增强唤醒，**非跳零** |
| 2 | **HBG-RP** | int8 **提案** payload；ep35 捕获为**二值**（勿混写） |
| 3 | **OGEC × PRRC → exact_capture** | 精确捕获预算门；**TMA-Agg** = 支撑织物 |

### Card G 增强器（已落地 RTL）
| 模块 | 一句话 |
|---|---|
| `c1s_tde3_prior` | 数字时差先验 → `dir/conf/tde_wake` |
| `c1s_wake_merge` | `op_stw \| tde \| delta` OR 合并（极小） |
| `c2s_tma_agg` | Tw 切分 + dir lookup 对齐 + 一致性 → early_exit / hyp_hint |

### Card H 四把 RED（已落地 RTL；增强 Fork A，不换岛）
| Order | 模块 | 刀角色 |
|---|---|---|
| H1 | `c2s_bisat_agg` | BiSAT — 双向时序织物，升级 TMA |
| H2 | `c1s_cfp_confgate` | CFP — conf → exact/PRRC 预算 |
| H3 | `c1s_sci_cleanexit` | SCI — scrub + exit → exact hold |
| H4 | `c2s_bui_guard_sdsa` | BUI — bit-guard → HBG / SDSA gate |

**禁止口径：** MX3P / 纯二值 ATLIF 换岛；DualRail-CIM / Loihi / 模拟硅声称；无联合仿真前勿写 AEE / µ²·mW 签核数字。

---

## 二、research 15–19c 索引（ideafromai）

| 文件 | 主题（中文优先） |
|---|---|
| `research/15_CROSS_AI_READ_GROK46_CODEX_VS_GROKBOT.md` | 三包对照：Grok Bot / Grok46 / Codex；ATLIF 身份冲突表 |
| `research/16_ATLIF_IDENTITY_VERDICT_GROKBOT.md` | ATLIF 裁决：ep35 二值捕获 vs HBG int8 **新共设** |
| `research/17_OPENROAD_PNR_SCOREBOARD_GROKBOT.md` | OpenROAD 记分板镜像（叶模块+pipe；**非签核**） |
| `research/18_CARD_G_TDE3_TMA_GROKBOT.md` | Card G 短记：TDE3 + TMA + 三刀 |
| `research/18_POST_TDE3_TMA_HARD_KNIVES.md` | TDE3/TMA 之后仍缺的硬刀（BiSAT/CFP/SCI/BUI 等 RED） |
| `research/19_CARD_H_IF_SKETCH_RED4.md` | Card H 端口级草图（H1–H4） |
| `research/19b_CARD_H_CFP_SCI_GROKBOT_POINTER.md` | H2+H3 落地指针 → HW `CARD_H_CFP_SCI` |
| `research/19c_CARD_H_BISAT_BUI_GROKBOT_POINTER.md` | H1+H4 落地指针 → HW `CARD_H_BISAT_BUI` |
| **`research/20_ISMD_INDEX_CARD_GH_GROKBOT.md`** | **本索引卡** |

相关前置（非 15–19，但常被引用）：`00–14` 轮次综合；`grok46_20260905/`；`codex_independent_20260905/`。

---

## 三、HW 树 Card G/H 文档（权威正文）

| 文档（HW） | 镜像到 ideafromai |
|---|---|
| `sdformer_c1c2star_grokbot/docs/CARD_G_TDE3_TMA_GROKBOT.md` | `ideafromai/docs/CARD_G_TDE3_TMA_GROKBOT.md` |
| `sdformer_c1c2star_grokbot/docs/CARD_H_CFP_SCI_GROKBOT.md` | `ideafromai/docs/CARD_H_CFP_SCI_GROKBOT.md` |
| `sdformer_c1c2star_grokbot/docs/CARD_H_BISAT_BUI_GROKBOT.md` | `ideafromai/docs/CARD_H_BISAT_BUI_GROKBOT.md` |
| `docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md` | 亦见 research/17 |
| `docs/OPENROAD_STATUS_GROKBOT.md` | 逐步尝试日志 |

---

## 四、路径清单（Codex / 人类 on ismd）

### ideafromai（调研 + 卡片指针）
```
/home/zhumd/work/sdformer_codex/ideafromai/research/15_CROSS_AI_READ_GROK46_CODEX_VS_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/research/16_ATLIF_IDENTITY_VERDICT_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/research/17_OPENROAD_PNR_SCOREBOARD_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/research/18_CARD_G_TDE3_TMA_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/research/18_POST_TDE3_TMA_HARD_KNIVES.md
/home/zhumd/work/sdformer_codex/ideafromai/research/19_CARD_H_IF_SKETCH_RED4.md
/home/zhumd/work/sdformer_codex/ideafromai/research/19b_CARD_H_CFP_SCI_GROKBOT_POINTER.md
/home/zhumd/work/sdformer_codex/ideafromai/research/19c_CARD_H_BISAT_BUI_GROKBOT_POINTER.md
/home/zhumd/work/sdformer_codex/ideafromai/research/20_ISMD_INDEX_CARD_GH_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/docs/CARD_G_TDE3_TMA_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/docs/CARD_H_CFP_SCI_GROKBOT.md
/home/zhumd/work/sdformer_codex/ideafromai/docs/CARD_H_BISAT_BUI_GROKBOT.md
```

### sdformer_c1c2star_grokbot（RTL / TB / flows / OR）
```
# Card G
rtl_c1star/c1s_tde3_prior.sv
rtl_c1star/c1s_wake_merge.sv
rtl_c2star/c2s_tma_agg.sv
tb_c1star/tb_c1s_tde3_prior.sv
tb_c2star/tb_c2s_tma_agg.sv
flows/oss/sim_tde3_prior.sh
flows/oss/sim_tma_agg.sh

# Card H
rtl_c1star/c1s_cfp_confgate.sv
rtl_c1star/c1s_sci_cleanexit.sv
rtl_c2star/c2s_bisat_agg.sv
rtl_c2star/c2s_bui_guard_sdsa.sv
tb_c1star/tb_c1s_cfp_confgate.sv
tb_c1star/tb_c1s_sci_cleanexit.sv
tb_c2star/tb_c2s_bisat_agg.sv
tb_c2star/tb_c2s_bui_guard_sdsa.sv
flows/oss/sim_cfp_confgate.sh
flows/oss/sim_sci_cleanexit.sh
flows/oss/sim_bisat_agg.sh
flows/oss/sim_bui_guard_sdsa.sh

# Docs + OpenROAD
docs/CARD_G_TDE3_TMA_GROKBOT.md
docs/CARD_H_CFP_SCI_GROKBOT.md
docs/CARD_H_BISAT_BUI_GROKBOT.md
docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md
docs/OPENROAD_STATUS_GROKBOT.md
out/synth/*_mapped.v
out/openroad/*_routed.def
```

**Box 工作副本：** `/workspace/ideafromai/` · `/workspace/sdformer_c1c2star_grokbot/`  
（ismd 同源路径通常为 `/home/zhumd/work/...`；由父代理 scp，本卡不自动同步。）

---

## 五、OpenROAD 口径（读记分板时）

- PDK：sky130hd · 工具：`/workspace/tools/openroad` · **无 PDN / 无 SPEF / 非签核**
- 默认 N_TILE=8 flatten（与 Card G/H 默认一致）
- STA @10 ns + IO max=1.0 / min=0.5 ns（placement parasitics）
- 权威表：`sdformer_c1c2star_grokbot/docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md`

---

## 六、给 teammate 的一句话

**三刀不变；Card G 用 TDE3+TMA 加深；Card H 四 RED（BiSAT/CFP/SCI/BUI）只增强 Fork A。** 读 15–16 先分清 ATLIF 身份，再读 18–19c 与 HW `docs/CARD_*`；P&R 数字只看 scoreboard，且标 **NOT signoff**。
