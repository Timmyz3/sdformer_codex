# ismd 可读索引 — Card G/H + idea 18/19 包

**日期:** 2026-09-06（Asia/Shanghai）  
**给谁看:** ismd-nemo 上的人 / 下次 scp 全包读者  
**box 源:** `/workspace/ideafromai/`  
**ismd 目标（惯例）:** `/home/zhumd/work/ideafromai/`  
**相关 RTL 树（另包）:** box `/workspace/sdformer_c1c2star_grokbot/` ↔ ismd `/home/zhumd/work/sdformer_c1c2star_grokbot/`  
**分工:** idea/调研 = 调研 bot；独立树 RTL/OpenROAD = iscas_ssh（**勿**改 nts07）

---

## 30 秒结论

| 项 | 状态 |
|---|---|
| 信件脊柱 | **仍 3 刀**：OP-STW(+TDE3) / HBG-RP(int8 *提案*) / OGEC×PRRC→exact |
| Card G | **绿** — TDE3-Prior + TMA-Agg（commit 约 `8773b44` 起） |
| Card H | **四刀全绿** — CFP → SCI → BiSAT → BUI（`73a995b`，regress 25/25） |
| Fork | **A only**；禁 MX3P / 纯二值换岛；禁空泛 CIM |
| OpenROAD | iscas_ssh 对新模块开；= 实现完整度，**非**硅基 PPA |

---

## 读序（推荐）

1. **本文件**（你在这儿）  
2. `18_POST_TDE3_TMA_HARD_KNIVES.md` — 为何这四把红刀、相对 EvQ/PredExit/BitHyp 的缺口  
3. `19_CARD_H_IF_SKETCH_RED4.md` — 端口级接口 / 接线 / prove-by / 勿 overclaim  
4. `18_CARD_G_TDE3_TMA_GROKBOT.md` — Card G 短记 → 详文在 RTL 树 `docs/`  
5. `19b` / `19c` pointer — 落地指针  
6. （可选）信件：`sdformer_c1c2star_grokbot/docs/TCASII_LETTER_CLAIM_DRAFT_GROKBOT.md`

---

## ideafromai/research — Card G/H 核心文件

| 文件 | 一句话 | 角色 |
|---|---|---|
| `18_POST_TDE3_TMA_HARD_KNIVES.md` | TDE3+TMA 之后仍缺什么；9 候选；**RED**：BiSAT / BUI / SCI / CFP | idea 主文 |
| `18_CARD_G_TDE3_TMA_GROKBOT.md` | Card G 动机 + 三刀短记；链到 RTL `docs/CARD_G_…` | Card G 索引 |
| `19_CARD_H_IF_SKETCH_RED4.md` | 四 RED 端口草图（CFP/SCI/BiSAT/BUI）+ 接线表 | Card H 规格 |
| `19b_CARD_H_CFP_SCI_GROKBOT_POINTER.md` | H2+H3 已落 RTL 指针 → `docs/CARD_H_CFP_SCI_…` | 落地 pointer |
| `19c_CARD_H_BISAT_BUI_GROKBOT_POINTER.md` | H1+H4 已落 RTL 指针 → `docs/CARD_H_BISAT_BUI_…` | 落地 pointer |
| **`20_ISMD_INDEX_CARD_GH_IDEA.md`** | **本索引** | scp 入口 |

### RTL 树内对应详文（scp RTL 包时一起带）

| docs（在 `sdformer_c1c2star_grokbot/docs/`） | 内容 |
|---|---|
| `CARD_G_TDE3_TMA_GROKBOT.md` | TDE3 + TMA 实现说明 |
| `CARD_H_CFP_SCI_GROKBOT.md` | CFP + SCI |
| `CARD_H_BISAT_BUI_GROKBOT.md` | BiSAT + BUI |
| `TCASII_LETTER_CLAIM_DRAFT_GROKBOT.md` | 三刀信件稿（含 H 增强表述） |
| `TCASII_INNOVATION_REVIEW_GROKBOT.md` | 新颖度 / 勿 claim |
| `OPENROAD_PNR_SCOREBOARD_GROKBOT.md` | P&R 记分板（完整度） |

### RTL 模块名（独立树）

| Card | 模块 |
|---|---|
| G | `c1s_tde3_prior`, `c2s_tma_agg`, `c1s_wake_merge` |
| H2 | `c1s_cfp_confgate` |
| H3 | `c1s_sci_cleanexit` |
| H1 | `c2s_bisat_agg` |
| H4 | `c2s_bui_guard_sdsa` |

实现序（已完成）：**CFP → SCI → BiSAT → BUI**。

---

## 四把红刀（信件里是增强器，不是第 4–7 刀）

| 刀 | 咬合点 | 勿 overclaim |
|---|---|---|
| **CFP-ConfGate** | conf → OGEC×PRRC×exact | 非 first occlusion OF |
| **SCI-CleanExit** | scrub/exit → exact_hold | SciFlow 是帧侧；勿搬 on-device 数字 |
| **BiSAT-Agg** | 双向时序升级 TMA → MFBD | 非 first temporal OF HW |
| **BUI-GuardSDSA** | bit-guard → HBG payload_en | PADE 是 LLM；勿称 first sparse-attn |

降级为消融/辅先验：EvQ-Win / PredExit-OF / BitHyp-Prior。

---

## 更早 rounds（背景，非本包必读）

| Round | 文件 |
|---|---|
| R1 | `01`–`04` + microarch `05` + plan `06` + cards A/B |
| R2 | `07`–`09` |
| R3 | `10`–`12`（含 event 栈 → TDE3/TMA/EvQ…） |
| R4 | `13`–`14` |
| 卫生 | `15` CROSS_AI · `16` ATLIF verdict · `17` OpenROAD scoreboard |

总 README：`../README_GROKBOT.md`（R2–R4 段；**本文件**补 G/H）。

---

## scp 提示（给 iscas_ssh）

建议最少同步：

```text
ideafromai/research/18_POST_TDE3_TMA_HARD_KNIVES.md
ideafromai/research/18_CARD_G_TDE3_TMA_GROKBOT.md
ideafromai/research/19_CARD_H_IF_SKETCH_RED4.md
ideafromai/research/19b_CARD_H_CFP_SCI_GROKBOT_POINTER.md
ideafromai/research/19c_CARD_H_BISAT_BUI_GROKBOT_POINTER.md
ideafromai/research/20_ISMD_INDEX_CARD_GH_IDEA.md   ← 入口
```

全包也可整目录 `ideafromai/` + 独立树 `sdformer_c1c2star_grokbot/`（含 `docs/CARD_*`）。  
**不要**把 nts07 / Codex 旧 HW 卷进来。

---

## DO NOT CLAIM（贴在 ismd 也适用）

- first event-OF HW / first temporal agg HW / first sparse-attn HW  
- DualRail-CIM 物理阵列 / Loihi 硅数字当本工作  
- MX3P / Grok46 纯二值换岛  
- OpenROAD µ²/mW 当硅基 PPA；无联合仿真报 AEE  
- 模块动物园 = 贡献列表（信件只 3 刀）



## RTL mirror

`ideafromai/rtl_c1c2star_grokbot/` — Card G/H + leaf RTL/TB/flows/docs（无 OpenROAD DEF 大包）。全量树仍见 `/home/zhumd/work/sdformer_c1c2star_grokbot/`。
