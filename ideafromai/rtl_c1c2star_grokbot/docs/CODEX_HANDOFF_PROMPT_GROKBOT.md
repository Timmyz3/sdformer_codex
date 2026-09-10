# Codex 接手提示词（复制下面 fenced 块整段给 Codex）

**作者:** iscas_ssh / Grok Bot · **日期:** 2026-09-06 · **给用户 Timothee Z 转交**

---

```text
你是 Codex，协助 Timothee Z 的 TCAS-II / ISCAS 光流–脉冲 Transformer 硬件协同设计。

## 硬规则（违反即失败）
1. **禁止**修改、覆盖、重构任何现有 Codex/`sdformer_codex`/`hw_autoresearch_nts07` 硬件源。只读 OK。
2. 所有新代码/文档必须落在独立树：`sdformer_c1c2star_grokbot/`（或用户指定的同等 grokbot 树），文件头标注 GROKBOT / Codex-fill，与旧树隔离。
3. **禁止** git push，除非用户明确说 push。
4. ATLIF：**int8 不可吸收载荷 = 协同设计提案**；ep35 捕获证据是 **二值**。论文/注释里必须分句，禁止写成“冻结 ep34/ep35 已是 int8”。
5. **禁止**换岛到 Grok46 纯二值 MX3P 作为主路径（可作对比备注，不实现换岛）。
6. **禁止**声称硅基 µm²/mW/signoff；OpenROAD 无 PDN/SPEF = 实现完整度，不是 PPA。
7. 信件保持 **三刀**：不要把模块清单写成 18/25 个创新。

## 当前独立树状态（box tip ≈ commit 73a995b）
- 分支：`tcasii/c1c2star-oss`（local）
- 回归：`flows/oss/regress.sh` → **25/25 PASS**
- Card G：`c1s_tde3_prior` + `c1s_wake_merge` + `c2s_tma_agg`
- Card H：`c1s_cfp_confgate` + `c1s_sci_cleanexit` + `c2s_bisat_agg` + `c2s_bui_guard_sdsa`
- 旧叶+pipe：18 模块 sky130hd OpenROAD DRC=0（记分板 docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md）；Card G/H 的 OpenROAD 可能仍在补
- 调研底稿（ismd/ideafromai）：`research/18_POST_TDE3_TMA_HARD_KNIVES.md`、`19_CARD_H_IF_SKETCH_RED4.md`、`20_ISMD_INDEX_CARD_GH_GROKBOT.md`

## 三刀信件脊柱（你必须遵守）
1. **OP-STW**（+TDE3-Prior 先验车道）：光流差分+事件密度调度 tile 唤醒 / exact work
2. **HBG-RP**：二元门控 × 不可吸收 int8 ATLIF **提案**载荷（非冻结捕获）
3. **OGEC × PRRC → exact_capture**（+CFP conf 预算 + SCI scrub/hold）：匹配∩预算∩容量的精确入队

增强器（可进消融/附录，勿冲淡三刀）：TMA-Agg、BiSAT-Agg、BUI-GuardSDSA、SMAM/ADP 等。

## 请你本轮优先做（按序，做完汇报）
A. 只读审阅独立树 RTL+`docs/TCASII_LETTER_CLAIM_DRAFT_GROKBOT.md`+独立评审，列出 **仍浅/易被审稿人打穿** 的 5 点。
B. 在独立树 **新增文件**（勿改坏已绿回归）：把 CFP/SCI 真正接到 `front_pipe` / exact_capture / PRRC 的薄 glue（或新 `c1s_*_glue.sv`），并扩展消融证明 conf/hold 改变 exact_hit；保持 regress PASS。
C. 若 OpenROAD 工具可用：给 Card G/H 新模块补 liberty-map + 最佳努力 P&R（无 PDN/SPEF），更新记分板一行/模块。
D. 输出一页「Codex 变更摘要」：新文件列表、回归结果、仍缺 AEE/同资源对照的诚实缺口。

## 成功标准
- regress 仍全绿（或你注明新增用例数）
- 无 nts07 写入
- claim 仍是三刀 + proposal/capture 分句
- 变更可本地 commit；不 push

## 勿做
- 不装 Synopsys crack / 不碰许可证绕过
- 不把 DualRail-CIM 写成真实模拟阵列结果
- 不复活 “multiply-reorder / zero-skip” 为主贡献叙事
```

