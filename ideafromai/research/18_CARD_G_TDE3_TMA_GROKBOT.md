# 18 — Card G TDE3-Prior + TMA-Agg（Grok Bot 短记）

**日期：** 2026-09-06（Asia/Shanghai）  
**源树：** `/workspace/sdformer_c1c2star_grokbot/`（勿碰 Codex/nts07 HW）

## 动机
独立评审新颖度 ~3.2。在**不**声称 DualRail-CIM / Loihi / MX3P 的前提下，用两块数字模块加深刀刃，并把信件压成 **3 刀**。

## 三刀
1. OP-STW（+TDE3-Prior 先验车道）  
2. HBG-RP（int8 **提案**；ep35 捕获为**二值**）  
3. OGEC×PRRC→exact_capture（TMA-Agg = 支撑）

## 模块一句话
- **TDE3-Prior：** 数字时差先验唤醒 OF tile（增强 OP-STW，非跳零）。  
- **TMA-Agg：** Tw 切分 + 按方向 lookup 对齐 + 一致性聚合 → early_exit / hyp_hint。  
- **EXACT 消融：** 同 MW wake，PRRC 预算把门 → exact_hit 49→19。

详情：`sdformer_c1c2star_grokbot/docs/CARD_G_TDE3_TMA_GROKBOT.md`
