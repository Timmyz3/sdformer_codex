# LUT-DLA

- uid/来源: `MAIN-R103`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/bn_state/next_training_proposal.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/bn_state/response_zero_one_page.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/hidden_queries/novelty_review.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/bn_state/support_service_notes.md
- 精读深度: §IV–VI/Algorithm1
- 题名: LUT-DLA: Lookup Table as Efficient Extreme Low-Bit Deep Learning Accelerator
- venue: HPCA 2025

## 可继承 A
LUT/查表式算术与编译到固定函数单元的先验，贴近 da4ml CSE/逻辑网思路。

## 强对照 B
通用 MAC 阵列；无表驱动折叠。

## 可差分 X线索
表驱动折叠借入；本地已有 CSE，单独搬 LUT-DLA 不构成标题。

## 与 F1–F7 / Stage B 关系
底座/对照；不抢 Stage B。

## 不可搬用边界
Algo1 级已读；未完整迁入训练/综合。

## 可复用 idea 点
- LUT/CSE 边界作为净费用合同的一列
- 与 lifting 编译图中间 RNE 对照

## 杀门建议
不降并集逻辑需求 → 不进入标题候选。

## 审计原文摘要（核对用）
- status: 训练/编解码解耦与表驻留作为完整强先验；原架构未全移植 / 原文方法已读；完整硬件尚未迁入
- reason: 不能只借低bit码却省略编码器、冷fill、共享表池和宽查表输出；零响应码候选当前H8零字为0，没有物理请求收益证据。 / forced-support 服务模型少 37.94%，但系数流量增加；仅 lookup 叶/本地缓存并非完整 LUT-DLA，不能宣称其体系在这里已输。
- what_untried: 分阶段训练完整流程、实际table bank/128bit字、LUT INT8同AEE、常量PSN尾编译；不把token频率当请求权重。 / 完整 CCM、队列、IMM/LS 及生产消费者排程；同字典/训练/资源的全链强基线
