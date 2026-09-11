# GustavSNN

- uid/来源: `MAIN-R002`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/architecture_followup_20260908.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/psn/gustavsnn_implementation_status_20260908.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/psn/rtl/gp_slice/README.md; /home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/report-source.md
- 精读深度: 审计§IV–VII + 本地镜像全文；证据见 gustavsnn_* / architecture_followup
- 题名: GustavSNN: Unleashing the Power of Gustavson's Algorithm on SNN Acceleration with Column-Parallel Tick-Batch Dataflow
- venue: 2026 IEEE International Symposium on High Performance Computer Architecture (HPCA) 2026

## 可继承 A
CPTB 柱并行时间批、每PE膜/状态驻留、NRV/NR4 双缓冲、L1D/merger、双口共享W与同k-ID屏障——作为供数与执行分母可继承到 lifting 源字对齐。

## 强对照 B
普通稠密供数；未对齐 Gustav 的 residual PSN；仅广播式弱分母（已被审计否定）。

## 可差分 X线索
无标题级 X：明确借入底座。差分只在「与 lifting 源字打包后的并集费用是否下降」。

## 与 F1–F7 / Stage B 关系
F5 底座；可与 Stage B 并行准备接口，但不抢 schedule_compare 档期。

## 不可搬用边界
不得冒用原作 PPA；本地 2×4 功能切片≠完整 8×8；θg/非因果 T10 适配未宣称等价原 LIF。

## 可复用 idea 点
- NRV 行与源∩W 计费分母固定后再比 lifting
- 同k-ID 屏障作为多消费者并集同步原语
- merger/L1D 边界作为 Stage B same-port 对照的状态口
- 负结果只描述本地切片，不外推原作分母

## 杀门建议
对齐后并集费用不降或不改 Stage B 结论 → 停「当标题」，保留底座。

## 审计原文摘要（核对用）
- status: 已纳入paper-guided CPU及2×4功能RTL；未完成原作全系统 / 纳入；不冒用原作 PPA
- reason: 原作不只是广播：CPTB、每PE膜驻留、NR4双缓冲、L1D/merger、双口共享W、同k-ID屏障及双指针均已逐项研究。切片已接C384→全部T10门；144/352任务功能通过不等于完整Gustav性能复现。；条件变化：早期只因接近广播降级已过时；固定BN/训练码使非因果消费者可编译，但不是将原LIF替换成原ep34 full-rank PSN而免费等价。 / 已完整读 CPTB/NRV/L1D/merger/局部膜与跨 PE 同步；作为迁移底座。F_cache/NR4/普通时间行共同优化后不能沿用旧弱分母收益。
- what_untried: 8×8完整调度/带宽；同址W与索引广播/驻留强控制；完整producer→FC1→PSN→FC2→BN2/shortcut。当前四ID重复读索引的负结果只能描述本地切片，不能作为原作分母。 / 完整适配本任务 θg、非因果 T10、实际存储/供数与生产成本尚未全部完成。
