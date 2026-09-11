# LoAS

- uid/来源: `MAIN-R003`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/fusion_delivery_20260907/report-source.md; /home/zhumd/work/sdformer_codex/ideafromai/research/same_workload_c1c2_20260907/c2/README_R4.md; /home/zhumd/work/sdformer_codex/ideafromai/research/same_workload_c1c2_20260907/README.md; /home/zhumd/work/sdformer_codex/ideafromai/codex_independent_20260905/02_literature_and_alternative_ideas.md
- 精读深度: 全文 FTP+Algo1；本地 r4 8bank 记录
- 题名: LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks
- venue: MICRO 2024

## 可继承 A
真正时间维并行（FTP）与 dual-sparse spMspM 作为时间打包强对照；Algorithm 1 级归约/分发先验。

## 强对照 B
串行 scatter 弱分母；特定共享组织（本地已测加法降但周期升）。

## 可差分 X线索
共享变体不作主创新；完整 FTP 作强控制。Vσ跨BN因R>T失败只停该表示。

## 与 F1–F7 / Stage B 关系
F2 对照（时间打包/共同完成）；不直接当 Stage B 主实验。

## 不可搬用边界
同负载功能诊断≠全网 DC/PT/FM；完整编码/神经元链未迁入。

## 可复用 idea 点
- 用完整 FTP 作 lifting 半步检查点的时间轴对照
- dual-sparse 计费与 r1 并集费用对齐时才谈共同完成
- R>T 条件变化后重测，不沿用旧弱分母收益

## 杀门建议
相对 direct 仅加法降而服务周期升 → 停该共享组织。

## 审计原文摘要（核对用）
- status: 已纳入真正T并行强对照；共享变体停止作主创新 / 强时间打包底座
- reason: r4已用真实8bank并发及每bank T10并行累加，修正了早期单源/串行scatter弱分母；特定共享相对direct加法下降8.9552%而服务周期增加0.23454%。；条件变化：后续训练K8改变自然R>T，值得把完整FTP继续作为强控制；失败的是特定共享组织。 / 完整时间签名归约/分发属于强先验；Vσ 跨 BN 保存因 R>T 失败只停止该表示。
- what_untried: 完整LoAS编码/控制/神经元链与checkpoint数值输入；目前同负载功能诊断不是全网或DC/PT/FM闭环。 / 完整 dual-sparse 数据流与当前非因果消费者联合迁移尚不完整。
