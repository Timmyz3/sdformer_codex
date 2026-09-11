# ExSpike / APEC

- uid/来源: `MAIN-R004`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/concept_reassessment_20260906/report-source.md; /home/zhumd/work/sdformer_codex/ideafromai/research/prosperity_fusion_20260906/README.md; /home/zhumd/work/sdformer_codex/ideafromai/research/same_workload_c1c2_20260907/README.md; /home/zhumd/work/sdformer_codex/ideafromai/research/handoff_reconciliation_20260906/report-source.md
- 精读深度: §III-A2/Fig.5 方法精读 + 本地 fusion 负结果
- 题名: ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event Compression
- venue: FPL 2026

## 可继承 A
ExSpike/APEC 式活动/乘积稀疏与多消费者共享接口（非当前失败的 G4 融合布局）。

## 强对照 B
Prosperity∪APEC-θ 当前 G4 完整算子（已慢于同资源 Prosperity）。

## 可差分 X线索
完整 ExSpike 非 C1 部分仍可能有未试接口；失败的是特定 G4 融合。

## 与 F1–F7 / Stage B 关系
F3 相关；第二队列；不抢 Stage B。

## 不可搬用边界
CPU 有限服务模型≠VCS 闭环；不得把 G4 负结果扩成家族失败。

## 可复用 idea 点
- 拆开 ExSpike 非C1 与 G4 融合，分布局复测
- 多消费者共享换挂到 lifting 后源活动再谈

## 杀门建议
再出现同资源慢于 Prosperity → 停该融合布局。

## 审计原文摘要（核对用）
- status: APEC进入C1融合CPU比较；完整ExSpike未迁入 / 完整先验，停止限定旧融合布局
- reason: APEC曾被误当事件坐标合并；已纠正为多个位置公共通道贡献。与Prosperity融合的特定静态基/残差组织未胜，不能据此否定APEC完整方法。；条件变化：arXiv v2写Accepted by FPL2026，旧“无会议预印本”标签应更新；算法可训练后支持集合不再等于旧捕获。 / APEC 是输入空间组共享贡献，再重建残差；固定小缓存负结果只约束该 C1 布局。
- what_untried: 完整事件压缩格式、APEC生成/消费、有限缓冲与全部执行路径；不能只用免费intersection或高稀疏toy。 / 作者完整压缩/检测/调度与适配缓存层次的联合迁移未完成。
