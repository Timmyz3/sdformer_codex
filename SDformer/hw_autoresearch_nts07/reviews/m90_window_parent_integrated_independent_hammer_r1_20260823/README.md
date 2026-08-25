# M90 window-parent integrated probe 独立打铁评审

## 结论

M90 的四个远端工件 exact SHA、两组 `1..40` record progress、十样本聚合、
origin/distance population、K4/K6 基线比较和全部预声明 gate 已独立复算通过。

- 评分 **79/100**，`P0=0 / P1=6 / P2=5`。
- **GO**：冻结 cohort 上的负筛选与算法反馈——window64 不应 promotion。
- **NO-GO**：window64 RTL、任何性能收益/倍速、matcher 可实现性/PPA、全网/系统及
  DATE headline。

独立脚本没有 import/执行 M90 producer，也没有重放 trace-level scheduler event；它能证明
证据包内部算术一致，不能独立证明 producer 的逐 record 决策。

## 核心数字

两组均有 3,240,000 次 parent choice，其中 window parent 788,279 次，比例
24.3295988%；origin 总和及 64-row distance population 独立闭合。

| 对比 | source cycles | integrated cycles | p95 | 结论 |
|---|---:|---:|---:|---|
| K4 M53 baseline | 68,847,096 | 79,869,808 | 8,139,624 | baseline |
| K4 window64 | 67,887,224 | 80,098,488 | 8,186,160 | source -1.3942%，但 integrated +0.2863% |
| K6 M89 baseline | 69,964,176 | 76,677,320 | 7,843,680 | baseline |
| K6 window64 | 70,076,488 | 77,341,480 | 7,907,496 | source +0.1605%，integrated +0.8662% |

K6 的三项 performance promotion gate 全部失败；窗口64必须 **NO-GO**。

## 为什么逐 destination objective 破坏 fusion

producer 先为每个 row 独立选择 parent，排序目标实际是：最小 bank-issue cycles、再最小
delta population、再固定 priority。之后 scheduler 才把已选 delta 做 K-way OR union 并决定
fusion group。单独较小但彼此不重叠的 delta 可能扩大 union work；新增 parent dependency
还会改变 ready set、parent-port wait 和 fusion overlap。因此算法侧下一目标应直接收费：

`group union bank cycles + parent dependency/readiness + port/wait cost`

而不是单独最小化每个 destination。现有结果足以证明当前 objective 在冻结 screen 上失败，
但没有 same-mask DAG ablation，尚不能把回退严格拆成 union 与 dependency 两部分。

## DAG 与 matcher

所有 window parent 都是同 timestep 内更小的 spatial index；selected-parent edge 与保留的
up-row edge 都是前向边，所以抽象 DAG 可证明无环、对负结论是安全的。不过无条件 up-row edge
偏保守，可能增加与实际 selected parent 无关的串行化。

更大的问题是 matcher 完全免费：每次选择最多扫描 64 个 row，计算 XOR、bank cycles、
population 和 reduction；没有查询延迟、端口、地址映射、比较/归约面积、时序、功耗或宏收费。
因此即使周期筛选为正也不能直接 promotion；本次既然性能为负，更不应写 RTL。

## 证据边界

合同虽在结果完成前落盘，但晚于 execution launch，不能称严格 preregistration；日志也无
exact command、合同/producer SHA、commit、host、Python/environment 或 start/end attestation。
结果只有 per-sample ledger，没有 per-record transaction ledger。后续若有 retained design，应
由预先冻结的 exact-SHA runner 发起并封存逐 record union/dependency/wait 分解。

此外，contract 中 M69 analyzer/result 是设计背景但并非 producer 的 executable input；建议
后续把“设计谱系”与“实际运行依赖”分开列出。

机器结论见 `m90_window_parent_integrated_independent_hammer_review.json`；复算入口为
`audit_m90_independent.py`，输出为 `m90_independent_recompute.json`。未修改 producer、contract、
receipt 或 remote artifact。
