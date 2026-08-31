# M517：H67 FC2 dense/sparse stratifier fast-kill

## 技术结论

**不开发新的 dense/sparse 分流 RTL。** 在冻结的 Motion/H67 ep35 FC2
负载上，96-bit tile 没有足够的高密度尾部；更根本地，在相同 8-bank、每 bank
每拍最多一个 source 的带宽下，dense 顺序路径每 tile 固定需要
`12 × output_blocks` 个 issue cycle，而 K8 sparse 的精确 issue 下界是
`max(bank_popcount) × output_blocks`，对每个 tile 都不大于 dense。

因此零税条件下，`36,480,000` 个 tile 中 dense 严格获胜为 **0**；只有
`6` 个 tile 打平。允许不分流时，最优方案就是现有 K8 no-op 路由，倍率
`1.000000×`，未达到预注册的 `1.10×` RTL 门。

## 真实密度不支持 Bishop 式双路径

本审计逐成员流式读取封存的 M51 `tar.zst`，对 120 个 FC2 payload 逐一校验
SHA256，并重算全部 5.58M token、36.48M 个 96-bit tile 和 143.895M 个事件。

| 指标 | H67 FC2 实测 | 对分流的含义 |
|---|---:|---|
| 全 tile 事件密度 | 4.1088% | 总体强稀疏 |
| 空 tile | 17,610,624 / 36,480,000 = 48.2758% | 现有 sparse 路径可直接吸收 |
| `nnz ≥ 24`（≥25% 密度） | 978,003 = 2.6809% | 所谓“高密度”仍很小 |
| `nnz ≥ 48`（≥50% 密度） | 1,922 = 0.005269% | 无法平衡第二条 core/path |
| `nnz ≥ 72`（≥75% 密度） | 0 | dense 主导区不存在 |
| tile 最大 nnz | 55 / 96 | 最密 tile 也只有 57.29% |
| 10 个 sample 的密度范围 | 4.0514%–4.1527% | 不是单一样本异常 |

按 stage 分层，stage 0/1/2/3 的事件密度分别为
3.1744% / 4.4832% / 4.7985% / 5.9485%。≥50% 的 1,922 个 tile 只出现在
stage 1（232）和 stage 2（1,690），stage 0/3 为零。

## 强基线与周期下界

比较基线不是 dense 或 K1，而是已经封存的 M216 full-vector K8
always-ready standalone FC2 frontend：`90,196,785` cycles。M517 从同一批原始
payload 独立复现其 full-vector bank-service floor：`70,657,362` cycles。

| 同一负载周期口径 | cycles | 相对 M216 K8 |
|---|---:|---:|
| M216 full-vector K8 frontend（强基线，含控制/碰撞） | 90,196,785 | 1.000× |
| full-vector K8 issue floor（无控制税） | 70,657,362 | 下界，不是实现 |
| 96-bit tile 强制分段 sparse floor（零 router 税） | 118,651,292 | 1.315× 更慢 |
| 所有 tile 走 dense sequential | 1,105,920,000 | 12.261× 更慢 |

强制以 tile 为调度边界会丢掉跨 tile bank 聚合：零税 sparse floor 从
70.657M 增到 118.651M（`1.679×`）。即使把 router、格式转换、顺序合并队列和
mode switch 全部当作零，分段候选仍只有 M216 的 `0.760×` 吞吐。一个可回退
的 router 可以选择不分流，从而回到 `1.000×`，但不能形成正加速。

## 控制税敏感性仍不足以翻盘

M216 实测 frontend 相对 service floor 有 19.539M 个 control/collision cycle。
把它极度乐观地均摊到每个非空 tile，是 `1.03551 cycle/tile`。再假设 dense
模式能把该份控制税全部消掉、且 router/格式/merge 都免费，只有 **6 个**
tile 会选择 dense，合计只省 `6.213` cycle；对 M216 的上界倍率为
`1.000000069×`。

这个均摊只是敏感性，不是逐 tile M216 归因。它已经偏向候选，仍比 1.10×
门槛低六个数量级，因此无需为更精细的归因开发 RTL。

## 能量方向也不支持新路径

dense 模式会读取被选 tile 的全部 96 个 source weight row；event-sparse
模式只请求非零 source。以 `nnz ≥ 48` 为例，1,922 个 tile 的均值只有
48.89 个非零，dense 仍要引入 340,266 个额外的
`source-row × output-block` 取数。它可能省掉少量 decoder 切换，但必须先覆盖：

1. 额外 zero-source weight SRAM 访问；
2. 在线 96-bit popcount/阈值树，或预计算 7-bit count + route metadata；
3. dense/sparse 格式转换和有序合并队列；
4. mode-switch 与新控制的时钟功耗。

当前没有这条候选的 matched SAIF/PTPX 和 SRAM 能量，因此不准入净能耗数字。
由于周期门已失败，也不应为能量单独开发双路径 RTL。

## 与顶会机制的正确关系

- [Bishop（ISCA 2025）](https://arxiv.org/abs/2505.12281) 的 stratifier 将
  高/低密度 TTB 分给专用 dense/sparse core，并强调阈值要平衡两个 core 的
  工作量；其论文报告的异构性收益建立在真实混合密度和额外 core 面积上。H67
  FC2 的 ≥50% tile 只有 0.0053%，不满足该前提。
- [FireFly-T](https://arxiv.org/abs/2505.12771) 用多 lane nonzero decoder
  同拍抽取多个 spike，并用 bank-aware load balancing 避免 conflict。这个机制
  支持继续收口现有 K8 sparse 路径，而不是证明应再加 dense bypass。

两篇论文的公开 speedup/energy 数字都没有转移到 H67；这里仅借鉴机制和决策
条件。

## 范围与限制

该结果是单一 DSEC sequence（zurich_city_09_a）、10 个冻结 sample、12 个 FC2
模块的 trace-only CPU fast-kill。它不是完整 FC2、FFN 或全网周期，也不包含
SRAM latency、accumulator/BN/residual commit、RTL、VCS、DC 或功耗。局部密度
结果不得写成系统倍速。

## 决策与后续

1. **KILL M517 dense/sparse router RTL。** 不占用 VCS/DC 队列。
2. 保留 M496/M495 shared-state K8 作为 FC2 收口主线，优先完成其物理、等价和
   能量证据。
3. FireFly-T 仅作为“多非零解码 + bank-aware 服务”的相关工作依据；Bishop
   作为“异构分流需要真实混合密度”的反例/消融依据。
4. 若未来 PAFT 或新 checkpoint 显著提高结构化高密度 bundle 比例，必须用新
   checkpoint 身份重跑 M517；旧 ep35 结论不能外推。

