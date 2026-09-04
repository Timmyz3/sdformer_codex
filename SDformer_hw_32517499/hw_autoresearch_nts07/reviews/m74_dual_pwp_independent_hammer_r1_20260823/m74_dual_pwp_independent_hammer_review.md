# M74 dual-PWP signed decomposition 独立打铁评审

结论：**当前架构 NO-GO RTL；只允许 beam1 加早停门控的算法探索。**

独立复算没有 import M43、M72 或 M74 生产分析器。评审器直接读取 M40 H67 ep35 r6 的样本 5--9 positive planes，独立完成 padding Conv3x3 im2col、16-bit partition、M72 centers 匹配以及 beam1/2/4/16 signed-pair 搜索。四档结果与生产 JSON 的整数账本和 operator 账本均为 0 mismatch：

| Beam | Vector-op speedup | Pair fraction | Matcher comparisons | 全部已选 PWP 读流量 |
|---:|---:|---:|---:|---:|
| 1 | 1.592328676x | 6.0070% | 1.24416B | 1.28716GB |
| 2 | 1.597795773x | 6.3646% | 2.07360B | 1.30084GB |
| 4 | 1.599159550x | 6.4477% | 3.73248B | 1.30426GB |
| 16 | 1.599725819x | 6.4697% | 13.68576B | 1.30568GB |

算术本身通过。逐 bit 验证了：

```text
plus:  residual = x-a-b,  a+b+residual = x
minus: residual = x-a+b,  a-b+residual = x
cost:  2 PWP reads + sum(abs(residual_bit))
```

beam16 对 1,033,278 个已选 partition-pattern 做了 16,532,448 次 bit identity check，0 mismatch；所有 beam 都满足 `dual_pwp_reads == 2 * pair_selected` 和完整 operation conservation。系数绝对值 2 确实按两个 unit vector operations 计费。

## 为什么不能写 RTL

最优 1.599726x 低于 M74 自己预设的 2x gate，而且相对 single-PWP 只有 1.064221x。matcher 成本完全没进入这个比值：beam1 每 vector 需要 48 次 16-bit Hamming comparison，beam16 需要 528 次。用一条 comparison 与一条 vector-op 等周期的压力归一化，beam1 也要超过 72 comparisons/cycle 才刚刚不输 bit-sparse；要维持 1.5x 则需要约 693 comparisons/cycle。beam16 对应约 786 和 7,092 comparisons/cycle。

双 PWP 也不是免费。每个 PWP 为 144B，pair path 需要两路 1152-bit 读。beam1/16 的总已选 PWP 读流量比 single-PWP reference 分别增加 21.26%/23.01%。串行两读会直接吃掉收益；并行两读则需要明确 banking/replication、refill、端口冲突和能耗。

beam16 还被 Pareto 淘汰：它的 matcher 是 beam1 的 11 倍，但只多省 134,837 vector ops；beam1 已保留 beam16 总抽象 operation 改善的 99.2254%。

## 算法反哺

1. 只保留 beam1，停止 beam2/4/16 硬件化。
2. 最近 center 的 Hamming distance 小于 2 时跳过 pair search：在本内部 holdout 上可跳过 28.804% vectors，pair 选择损失为零。
3. `distance >= 3` 是更激进的候选门：只搜索 12.150% vectors，可覆盖 beam1 80.614% 的 pair 选择数量；必须重新计算 operation saving 和跨序列泛化，不能只凭 count 晋级。
4. 训练直接 signed-pair catalog 或 pair-ID predictor，消灭 runtime `beam × Q` 枚举。
5. PAFT/catalog 优先提高 single-PWP exact hit；只有预计 correction saving 能偿还第二次 144B 读和 merge 时才允许 pair。
6. plus 占 pair 选择约 80--83%，应做 plus-only ablation；minus 若不能在 charged cycle 中保住收益就删除。
7. 2.304KB 的单 partition/单 output-block working set 可以尝试双 bank 或复制，但必须计入复制容量、31.85MB 全 catalog refill、1152-bit 读能耗。

另有一处 provenance 风险：M72 文本称来源为 `local_DSEC_valid825_first_ten_samples`，实际 pinned SHA 对应 M40 H67 ep35 S10 manifest。算术复算绑定 manifest，不接受该自由文本作为算法线或数据身份。

评分：算术正确性 100，机制创新性 72，抽象性能 52，物理性能证据 15，RTL readiness 10，当前 DATE best-paper readiness 18，综合里程碑 39/100。
