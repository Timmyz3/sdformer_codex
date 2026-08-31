# M225 H67 FC1 held-weight context multicast 筛选

M225 在 M224 已独立确认的 100 份 FC1 bitpack 上，改用跨 context 的权重复用：每次只读取一条 `96×INT8 = 768 bit` 权重向量，将其广播到 K 个相邻 context 中最多 F 个 accumulator；如果同一 source 还命中更多 context，权重保持在寄存器中继续服务，不重新读 SRAM。

所有点都固定 768-bit weight supply。额外资源只有 `96×F` signed event-add lanes、`K×96×19 bit` 最小 accumulator state，以及 context-mask/held-weight 控制。Cycle recurrence 串行收费 weight DMA、current/candidate scan、choice metadata、parent seed、group descriptor、service 和 final commit，overlap credit 为零。

## 主要结果

强参考是 raw K1/F1，serial cycles 为 1,087,104,872。

| 候选 | Parent | Product lanes | Acc19 state | 完整 serial ratio | 同parent multicast factor | Weight-read reduction | Slot utilization |
|---|---|---:|---:|---:|---:|---:|---:|
| K8/F2 | spatial | 192 | 14,592 bit | **1.802367×** | 1.523755× | 2.089814× | 76.90% |
| K8/F4 | spatial | 384 | 14,592 bit | **2.248493×** | 1.900919× | 2.089814× | 49.64% |
| K8/F8 | raw | 768 | 14,592 bit | 2.481930× | 2.481930× | 2.580060× | 32.25% |

K8/F2 的十样本范围为 1.768939–1.842518×；K8/F4 为 2.212879–2.305815×。因此不是单个稀疏样本造成的峰值。

K8/F8 虽然倍率最高，但 lanes 增至 8×且利用率只有 32.25%，不进入首轮 RTL。M226 应同时实现 K1/F1、K8/F2 和 K8/F4 的同协议 service island，用 Synopsys VCS 证明 context mask、held replay、signed Acc19 和 stall/fault，再用同一 3 ns TSMC28 logic-only DC 定价。最终选择必须依据 throughput/area 与 SAIF/PTPX，而不是只看 2.248×。

## 创新点边界

这一结果与 M224 的 multi-source K-bank 不同：M224 同拍读取多个不同 source 的权重并因 destination slicing 变慢；M225 每次只读一个 source 的96-lane权重，通过跨 context 广播增加有效更新。性能来自消除重复权重读取和复用 held weight，不来自增加 SRAM 读宽。

当前结果仍是 M51-s10 trace premodel，不是 RTL throughput、面积效率、完整 FC1/FFN、系统加速或 macro-aware PPA/energy。Stage 3 两个非二值 FC1 仍保持 conventional fallback；论文正文和 `docs/359` 未修改。
