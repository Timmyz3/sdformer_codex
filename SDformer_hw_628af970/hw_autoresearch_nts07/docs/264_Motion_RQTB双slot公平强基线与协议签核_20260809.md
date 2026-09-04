# Motion RQTB 双 slot 公平强基线与协议签核

## 1. 本轮结论

本轮不再使用单 slot/cycle 的弱基线，而是在同一个双 slot
前端、双类别更新后端、同一 FIFO 深度和同一反压序列下对比
`Fixed2S` 与 `RQTB2S`。

分层结论如下：

1. `[rtl]` `20,841` 个 gated-K 输出和 `4,416` 个 synthetic Acc32
   checksum 零失配；
2. `[rtl]` 公平主基线 `111,807 -> 94,348` 周期，加速 `1.185x`，
   周期下降 `15.62%`；
3. `[open-pnr代理]` RQTB2S 标准单元面积比 Fixed2S 高 `1.82%`，
   直接组合的面积归一吞吐代理为 `1.164x`；
4. `[open-pnr代理]` RQTB2S 在 `5 ns` 下 post-route WNS 为 `+0.0686 ns`，
   Fixed2S 为 `-0.5118 ns`且 setup violation flow metric 为 `20`，因此两者尚不是
   同频闭合 PPA；
5. `[DATE待审阅]` RQTB 已具备一个可辩护的 exact temporal quotient
   机制，但当前只有单样本、单窗口和无功耗的 row-slice 证据，不能单独
   支撑 DATE 接收。

这一结果不意味着 Motion 线停止。Motion 仍然是独立推进线，后续补多样本、
功耗和系统边界，并继续筛选与真实 workload 匹配的新机制。Local5 同时推进，
不因 Motion 本轮正结果而降级。

## 2. 机制与精确性合同

### 2.1 数据流

```text
Q0/Q1 + K0/K1
      |
      v
双路 Motion-XOR Q7 score
      |
      +-- Fixed2S：每个 temporal pair 始终写两个 slot
      |
      +-- RQTB2S：score0==score1 时写一个 {score, temporal_mask=11}
      |
      v
每拍最多两个 slot 的 FIFO
      |
      v
双类别 weighted-SCS histogram update
  multiplicity = popcount(temporal_mask)
      |
      v
Shiftmax exp / denominator / active descriptor
      |
      v
K0/K1 双 bank 同步读取
      |
      v
在 gated-K 边界按 time0 -> time1 延迟展开
```

### 2.2 RQTB 不是近似剪枝

RQTB 对的是量化后 Q7 score 的精确等价关系：

- `score0 == score1` 时只存一份 score，但 `temporal_mask=11` 保留两个时间位置；
- SCS 通过 `popcount(mask)=2` 保留 Shiftmax 分母的 multiplicity；
- `active_mask` 独立保留 K0/K1 的有效性；
- gated-K 边界按原时间顺序读取并展开 K bank；
- `score0 != score1` 时自动退化为两个 slot，不需要有损 fallback。

所以论文可使用的差分是：

> 面向 T=2 Motion attention 的无损 post-quantization temporal score quotient；
> 它在归一化域保留 multiplicity，并把 token 展开延迟到 gated-K 边界。

不能声称发明 TTB，也不能把“精确等价关系”和“RQTB 数据结构”拆成两项贡献。

## 3. 公平强基线

### 3.1 公平边界

Fixed2S 和 RQTB2S 共享：

- 两路 Motion-XOR score 前端；
- 每拍最多两个 slot 的接口；
- 双类别 SCS histogram update 带宽；
- 相同的 FIFO 深度、K bank 端口和 Shiftmax backend；
- 相同的 16-bit 固定种子 LFSR 反压，每个 head-row 重新播种；
- 相同的 Q/K、gate、gated-K 和 Acc32 检查口径。

主配置使用 FIFO depth=`32`。深度 DSE 是另一组同深度对照，不与
depth32 的物理数字混用。

### 3.2 周期和工作量

| 项目 | Fixed1S | RQTB1S | Fixed2S | RQTB2S |
|---|---:|---:|---:|---:|
| 总周期 | 154,176 | 105,927 | 111,807 | 94,348 |
| 同带宽加速 | 1.000x | 1.455x | 1.000x | **1.185x** |
| 行周期 mean | 1,117.22 | 767.59 | 810.20 | 683.68 |
| 行周期 p95 | 2,040.50 | 1,521.15 | 1,719.75 | 1,381.00 |
| 行周期 p99 | 2,121.12 | 1,736.20 | 1,804.16 | 1,511.26 |
| 行周期 max | 2,148 | 1,792 | 1,830 | 1,570 |

RQTB2S 在 `138/138` 个 head-row 中都比 Fixed2S 快。逐行加速 mean/p95/p99/max
为 `1.187/1.286/1.312/1.344x`。

| 工作项 | Fixed | RQTB | 变化 |
|---|---:|---:|---:|
| slot | 62,100 | 34,052 | -45.17% |
| exp 事务 | 22,133 | 17,255 | -22.04% |
| gated-K 检查 | 20,841 | 20,841 | 不变 |
| synthetic Acc32 checksum | 4,416 | 4,416 | 不变 |

早期 1S 结果把单口解码瓶颈也算进 RQTB 收益。Fixed 单纯从 1S 改为 2S 已带来
`1.379x`，而 RQTB 从 1S 到 2S 只有 `1.123x`。因此主文只能报
`Fixed2S -> RQTB2S = 1.185x`。

## 4. 等价、协议和反压验证

### 4.1 真实 trace 范围

| 项目 | 数值 |
|---|---:|
| checkpoint | H67 fullres epoch30 |
| 样本/窗口 | sample0/window0 |
| attention block | 12 |
| T450 head-row | 138 |
| token | 62,100 |
| gated-K 检查 | 20,841 |
| synthetic Acc32 checksum | 4,416 |
| checksum mismatch | 0 |
| 2S SVA row | 138 |

Acc32 使用确定性人工 lane weight，只证明 Fixed/RQTB 的最终整数累加一致，
不是真实 projection 权重的端到端回放。

### 4.2 关键机制 coverage

| coverage | hit |
|---|---:|
| 跨 pair 同拍处理 | 7,302 |
| 同 class 双更新冲突 | 31,710 |
| 双 active append | 10,104 |
| FIFO 同拍 push/pop | 50,081 |
| 双 K bank 读取 | 4,878 |

### 4.3 非法 restart 的 fail-closed 反例

早期测试只在 emitter 保持输出时插入 `window_start`，无法 mutation-kill
“构建阶段仍可合法接收 pair 时被 restart 破坏”的故障。本轮补了两类独立反例：

1. `PAIRS=2`：已提交 pair0，在 pair1 仍合法 ready 的构建阶段同拍提交
   `pair_valid + window_start`；
2. emitter 已有 held output 时提交非法 `window_start`。

两类情形均要求：外部 ready 为 0，内部 encoder valid 为 0，不产生 encoder
commit、性能计数或 FIFO 变化，已完成 Icarus 与 Verilator+SVA 回归。

## 5. FIFO 深度 DSE

| depth | FIFO bit | Fixed 周期 | RQTB 周期 | 同深度加速 | RQTB 较 depth32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 112,996 | 96,547 | 1.170x | +2.33% |
| 4 | 64 | 111,807 | 94,832 | 1.179x | +0.51% |
| 8 | 128 | 111,807 | 94,468 | 1.184x | +0.13% |
| 16 | 256 | 111,807 | 94,368 | 1.185x | +0.02% |
| 32 | 512 | 111,807 | 94,348 | 1.185x | 0.00% |

按“相对 depth32 周期劣化不超过 1%”的内部门槛，depth=`4` 就足够，容量从
`512 bit` 降为 `64 bit`。这只是容量 DSE，不是独立新颖性，也不证明多上下文、
SRAM 面积或功耗收益。本轮 OpenROAD 仍使用 depth=`32`，不与 depth4 数字混合。

## 6. OpenROAD 物理代理

### 6.1 口径

- Nangate45 开放库；
- `5 ns` 时钟，输入/输出各留 `0.5 ns`；
- T450 K store、FIFO 和 directory 全部映射为 flop，`macro_count=0`；
- 完成详细布线、post-route RC 提取和 STA；
- 未做 DC、PrimeTime、PTPX、目标 SRAM 宏或 GDS 签核。

### 6.2 同约束结果

| 候选 | 标准单元面积 | 单元数 | WNS | setup/hold flow metric | max-cap | DRC | 线长 | via |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed2S | 273,360 um2 | 111,313 | -0.5118 ns | 20/0 | 72 | 0 | 2,496,414 um | 829,998 |
| RQTB2S | 278,348 um2 | 112,497 | +0.0686 ns | 0/0 | 49 | 0 | 2,516,992 um | 841,118 |

RQTB2S 的面积、单元数、线长和 via 相对 Fixed2S 分别变化
`+1.82%/+1.06%/+0.82%/+1.34%`。结合 RTL 周期得到的面积归一吞吐代理是
`1.164x`。

但 Fixed2S 并未在 5 ns 下闭合，两者也都有 max-cap 违例。因此：

- 可作为“同开放工艺、同 SDC、同全 flop-memory 规则下的物理可行性筛选”；
- 可在完整列出违例的前提下报面积归一吞吐代理；
- 不可声称同频闭合 PPA、ASIC 面积、节能或完整 encoder 吞吐。

未约束端点已精确审计：Fixed2S 仅为 `perf_k_read_bits[0:4]` 和
`perf_slots[0]`，RQTB2S 仅为 `perf_k_read_bits[0:4]`；主数据输出不在未约束集合中。

setup/hold 计数取自 `6_report.json` 的
`finish__timing__drv__setup/hold_violation_count` ORFS flow metric。文本日志中
`find_timing_paths` 未指定多路径数量，默认只列一条最差负裕量路径；该行的
`1` 不是违例总数，也不与 JSON flow metric 混用。

## 7. 负结果与证据边界

1. 1S 弱基线高估 RQTB 收益，约一半早期周期收益不是 RQTB 独有；
2. FIFO depth4 容量 DSE 是工程选参，不是 DATE 贡献；
3. Fixed2S 在 5 ns 下未闭合，当前不存在公平的同频功耗或 EDP 表；
4. 详细布线 DRC 为 0，但 max-cap 仍为 `72/49`；
5. 存储全部为 flop，不代表 SRAM 端口、宏面积和能量；
6. 真实 trace 只有 sample0/window0，不能外推多样本 mean/p95/p99；
7. 当前是 attention row slice，不是完整 Motion 或 full encoder 加速器；
8. 无 SAIF/PTPX，不报动态能量、时钟功耗或 EDP。

## 8. 独立 DATE 审阅

### 8.1 结论与评分

- 总评分：`3.0/5`；
- Recommendation：`Weak Reject / Major Revision`；
- 新颖性：`3.0/5`；
- 正确性：`4.0/5`；
- 实验：`2.5/5`；
- 物理实现：`2.0/5`；
- 系统完整度：`1.5/5`。

审稿人的定位是：

> RQTB 已经是一个验证充分、收益明确的 exact 微架构机制，但当前只能
> 作为主架构下的重要子机制，不能单独承担整篇 DATE 论文的顶层架构贡献。

建议名称为 `Exact Normalization-Domain Temporal Quotient`。它不应放入附录，
因为 `15.62%` 的公平周期收益和 exactness 足以进入主文；但也不应单独
写成一项顶层架构贡献。

### 8.2 评审认可的差分

1. Fixed2S/RQTB2S 的 RTL 周期公平性基本成立，`1.185x` 可作为 `[rtl]` 结果；
2. RQTB 在 post-Q7 score 域取商，而不是以 token-time 密度打包；
3. multiplicity、temporal mask 和 active-K mask 保留了 Shiftmax 与 gated-K 语义；
4. 延迟展开是跨 SCS/Shiftmax/gated-K 的算子间数据流差分，不是 ECP 近似剪枝。

### 8.3 问题分级

`P0`：无。未发现推翻 exactness 或导致协议错误的 RTL 缺陷。

`P1`：

1. Fixed2S 在 5 ns 下未闭合，`1.164x` 只能作为未同频闭合的面积-周期筛选代理；
2. 需补多 sample/window 的周期、slot、exp 和 FIFO occupancy mean/p95/p99；
3. 需使用真实 projection 权重闭合 gated-K 到 Acc32 的 miter；
4. 需补 SAIF 活动、功耗分账和 SRAM 宏合同；
5. 需补“FIFO 压缩 -> weighted-SCS -> late expansion”的分级机制消融；
6. Motion 主架构仍需包含调度、存储、projection 和 ATLIF 边界。

`P2`：

1. setup violation flow metric 与文本默认单路径的口径必须分开，本文已整改；
2. 后续增加多组可复现随机反压种子；
3. depth4 只能暂定，须在多 trace 上复验；
4. SVA 为动态仿真断言，不得写成 formal verification。

## 9. 下一步

Motion 线按以下顺序继续：

1. 先根据独立 DATE 审阅修正 P0/P1；
2. 补多 sample/window 的 RQTB 等价与周期分布；
3. 在同频闭合约束下做 Fixed2S/RQTB2S 比较，并补 SAIF 活动证据；
4. 把 depth4 释放的容量只用于有真实 workload 门槛的 exact 跨窗重叠候选；
5. 继续筛选新机制，但必须先有强基线和 workload 上界，不以增加 RTL 数量代替收益证据。

Local5 并行进入端到端闭环：精确语义冻结后优先连通
`score/Shiftmax5 -> relation transpose -> source-major term -> TCFM5 -> Acc32`，
然后再根据同窗全 head profile 决定 RelationMemo 是否进入主 RTL。

## 10. 关键产物

- 公平 RTL 报告：`results/h67_rqtb_strong_baseline_2s_t450_20260809/report.{md,json}`
- FIFO DSE：`results/h67_rqtb_fifo_depth_dse_t450_20260809/report.{md,json}`
- 物理代理：`results/h67_rqtb_2s_openroad_proxy_t450_20260809/report.{md,json}`
- 公平 RTL 一键回归：`sim_h67/run_h67_rqtb_strong_baseline_checks.sh`
- FIFO DSE 入口：`sim_h67/run_h67_rqtb_fifo_depth_dse.sh`
- OpenROAD 入口：`openroad_hifp/run_openroad_rqtb_2s_t450.sh`
- 端点审计：`openroad_hifp/run_check_setup_rqtb_2s_verbose.sh`

## 11. 当前允许写入论文的句子

> 本文对 T=2 Motion attention 的量化后 temporal score 等价类取商：
> 相等 Q7 score pair 只保留一份 score 和一个 temporal mask，weighted-SCS
> 使用 mask popcount 保留 Shiftmax 归一化 multiplicity，并在 gated-K 边界按
> active mask 无损恢复原时间顺序。

> 在 H67 epoch30 的 sample0/window0、12 个 attention block、138 个 T450 head-row
> 上，RQTB2S 与等带宽 Fixed2S 的 gated-K 和 synthetic Acc32 checksum 一致；
> slot、exp 事务和周期分别减少 45.17%、22.04% 和 15.62%。

不得写“Motion 端到端加速 1.185x”、“ASIC 面积归一吞吐 1.164x”、
“RQTB 节能”或“首个 temporal TTB”。
