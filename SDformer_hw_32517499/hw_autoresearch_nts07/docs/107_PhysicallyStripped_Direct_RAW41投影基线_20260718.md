# Physically-stripped Direct RAW41投影基线

## 一、目的

此前RAW-only实验只是在完整single-context顶层中选择RAW运行路径，未激活的IPD、resident、cache和多源路由逻辑仍存在，因此只能用于周期比较，不能用于面积或功耗比较。

本轮新增独立顶层`gatestack_direct_raw_multihead_projection_top`，从RTL源集合和综合层次中物理删除：

- resident replay joiner；
- IPD32W decoder；
- FADC24 decoder；
- Adaptive CSR首字分派；
- 三源replay mux和IPD cache-fill接口。

保留项为RAW41 decoder、尾事件修正、单事件term/event适配、与GateStack相同的multihead TDR multicast backend、banked AccTile以及weight/bias/final接口。

## 二、数据流

```text
RAW41 slot words
      |
      v
RAW41 decoder -> tail retimer -> one-event term adapter
                                      |
                                      v
                  multihead TDR multicast backend
                                      |
                                      v
                              banked AccTile
```

`tail retimer`不能删除：最后一个非零K event之后可能还有K-zero token，decoder的最后一个record并不一定是最后一个event。基线TB专门构造token 2和159有效、token 160和161为K-zero，验证token159仍被标记为真实head-last。

## 三、RTL验证

入口：`sim_hitflow/run_gatestack_direct_raw_physical_baseline_checks.sh`。

| 项目 | 结果 |
|---|---|
| Icarus自检TB | PASS，478周期 |
| Verilator `-Wall --assert` | PASS，478周期，0 warning/error |
| RAW records/events | 162/2 |
| projection terms/completed | 2/2 |
| final输出 | 162，逐元素零mismatch |
| done/protocol/overflow | 0错误 |
| Erie lint | 0 error，0 warning |
| Yosys综合层次禁用模块检查 | resident/IPD/FADC/Adaptive/replay mux均不存在 |

该TB验证基线的数值和生命周期合同，但不是H67真实trace周期测试。真实RAW-only周期仍应引用`results/gatestack_real_trace_ablation_20260717/report.md`中的同顶层回放。

## 四、同流程结构代理

三个projection slice统一执行：

```text
proc; flatten; opt; memory -nomap; stat
```

| Projection slice | Yosys generic cells | 相对Direct |
|---|---:|---:|
| Direct RAW41 physically-stripped | 1293 | 1.000x |
| GateStack IPD32W | 2188 | 1.692x |
| GateStack Adaptive CSR | 3183 | 2.462x |

Direct相对IPD减少约`40.9%` generic cells，相对Adaptive减少约`59.4%`。Adaptive相对IPD增加约`45.5%`，主要来自第二个流式decoder、256-bit reservoir、bitmap扫描和首字分派。

这些数字的意义是：

1. 多格式支持不是“零面积”抽象，目标PPA必须计入；
2. Direct面积较小不代表EDP较好，S3真实trace中RAW展开会显著增加term与周期；
3. 最终论文应画面积、周期和能量Pareto，而不是只报某一项；
4. generic cell不含标准单元映射、SRAM macro舍入、时钟树和布线，不能作为芯片面积主表。

## 五、公平性与边界

### 5.1 已保持一致

- tile/head调度语义；
- TDR multicast backend；
- banked accumulator；
- weight request/response；
- bias和final输出；
- gate、weight、accumulator位宽；
- 默认TOKENS、LANES、OUT_TILE和BANKS参数。

### 5.2 有意不同

- Direct只接受RAW41，不提供CSR/resident能力；
- Direct删除多源route和cache-fill端口；
- 当前比较边界是projection slice，不含single-context head-slot SRAM、descriptor cache、control plane和完整encoder。

因此本轮关闭的是“没有物理裁剪Direct projection baseline”的缺口，不是“整加速器物理公平基线全部闭合”。后续仍需physically-stripped IPD/FADC/Adaptive全套目标库网表、SRAM宏和head-major spill基线。

## 六、DATE Claim

当前可写：

> 在相同projection backend边界下，physically-stripped RAW41基线的开放综合结构为1293 generic cells；IPD和Adaptive分别为2188和3183，证明精确压缩与异构重放具有可量化的逻辑代价。结合真实trace周期，后续以PPA/EDP Pareto决定最终配置。

当前不可写：

- Direct面积为某个`um²`；
- Adaptive面积增加45.5%或Direct面积减少59.4%；generic cell不是面积；
- Direct功耗或能量更低；
- Adaptive的S3周期收益已经抵消双decoder目标库面积；
- full encoder或芯片基线已经物理公平。

## 七、下一步

1. 实现head-major partial-sum spill RTL和SRAM traffic计数；
2. 给IPD-only、FADC-only、Adaptive和Direct建立同一DC/STA/SAIF脚本；
3. 明确slot/cache SRAM宏舍入，不能只比较decoder逻辑；
4. 扩大真实trace并按stage报告周期和格式尾分布；
5. 在format-aware residency完成前，禁止`Adaptive + residency`非法组合。

## 八、入口

- RTL：`rtl_hitflow/gatestack_direct_raw_multihead_projection_top.sv`；
- TB：`tb_hitflow/tb_gatestack_direct_raw_multihead_projection_top.sv`；
- 回归：`sim_hitflow/run_gatestack_direct_raw_physical_baseline_checks.sh`；
- 报告：`results/gatestack_direct_raw_physical_baseline_20260718/report.{md,json}`；
- 综合日志：`build_hitflow/gatestack_direct_raw_physical_baseline/`。
