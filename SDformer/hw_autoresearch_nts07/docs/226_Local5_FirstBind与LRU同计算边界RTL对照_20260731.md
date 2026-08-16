# Local5 First-Bind 与 LRU 同计算边界 RTL 对照

> 日期：2026-07-31  
> 证据等级：`[rtl]` + `[open-synth]`。不是 DC/STA/SAIF 或目标工艺 PPA。

## 1. 为什么要做这组实验

前一轮 VL-GS-TTB 只有 allocator/decoder，而 LRU 强基线包含 product SRAM、
4 个 gate-weight 乘法器和完整输出路径。两者计算边界不一致，不能比较。

本轮在同一个 `qfit_lane_product_cache_leaf` 中加入综合期参数：

```text
NO_REPLACE=0: lane-local LRU
NO_REPLACE=1: lane-local first-bind，满表后精确旁路
```

两种策略共享：

- 同一 1498-term ordered producer；
- 同一 weight loader；
- 同样 4 个 `10x8->17` 结果乘法器；
- 同样每 way 一个 `68 bit x 32 entry` 同步 1RW product bank；
- 同一输出 skid、随机反压和逐项整数 product 检查；
- 同一 Verilator SVA 与 Yosys 命令。

因此本节首次把 Local5 first-bind 与 LRU 放到了相同 product 计算边界。

## 2. 微架构差异

### 2.1 LRU

```text
(lane, gate)
  -> W-way tag compare
  -> hit: 读product SRAM
  -> miss: 计算product，写victim，更新所有相关LRU age
```

### 2.2 First-bind

```text
(lane, gate)
  -> W-way tag compare
  -> hit: 读product SRAM
  -> empty: 计算product并绑定空slot
  -> full: 计算product但不替换，直接exact bypass
```

first-bind 不是近似 admission：满表 gate 仍计算精确 product，只是不污染已稳定
slot。其价值假设是本网络 gate vocabulary 的早期出现顺序稳定，避免 LRU
metadata 写和无效替换。

## 3. 1498-term RTL 结果

| 策略 | W | hit | product start | product SRAM write | LRU metadata write | tag compare | stall |
|---|---:|---:|---:|---:|---:|---:|---:|
| LRU | 4 | 1036 | 462 | 462 | 4553 | 5992 | 483 |
| First-bind | 4 | 1114 | 384 | 96 | 0 | 5992 | 483 |
| LRU | 6 | 1345 | 153 | 153 | 5022 | 8988 | 483 |
| First-bind | 6 | 1345 | 153 | 144 | 0 | 8988 | 483 |
| LRU | 8 | 1345 | 153 | 153 | 5031 | 11984 | 483 |
| First-bind | 8 | 1345 | 153 | 153 | 0 | 11984 | 483 |

First-bind 的 fill/bypass 分账：

| 配置 | fill | bypass |
|---|---:|---:|
| S4 | 96 | 288 |
| S6 | 144 | 9 |
| S8 | 153 | 0 |

关键差值：

- S4 对 W4 LRU：product start 减少 78 次，即 **16.88%**；
- S4 product SRAM write：462 -> 96，减少 **79.22%**；
- S6 对 W6 LRU：product start 完全打平；
- S6 product SRAM write：153 -> 144，只减少 **5.88%**；
- first-bind 三种容量均消除该实现中的 LRU age write；
- tag compare 和 stall 不变。

## 4. 一个重要负结果：当前没有周期加速

两种策略的 stall 都是 483，term 接收/退休周期也相同。原因是当前叶模块：

1. hit 与 miss 都是一项一拍接收；
2. 4 个乘法器是组合 miss 路径，没有额外迭代周期；
3. 同一个单项 skid 决定吞吐；
4. first-bind 的 288 次 S4 bypass 没有建模成独立 exception FIFO。

因此本轮只能支持活动减少和结构简化机会，不能声称 throughput speedup。
如果后续加入真实 SRAM latency、乘法流水或 exception FIFO 后 S4 周期恶化，
S4 必须被淘汰。

## 5. 开放综合结构代理

同一 Yosys 流程下：

| 策略 | W | Yosys cell | wire bit |
|---|---:|---:|---:|
| LRU | 4 | 501 | 8471 |
| First-bind | 4 | 326 | 6288 |
| LRU | 6 | 663 | 11259 |
| First-bind | 6 | 398 | 7839 |

first-bind 通过综合期删除 LRU age 状态及更新逻辑。该结果说明 RTL 结构方向符合
预期，但 Yosys flatten 后 `num_memory_bits=0`，不能把 cell 数当作面积或据此
计算 EDP。product SRAM 宏容量在架构合同上仍与相同 W 的 LRU 一致。

## 6. 对 VL-GS-TTB 创新性的实际影响

这组结果把 Local5 主张从“模型中的 no-replace 可能更好”提升为：

> 在相同 product SRAM、乘法器和输出边界下，由 gate vocabulary 生命周期驱动的
> first-bind admission 可以保持 bit-exact，并在 S4 上减少 16.88% product
> starts、79.22% product SRAM writes，同时消除 LRU metadata updates。

但它仍不是完整 VL-GS-TTB：

- 输入仍是 raw `(lane, gate)`，还没有物理 slot-key 窄链路；
- W-way gate compare 仍在该叶模块内，尚未证明与 FCSR 上游目录融合；
- S4 的 288 次 bypass 尚未经过 primary/exception FIFO；
- 单个 W6 定向 trace 不代表 fullres 多样本。

因此论文中可把它作为“deterministic no-replacement admission”的硬件机制证据，
不能把整套 Gate-Slot TTB 宣称为已经闭环。

## 7. 参数晋级决策

- **S8：淘汰。** 与 S6 product start 相同，tag compare 和存储更大。
- **S6：低风险基线。** 与 LRU W6 乘法打平，主要机会是去除 LRU 写和窄链路。
- **S4：高收益/高旁路候选。** 计算和写活动更低，但必须证明 288 次 bypass
  在真实 FIFO 与反压下不吞噬收益。

双候选继续保留，不能只凭本轮结果冻结 S4。

## 8. 下一步

1. 把 S4/S6 first-bind 接到真实 primary/exception FIFO，而不是叶内零成本旁路；
2. 统计完整 command bit、FIFO occupancy、总周期和异常流领先距离；
3. post-G0 fullres 多样本报告 mean/p95/p99；
4. 对 Motion S4 做同样的完整包与 header/body 重叠实验；
5. 有工艺库后统一 DC/STA/SAIF，再判断能否进入 DATE 主表。

