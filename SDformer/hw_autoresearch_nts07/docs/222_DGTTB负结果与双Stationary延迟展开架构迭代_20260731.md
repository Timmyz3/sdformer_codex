# DG-TTB负结果与双Stationary延迟展开架构迭代

## 1. DATE独立复审结论

DG-TTB独立新颖性评分为2.8/5，结论是：

> 当前DG-TTB仍是普通header/body字典编码，并且被原始descriptor直传和
> 因子化descriptor驻留明显支配，不应优先进入RTL。

关键对照：

| 表示 | 单W6 trace逻辑bit |
|---|---:|
| 37-bit flat term | 55,278 |
| DG-TTB固定header/body | 23,544 |
| 原始descriptor直传 | 3,564 |
| 变长`K bitmap + gate/mask dictionary` | 约3,146 |

因此“55,278降到23,544”虽然算术成立，但不是Pareto点。DG-TTB作为独立
候选封存为负结果，不进入G1 RTL。

## 2. 真正的结构性质

每个Local5 source descriptor产生的term不是任意稀疏列表，而是：

```text
active K lane set × unique {gate, destination mask} set
```

当前builder按以下顺序物化笛卡尔积：

```text
for lane in active_K:
    for gate in unique_gate:
        emit(lane, gate, mask)
```

这使lane/weight地址稳定，但gate和mask几乎每term切换。

循环交换后：

```text
for gate in unique_gate:
    for lane in active_K:
        emit(lane, gate, mask)
```

term multiset不变，每个lane看到的gate子序列也不变，因此lane-local cache
的hit/miss完全不变；gate/mask可以在一段active-lane扫描中驻留。

## 3. 真实Trace活动分账

| 活动 | lane-major | gate-major | 变化 |
|---|---:|---:|---:|
| gate切换 | 1,433 | 82 | -94.28% |
| gate Hamming | 4,746 | 252 | -94.69% |
| mask切换 | 1,440 | 89 | -93.82% |
| mask Hamming | 4,249 | 239 | -94.38% |
| lane切换 | 595 | 1,493 | +150.92% |
| lane Hamming | 1,262 | 3,174 | +151.51% |
| gate+mask+lane Hamming | 10,257 | 3,665 | -64.27% |

product cache不变量：

| cache | product start | lane-major weight-vector load | gate-major |
|---|---:|---:|---:|
| W4 LRU | 499 | 317 | 499 |
| W6 LRU | 156 | 108 | 156 |
| W8 LRU | 156 | 108 | 156 |

结论不是“gate-major获胜”。lane-major减少weight SRAM读取和lane/tag set切换；
gate-major减少gate/mask控制与多播活动。没有目标宏和SAIF系数时不能把
Hamming和读取次数直接相加。

## 4. DS-FLM定义

候选名称：

> **DS-FLM：Dual-Stationary Factorized Late Materialization**

核心结构：

```text
raw source descriptor
    |
factorized resident context
  - source metadata
  - active K bitmap
  - unique {gate,mask} dictionary
    |
single physical materializer
  mode 0: lane-stationary
  mode 1: gate-stationary
    |
shared product cache / four narrow multipliers
    |
TCFM-5
```

它不是lane核与gate核两套异构硬件，而是一套priority encoder、bitmap状态、
gate index和term输出寄存器，在descriptor边界选择循环方向。

## 5. 两种模式

### Lane-Stationary

- 选中一个active lane；
- weight vector和cache set保持不变；
- 遍历全部unique gate；
- 对多个miss可只加载一次weight vector。

### Gate-Stationary

- 选中一个unique gate/mask；
- gate、mask及下游destination角色保持不变；
- 遍历全部active lane；
- 适合gate/mask互连或多播活动占主导的场景。

## 6. Mode选择不能只靠命名

可接受的第一版selector必须是确定性、低成本且可审计的，例如：

```text
mode = f(unique_count, active_lane_count, cache_epoch_occupancy)
```

不能使用未实现的“完美能耗预测器”。

候选策略：

1. per-stage静态mode：由profile/train冻结，硬件零决策开销；
2. `unique_count`阈值：字典项多时lane-major优先；
3. cache warm位：冷cache优先lane-major，warm后允许gate-major；
4. 二维小LUT：`unique_count × active_lane_bucket -> mode`。

最终选择必须由目标SRAM/tag/互连能耗系数或代表性SAIF训练，测试集只评估。

## 7. Exact边界

循环交换保持term multiset，但改变term全局顺序。bit-exact成立需要：

1. TCFM更新是整数加法；
2. destination mask与gate绑定不变；
3. 每个`lane×gate×destination`恰好执行一次；
4. Acc不采用顺序相关的饱和或舍入；
5. `descriptor_last`跟随新顺序的最后一个term；
6. 外部只观察最终Acc，不依赖中间term顺序。

若下游存在顺序相关饱和、early read或跨descriptor可见状态，gate-major必须
禁用或补等价证明。

## 8. 与现有工作的边界

| 范式 | 借鉴 | DS-FLM差分 |
|---|---|---|
| output/weight-stationary dataflow | loop order与驻留分析 | 在event stencil的`K-set×gate-set`因子上双模式 |
| Balanced SA | 利用率与映射选择纪律 | 选择对象是控制/weight活动，不是通用GEMM阵列 |
| StreamTensor | 保持迭代空间因子化 | 硬件descriptor context内延迟展开 |
| Bishop TTB | metadata-first | 不使用TTB命名、ECP或dense/sparse双核 |

不能宣称发明loop interchange、stationary dataflow或descriptor residency。
可辩护主张必须是：

> 利用Local5 exact source quotient形成的笛卡尔积执行域，在同一物理
> materializer内动态选择lane/weight驻留或gate/mask驻留，并保持
> lane-local product reuse不变量。

## 9. 是否适用于Motion

Motion H67的row gate目录也可形成：

```text
active K lane set × occupied gate/class set
```

因此DS-FLM概念上可共享。但Motion现有SCS/GateStack是否已经按gate-major
消费目录需要先读真实RTL与ordered trace；若已有同序，则只能共享硬件，不是
新增创新。Local5数据不能外推Motion。

## 10. RTL前门槛

1. 独立DATE审稿确认双stationary不是普通loop interchange包装；
2. ASIC审阅冻结单context FSM与term-last合同；
3. 建立lane-major、gate-major、static-best和oracle-selector四个模型；
4. 用目标宏估算weight read、tag compare、gate/mask bus toggle能耗；
5. 若任一简单静态模式跨样本稳定胜出，则双模式可能没有必要；
6. 只有双模式在面积开销后仍降低mean与p95能量/EDP，才进入论文贡献。

当前证据等级为`[prof]+[模型]`，尚未晋级RTL。
