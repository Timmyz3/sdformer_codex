# M333：M328 dynamic cumulative-B 独立打铁评审

结论：`90/100`，`P0=0、P1=2、P2=3`。M328 的 frozen CPU 正确性和**冻结串行成本模型下的 NO-GO**成立；但“没有值得实现的执行路径”没有被证明。M328 没有在同一个 total 内重复加入 scan、active-list 和 metadata 三种成本；真正的悲观项是 selector 与 K8 issue 被强制完全串行。

本次只读复核，没有调用 M328 分析器、GPU 或新思，没有修改分析器、源、RTL、合同和 `docs/359`。

## 身份与 population

合同、分析器、checkpoint、M51 manifest、M287、M293、M324 和 `docs/359` 当前 SHA 全部匹配。M328 的 source/output manifest 重放通过。

独立读取并校验全部 110 个相关 packed payload：

- FC1：10 个模块、100 条记录、5,520,000 operation-token、112,213,979 active source，100/100 payload SHA 与 active count 匹配。
- Conv：10 条记录、7,680,000 output token。独立用 channel population 和 padded 3×3 receptive field 复算得到 452,261,964 source contribution、最大 active fan-in 448，10/10 payload SHA 匹配。

## 正确性

- 28 个 B0 module/aggregate group row 全部 exact：drop=0、raw error=0、selector overhead=0、candidate=baseline。
- JSON 与 CSV 各 224 条、224 个唯一 policy key，字段 mismatch=0。
- 224 条 module/aggregate row 均满足 `max |raw INT8 error| <= max cumulative beta <= B`，报告 violation=0。
- group4/B1024 最坏 raw signed INT8 error 为 992，最大 cumulative bound 为 1024。

其数学边界成立：对 destination group 中任一输出 `j`，`beta_i=max_j |Wq[j,i]|`，所以 `|sum dropped Wq[j,i]| <= sum beta_i <= B`。这只约束逐行量化后的整数 accumulator；不同 row 的反量化 scale、网络输出误差和 accuracy 均未证明。

dynamic witness 的定义也有效：同一个固定 `module/source/destination-group` pair，只有在 active occurrence 中既出现 drop 又出现 keep 才计为 witness。最佳 group4/B1024 有 819,815 个 witness；5,148,082 是跨多个 budget/group policy 的重复累计，不是 unique pair。

## 周期模型复核

独立从 packed bits 复算 K8 baseline 与 active-list population，全部对账：

| group4 scope | K8 baseline | scan96 | active-list8 | metadata16B |
|---|---:|---:|---:|---:|
| FC1 | 4,802,956,800 | 1,843,200,000 | 3,375,839,616 | 33,177,600,000 |
| Conv | 1,878,719,472 | 1,658,880,000 | 1,433,001,240 | 29,859,840,000 |
| Combined | 6,681,676,272 | 3,502,080,000 | 4,808,840,856 | 63,037,440,000 |

公式与合同一致：

- K8：每 token/group 取 8 个 `source_id mod 8` bank population 的最大值。
- scan96：`token × group × ceil(Nsource/96)`。
- active-list8：`group × sum_token ceil(Nactive/8)`。
- metadata16B：`token × group × ceil(3*Nsource/16)`。

三种 selector/read 模型是三个替代 total，没有彼此相加，因此不存在 scan+active+metadata 的算术双计。每个模型只把自己与 kept-source issue 相加。

## 冻结最佳点与 NO-GO

combined group4/B1024：drop task 45.4944%，baseline/candidate 为 6,681,676,272/3,787,146,191 cycles，ideal K8 `1.7643x`。串行加入 selector 后：

- scan96：7,289,226,191 cycles，`0.9167x`；
- active-list8：8,595,987,047 cycles，`0.7773x`；
- metadata16B：66,824,586,191 cycles，`0.1000x`。

B0、零 violation、dynamic witness、ideal≥1.15 均通过；scan>1、metadata>1、metadata≤4x 三门失败，passing policy=0。因此 M328 自己的 frozen `NO_GO` 是正确的。

24x 计算也正确：997,632 pair 的一位 mask 是 124,704 bytes，`uint8 beta + uint16 ID` 是 2,992,896 bytes，比例正好 24。但它只是朴素布局。按各模块真实 source 数使用 7-bit beta 和 7–10-bit ID，仍是加权 `15.91 bits/pair`，虽然小于 24x，仍过不了 4x 门。

## 可翻转的 streaming trick

如果 selector 和八 bank MAC 是独立硬件，并用 ping-pong cutoff/queue 跨 token/group 重叠，稳态下界应接近 `max(selector, issue)`：

- scan96 perfect overlap：`max(3.7871B,3.5021B)=3.7871B`，proxy 回到 `1.7643x`；只需隐藏 scan 的 17.35% 就能超过 1x，隐藏 42.23% 可达到 1.15x。
- active-list perfect overlap：`max(3.7871B,4.8088B)=4.8088B`，为 `1.3895x`；需隐藏 active-list 工作的 39.81% 才超过 1x。
- 16 B/cycle metadata 即使完全重叠仍只有 `0.1060x`，不能翻转。

可落地条件有两条路线：

1. Exact lossless：metadata 总 footprint 必须压到不超过 498,816 bytes，即包含 index/restart state 后平均≤4 bits/pair；同时 96 pair/cycle 解码。朴素 3-byte 格式要喂满 scan96 需 288 B/cycle，且至少隐藏 607,549,919 scan cycles。
2. Conservative 4-bit beta：active list 提供 source ID，只存每 source/group 一个 4-bit 保守上界码，footprint 恰为 4x；用 16 类稳定 bucket、8 lookup/cycle、独立 bank FIFO 和最大 448-active 的 ping-pong state。由于上界量化会改变 order/drop prefix，必须重跑 DSE；combined candidate 必须≤5,810,153,280 cycles 才保住 ideal≥1.15，并继续满足 B0 exact、零 bound violation 和 dynamic witness。

两条路线都必须显式计入 sorter/bucket、FIFO bank conflict、activation/metadata 端口、fill/drain 与 commit。满足这些条件只够启动一个新的 streaming predesign milestone，不能把 M328 直接改判为 RTL/accuracy/hardware GO。

## 问题分级

- P1：完全串行 selector+issue 对解耦流水过度悲观，M328 只能否定冻结串行机器。
- P1：24x 是朴素 representation，不是动态 metadata 的不可突破下界。
- P2：active-list 忽略排序、bucket 与 queue conflict；没有 cutoff-rank/early-stop 分布；M328 原生 output manifest 可重放但没有二级 seal。

M333 的 SHA manifest 会直接绑定 M328 输入、结果、CSV、合同、分析器及本评审，并另行 seal；`docs/359` 保持 `dedde7ce...`。
