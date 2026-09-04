# 双线创新筛选：MSSB5 晋级与 Local5 Source Wavefront 否决

## 1. 本轮边界

本轮只回答两个问题：

1. Motion 的五充分统计量 score front 能否在真实 RQTB 行顶层保持 bit-exact，
   并量化相对 CSE7 与 SSR5 两级基线的真实增量；
2. Local5 的 source-owned Q-silent wavefront 是否打赢同吞吐、单 popcount
   sidecar 强基线。

不改 `docs/359` 的封存主表，不产生新的 full-encoder 加速数字。

## 2. Motion：MSSB5 晋级为主贡献支撑机制

### 2.1 执行边界

常规 CSE7 score front 需要七个精确计数：双时间的 Q popcount、K popcount、
overlap，以及共享 `K0 xor K1` motion。MSSB5 直接归约五个充分统计量：

```text
{overlap0, same-zero0, overlap1, same-zero1, motion}
                           |
                      dual Q7 score
                           |
             RQTB {score,multiplicity,mask}
```

`same-zero` 直接由逐 lane `~(Q|K)` 产生，不再先分别形成 Q/K count 后重构。
因此它改变 score front 的归约对象；RQTB 改变归一化前的存储和调度对象。
两者合并成一条 exact score-to-quotient 数据流，但仍只计作一条 Motion 主贡献。

### 2.2 RTL 证据

- `[rtl]`：Direct/CSE7/MSSB5 三方 packet miter；
- Icarus 与 Verilator 各覆盖 ep35 真实 138 行、31050 个 temporal pair；
- packet、slot count、commit、计数器 mismatch 均为 0；
- `[rtl]`：MSSB5 行顶层公平回放仍为
  `112589/94891/34099/28001`，default-direct 现编现跑结果完全相同。

这表示 score-front 替换没有改写封存的 `1.1865x`，也没有依靠更宽接口或不同
反压序列获得周期优势。

### 2.3 两级基线与纠偏

3 ns 时序驱动开放映射代理：

| packet encoder | cells | 面积代理 | 关键路径(ns) | slack(ns) |
|---|---:|---:|---:|---:|
| CSE7 | 1705 | 2470.608 | 2.139979 | 0.824446 |
| MSSB5 | 1431 | 2095.016 | 1.772563 | 1.191225 |

MSSB5 相对 CSE7：面积代理下降 `15.20%`，关键路径下降 `17.17%`，
面积延迟积代理下降 `29.76%`。2 ns 代理中 MSSB5 `MET`，CSE7
`VIOLATED`。

但 CSE7 不是最强算术基线。既有叶级 SSR5 已直接形成相同五个充分统计量：

| 叶级强基线 | 面积代理 | 关键路径(ns) |
|---|---:|---:|
| SSR5 | 2292.388 | 0.959107 |
| MSSB5 | 2242.912 | 0.953343 |

MSSB5 相对 SSR5 仅减少 `2.16%` 面积代理和 `0.60%` 关键路径。由此必须把
15.20% 行顶层结果解释为“五充分统计量替代七计数”的 domain CSE 收益，不能
解释为 packed reduction tree 的独立架构收益。

必须同时保留负面敏感性：面积优先 `abc -fast` 映射曾得到面积下降但路径增长，
所以这里只能称 `[开放逻辑映射代理]` 与 `[开放网表STA代理]`；最终主张须由
同 SDC 的 DC/STA 复核。

### 2.4 论文位置

MSSB5 不能单列成“新蝶形网络”。可辩护表述是：

> Motion attention 先以五个双时间充分统计量生成两个精确 Q7 score，再把相等
> score 映射为带 multiplicity 和 temporal mask 的可逆 quotient packet；前者
> 去除 score 归约冗余，后者去除归一化/目录冗余。

SSR5 强基线进一步限定：MSSB5 只作为 RQTB 主贡献的 score-front 实现支撑，
不能独立抬高 DATE 架构创新评分。

证据入口：

- `results/h67_mssb5_rowtop_integration_20260814/report.md`
- `results/h67_mssb5_slot_ep35_rtl_20260814/`
- `results/h67_mssb5_fair_ep35_rtl_20260814/`
- `results/h67_mssb5_slot_integration_openproxy_20260814/`

## 3. Local5：source-owned Q-silent wavefront 否决

### 3.1 成立的 workload 性质

`[prof]` 100 sample/group：

- Q-silent exact edge：190575；
- inverse-stencil K mismatch：0，score mismatch：0；
- popcount evaluation：190575 -> 45000，下降 `76.39%`；
- score-side K read bit：6816000 -> 1861050，下降 `72.70%`。

这些结果证明 `Q=0` 时使用 source-owned `popcount(K)` 充分统计量在数学上成立，
但工作量下降不是周期或能量结果。

### 3.2 强基线否决

同一 OUT_DIM=2 score+projection tile 的 CPU 有限资源模型：

| 当前 RTL 锚点 | source-owned | 单 popcount sidecar 强基线 | source 相对强基线 |
|---|---:|---:|---:|
| q0_serial 191424 | 155188 | 151167 | 慢 2.66% |
| q0_ident_overlap 183379 | 147143 | 143122 | 慢 2.81% |

sidecar 在 K 进入公共 store 或被首次读取时只生成一次 6-bit popcount，
destination pipeline 随后读取五个 6-bit stripe tap。它保留 destination-owned
归一化顺序，不需要 source scatter、五 destination context 或 mixed join。

开放 Nangate45 结构代理中：

- source router：2234.400 area proxy；
- sidecar store + score leaf：1644.412 area proxy；
- sidecar/source 比为 `0.736x`。

该比较尚未包含真实 SRAM macro，且当前 row-wide K ingress 可能需要 staging；
但 source-owned 既没有周期优势，也没有结构代理优势，缺乏继续做完整 wavefront
RTL 的 Pareto 余量。

### 3.3 裁决

`NO-GO_AS_DATE_MECHANISM`：

- 不接入 Local5 主 RTL；
- 不新建第三条 leftover/exact 路径；
- 不把 76.39% popcount 或 72.70% K-read bit 写成周期/能量收益；
- 保留 profile、Icarus population 与 sidecar Icarus/Verilator smoke，作为
  source-owned vs destination-owned 数据流消融；
- 该方向与 `docs/190/191` 的 SOSW 是同一旧候选的本土化筛选，不换名复活。

## 4. 本轮 DATE 影响

| 维度 | 影响 |
|---|---|
| Motion 创新 | RQTB 主张获得五充分统计量 score front 支撑；SSR5 对照表明不新增独立架构贡献 |
| Local5 创新 | 没有增加；通过强基线否决减少 scope creep |
| 验证 | Motion 新增双仿真器三方 packet miter 与真实公平行顶层 |
| 实验完整度 | 新增强基线和映射敏感性；仍缺 DC/SAIF、真实存储和 full encoder |
| 主表 | 完全不变 |

下一轮 Local5 不再围绕 Q-silent 充分统计量扩展。创新筛选应回到现有
`Shiftmax5 -> DiSEP/inverse-stencil -> source-major term -> TCFM5` 边界，寻找
能够打赢实体 transpose 和同端口 destination-major 的新存储对象或调度合同；
若没有强基线优势，就优先补系统完整度而不是增加名字。
