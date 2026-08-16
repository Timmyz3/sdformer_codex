# 双线接棒：Local5 Query-Silent 精确级联与 Motion LAWS 筛选

> 日期：2026-08-13  
> 范围：接 Codex `019f365d` 停点后的双线推进  
> 原则：新文件旁路；不把模型写成 RTL；不把 Yosys 写成 ASIC PPA

## 1. 接棒时的停点

Codex 已封存：

- Motion ep35 真实权重两通道 Acc32 miter（138 行，Icarus+Verilator/SVA）
- Local5 score/Shiftmax5 并入部署周期后，端到端从 post-score `1.49x` 掉到 `1.20x`

公开瓶颈已经清楚：Local5 公共 score 前端约占 TCFM 周期的 59%。再抠 TCFM 映射收益有限。

## 2. 新机制筛选

对封存向量做只读 profile，不先写 RTL：

| 候选 | 数据 | 裁决 |
|---|---|---|
| Local5 邻域密度双路径 | valid popcount 全是 3/4/5，没有 0/1/2 | 否决 |
| Local5 Shiftmax pattern memo | 92.2% valid score 全相等，但是 Shiftmax5 已是组合逻辑 | 单独做 memo 不省周期 |
| Local5 **Query-Silent** | **89.46% destination 的 Q==0**；此时 AXNOR 精确等于 `32-popcount(K)`，190575 个有效分数 0 mismatch | **晋级 RTL** |
| Motion 整核双工作区 | 同块双发射模型 `1.87x`，但整核复制 ANT≈0.94 | **否决为贡献** |
| Motion 共享后端 row-pipeline | FIFO p50=0，encode 与 emit 可重叠 | 保留为下一档候选，本轮不做整核复制 |

Query-Silent 能包装的原因不是又发明了一个 Shiftmax，而是：

> 光流局部注意力在 Q7 部署域里大量 query 静默。静默 query 的分数不再需要 32-lane AXNOR/XOR 扫描，只要 popcount(K)。这是 exact metadata-first cascade，和 Motion 的 score-class 合并用的是同一类量化域冗余。

## 3. Local5 Query-Silent RTL

旁路文件：`rtl_qfit/qfit_local5_qsilent_score_leaf.sv`  
`ARCH_QSILENT=0` 时对原 leaf 透明；`=1` 时 Q==0 走 2 拍快路径，Q!=0 仍走原 leaf。

证据包：`results/local5_qsilent_score_rtl_20260813`

| 检查 | 结果 |
|---|---|
| leaf miter Icarus+Verilator/SVA | 68 向量，score/gate/K/mask 全一致 |
| 100-group TCFM5/Linear5 Acc32 | 每配置 90,000，合计 270,000，0 mismatch |
| Yosys check | 0 problem |

周期（相对刚封存的 residual score 前端）：

| 配置 | 原周期 | Q-silent | 加速 | service |
|---|---:|---:|---:|---:|
| TCFM5 L1 | 324,605 | 191,424 | **1.6957x**（−41.03%） | 57,317 → 4,650（−91.9%） |
| Linear5 L1 | 390,325 | 257,144 | 1.5179x | 同上 |
| TCFM5 L2 | 328,153 | 194,972 | 1.6831x | 同上 |

因为公共前端变短，TCFM5/Linear5 的相对加速从 `1.2025x` 升到 **1.3433x**。后端差额没变，是 Amdahl 分母变小。

## 4. 现在可以写进 DATE 的贡献口径

### Motion（不变）

可逆时间 score-class 合并。主锚点仍是 ep35 单窗 Fixed2S→RQTB2S `1.1865x` `[rtl]`。

### Local5（本轮升级）

一条系统数据流，现在有两段可量化机制：

1. 固定五邻域 inverse-stencil + 五色无回放投影（原 post-score `1.49x`）
2. **Query-Silent exact cascade**：Q==0 时分数退化为 K popcount，跳过 residual XOR

二者合成后的公平部署切片是 TCFM5 L1 **191,424 vs Linear5 257,144 = 1.343x**，并且相对旧 residual 前端自身是 **1.70x**。

普通 banking、FIFO、定点化、验证框架仍不单列。

## 5. 明确不写

- Motion 整核双发射 `1.87x`：模型成立，但 ANT<1，不是贡献
- “发明 Shiftmax / 发明 TTB / 发明密度分层核”
- full encoder、能量、ASIC PPA
- 把 89.5% Q==0 写成算法稀疏率；它是 **部署域 Q7 的 query 静默**

## 6. 独立 DATE 复审（本轮证据包，不是整篇论文）

| 项 | 分 | 说明 |
|---|---:|---|
| 问题动机 | 4.3 | 针对已测量的 59% 前端，而不是再发明后端缩写 |
| 机制新颖性 | 3.4 | exact silence cascade 可讲，但是 metadata-first / skip 家族；要靠 workload 本土化撑住 |
| 证据纪律 | 4.6 | 与密封 baseline 同向量、miter、Acc32、Yosys；边界写清楚 |
| 系统完整度 | 2.8 | 仍缺 12-block 连续调度、DC/SAIF、encoder 分账 |
| **证据包** | **4.1 / 5 Weak Accept** | 可以作为 Local5 主贡献的前端补强 |
| **整篇 DATE 论文** | **3.3 / 5 Borderline** | 还差 PPA 与系统边界 |

## 7. 下一步

1. Motion：共享 encoder/Shiftmax 的 dual-directory row-pipeline，不要复制整核
2. Local5：把 Q-silent 与 12-block job 调度接到同一 wall-time
3. 有 DC 机器后：Q-silent 开/关同 SDC 对照 STA/SAIF
