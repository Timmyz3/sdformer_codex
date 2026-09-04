# DATE 第十九轮 XBF-QFSA 架构迭代与 RTL 负结果

> 后续状态：第十九轮独立复审发现 T8 方向密度路由存在合法 hot-bank
> 反例；该问题已在 `docs/207` 中以 DBDR 双上界路由整改。本文件保留为
> 发现反例前的阶段证据，不能作为当前最终方案。

## 1. 本轮结论

本轮没有把“跨方向动态借 lane”直接包装成成功贡献，而是先支付实际 RTL
代价。结果表明：

1. 全局 `128->4` tagged compactor 是明确负结果，不能作为 Local5 主候选；
2. 把固定五点 stencil 的方向标签折入 lane-bank 映射后，可以用四个局部
   选择器替代全局压缩网络；
3. 新候选命名为 **XBF-QFSA（XOR-Banked Fixed-incidence QFSA）**；
4. XBF-QFSA 已完成可切换 RTL、整数金参考、随机反压和开放综合代理；
5. 在相同 T8 exact density route 下，XBF-QFSA 仍比 4xW1 多 `7.20%`
   generic cells。因此它只是“等待真实 trace 偿还成本”的架构候选，不是已证明
   的 DATE 贡献。

QFIT 仍是总架构抽象：

- score 行方向使用 XBF-QFSA 做固定关联图上的商差执行；
- relation 列方向使用 FCSR 做闭式最后消费者退休；
- 二者共享同一 stencil 坐标、边界和方向语义。

SCS-Shiftmax、term、DCTF、FIFO 和双 context 仍只作为算子或实现机制，不单独
列为架构贡献。

## 2. 为什么全局 QFSA 失败

Local5 的四个邻方向各有 32 个 residual 候选。最直接的 QFSA 把它们合并成
128 项 tagged 域，每拍全局选择四项：

```text
4 x 32 residual masks
        |
        v
global 128-to-4 selector
        |
        v
4 tagged delta lanes
```

这种结构在周期模型中看起来可以跨方向借 lane，但在 RTL 中必须支付大规模优先
选择、tag 路由和多路器代价。Yosys generic mapping 结果为：

| 变体 | cell 数 | 相对同合同 4xW1 |
|---|---:|---:|
| 4xW1，完美 16-mask route | 19,866 | 基线 |
| 全局 QFSA，一拍 compactor | 24,719 | `+24.43%` |
| 全局 QFSA，两级 compactor | 24,834 | `+25.01%` |

两级版本只增加寄存器，没有消除选择网络。因此“给全局 compactor 加一级流水”
不是结构修复。该候选冻结为论文负结果，用于说明动态共享必须与固定图拓扑共同
设计，不能只在抽象模型里合并 lane。

## 3. XBF-QFSA：方向感知 XOR-bank 固定交换

### 3.1 固定置换

每个 residual 事件由 `(direction, lane)` 唯一标识。本轮采用：

```text
bank = lane[1:0] XOR direction[1:0]
```

四个 bank 各自每拍最多选一个事件。对任意固定方向，连续四个 lane 被一一映射
到四个 bank；对同一 lane，四个方向同样被一一打散。该映射不需要学习、预测、
排序或运行时路由表。

```text
N/S/E/W residual masks
        |
        v
fixed XOR permutation
   /    |    |    \
B0     B1   B2     B3
 |      |    |      |
local  local local  local selector
   \     |    |     /
       4 delta lanes
```

### 3.2 不是照搬蝶形 zero skipper

复旦 ISSCC 2023 天溪工作的 in-memory butterfly zero skipper 提供了“用规则
交换网络分散稀疏有效项”的启发；Bishop 提供了 density stratifier 的启发。
本项目没有主张首次蝶形网络，也没有照搬其存内零跳过结构：

- 天溪面向非结构化剪枝权重和 CIM；XBF 面向 Local5 固定方向 incidence；
- XBF 只使用一次方向感知 XOR 置换，不实现多级通用 butterfly；
- 路由阈值只选择 exact direct 或 exact residual 执行，不删除候选；
- 所有候选最终均进入相同四个 13-bit accumulator，并只做一次 RNE；
- 方向 tag 不是网络附加元数据，而是 Local5 数值语义的一部分。

因此可辩护的表述是：

> 我们将固定 stencil 的方向语义编译进一个 XOR-bank 局部交换层，使跨方向
> residual lane pooling 从全局 128 项压缩变为四个有界局部选择问题。

不能写“提出蝶形网络”或“首次把蝶形用于 SNN”。

### 3.3 exact density stratifier

每个方向先计算 `popcount(K_neighbor XOR K_self)`。T8 路由规则为：

- `delta_count > 8`：走共享 direct32 engine；
- `0 < delta_count <= 8`：走 XBF residual lane；
- `delta_count = 0`：不发 residual work；
- valid candidate 无论走哪条路径都产生完全相同的 raw score、RNE score 和
  Shiftmax gate。

它借鉴 Bishop 的 density stratification，但不使用异构 dense/sparse core，也
不使用 ECP 或任何有损 pruning。direct 与 residual 后端在 anchor 之后并行，
服务周期按两者最大值记账，不能相加夸大基线。

## 4. 已完成 RTL

### 4.1 文件与参数

- `rtl_qfit/qfit_tagged_compactor4.sv`：全局 128-to-4 负结果基线；
- `rtl_qfit/qfit_xorbank_compactor4.sv`：四个 XOR-bank 局部选择器；
- `rtl_qfit/qfit_local5_score_leaf.sv`：4xW1/QFSA/XBF 可切换 score leaf；
- `tb_qfit/tb_qfit_local5_score_leaf.sv`：自校验 testbench；
- `sim_qfit/run_qfit_score_leaf_checks.sh`：Icarus 与 Verilator 入口；
- `rtl_qfit/filelist.f`：独立 RTL 文件清单。

顶层参数：

| 参数 | 作用 |
|---|---|
| `ARCH_QFSA=0` | 四方向固定 W1 |
| `ARCH_QFSA=1, XBF_BANKED=0` | 全局 tagged QFSA |
| `ARCH_QFSA=1, XBF_BANKED=1` | XBF-QFSA |
| `PIPE_COMPACTOR=1` | selector 与 delta reduction 间插寄存器 |
| `USE_THRESHOLD_ROUTE=1` | 使用 exact 阈值路由 |
| `ROUTE_THRESHOLD=8` | 当前 T8 候选 |

### 4.2 功能证据

同一个 testbench 同时实例化：

1. 4xW1；
2. 全局 QFSA 一拍版；
3. 全局 QFSA 两级版；
4. XBF-QFSA T8 版。

已完成：

- 300 个随机五候选向量；
- 16 个 direct-mask 定向向量；
- 每个 valid candidate 的 Q7 score 与 direct golden 逐项相等；
- 五个 Q1.7 Shiftmax gate 逐项相等；
- 随机输出反压期间 payload 稳定；
- Icarus 仿真通过；
- Verilator lint 通过；
- QFSA 整数参考 10,000 组随机输入、50,000 个 score 零失配；
- 本轮相关 Python 单元测试 66 项通过（使用 `sdformerflow` 环境并禁用
  CUDA）。

这些是 `[rtl]` 与 `[整数参考]` 证据，不等于部署 fullres RTL-exact。

## 5. 开放综合代理与公平对照

### 5.1 compactor 单体

| compactor | generic cells | 相对全局 |
|---|---:|---:|
| global tagged 128-to-4 | 4,584 | 基线 |
| XOR-bank 4 x local selector | 1,002 | `-78.14%` |

固定置换确实消除了大部分全局选择代价。

### 5.2 score leaf

| 变体 | cells | wire bits | register cells |
|---|---:|---:|---:|
| 4xW1 + T8 route | 18,579 | 23,238 | 484 |
| XBF-QFSA + T8 route | 19,916 | 24,991 | 513 |

XBF-T8 相对 4xW1-T8：

- cells：`+7.20%`；
- register cells：`+5.99%`；
- 当前没有目标库 Fmax 和切换功耗。

因此不能只引用 compactor 单体 `-78.14%`，必须同时报告完整 score leaf
`+7.20%`。前者证明拓扑本土化有效，后者证明它尚未自动转化为系统收益。

## 6. 能否列为 DATE 架构创新

### 6.1 当前可列为候选的贡献

**固定关联图编译的 exact 双路径 score 架构：**

1. self anchor 与四方向 residual 的 exact 商差分解；
2. 方向感知 XOR-bank 固定交换，避免全局 tagged compaction；
3. 基于真实 delta density 的 direct/residual exact stratification；
4. 四候选共享 lane，但保留独立 accumulator 和最终 RNE 数值边界；
5. 与 FCSR 共享固定 incidence graph，形成 QFIT 行/列双向数据流。

这里真正的架构层内容是“拓扑映射、资源组织、工作分流和双向关系数据流”的
组合，不是 XOR 门、阈值比较器或 Shiftmax LUT 本身。

### 6.2 还不能写成已证明贡献

- 不能声称 XBF-QFSA 已降低能耗或 EDP；
- 不能用随机 RTL 向量替代 Local5 fullres ordered trace；
- 不能把 Yosys generic cell 当 DC 面积；
- 不能把 fullres 精度尚未通过的 Local5 宣布为论文唯一主线；
- 不能将 FCSR、XBF 和 SCS 分拆成三个互不相关的“首次”。

## 7. post-G0 晋级门槛

Local5 fullres exact 完成后，profile 必须输出每个 stage、sample、window 的：

- 四方向 `delta_count` 与真实 4x32 delta mask；
- 4xW1、global-QFSA、XBF-QFSA 的 ordered residual waves；
- T4/T8/T12 direct mask、direct cycles 和 residual cycles；
- XOR-bank 每 bank load、max/mean imbalance 与空槽率；
- score service mean/p95/p99；
- route 模式切换次数和 direct/residual 后端活动率；
- Shiftmax valid mask 与最终 gate；
- FCSR retire burst、FIFO occupancy、stall 和 Acc conflict。

XBF-QFSA 对 4xW1 的晋级条件：

1. 四 stage 加权 score 周期至少下降 `12%`；
2. p99 score 延迟不退化超过 `5%`；
3. mapped SAIF 下 score leaf energy 至少下降 `10%`；
4. 同 SDC、同 SRAM 合同下 EDP 至少下降 `15%`；
5. fullres hardware-order score/gate 零失配。

若周期收益小于 `8%`，直接淘汰 XBF，保留 4xW1；`8%~12%` 只作消融；
达到 `12%` 后才允许进入 DC/SAIF。这样避免为了“架构新意”保留不赚钱的网络。

## 8. 下一步

1. 等待 Local5 fullres exact follower 完成，自动跑 post-G0 profile100；
2. 用真实 ordered mask 比较 4xW1、global-QFSA 和 XBF-QFSA；
3. 实现 Dynamic Frontier 与 FCSR 同合同关系转置 RTL；
4. 将 score leaf 与 relation leaf 接成最小 QFIT tile；
5. 在同一 SDC、同 macro 合同下做 DC/STA/SAIF；
6. 只在上述门槛通过后，把 QFIT 写进 DATE 主贡献列表。

本轮最重要的产物不是新增一个名字，而是得到一个可复现负结果：全局动态 lane
pooling 在 RTL 中不划算；只有把 Local5 固定图的方向语义编译进局部交换网络，
才把额外成本压到有机会由真实 workload 偿还的范围。
