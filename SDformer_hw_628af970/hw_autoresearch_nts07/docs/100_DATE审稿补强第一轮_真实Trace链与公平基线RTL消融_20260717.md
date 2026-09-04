# DATE审稿补强第一轮：真实Trace链与公平基线RTL消融

> **状态更新（2026-07-17）**：本文记录第一轮 trace-shaped 消融。真实四stage trace、真实权重候选整数回放和 FADC24 架构再冻结随后已经完成，最新结论以 `docs/101_H67真实四Stage消融与GateStack架构再冻结_20260717.md` 为准。本文中“GPU阻塞”“真实trace未运行”等表述仅代表当时状态，不再是当前状态。

## 一、本轮目标

本轮直接针对 `docs/97` 的两个最高优先级拒稿原因迭代：

1. 现有回放不是 H67 真实网络 bit trace；
2. GateStack 缠在一起，缺少同接口 RAW-only 与 no-residency 公平消融。

架构与 RTL 两个独立 orchestrator 已分别输出：

- `docs/98_H67_GateStack_DATE补强架构签核规格_20260717.md`；
- `docs/99_GateStack公平基线与真实Trace_RTL实施规格_20260717.md`。

本文件记录主线程随后完成的代码、RTL、验证和论文判断更新。

## 二、真实 H67 位级 Trace 采集链

### 2.1 已实现

新增 `h67_bit_trace.py`，在 H67 attention 的真实推理点采集：

- `Q` 二值位图，布局 `[T=2, window, head, spatial_token, lane]`；
- `K` 二值位图，布局与 Q 对齐；
- RTL Shiftmax 之后的真实 Q1.7 gate code，范围 `[0,256]`；
- checkpoint 中的 `attn.proj.weight` 和 bias；
- 一个候选逐输出通道 dyadic INT8 权重编码；
- 与该候选权重 scale 对齐的 bias accumulator code；
- 文件 SHA256、shape、活性和 stage 覆盖 manifest。

profile 入口新增：

```text
--bit-trace-dir
--bit-trace-samples
--bit-trace-windows
--bit-trace-all-blocks
```

默认只导出一个样本、每个模块一个窗口和四个 stage 的 `B0`，控制数据量且覆盖 S0/S1/S2/S3。

### 2.2 数据质量审计

新增 `scripts/audit_h67_bit_trace.py`，从 NPZ 重新检查和计算：

- SHA256、Q/K/gate/weight/bias shape；
- Q/K bit pack/unpack 一致性；
- gate Q1.7 合法范围；
- dyadic INT8 权重半步长误差界；
- 真实 K active-lane direct work；
- 真实 `(gate_code,lane)` 等价类 term；
- 每 row class、fanout、IPD bits、RAW41 bits 和容量模式；
- 四 stage 覆盖完整性。

采集器和审计器共 6 个 CPU 单元测试通过；`bsa_attention.py` 原有 57 个测试全部通过。

### 2.3 GPU 状态与运行入口

当前 GPU 使用约 `65.7 GiB/80 GiB`，利用率接近满载。为避免与训练抢卡，没有启动真实模型采集。

新增入口：

```bash
sim_hitflow/run_h67_real_bit_trace_capture.sh
```

该入口在 GPU 使用超过 8192 MiB 时返回 75 并停止，不创建后台 watcher。GPU 空闲后，它将一次完成：

```text
H67 ep19 RTL-exact单样本推理
  -> 四stage B0真实bit trace
  -> manifest
  -> 强制四stage数据质量审计
```

当前状态必须写成：

> `[代码完成/运行阻塞]` 真实 trace 采集与审计链已实现，真实四 stage 数据产物尚未生成。

### 2.4 权重量化边界

当前导出的逐输出通道 dyadic INT8 是候选合同：

```text
scale[o] = 2 ^ ceil(log2(max_abs(weight[o,:]) / 127))
code[o,i] = RNE(weight[o,i] / scale[o])，饱和到[-127,127]
```

它满足编码误差界，但尚未进入 valid825 全模型推理，也未冻结 projection BN folding、bias/requant 和 residual scale。因此不能把它写成“已验证 INT8 部署”。

## 三、公平基线 RTL 实现

### 3.1 编译期 no-residency

在以下层次加入 `ENABLE_RESIDENCY` 编译期参数：

- `gatestack_replay_plan_builder.sv`；
- `gatestack_replay_control_plane_top.sv`；
- `gatestack_single_context_execution_top.sv`。

`ENABLE_RESIDENCY=0` 时：

- CSR head 每个 output tile 都走 IPD32W；
- 不执行 cache lookup、promotion 或 cache release；
- descriptor cache 与 auto-fill adapter 从 elaborated hierarchy 删除；
- IPD fill sideband 被无损消费，不阻塞主 decoder；
- 顶层功能接口、权重后端、AccTile、反压和输出合同保持不变。

### 3.2 RAW41-only 运行路径

trace 生成器新增 `--force-raw`，把相同构造 gate/K 数据全部编码成每 head `162×41=6642 bit` RAW41。

该基线：

- 使用相同 single-context execution top；
- 使用相同 scheduler、weight response、TDR、multicast 和 AccTile；
- 所有 head 每个 tile 完整 RAW41 replay；
- 数值输出与 GateStack 相同。

严格边界：当前 RAW-only 是同顶层的运行路径基线，完整顶层仍包含未激活的 IPD/cache 逻辑。它可以用于周期与流量比较，不能用于物理面积比较。后续还需要 physically-stripped Direct top。

### 3.3 回归矩阵

新增：

```bash
sim_hitflow/run_gatestack_p0_baseline_ablation.sh
```

三种模式均通过：

- Icarus 功能仿真；
- Verilator `--assert -Wall`；
- 现有 SVA bind；
- `T162 × H24 × O24 × L32`；
- 576 个 head session；
- 3,888 个 final token；
- 整数输出 `mismatch=0`；
- `protocol_error=0`。

## 四、第一批公平消融结果

结果路径：`results/gatestack_p0_baseline_ablation_20260717/report.md`。

| 模式 | Verilator周期 | 相对RAW加速 | payload words | 相对RAW减少 | projection terms | 相对RAW减少 | slot replay | cache hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW41-only运行路径 | 463,514 | 1.000x | 59,904 | 0.00% | 64,848 | 0.00% | 576 | 0 |
| IPD无驻留 | 242,599 | 1.911x | 15,264 | 74.52% | 30,960 | 52.26% | 576 | 0 |
| GateStack完整机制 | 236,382 | **1.961x** | 8,663 | **85.54%** | 30,960 | **52.26%** | 415 | 529 |

这里的周期是 `[RTL trace-shaped]`，不是整 encoder FPS；payload word 是逻辑 64-bit 传输，不是能量。

### 4.1 机制分账

**等价类/IPD 是当前主要周期收益来源。**

相对 RAW-only，IPD-no-residency 已获得：

- `1.911x` 周期加速；
- projection terms 减少 `52.26%`；
- payload words 减少 `74.52%`。

**Residency 的吞吐增益较小，数据移动增益较大。**

GateStack 相对 IPD-no-residency：

- 周期从 242,599 降到 236,382，改善约 `2.56%`，速度约 `1.026x`；
- slot replay 从 576 降到 415；
- payload words 从 15,264 降到 8,663，改善 `43.25%`；
- projection terms 不变。

因此 residency 当前不能作为主要吞吐创新。它是否能成为能效创新，必须由目标库 SRAM/mapped SAIF 证明。

### 4.2 开放结构趋势

同版本 Yosys `memory_collect` 结果：

| 构建 | generic cell | logical memory | `$mul` | `$mux` |
|---|---:|---:|---:|---:|
| GateStack | 3,984 | 12 | 43 | 1,288 |
| no-residency | 3,706 | 9 | 38 | 1,191 |

编译期关闭 residency 后 generic cell 减少约 `6.98%`，logical memory 从 12 降到 9。这说明 residency 有真实结构代价，更需要 SRAM 能耗收益来证明净 EDP。

Yosys generic cell 不是标准单元面积，不能进入 PPA 主表。

## 五、对架构创新的修正

### 5.1 不再平均强调三个机制

本轮证据不支持把 residency、output-stationary 和多格式共享后端平均写成三个主创新。更合理的层次是：

**主架构机制：final-gate 等价类驱动的精确因子化投影执行。**

- final Q1.7 gate 与 K lane 共同形成唯一 product key；
- 一个 product 通过 destination list/multicast 服务多个 token；
- IPD32W 把等价类 descriptor 与 token destination 分离；
- RAW41 保证任何容量越界仍然 exact。

**数据流机制：output-tile-stationary 的 head-stacked replay。**

- 一个 AccTile 在所有 input head 期间驻留；
- 避免 head-major partial-sum spill；
- 仍需 head-major RTL/模型公平对照。

**能效辅助机制：decode-once residency。**

- 当前主要减少 descriptor/payload 数据移动；
- 只有目标库 EDP 改善达到 15% 门槛时才保留在标题贡献中；
- 若 EDP 不过线，应降为实现优化。

### 5.2 对 C1 双 Context 的影响

`docs/98` 条件推荐双 context C1，但本轮结果显示 resident promotion 对 execute latency 的直接收益只有约 2.56%。双 context 是否值得，取决于它能否隐藏尚未计入的 build/commit 和真实 SRAM latency。

因此 C1 继续保留为条件候选，但不能仅凭 cache hit 率晋级。必须满足：

- 真实 trace 上 `L_cold_total` 相对 C0 至少改善 10%；
- 端口冲突和双份 slot/cache 后，子系统 EDP 至少改善 15%；
- full-encoder throughput 有可见 Amdahl 收益。

## 六、DATE 审稿缺口状态更新

| P0 项目 | 本轮状态 | 仍缺什么 |
|---|---|---|
| 四 stage 真实 bit trace | 采集/审计代码完成 | GPU 空闲后运行、生成真实产物 |
| 真实量化权重/bias | 候选编码器完成 | BN folding、requant、valid825 精度 |
| IPD no-residency 基线 | **同接口 RTL 完成** | 真实 trace、目标库 PPA |
| RAW41-only 基线 | **同顶层周期基线完成** | physically-stripped Direct top |
| head-major spill 基线 | 未完成 | scheduler、partial-sum SRAM 接口与RTL |
| 目标库 PPA | 未完成 | `dc_shell`、`.db`、PVT、SRAM macro、mapped SAIF |
| full-encoder 闭环 | 未完成 | SCS/ATLIF/skip/外存统一周期和能量分账 |

## 七、当前允许与禁止的 claim

当前新增允许表述：

- 在一个 H67 ordered-statistics-shaped stage3 workload 上，GateStack、IPD-no-residency 和 RAW41-only 三种同顶层路径均完成默认规模 RTL 回放且零 mismatch；
- GateStack 相对 RAW41-only 运行路径获得 `1.961x` RTL execute-cycle speedup；
- IPD 等价类减少 `52.26%` projection terms；
- residency 相对 no-residency 减少 `43.25%` payload words，但周期只改善约 `2.56%`；
- no-residency 编译变体已删除 descriptor cache/auto-fill 层次。

仍禁止表述：

- “真实 H67 trace 已完成回放”；
- “GateStack 节能 85.54%”；
- “Direct baseline 面积已测”；
- “residency 是主要吞吐创新”；
- “达到 30 FPS”；
- “DC/PPA 已完成”。

## 八、下一阶段

```text
1. GPU空闲后运行真实四stage bit trace
2. 将真实trace转换成IPD32W/RAW41与整数金参考
3. 用真实trace重跑RAW-only/no-residency/GateStack矩阵
4. 实现physically-stripped Direct RAW41 top
5. 实现head-major + partial-sum spill公平基线
6. 冻结projection INT8/bias/requant并跑valid825
7. 获取目标库与SRAM macro后跑DC/STA/mapped SAIF/LEC
8. 再次由独立DATE审稿人复审
```

## 九、阶段结论

本轮已经把“没有公平基线”的拒稿问题推进到部分闭环，并改变了架构贡献排序：当前最强证据支持 final-gate 等价类因子化执行，而不是 descriptor residency 本身。

真实 trace、物理 Direct baseline、head-major spill 和目标库 PPA 仍是下一轮直接阻塞项。在这些证据完成前，当前结论只属于有序统计塑形 workload 的 RTL 微架构消融。
