# FADC24流式Decoder与四Stage同顶层RTL迭代

> 后续状态：`docs/105_统一AdaptiveCSR运行时双格式架构与RTL验证_20260718.md`已经把IPD32W/FADC24从两个编译期配置合并为单一可综合运行时前端，并完成同一context的IPD/FADC/RAW交错回放。因此本文关于“运行时双格式尚未实现”的描述仅代表FADC24单格式迭代时点。

## 一、结论先行

本轮将 `docs/101` 中只有 Python 金参考和容量模型的 FADC24 候选推进为真实 RTL，并接入 H67 single-context 投影执行顶层。四个 stage 均使用真实 Q/K、真实 Q1.7 gate、checkpoint projection 权重的候选 dyadic INT8 码和真实 bias 候选码，完成 Icarus 与 Verilator/SVA 双模拟器回放：

- S0/S1/S2/S3 全部 `mismatch=0`；
- `group_done_error=0`、`protocol_error=0`；
- `count_error_aborts=0`、`count_timeout_aborts=0`；
- S3 的 projection term 从 IPD 无驻留的 `30,960` 降为 `12,888`；
- S3 周期从 IPD 无驻留的 `259,122` 降为 `169,703`，对应同顶层速度比 `1.527x`；
- FADC24 在 S0/S2 没有减少 term，周期反而分别比 IPD 无驻留慢约 `5.6%` 和 `4.5%`。

因此，FADC24 **不应全 stage 无条件替换 IPD32W**。当前真实证据支持的架构决策为：

> S0/S1/S2 保留 GateStack-IPD/residency，S3 使用 FADC24；格式选择由 stage/block descriptor 冻结，RAW41 继续作为容量安全的精确回退。

四 stage 单窗口 trace bundle 中，该组合的周期和为 `195,149`，相对当前 GateStack 的 `278,388` 为 `1.427x`。该数字只用于同一组真实 trace 的架构筛选，不是完整 encoder 延迟或 FPS。

## 二、实现内容

### 2.1 FADC24格式

每个精确等价类 term 的 key 为：

```text
(final gate code[8:0], K lane[4:0])
```

24-bit descriptor 为：

```text
gate[8:0] | lane[4:0] | destination_count[7:0]
| bitmap_mode | reserved
```

每个 term 的 destination set 按 fanout 无损选择：

- `fanout <= 21`：8-bit token ID list；
- `fanout > 21`：162-bit bitmap，占 21 byte，最高 6 个 padding bit 必须为零；
- 整个 head 超过物理槽容量：回退 RAW41。

该格式不改 gate、K、权重、乘积或累加数学语义，只改变一个共享乘积的目的集合表示。

### 2.2 流式decoder

新增主实现：

- `rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv`。

数据通路为：

```text
64-bit slot word
  -> 256-bit byte reservoir
  -> 16-byte header校验
  -> 顺序24-bit descriptor解包
  -> list字节流 / 21-byte bitmap装载
  -> 9×18-bit分段bitmap scan，每周期最多4个token
  -> 统一term/event/done接口
  -> TDR + product + multicast + AccTile
```

全 head buffer 版本 `gatestack_fadc24_replay_decoder.sv` 仅作为结构参考，不进入主数据流。流式版本避免 832-byte 随机访问 buffer，并保留 header、长度、tag、word index、last、descriptor、event sum、bitmap padding 和尾部剩余字节校验。

### 2.3 同顶层接入

以下模块新增编译期 `CSR_FORMAT_FADC24` 参数：

- `gatestack_multihead_decoder_projection_top.sv`；
- `gatestack_single_context_execution_top.sv`。

FADC24 与 IPD32W 共用既有 term/event/done 接口和投影后端。当前 FADC24 变体设置 `ENABLE_RESIDENCY=0`，目的是先隔离格式和 decoder 的净收益；尚未声称已支持 FADC24 descriptor residency 或运行时双格式切换。

### 2.4 物理槽合同修复

本轮拆分：

```text
RAW_PAYLOAD_BITS   = 6642
SLOT_CAPACITY_BITS = 104 × 64 = 6656
```

修复涉及：

- `gatestack_head_slot_sram_adapter.sv`；
- `gatestack_replay_plan_builder.sv`；
- `gatestack_replay_control_plane_top.sv`；
- `gatestack_single_context_execution_top.sv`；
- 对应 SVA 和 head-slot TB。

修复后：

- storage adapter 允许 CSR/FADC 使用完整 `6656 bit` 物理槽；
- replay planner 仍要求 RAW payload 精确等于 `6642 bit`；
- 压缩格式容量检查使用 `SLOT_CAPACITY_BITS`；
- head-slot TB 新增 104 个完整物理 word 的 CSR commit/replay/release，双模拟器通过。

## 三、真实四Stage结果

完整报告：`results/gatestack_fadc24_fulltop_20260718/report.md`。

| Stage | FADC周期 | 相对IPD无驻留 | 相对GateStack | payload words | 相对IPD减少 | terms | 相对IPD减少 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 2,592 | 0.947x | 0.924x | 159 | 14.52% | 186 | 0.00% |
| S1 | 1,729 | 1.000x | 0.970x | 72 | 0.00% | 0 | 0.00% |
| S2 | 23,479 | 0.957x | 0.910x | 1,560 | 15.03% | 1,956 | 0.00% |
| S3 | 169,703 | **1.527x** | **1.490x** | 13,608 | 10.85% | 12,888 | **58.37%** |

正确解释如下：

1. S0/S2 的 IPD32W 本来就能容纳所有 head，FADC24 只减少 payload word，没有减少执行 term；复杂 decoder 开销导致周期变差。
2. S1 是全空 K 场景，两种格式都只解析空 header；FADC24 相对 IPD 无驻留周期相同，但不如已有 descriptor residency。
3. S3 的一个高扇出 head 在 IPD32W 下回退 RAW41；FADC24 用 bitmap 装入原物理槽，恢复了 factorized term 执行，因此收益来自避免 fallback，而不是一般性的 header 压缩。
4. FADC24 当前关闭 residency，所以相对完整 GateStack 的 payload word 可能更高；周期收益不能直接写成 SRAM 能量收益。

## 四、验证与缺陷发现

### 4.1 叶级验证

入口：`sim_hitflow/run_gatestack_fadc24_leaf_checks.sh`。

覆盖：

- 真实 S3/head4，61 term、814 event、15 bitmap term；
- 随机 term/event/done backpressure；
- 无 backpressure；
- bitmap padding bit 错误注入；
- buffered reference 与 streaming implementation；
- Erie lint、Verilator `-Wall --assert`、Yosys 综合可读。

新增专属断言：

- descriptor/term/event/done 在 stall 下稳定；
- `event_count` 与 token-valid mask 一致；
- token-valid 为前缀掩码；
- head-last 必须同时为 term-last；
- descriptor、term、event、done 输出相位互斥；
- protocol error 置位后保持。

### 4.2 同顶层验证

入口：`sim_hitflow/run_gatestack_fadc24_fulltop_ablation.sh`。

四个 stage 均执行：

- Icarus 功能仿真；
- Verilator `-Wall --assert`；
- 既有 scheduler、slot、control、lifecycle、mux、projection SVA；
- 新 FADC24 decoder SVA；
- 真实候选 INT8 权重、bias 与逐元素 32-bit accumulator 金参考。

Icarus与Verilator的功能计数一致，但group周期分别相差S0 `2`、S1 `1`、S2 `1`、S3 `4`周期。结果主表统一使用Verilator周期；当前只声明功能等价，不声明双模拟器cycle-exact。

### 4.3 本轮实际发现的两个缺陷

1. **测试向量双tag错误**：FADC header 最初使用 `0xFA...` 调试 tag，而 lifecycle 要求 decoder 返回 slot payload tag `0x6800_0000+head`。硬件正确触发 abort；生成器已修复。
2. **纯list header合同错误**：原 header 检查错误要求非空 head 必须至少有一个 bitmap term，导致合法纯 list head 被拒绝。已改为“空 head 的 bitmap 数必须为零；非空 head 允许 bitmap 数为零”。S0 纯 list head和S1全空head已在完整顶层覆盖。

这两个问题说明完整 lifecycle 回放和跨稀疏形态测试不能由单个高扇出 leaf TB 替代。

## 五、结构代价

统一使用 Yosys `proc; opt; memory -nomap` 的结构代理：

| Decoder | generic cells |
|---|---:|
| RAW41 | 293 |
| IPD32W | 448 |
| FADC24流式 | 954 |
| FADC24全buffer参考 | 4,611 |

流式实现相对全 buffer 参考减少约 `79.3%` generic cells，但仍约为 IPD32W 的 `2.13x`。这不是目标库面积，也没有包含布线、SRAM macro、时钟树和功耗。

当前环境仍没有 `dc_shell`、目标 `.db`、PVT 和 SRAM macro，因此不能给出 DC 面积、频率、功耗或 EDP 主表。

## 六、架构创新边界

可以主张的机制不是“发明 list/bitmap”，而是：

> 面向 H67 final-gate 等价类的 fanout-adaptive exact destination dataflow：以 `(gate code, K lane)` 为可复用乘积键，按目的 fanout 选择 list/bitmap，并直接驱动共享 product 的 token multicast；任何容量越界回退 RAW41，保持网络语义不变。

该机制相对一般稀疏格式的区别在于编码单位和执行单位都是一个可复用乘积的 destination set，格式选择会决定是否保留跨 token 的乘积复用，而不仅是压缩索引存储。

但当前还不能声称：

- list/bitmap 自适应编码为首次提出；
- 已实现运行时 stage-adaptive 双 decoder；
- FADC24 已支持 descriptor residency；
- profile100 的 ambiguous 容量已消歧；
- 完整 encoder、valid825 部署精度和目标库 PPA 已闭环。

## 七、下一阶段

按 DATE 拒稿风险排序：

1. **physically-stripped Direct RAW41 top**：同接口、同 lane、同 SRAM，综合时真正删除 IPD/resident/cache/fill，而不是只跑 RAW runtime path。
2. **head-major partial-sum spill baseline**：至少实现 scheduler、PSUM SRAM adapter 和 cycle/traffic RTL计数，证明 output-tile-stationary 的价值。
3. **扩大真实 bit trace**：覆盖更多 sample、block、window，报告各 stage 的 FADC fallback、周期 p50/p95/p99 和最坏值。
4. **FADC24 residency评估**：优先只驻留解码 descriptor，目的 list/bitmap仍从 slot 流式读取；若控制或地址对齐代价过大则保留 no-residency S3 专用路径。
5. **部署量化闭环**：valid825 上冻结 projection BN folding、weight/bias/requant、饱和、残差 scale 和最终精度。
6. **目标库签核**：获得 `.db`、PVT、SRAM macro 后跑 DC/STA、mapped SAIF、netlist LEC，再决定双格式 decoder 的面积和 EDP 是否值得。

## 八、证据入口

- `results/gatestack_fadc24_real_trace_20260717/analysis.md`；
- `results/gatestack_fadc24_profile100_20260717/analysis.md`；
- `results/gatestack_fadc24_fulltop_20260718/report.md`；
- `results/gatestack_fadc24_real_trace_vectors_20260718/manifest.json`；
- `rtl_hitflow/gatestack_fadc24_streaming_replay_decoder.sv`；
- `sim_hitflow/run_gatestack_fadc24_leaf_checks.sh`；
- `sim_hitflow/run_gatestack_fadc24_fulltop_ablation.sh`。
