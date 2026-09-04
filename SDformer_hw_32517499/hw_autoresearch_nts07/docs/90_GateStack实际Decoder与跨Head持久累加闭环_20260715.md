# GateStack 实际 Decoder 与跨 Head 持久累加闭环

## 1. 本轮回答的问题

前一阶段已经证明三种 replay 表示可以共享一套投影后端，但仍有两个关键证据缺口：

1. 统一流是否真的来自 Resident、IPD32W、RAW41 三种 Decoder，而不是测试平台人工构造；
2. 多个输入 head 是否在同一个输出 tile 上持续累加，并且只在最后一个 head 后加一次 bias。

本轮分别闭合这两个问题，并进一步完成三路 replay 路由与多 head 持久累加的组合验证。

当前结论是：

- 实际三 Decoder 到共享 TDR、多播、累加器、bias 的单 head 数值链已经闭合；
- 统一流输入下，tile 级跨 head 持久累加已经闭合；
- 三种 replay 路由跨 head 切换、共享同一投影后端已经闭合；
- 实际三 Decoder 与 routed multihead 控制器已经封装为同一 wrapper，并完成三 head 数值回归；
- slot、cache、launch、lifecycle、双 context 和 output-tile 调度仍在最终加速器边界之外。

## 2. 实际三 Decoder 集成

模块：

```text
gatestack_decoder_projection_top
  +-- resident replay joiner
  +-- IPD32W replay decoder
  +-- RAW41 replay decoder
      +-- RAW tail retimer
      +-- RAW issue adapter
  +-- session-locked replay mux
  +-- TDR product/bitmap backend
  +-- segmented multicast
  +-- banked accumulator
  +-- bias commit
```

三种表示只复制格式恢复逻辑，不复制乘法、多播和 accumulator。实际 TB 依次输入：

| 路径 | 输入内容 | 目标 token | gate/lane |
|---|---|---|---|
| Resident | descriptor 与 token-only replay | 0、7 | gate=2，lane=0 |
| IPD32W | 4 个 64-bit 完整 payload word | 1、6 | gate=3，lane=1 |
| RAW41 | 8 个 41-bit token record | 2、5 | gate=4，lane=2 |

最终结果对所有 8 个 token、每个 token 的 2 个输出 lane 做整数逐元素比较。测试不是只比计数或 done。

严格回归结果：

```text
PASS: actual three-decoder projection terms=4 completed=4 bias=24 cycles=104
```

这里 bias=24 是三次独立单 head group 各提交 8 个 token，用于验证每条实际 Decoder 路径。它不是最终多 head 投影的正确 bias 次数。

## 3. RAW 尾事件语义缺陷与修复

### 3.1 原问题

RAW41 Decoder 原先只在“最后一个 token record 自身产生 direct event”时拉高 `direct_head_last`。真实数据可能出现：

```text
token 2: active
token 5: active
token 6: K-zero
token 7: K-zero，且为最后 record
```

此时 token 5 是最后一个有效事件，但原 Decoder 不会把它标为 `head_last`。TDR backend 已正确要求非空 head 必须看到最后 bitmap，所以该边界会导致 backend 等待，而不是错误地提前释放资源。

### 3.2 修复方法

新增 `gatestack_raw_tail_retimer`：

1. 始终保留当前最后一个 direct event；
2. 下一个 direct event 到来时，释放前一个，并标记为非末尾；
3. Decoder done 到来时，释放保留事件，并标记为真正的 `head_last`；
4. 空 head 不制造伪事件，直接传递 done；
5. 输出或 done 背压时保持全部 payload 稳定。

这样做不修改任何 gate、lane、token 或乘积值，只修复流式协议中的“最后有效事件”判定。

专用 TB 覆盖了普通事件、尾随 K-zero、空 session 和随机背压，Icarus、Verilator assertion、Yosys、Erie 均通过。

## 4. 跨 Head 持久累加

新增模块：`gatestack_multihead_tile_projection_top`。

### 4.1 生命周期层级

```text
tile_start
  -> accumulator clear，一次
  -> head 0: TDR session -> backend_done
  -> head 1: TDR session -> backend_done
  -> ...
  -> last head: TDR session -> backend_done
  -> token 0..T-1 bias commit，一次
  -> final output
  -> tile_done
```

关键约束：

- accumulator group 的生命周期是 output tile，不是 input head；
- TDR backend 的生命周期是一个 input head；
- `backend_done` 表示 Decoder done 且所有 outstanding multicast 已完成；
- 只有最后一个预期 head 完成后才能进入 bias 状态；
- `head_index` 必须从 0 连续递增；
- `head_last` 必须与 `tile_start_head_count` 推导出的最后 head 一致；
- 每个 head 的 `input_channel_base + lane_id` 形成全局输入通道索引。

### 4.2 数值验证

定向 TB 使用 3 个 head：

- head 0：一个非空 term；
- head 1：空 head；
- head 2：两个 term；
- 多个 term 写入重叠 token，验证跨 head 求和；
- weight response、head done、bias 和 final output 均施加背压。

结果：

```text
PASS: GateStack multihead tile heads=3 terms=3 bias=8 cycles=53
```

8 个 token、每个 token 的 2 个输出 lane 全部与独立整数期望一致。最重要的协议结果是：3 个 head 只提交 8 次 bias，而不是 24 次。

## 5. 三路表示跨 Head 路由

新增模块：`gatestack_routed_multihead_tile_projection_top`。

它在每个 head 开始时原子获取两类资源：

1. replay mux 的 route session；
2. 多 head 投影控制器的 TDR head session。

只有二者同时 ready，`head_start` 才能握手。非法 route 不会被接收，避免 mux 未激活而 backend 已启动造成死锁。

定向 TB 的三个 head 分别走 source 0、1、2，并写入重叠 token。检查项包括：

- route 在一个 head 内保持锁定；
- 未选 source 的 ready 不会误拉高；
- 每个 head 完成后才允许切换 route；
- 三个 route 的结果在同一 accumulator 中累加；
- 最后一个 head 后只提交一次完整 bias；
- 最终逐 token、逐 lane 整数一致。

结果：

```text
PASS: GateStack routed multihead heads=3 terms=3 bias=8 cycles=49
```

Icarus 与 Verilator 的周期打印可能因 TB 时序调度差 4 个周期，但两者的握手数、最终数值和协议断言一致。论文不能直接引用该小 TB 周期作为吞吐结果。

## 6. 严格验证流程

新增入口：

```bash
sim_hitflow/run_gatestack_raw_tail_retimer_checks.sh
sim_hitflow/run_gatestack_decoder_projection_checks.sh
sim_hitflow/run_gatestack_multihead_tile_projection_checks.sh
sim_hitflow/run_gatestack_routed_multihead_tile_projection_checks.sh
sim_hitflow/run_gatestack_multihead_decoder_projection_checks.sh
```

每个新入口包含：

- Icarus 自检 TB；
- Verilator `--assert -Wall`；
- Verilator 0 warning/error 门槛；
- Yosys `hierarchy/proc/opt/check/stat`；
- Erie 独立静态 lint。

多 head SVA 额外检查：

- tile start 在背压下 payload 稳定；
- head start 在背压下 tag/index/base/last 稳定；
- head done 在背压下 payload 稳定；
- tile done 在背压下 tag 稳定。

实际三 Decoder 多 head 顶层使用同一个 tile、同一个 tag，依次执行 Resident、IPD32W、RAW41 三个 head。RAW 路径继续保留尾随 K-zero 场景。结果为：

```text
PASS: actual three-decoder multihead heads=3 terms=4 bias=8 cycles=79
```

该回归在最外层 wrapper 和内部 tile controller 两个层级分别绑定 start/done 稳定性断言，覆盖 Decoder 未 ready 时的外部 head-start 背压。

### 6.1 162-token 尺度回归

另设 `TOKENS=162, LANES=32, SEGMENT_TOKENS=18` 的完整顶层用例：

- Resident 目标 token 为 0、161；
- IPD32W 目标 token 为 1、160；
- RAW41 输入完整 162 个 record，共 6642 bit、104 个 64-bit SRAM word；
- RAW 活动 token 为 2、159，token 160、161 为尾随 K-zero；
- 三个实际 Decoder 仍在同一个 tile 内跨 head 累加；
- 检查全部 162 个 token、每个 token 的 2 个输出 lane；
- bias 和 final output 都带周期性背压。

结果：

```text
PASS: actual three-decoder multihead scale162 heads=3 terms=4 bias=162 cycles=529
```

Icarus 与 Verilator/SVA 均通过，Verilator 编译 0 warning/error。该结果证明当前 GateStack 新顶层自身可运行 162-token 尺度，不再借用旧 G1 的 T162 回归作为替代证据。529 周期仍是定向小工作量 TB，不是 profile100 端到端吞吐。

## 7. Yosys 结构审计

默认参数下的 generic 结构：

| 顶层 | generic cells | memory bits | `$mul` |
|---|---:|---:|---:|
| 多 head tile projection | 1159 | 41472 | 8 |
| 三路 routed 多 head projection | 1268 | 41568 | 8 |
| 实际三 Decoder 单 head projection | 2393 | 45664 | 8 |
| 实际三 Decoder 多 head projection | 2456 | 45664 | 8 |

这组数字只支持以下结构性结论：

- 从单 head 扩展到多 head 没有复制乘法器；
- 增加三路 replay mux 后仍只有 8 个真实 `$mul`；
- 实际三 Decoder 主要增加控制、解码与少量 replay 存储；
- accumulator 仍是主要逻辑 memory bits 来源。

这些数字不是标准单元面积，也没有 SRAM macro、时序和功耗含义，不能作为 DATE PPA 主表。

## 8. 对架构贡献的实质推进

本轮使以下贡献具备 RTL 证据：

1. **表示异构、计算统一**：Resident、IPD32W、RAW41 三种无损表示共享一个 TDR、多播和乘加后端；
2. **双层资源生命周期**：Decoder route 可在流消费完后释放，但 head slot 必须等 outstanding multicast 全部完成后才能释放；
3. **tile-lifetime persistent accumulation**：跨 input head 保持 accumulator 驻留，消除每 head 清零、写回和重复 bias；
4. **原子 route/backend 获取**：防止 replay 路由与计算 session 启动相序不一致；
5. **尾事件语义恢复**：RAW trailing K-zero 下仍能无损构造真正的最后事件，不依赖近似剪枝。

其中第 3 点属于完整投影数据流的必要架构机制，不再只是单行 attention operator 优化。

## 9. 尚未闭合的边界

按严重度排序：

1. output-tile 循环尚未实现，当前一次 tile session 只处理一个输出 tile；
2. descriptor cache 与 head-slot 的 fill/replay 同拍关系尚未接入；
3. 双 context 的 build/replay 重叠尚未进入 ordered cycle RTL；
4. malformed 输入的统一 abort/drain FSM 尚未实现；
5. bias 独立位宽、requant、饱和与输出格式尚未冻结；
6. 尚无 ordered trace transcript、门级 SAIF、目标库 DC、LEC、WNS 和 SRAM macro PPA；RTL VCD 已生成，但不能替代上述证据。

## 10. 下一步实施顺序

1. 增加 output-tile controller，使 descriptor 在多个输出 tile 间驻留并只切换 weight tile；
2. 接入 slot/cache/launch/lifecycle，补统一 abort/drain；
3. 生成 profile100 ordered replay transcript，比较 direct、串行 GateStack、TDR GateStack 周期；
4. 扩展 `TOKENS=162, LANES=32` 到随机多 term、多 route 长回归；
5. 冻结 bias/requant 合同并对 H67 dyadic 部署结果逐元素比较；
6. 有目标库后从综合网表重放活动、生成 SAIF，再运行 DC/LEC/PPA。

架构保留门槛不变：真实 ordered trace 吞吐提升、同约束 EDP 改善和全路径 bit-exact 必须同时成立。当前 RTL 已经证明机制可实现，但尚不能替代目标工艺 PPA 签核。

## 11. RTL 代码审阅结论

本轮在功能回归之外做了参数化与异常路径审阅，发现并修复两项真实问题：

### 11.1 replay mux 的伪参数化

`gatestack_replay_mux` 对外暴露 `SOURCES`，但 unpack 与 reset 循环原先硬编码为 3。主配置刚好是 3，因此不会在普通回归中暴露；若做 2-source 消融，会出现未覆盖数组项或 elaboration 风险。

修复后两个循环都以 `SOURCES` 为上界，并新增 `SOURCES=2` 空 session 双路回归。Icarus、Verilator assertion 和原 3-source 回归全部通过。

### 11.2 非法 head 元数据的资源占用风险

原多 head 控制器会接收 tag/index/last 不一致的 head，再依赖 sticky error 报告。若 tag 错误，TDR 产生的 update tag 与 accumulator tile tag 不同，可能被永久拒绝。

修复后，以下条件成为 head-start 准入条件：

```text
head_tag == tile_tag
head_index == heads_completed
head_last == (heads_completed == expected_heads - 1)
```

非法 head 不会启动 replay mux、Decoder 或 TDR backend，并立即反映为 `protocol_error`。合法路径的统一流、多路由、实际三 Decoder 和 T162 回归全部重跑通过。

仍未解决的是 session 已经启动后内部 term/event 损坏时的统一 abort/drain；这项继续保留为最终 full top blocker。

## 12. T162 RTL 活动文件

新增可重复入口：

```bash
sim_hitflow/run_gatestack_scale162_activity.sh
```

该入口先重跑实际三 Decoder 多 head 与 T162 严格回归，再以 `+dump_vcd` 导出 DUT 全层次 VCD，并生成：

- `results/gatestack_scale162_activity_20260715/activity.json`；
- `results/gatestack_scale162_activity_20260715/activity.md`。

当前 T162 定向用例结果：

| 指标 | 值 |
|---|---:|
| VCD 大小 | 0.42 MiB |
| 声明变量 | 1278 |
| 有更新变量 | 1278 |
| 已知值 bit toggle | 41577 |
| 主要活动层次 | routed projection/TDR/accumulator |

该统计证明切换活动文件非空、主要计算路径被激活、后续功耗输入可重复。它没有标准单元电容、时钟树、布线和 SRAM macro 信息，因此不能换算为 mW、uJ 或 EDP。当前环境也没有 `dc_shell`、PrimeTime、`vcd2saif` 或目标 `.lib/.db`。
