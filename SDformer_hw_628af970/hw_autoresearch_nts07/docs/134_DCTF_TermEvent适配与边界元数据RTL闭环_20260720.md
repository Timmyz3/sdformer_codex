# DCTF Term/Event 适配与边界元数据 RTL 闭环

## 1. 关闭的语义缺口

原 DCTF command 只有一个 destination token，无法表达真实 decoder 的“一条 term 对应多个 event beat”。本轮完成两项修订：

1. 新增完整 term 验证与 destination 串化 adapter；
2. fabric entry 增加 `term_issue_seq/term_first/term_last/head_last`，并逐 bank 保序携带。

现在 command 合同为：

~~~text
{group_tag, cmd_sequence, term_issue_seq,
 input_channel, gate_code, lane_id, destination_token,
 term_first, term_last, head_last}
~~~

`cmd_sequence` 保证每个 bank 的 destination command 顺序；`term_issue_seq` 和 first/last 使 bank-local backend 知道何时请求一次权重、复用 product 和释放 term。

## 2. 为什么采用完整 term 验证后发射

如果 adapter 边接收 event 边向 bank 发 command，后续 event 才发现重复 token 或计数错误时，前面的 partial command 可能已经更新 Acc。为了让错误 term 在进入 bank 前被完整阻断，本轮使用保守两阶段：

~~~text
COLLECT + exact validate -> token buffer -> EMIT commands
~~~

默认 `TOKENS=162, TOKEN_ID_W=8` 时缓存：

- token buffer：`162 x 8 = 1296 bit`；
- seen bitmap：`162 bit`；
- 合计 `1458 bit`，另加 term 元数据和控制寄存器。

合法最后一个 event 握手后的下一周期才可能输出首 command；term 收集和发射不重叠。这个实现优先保证 exact，后续必须通过真实 trace 判断周期代价是否可接受。

## 3. Adapter exact 检查

adapter 检查：

- destination count 非零且不超过 TOKENS；
- event gate/lane/issue sequence 与 term 一致；
- first/last/head-last 边界一致；
- event count 等于 token-valid popcount；
- token 范围合法；
- term 内无重复 token；
- 实际 destination 数等于 descriptor count。

错误发生在非 last event 时进入 drain，只接收并丢弃到 `event_term_last`；flush/reset 可从缺失 last 的上游错误恢复。`protocol_error` 保持 sticky，完整系统应由 context abort reset 清除。

## 4. 验证结果

### 4.1 Adapter

| 指标 | 结果 |
|---|---:|
| 周期 | 83 |
| 合法command | 9 |
| 错误类型 | 8 |
| mismatch | 0 |
| multi-beat/single | 命中 |
| duplicate/metadata/count/range | 命中 |
| drain/flush/backpressure | 命中 |
| Icarus | PASS |
| Verilator + SVA | PASS |
| Yosys default + EVENT_WAYS=2 | PASS |
| Erie | 0 error / 0 warning |

审阅时发现并修复 `EVENT_WAYS` 循环被硬编码为4的问题，增加 `EVENT_WAYS=2` 的综合检查。

### 4.2 Fabric

| Q | 周期 | accepted | retired | input stall | max occupancy |
|---:|---:|---:|---:|---:|---:|
| 2 | 402 | 260 | 256 | 97 | 2 |
| 3 | 391 | 260 | 254 | 86 | 3 |
| 4 | 387 | 260 | 252 | 81 | 4 |

测试包含两次随机 flush，retired 小于 accepted 是 flush 丢弃 in-flight command 的定义结果，不是数据丢失。Q2/Q3/Q4 均通过 Icarus；Q4 通过 Verilator动态SVA；Yosys和Erie通过。

## 5. 开放逻辑面积代理

| 模块 | logic area | cells | `$mem_v2` |
|---|---:|---:|---:|
| 完整term adapter | 5731.236 | 3886 | 1 |
| Q2 command fabric | 4475.184 | 2722 | 0 |

两者逻辑面积合计 `10206.420`。Adapter 的 token buffer 保留为一个未映射 memory，其面积未计。因此该数字只能用于结构筛选，不能作为 DC 面积或总开销。

相对 Central96 与 3xIndependent32 的开放逻辑面积差 `66744.188`，前端逻辑代理约占该差值的 `15.3%`。这说明 DCTF 控制前端在逻辑层面仍有容纳空间，但 memory、bank backend、宽网和时序未计，不能据此宣称面积净收益。

## 6. 仍未关闭的边界

- fabric retire 仍只表示三 bank dispatch 完成，不表示计算完成；
- 尚未接入32-lane weight/product/Acc bank backend；
- adapter收集和发射不重叠，真实H67周期代价未知；
- 缺 bank response skew、term completion、head completion和abort联动；
- 缺真实 S0-S3、SRAM macro、STA、SAIF和LEC。

下一步是实现单 bank executor：`term_first` 请求一次权重，product 在本地保持并服务该 term 全部 destination，`term_last` 的 Acc update 接受后才发 term-complete。三个 executor 与 Q2 fabric、三个本地 AccTile 接通后，DCTF 才从控制叶模块晋级为数据流原型。

## 7. 复现

~~~bash
bash sim_hitflow/run_gatestack_dctf_term_event_adapter_checks.sh
bash sim_hitflow/run_gatestack_dctf_term_fabric_checks.sh
bash dc_handoff/scripts/run_gatestack_dctf_frontend_nangate45_mapping.sh
~~~

结果位于 `results/gatestack_dctf_frontend_20260720/`。
