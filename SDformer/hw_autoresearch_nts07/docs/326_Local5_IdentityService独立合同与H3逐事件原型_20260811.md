# Local5 Identity-Service 独立合同与 H3 逐事件原型

## 1. 本轮裁决

本轮只闭合 Local5 verification-only 服务合同，不修改数值 DUT RTL，不宣称 formal
G0、ASIC PPA 或架构性能。

```text
reviewfix3 软件包        PASS [软件确定性服务合同]+[独立软件重算]
sealed v7 H3             PASS [rtl]+[软件整数金参考]+[rtl-build-provenance]
sealed v8 H3 trace-v2    PASS [rtl]+[软件整数金参考]+[rtl-build-provenance]
sealed release v6        FAIL，保留为负结果
formal G0                DENY
```

## 2. 为什么 v6 失败

`results/local5_erep_numeric_rtl_release_v6_identity_20260811` 完成了四种 head 规模的
密封构建，但首次 H3 canary 在第 3,634 个验证周期失败。失败原因是 TB 把
`response_valid` 首次出现和 `response_valid && response_ready` 接受错误地要求为同一
周期；真实 DUT 可以在 response available 后因下游反压延迟接受。

该失败不表示数值 DUT 错误，但说明 v6 的验证合同错误。v6 不得改写、不得计入
PASS，也不得作为后续论文证据。

## 3. 软件服务合同修复

最终有效包：

`results/local5_identity_service_tables_sample2_h3_v4_reviewfix3_20260811`

相对被独立评审判为 2/5 的前一版，reviewfix2/reviewfix3 完成：

1. verifier 不导入 generator 或项目 identity oracle，自行重算 canonical JSON、
   length-prefix framing、transaction digest、delay、ordered/multiset ledger 和
   multiplicity；
2. task plan 与 generator/oracle/verifier 源码复制进包，删除关键绝对路径依赖；
3. producer receipt 与 independent verification receipt 分离；
4. artifact 集、NPZ dtype/shape、source role、runtime count、flat index、握手合同和
   boundary 均为精确匹配；
5. seed 冻结为 `20260810`，非默认 seed fail-closed；
6. 增加 memh、artifact map、runtime 元数据、source role、NPZ dtype 和 receipt
   篡改测试。

冻结 SHA：

| 项 | SHA-256 |
|---|---|
| manifest | `d9bb3287eff11e925a18cca3bd95f5af76cbdc96904bc08054120dc037c6a2a2` |
| independent receipt | `8edb5bc60902f7a9ba52eda41f809364f6e3d9f6beb0bfd0b49076b6e1eae7f7` |

reviewfix2 的 13 项回归中 12 项首轮通过，唯一失败是错误消息 regex；修正后单独
复跑通过。reviewfix3 增加执行 verifier SHA 和精确文件集检查，14/14 一次通过。

## 4. 索引与握手合同

flat index：

```text
relation = input_head * 450 + source_id
weight   = (((output_tile * heads + input_head) * 32 + lane) * 32) + out
final    = ((output_tile * 450 + source_id) * 32) + out
```

relation 的 1,350 项表在三个 output tile 间复用，运行时查询为 4,050 次。

relation/weight：

- request accept：`posedge(request_valid && request_ready)`；
- response available：request accept 后恰好 `delay+1` cycle 的首次 `response_valid`；
- response accept：`posedge(response_valid && response_ready)`，允许因反压晚于
  available，不允许更早；
- 每条 stream 至多一个 outstanding。

final consumer：

- 每个有序结果首次 `tile_result_valid` 是 request；
- valid 必须保持到接受；
- 接受恰好位于 request 后 `delay+1` cycle。

## 5. 密封 v7 H3 逐事件验证

结果目录：

`results/local5_identity_service_h3_canary_v7_sealed_20260811`

release：`results/local5_erep_numeric_rtl_release_v7_identity_20260811`

release manifest SHA 为
`0f0a9fb2a98cb35365cad224bb6c31e1a196323be4a6c305cdef14292559addc`。
H3/H6/H12/H24 executable 均密封，本轮只运行 H3；运行前后均从 `/tmp` 验证
source tree、bundle、工具绑定、compile argv 和 executable SHA。

真实坐标为 sample2/stage0/block0/window249，输入、INT8 checkpoint 权重和软件
expected Acc32 复用 v5 正式 numeric canary 的冻结向量。DUT 数值 RTL 未改，只有 TB
服务模型和 trace monitor 改动。

| 指标 | 结果 |
|---|---:|
| relation request/available/accept | 4,050 / 4,050 / 4,050 |
| weight request/available/accept | 9,216 / 9,216 / 9,216 |
| final request/accept | 43,200 / 43,200 |
| group/tile/head start-done | 1 / 3 / 9，顺序一致 |
| 原始事件 | 844,075 |
| Acc32 | 43,200 |
| mismatch / max abs error | 0 / 0 |
| 验证周期 | 681,519 |

关键 SHA：

| 项 | SHA-256 |
|---|---|
| identity trace | `263308420280aed170e8fdd3694610013386df113b594927d8d28a8900c1d895` |
| actual Acc32 | `ec33e970a35a77b10c7a280491d4336ea6eab59c3b5a687b69aaafcd66a8507e` |
| trace verification | `58731cc0a422c15fd96101a0347a885616a7eee863ad6f3e50ee448724dc8c3f` |
| Verilator log | `eea717887cf6859f58b5fdcdeae43a72872206197cb7b7fa9c5717096402687f` |

`scripts/verify_local5_identity_service_rtl_trace_v1.py` 逐行检查 logical identity、flat
index、delay、available/accept 周期、单 outstanding、边界顺序和 Acc32。trace 中显式
绑定 manifest 与 independent receipt SHA。

## 6. 证据边界

- 681,519 cycle 是带确定性服务延迟和大规模文本 trace 的验证环境延迟，不是架构
  throughput；
- v7 executable 已密封，可写 `[rtl-build-provenance]`，但只覆盖 H3 单窗；
- 只覆盖 H3 单窗，不代表 H6/H12/H24 或 100-sample formal；
- phase template、tile patch 和 462,600 条 formal phase ledger 尚未证明；
- formal G0 保持 `DENY`。

## 7. v8 trace-v2 加固

v7 已证明单窗数值和逐请求时序，但它的离线 verifier 未比较 available/accept payload，
对 response-accept 的 metadata 检查不完整，且内部状态事件只要求非零。v8 不改数值
DUT，专门关闭这些证据缺口：

| 项 | v8 结果 |
|---|---:|
| relation payload 配对 | 4,050 / 4,050 一致 |
| weight payload 配对 | 9,216 / 9,216 一致 |
| final payload 配对 | 43,200 / 43,200 一致 |
| `tx_state` | 129,634，计数与有序摘要一致 |
| `acc_state` | 172,801，计数与有序摘要一致 |
| `head_state` | 415,414，计数与有序摘要一致 |
| 状态事件合计 | 717,849，联合有序摘要一致 |
| Acc32 | 43,200，零失配 |

状态参考由已接受的 v7 trace 冻结，绑定 v7 trace、release manifest 和 complete SHA；
v8 release 将状态参考和 trace-v2 verifier 纳入只读 source closure。v8 complete 又直接
绑定 release manifest、H3 executable、compile argv、verifier、状态参考、输入、权重、
vector manifest、软件 expected NPZ、identity task plan/manifest/receipt 和全部运行产物，
共 22 项，当前无失配。

定向负测试结果：修改 relation payload、修改 weight response metadata、删除一条
`tx_state`、修改一条 `head_state` 均被密封 verifier 拒绝。详细证据见
`docs/327_Local5_H3_trace_v2证据闭环_20260811.md`。

## 8. 下一步闸门

1. 对密封 v8 trace-v2 做独立 DATE 复审；
2. 只有复审通过，才开始 OUT_DIM32 phase-template + tile-patch canary；
3. formal G0 继续保持 `DENY`。
