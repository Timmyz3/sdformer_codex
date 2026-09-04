# Local5 Identity-Service 服务合同冻结

## 1. 裁决

OUT_DIM32 phase-template canary 暂不直接复用旧 numeric TB 的
transaction-indexed 延迟。原因是：

- 旧 TB 按第 `n` 次 token/weight/result 事务和 seed 生成延迟；
- `local5_erep_identity_service_v4` 明确禁止 transaction index 进入逻辑身份；
- 两者即使延迟分布相似，也不是同一服务合同。

本轮先冻结真实 H3 窗口的 identity-derived 服务表。前一版 reviewfix 因 verifier
与生成器共享 identity oracle，被独立评审判为 2/5；reviewfix2 获 4/5 Accept 后又
关闭两项 P2，当前最终有效包为 reviewfix3：

```text
identity service tables = PASS_IDENTITY_SERVICE_TABLES_NOT_G0
numeric RTL              = unchanged
formal G0                = DENY
```

## 2. 冻结身份

真实窗口坐标：

| 字段 | 值 |
|---|---:|
| sample | 2 |
| stage/block | 0/0 |
| window | 249 |
| head / output tile | 3 / 3 |
| token | 450 |
| lane / out dim | 32 / 32 |

`sample` 在 identity-service 中必须是非空字符串。冻结映射为：

`local5/profile100/<ordered_term_manifest SHA256>/sample/002`

本次完整字符串中的 manifest SHA 为：

`db92881db34b62cfd0bf62eccddbf4e860c670076814bb8483f7a7b219874f52`

这使 sample 名称同时绑定正式 profile100 来源和 numeric sample id。

## 3. 三类服务事务

### Relation

身份字段：

```text
sample, stage, block, window, input_head, source_id
```

output tile 不进入 relation identity，因为同一个 input head 的同一个 source 在三个
output tile 上复用同一逻辑 relation。H3 recompute runtime 共查询 4,050 次，对应
1,350 个唯一身份，每个身份 multiplicity 恰为 3。

### Weight

身份字段：

```text
sample, stage, block, window, input_head, output_tile, lane, out
```

H3 HxH 共 9,216 个事务，9,216 个唯一身份。

### Final

身份字段：

```text
sample, stage, block, window, output_tile, source_id, out
```

H3 共 43,200 个事务，43,200 个唯一身份。

## 4. 延迟与顺序

每个事务的 canonical JSON 与 schema、seed、kind 一起进入 SHA-256。取 digest 前
8 byte 后模 4，得到 `delay=0..3`；注册服务响应延迟为 `delay+1`，即 1..4 cycle。

本轮 seed 固定为 `20260810`。运行顺序冻结为：

- relation：`output_tile,input_head,source_id`；
- weight：`output_tile,input_head,lane,out`；
- final：`output_tile,source_id,out`。

顺序只用于审计 ordered ledger 和索引服务表，**不进入逻辑身份的哈希字段**。
flat index 冻结为：

```text
relation = input_head * 450 + source_id
weight   = (((output_tile * heads + input_head) * 32 + lane) * 32) + out
final    = ((output_tile * 450 + source_id) * 32) + out
```

relation 在不同 output tile 间复用同一个 flat index，不能按 4,050 次 runtime
lookup 线性读取 1,350 项 memh。

握手边界冻结为：relation/weight 的 request edge 是
`posedge(valid && ready)`，response available 必须恰好位于 `delay+1` 个周期后；
response accept 可因下游反压晚于 available，但不得更早。final 由 consumer 建模，
每个有序结果第一次 `tile_result_valid` 为 request，valid 保持期间在 `delay+1`
周期后接受。所有 response 的 valid 与 payload 从 available 到 accept 必须保持稳定；
每个 relation/weight stream 至多一个 outstanding。

| stream | transaction | identity | multiplicity |
|---|---:|---:|---|
| relation unique | 1,350 | 1,350 | 1 |
| relation runtime | 4,050 | 1,350 | 3 |
| weight runtime | 9,216 | 9,216 | 1 |
| final runtime | 43,200 | 43,200 | 1 |

manifest 同时保存每个 stream 的 ordered ledger digest、unordered multiset digest、
identity multiplicity SHA 和 multiplicity histogram。

## 5. 产物

结果目录：

`results/local5_identity_service_tables_sample2_h3_v4_reviewfix3_20260811`

主要产物：

- `identity_service_tables.npz`：delay 与每个 identity 的 32-byte transaction digest；
- `relation_delay.memh`：1,350 项；
- `weight_delay.memh`：9,216 项；
- `final_delay.memh`：43,200 项；
- `manifest.json`：身份、顺序、数量、ledger digest、artifact SHA。
- `producer_complete.json`：只证明生成器原子发布完成，不能代替独立验证；
- `verification_receipt.json`：包内独立 verifier 成功后才生成；
- `task_plan.json` 与 `source/`：自包含任务合同和 generator/oracle/verifier 源码。

代码与测试：

- `scripts/generate_local5_identity_service_tables_v4.py`
- `scripts/verify_local5_identity_service_tables_v4.py`
- `tests/test_generate_local5_identity_service_tables_v4.py`
- reviewfix2 的 13 项回归中 12 项首轮通过，唯一失败是错误消息 regex；修正后单独
  复跑通过。reviewfix3 新增执行 verifier SHA 与包内角色绑定、精确文件集检查，
  最终 14/14 一次通过。覆盖 later-tile 外来 group、非默认 seed、重复输出目录、布尔字段、
  大写 SHA、memh、artifact map、runtime 合同、source role、NPZ dtype 和 receipt 篡改。

独立 verifier 不导入 generator 或 `local5_erep_identity_service_v4`，自行实现 canonical
JSON、length-prefix framing、transaction digest、delay、ordered/multiset ledger 和
multiplicity 重算。它从 `/tmp` 读取包内源码验证通过：

```text
manifest SHA             d9bb3287eff11e925a18cca3bd95f5af76cbdc96904bc08054120dc037c6a2a2
verification receipt SHA 8edb5bc60902f7a9ba52eda41f809364f6e3d9f6beb0bfd0b49076b6e1eae7f7
```

## 6. 证据边界

- `[软件确定性服务合同]`：identity、delay、digest、multiplicity 和 memh 来自冻结
  identity-service v4；
- 服务表是验证输入，不是 DUT 内部状态，也不是新的架构机制；
- v8 密封 numeric TB 已成功消费该表，并对 available/accept payload、完整响应身份和
  内部状态序列做离线精确验证，可标为 `[rtl]` 服务合同证据，但不能扩大为 formal
  G0 或架构性能；
- 当前 H3 release v5 的 691,588 cycle 仍属于旧 transaction-indexed 服务条件；
- release v6 identity 原型已密封，但首个 H3 canary 因 TB 把 response available 与
  response accept 错当同一周期而失败；v6 与其失败 trace 仅作负结果，不计 PASS；
- v7 的“握手逐条验证 + 状态采集”已被 v8 的 trace-v2 取代；v8 进一步冻结
  717,849 条状态事件的精确计数和有序摘要，并通过四类定向篡改反例；
- formal G0 保持 `DENY`。

## 7. 下一步

1. 独立复审 v8 密封 H3 trace-v2 canary；
2. 复审通过后建立 expanded reference 与 template/patch 两条独立路径；
3. phase-template 通过后再决定是否扩 H6/H12/H24 identity-service 运行。
