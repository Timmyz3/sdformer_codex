# M1672｜M1671 ep34 Decoder D0 recoverable-shard source 独立审阅

## 裁决

**FAIL-CLOSED 84/100**：`FAIL_M1672_M1671_DECODER_FULL_D0_SOURCE__NO_M1673_EXECUTION_RELEASE__SUCCESSOR_EXECUTION_CLOSURE_REQUIRED`。P0=0、P1=3、P2=1。

M1671 的静态计算模型成立，但 execution closure 不成立，因此禁止直接生成 M1673 execution release。允许的下一步仅是一个保留现有网格、scheduler 和 miter 的最小 execution-closure successor，再由不同作者审阅。

## 已验证成立的部分

- D0 call ordinal 为 `0,4,...,116`，共 30 call；每 call 10 timestep。
- 每 timestep 1,200 destination、4 output block。
- 每 timestep 为 28 个 42-destination shard 加最后一个 24-destination shard，即 29 shard；总数 `30×10×29=8,700`。
- 8,700 个坐标无重复、无 gap、无 overlap；首 shard 是 call0/t0/d0..41，末 shard 是 call116/t9/d1176..1199。
- 三配置顺序固定为 dense typed K8、bit equal-service K1×8、bit typed K8，禁止 product-capture 配置。
- `ShardSession.accept` 保留 reference↔compact 的逐 request cycle/port/outstanding miter；`finish_destination` 保留累计 request/kind/byte/address/commit/cache/port 状态 miter。
- RSS 上限仍为 2 GiB absolute、512 MiB increment，模型在每个 destination 采样。
- 纯 reducer 只在恰好 8,700 个顺序 shard 上返回整数 numerator/denominator，采用 ratio-of-sums；输出明确是 shard-isolated D0，不是 monolithic call、full decoder 或 system。

双 Python 对上述网格、合成三配置 session 和完整 8,700-row reducer 重算一致；9/9 个 incomplete/order/config/resource/miter/commit/cycle 负突变被拒绝。整个评审安装了 bitpack open guard，canonical payload open attempt 为 0；没有运行 replay、GPU、EDA 或生产 reducer。

## 阻断问题

### P1-1：execution closure 缺失

与已成功执行的 M1656 不同，M1671 没有：

- payload-to-shard 私有执行 target；
- `RESULT/ATTEMPT/WORK/FAILURE` 精确命名空间；
- attempt consumer；
- immutable payload FD/hash 入口；
- 每 shard 原子 seal/publish、失败隔离和 resume verifier。

因此 `attempt_before_payload=true` 与 `automatic_retry=false` 目前只是合同/describe 字段，不是可执行顺序。M1673 若直接授权该 identity，将没有可安全调用的执行目标。

### P1-2：reducer 接受未封且不完整的 metrics

`reduce_complete_shards` 接受普通内存 dict，没有验证 shard manifest/outer seal。`validate_three_configuration_metrics` 也未要求非负 `request_count/byte_counts`，未要求 address digest、destination-state chain 或 `paper_result=false`。

独立攻击把三配置的 request count 设为 −7、commit bytes 设为 −99，并移除 address/destination-chain 字段，现有 validator 仍接受。于是伪造的 8,700-row 人口可以通过坐标/order gate 并污染 totals。

### P1-3：M1666 predecessor 并非实际递归闭合

M1666 已列出的 flat members、manifest 和 outer seal 均重算通过，但目录含未封 `__pycache__` 文件。M1671 的 `verify_flat_tree` 不比较实际递归人口，因此其“exact recursively sealed M1666”声明不可执行。

## 最小修复，不重做算法

1. 直接复用 M1671 的网格、`ShardSession` 和 ratio-of-sums 数学。
2. 增加 exact private payload-to-shard runner 与 immutable FD/hash。
3. 固定 fresh result/attempt/work/failure namespaces，并在第一次 payload open 之前原子消费 attempt；失败后禁止 retry。
4. 每 shard 原子双封；resume/reducer 必须核全量递归人口与 seal。
5. shard schema 必须验证非负 request/byte、address/commit/destination-state digest 和三配置共同 resource identity。

修复后仍须保持严格口径：完整结果只是 full-D0 population under shard-isolated cycle model；D1 排除，D2/D3 单独 rebind；不得写成 monolithic、full decoder、Table-A 或 system speedup。
