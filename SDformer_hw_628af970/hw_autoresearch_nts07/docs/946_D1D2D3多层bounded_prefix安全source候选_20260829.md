# M946｜D1/D2/D3 多层 bounded-prefix 安全 source 候选（2026-08-29）

## 结论

M946 只新增一个按 `layer + sample + config + timestep` 选择冻结 M686 row 的外层选择器，以及 `1K / 10K / 100K` expanded-request 前缀预检。它没有改 M785 mapper、resource、transaction/address order，也没有改 M896 RUN-GTLS recurrence 和六类 cycle priority。

当前身份是 `DRAFT_SOURCE_ONLY__INDEPENDENT_FRESH_HAMMER_REQUIRED`。它不是执行 release，不允许 full-row、production、结果目录、EDA、GPU 或 remote，也不得生成论文 Table-A 行。

解释器也是冻结输入：只接受 `/opt/anaconda3/envs/pytorch310/bin/python3.10`，版本 3.10.18，SHA-256 `9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115`。本机默认 `/usr/bin/python3` 为 3.6.8，明确不支持；source/checker 在加载 M896 前即 fail closed。

## 数值边界

| Layer | M946 route | 合法解释 |
|---|---|---|
| D1 | `COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE` | 输入虽为精确 `{0, theta}`，但现有 Acc24/INT8 路径没有通过冻结 FP32 bit-exact bridge；只允许按公共资源全形状收费并单列诊断 |
| D2 | `EXACT_BINARY_SUPPORT` | 冻结 binary-support payload，可做 bounded exact scheduler miter |
| D3 | `EXACT_BINARY_SUPPORT` | 冻结 binary-support payload，可做 bounded exact scheduler miter |

因此有效论文措辞仍是“D0/D2/D3 exact-binary support subset，D1 separately charged diagnostic”。任何“four exact accelerated layers”“decoder complete”“full decoder/system speedup”均被 source 和合同显式拒绝。

## 第一性原理设计

选择器不复制交易生成逻辑。它直接调用冻结 `M785.iter_record_transactions`，再用冻结 `M890.truncate_transactions` 截断 expanded-request 前缀。每个前缀同时由 M896 RUN-GTLS 与 M890 GTLS 调度；`<=10K` 时再下探 M768/M861 reference。逐字段核对：

- total cycle 与六类 cycle 分解；
- expanded request 与 compressed transaction 数；
- 每个 scheduled request、compressed schedule；
- transaction address hash 与 commit hash；
- terminal readiness、port calendar 和同拍 response-slot reuse。

这证明前缀选择没有偷换冻结执行语义。前缀 cycle 仍只是 diagnostic，不是 full-row latency。

## Scalability gate

100K 才是 authoritative preflight；1K/10K 只用于快速 exactness 与趋势检查。projection 同时报：

1. measured elapsed 与 process max RSS；
2. scheduler state 按目标 request proxy 线性外推；
3. `2 × projected peak` 是否同时小于 `MemAvailable` 和 commit headroom；
4. `2 × projected elapsed` 是否小于 6 小时 cap。

D1 的 16,688,570 requests 是闭式精确计数；D2 的 151,879,626 和 D3 的 504,012,937 只是 M942 sizing proxy，不是 cycle 预测。即使两道 gate 均 PASS，`full_row_authorized` 仍固定为 false，必须重新走独立 hammer 与 release。

## Author 轻量验证

本里程碑只允许运行 Python compile、synthetic 1K 和 real D1/D2/D3 sample0/A1/t0 的 1K exact miter；额外允许至多一个 D1 100K scalability preflight。禁止触发 D2/D3 full contributor/full row。

Author preflight 实测 6/6 unittest 通过；D1/D2/D3 1K 均通过 M768/M861/M890/M896 exact miter。D1 100K 得到 100,000 requests、3 transactions、147,023 diagnostic cycles，elapsed 7.555 s、max RSS 779,408 KiB；按 D1 精确 16,688,570-request 目标外推，2× timeout 为 2,522 s，2× projected memory 为 1,613,402,396 B，两门均通过。该结果仍明确 `full_row_authorized=false`。1K/100K 均尚未到首个 commit，故 empty commit-sequence SHA 是预期的 prefix 边界，不是 commit coverage。

冻结 source candidate：

- `system_simulator/scripts/analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py`
- `system_simulator/scripts/check_m946_decoder_multilayer_bounded_prefix_source_candidate.py`
- `system_simulator/tests/test_m946_decoder_multilayer_bounded_prefix_source_candidate.py`
- `contracts/m946_decoder_multilayer_bounded_prefix_source_contract_DRAFT_r0_20260829.json`

## 后续门

独立 subagent 必须先做 fresh source hammer，核对递归身份、fail-closed CLI、三层 selector 与 exact-miter 覆盖，再决定是否生成执行合同。M946 本身不修改 M925 attempt/result，也不修改 `docs/359_DATE终局冻结_20260813.md`。
