# M1100 M1095 C1 preflight failure 独立审计

结论：**PASS failure audit，100/100；M1095 DO_NOT_RETRY。** 失败来自 M1086 的 work-domain gate 过严，不是 canonical trace 错误，也不是 present short-work 的 1RW/RAW 几何错误。允许另立 additive successor；当前没有周期、speedup 或 paper result。

## 失败边界

- attempt seal 有效：`CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS`，`maximum_attempts=1`、`automatic_retry=false`。
- quarantine seal 有效：`FAILED_OR_INTERRUPTED__NO_RETRY`，阶段 `PREFLIGHT_THEN_FULL_REPLAY`。
- traceback 终点：`RuntimeError: unsupported positive work interval 1..14`。
- raw result 未发布；M1095 不得重试。

## 首个短正工作

| 字段 | 值 |
|---|---:|
| task/sample/operator/chunk/partition | `2313 / 0 / 0 / 5 / 153` |
| row / row_count / file_offset | `9 / 64 / 4,133,880` |
| design / work | `candidate / 8` |
| shared preprocess | `146` |
| raw-row SHA256 | `a88f07c79d3f5185159ccc519f0d5ccd042d04025e34a330ea95253c0e80aecd` |
| provenance SHA256 | `9f3c2acd72681cef0f7d5191d785969c064f99437a0e040d44f27df054a90a17` |

同一 task 的三个 design 都是 work=8。

## 完整 812,160×3 扫描

`1..14` 中实际只出现 **8**：

| design | work=8 次数 | work 1..7 / 9..14 | 最小正 work | 最大 work |
|---|---:|---:|---:|---:|
| candidate | 4,174 | 0 | 8 | 3,680 |
| strongest_zero | 4,174 | 0 | 8 | 7,360 |
| same_coordinate_bit | 4,174 | 0 | 8 | 7,360 |

总计 12,522 个 work=8；其余短值全部为零。三个 design 的全域最小值均为 0、最小正值均为 8。扫描只调用 frozen `CanonicalRowReader.derive`，没有调用 production preflight、full iterator 或 full cycle replay。

## Frozen M1056 bounded 几何

对全部 12,522 个 work=8 occurrence 分别执行：

1. fresh `last_write`；
2. 对同一 packed address 注入延迟 predecessor 的 RAW 压力。

两档均为 12,522/12,522 PASS；`raw_dependencies_pass` 12,522/12,522，负 dependency 为 0，最小 dependency delay 为 0。原因是 work=8 时 `span=1`，八个 bank 的 write/read delay 都是 0，不会重现 zero-work 的 write-before-read 问题。

这也是源码内部一致性问题：M1086 的 `source_small_oracle` 已把 work=8 当作合法 positive delegation 检查，但 `validate_production_work` 又把完整 `1..14` 禁掉。

## 最小 additive repair 门

1. M1086/M1094/M1095、attempt 与 quarantine 全部冻结；新建 source/contract/release/result/attempt/lock namespace。
2. domain 精确改为 `work == 0 or work >= 8`；继续拒绝 1..7、bool、nonfinite 和负值。
3. work=0 语义完全不变；所有 work≥8 直接委托 frozen M1056。
4. 新 source 必须再次穷举 2,436,480 个 work，绑定 counts/digest/provenance，并对 12,522 个 work=8 做 fresh+delayed-RAW 几何回归。
5. 不同作者 source hammer 通过后，才可授权唯一一次新 CPU attempt；不得重用 M1095。

本审计未修改既有证据或 `docs/359`；后者 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
