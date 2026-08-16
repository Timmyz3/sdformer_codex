# Local5 正式 Preflight 首次复审整改

> 后续状态：第二次独立复审接受 H×H topology P0，但指出正式 manifest/payload/cohort/
> projection 的正路径身份绑定和 runner 防伪断言仍不足。后续整改见
> `docs/303_Local5正式Preflight第二次复审整改_20260810.md`；本文件结果目录为历史版。

## 1. 评审裁决

首次独立评审对 HxH topology preflight 给出 `4/5 Weak Accept`，H×H P0 已接受，
formal G0 继续 DENY。评审留下两个 P1 与两个 P2：

1. `3.0==3` 使 JSON 拓扑整数类型不够 fail closed；
2. runner 只接受 manifest 缺失的 DENY 路径；
3. manifest group 允许冲突的 shadow `input_head` 字段；
4. task SHA 只输出、不与冻结值比较。

本轮全部整改，不修改 GPU producer 或任何 RTL。

## 2. 严格类型与字段集合

selection 的 sample/stage/block/window/heads/batch-windows 必须是非 bool Python
整数，probability/analysis-weight 必须是 JSON float。manifest group 的 19 个字段
集合完全冻结；因此增加 `input_head=99` 会直接因未知字段失败。

manifest 的 tag、head、heads、lanes、tokens、time-planes、plane-tokens、spatial-side、
flat-group 与 batch-windows 全部严格检查整数类型；empty 必须为 bool。module、
selection、plane execution 与 ordered-item SHA 也逐项冻结。

projection block 字段集合完全冻结；heads/head-dim 与 weight-shape 元素必须为整数，
因此 `[96.0,96.0]` 不再等价于 `[96,96]`。NPZ shape/dtype/array set 检查保持不变。

## 3. Canonical Task Digest

`enumerate_hxh_tasks()` 现在先要求 1200 个 window 严格保持 sample-major、
stage/block canonical 顺序，再枚举 210600 项。最终 digest 必须等于：

```text
5e894781aaca24b307fc0c33ddb116b28082694f484e3bb15784b8da7a6b07c6
```

把 window 输入整体反序，即使数量和唯一性不变，也会因 canonical order 失败。

## 4. Runner 双状态

runner 现在接受且只接受：

```text
DENY_FORMAL_MANIFEST_ABSENT
PREFLIGHT_PASS_NOT_G0
```

两种状态都强制 `admission_generated=false`、window=1200、head-group=13800、
task=210600。正式 manifest 到达后，正向 preflight 可以打包，但仍不能越级生成 G0。

## 5. 回归结果

结果目录：

```text
results/local5_erep_formal_preflight_v4_reviewfix_20260810
```

| 检查 | 结果 |
|---|---:|
| 单测 | 7/7 PASS |
| float heads/shape/lanes | REJECT |
| shadow input_head | REJECT |
| reversed task windows | REJECT |
| result SHA / complete receipt | PASS / PASS |
| 当前 formal status | DENY_FORMAL_MANIFEST_ABSENT |
| admission generated | false |

本轮仍只属于 `[契约审计]`。formal G0、底层 ledger 重放、T450/OUT_DIM32 miter、
EREP 候选 RTL 和 ASIC PPA 均未放行。
