# Local5 正式 Preflight 第二次复审整改

> 后续状态：第三次独立评审给出 `3/5 Reject`，发现 manifest-present 时单测仍写死
> “manifest 必须缺失”，导致 runner 正路径不可达。本文件中的“正路径打包已闭合”因此为
> 历史结论；修复与双分支集成证据见
> `docs/304_Local5正式Preflight正路径Runner整改_20260810.md`。

## 1. 本轮边界

第二次独立 DATE 复审维持 `4/5 Weak Accept`，接受 1200 window、13800 input-head
group 和 210600 H×H projection task 的拓扑 P0，但留下两个 P1：

1. 正路径报告没有绑定正式 manifest、ordered payload、cohort 和 projection 文件身份；
2. runner 对正路径只检查几个标量，伪造的最小 `PREFLIGHT_PASS_NOT_G0` 报告也可能通过。

此外有两个可修 P2：顶层 JSON 字段集合与 seed 类型未完全冻结，helper 入口仍可把
浮点计数经 `int()` 静默转为整数。Git 是否纳入版本控制不属于本轮代码语义整改。

本轮只关闭上述正式输入契约和打包防伪问题，不修改运行中的 GPU producer、不生成
formal G0 admission，也不实现 EREP 候选 RTL。

## 2. 正式产物身份链

正路径现在必须同时满足以下绑定：

| 产物 | 绑定内容 |
|---|---|
| formal manifest | 文件 SHA-256 与报告顶层 SHA 一致 |
| ordered payload | manifest 中 basename、SHA 与 profile 内真实文件一致 |
| cohort | manifest 中 basename、文件 SHA 与 profile 内真实文件一致 |
| projection JSON | 必须就是冻结的 12-block contract，文件 SHA 一致 |
| projection NPZ | manifest、projection JSON 和真实 NPZ 三方 SHA 一致 |

所有被 manifest 引用的文件都必须是 profile 目录内 basename；绝对路径、`../` 与目录
逃逸均 fail closed。报告使用仓库相对路径，避免把机器特定绝对路径当成可复现身份。

正式 manifest 还必须与冻结 selection/cohort 对齐：sampling method、整数 seed、
selection-plan SHA 和 cohort SHA 均一致。13800 个 group 额外输出：

- producer 原始 group key 顺序 SHA；
- `tag + key + ordered_item_sha256` 顺序身份 SHA；
- canonical sorted key SHA；
- exact key coverage 标记。

这三类 digest 分别区分“集合正确”“producer 顺序正确”和“每项 payload 身份正确”。

## 3. Runner 防伪

runner 不再自行维护一套弱 shell 判断。它读取生成的 JSON 后重新调用
`validate_report_for_packaging()`；该函数从冻结输入重新执行完整 preflight，要求报告
与独立重放结果逐字段完全相同。

如果正式 manifest 已到达，runner 还会再次读取报告中的五项 artifact binding，逐文件
复算 SHA，并将这些正式输入加入 `source_input_sha256.txt`。因此伪造状态、删去字段、
替换 payload 或在 preflight 与打包之间改文件都会导致打包失败。

## 4. 严格类型与字段冻结

新增 fail-closed 检查包括：

- selection、formal manifest、projection contract 顶层字段集合完全冻结；
- selection seed 必须是 JSON 整数，`20260809.0` 不再被接受；
- H×H helper 的 sample/stage/block/window、input-head-count 和 output-tile-count 均使用
  严格非 bool 整数检查；
- 直接向 helper 传入 `0.0` 或 `3.0` 也会失败，不能绕过上游 validator。

## 5. 验证结果

结果目录：

```text
results/local5_erep_formal_preflight_v4_bindingfix_20260810
```

| 检查 | 结果 |
|---|---:|
| Python 单测 | 10/10 PASS |
| py_compile | PASS |
| float seed/helper topology | REJECT |
| 顶层 shadow 字段 | REJECT |
| profile 路径逃逸 | REJECT |
| 伪造最小正报告 | REJECT |
| 独立报告重放 | PASS |
| result SHA / receipt SHA | PASS / PASS |
| 当前正式 manifest | 缺失 |
| 当前 formal 状态 | `DENY_FORMAL_MANIFEST_ABSENT` |
| admission generated | false |

固定任务规模和 digest 未改变：

```text
window = 1200
input-head group = 13800
H×H task = 210600
task SHA-256 = 5e894781aaca24b307fc0c33ddb116b28082694f484e3bb15784b8da7a6b07c6
```

## 6. 证据边界与下一步

本轮证据仍是 `[契约审计]`。它证明正式输入一旦生成，preflight 的正路径不能脱离真实
artifact 身份，也不能被最小假报告绕过；它没有证明正式 13800 group 已生成，更没有
证明 H×H task 已做 RTL replay。

formal G0、底层 event/resource ledger 重放、T450/OUT_DIM32 Acc32 miter、EREP
候选 RTL 和 ASIC PPA 继续为 `[待验证]`。在第三次独立复审接受本轮 P1/P2 后，下一轮
最高优先级才转向 anti-self-report：由 head-phase/window-schedule/command ledger 重算
C0--C4，而不是比较同源聚合标量。
