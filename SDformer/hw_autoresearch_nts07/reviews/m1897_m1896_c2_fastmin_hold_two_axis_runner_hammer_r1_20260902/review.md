# M1897｜M1896 C2 双轴 fast-min hold-repair runner 独立打铁

**FAIL CLOSED，61/100，P0=0，P1=5，P2=0。** 本次仅做静态只读审阅；没有查询 license，没有创建 attempt，没有运行 DC、PrimeTime、Formality、VCS 或 PTPX，也没有修改 runner、Tcl、任何前序证据或 `docs/359`。因此不得创建 M1896 attempt，更不得引用新的 hold、setup、area 或 PPA 数字。

## 已核对成立的部分

- M1811 的 K8/K1x8 DDC 与 mapped SDC 四个 SHA 精确；M1811、M1830、M1893R2 的目录双封均通过。
- M1892 Tcl SHA 为 `b01b2266...9885f`；runner SHA 为 `d1ec9d22...e097`；`docs/359` 仍为 `dedde7ce...dfc4`。
- 两轴白名单固定为 `k8,k1x8`，各自 DDC/SDC、design 名称和基线面积正确。面积 ceiling 精确为各自 M1811 基线的 1.05 倍：`137363.9139348` 与 `614811.72022515 um2`。
- Tcl 只做一次 `set_fix_hold` 与一次 `compile -incremental_mapping -only_hold_time`；优化时 hold uncertainty 为 0.070 ns，随后恢复 0.050 ns；3.000 ns clock 与 0.200 ns setup uncertainty 未漂移。
- runner 固定两次 DC，要求两轴 setup/hold `status=MET` 且 `violating_paths=0`；结果口径仍明确为 logic-only raw result，Formality/PT/power/system speedup 均为 false。

这些正项不等于 launch authority。下面五项任一项都足以阻止 M1896。

## P1 阻塞项

1. **review/runner 自 pin，没有独立 release。** `M1896_EXPECTED_M1893R2_REVIEW_SHA256`、`M1896_EXPECTED_RUNNER_REVIEW_SHA256` 和 `M1896_EXPECTED_RUNNER_SHA256` 全由调用者传入；状态和 runner 绑定仅用 substring grep。替换 runner 与两个 review 目录、重新封存后，再传入相互匹配的 SHA，可以绕过当前权威门。M1893R2 明确要求独立封存的 release，M1896 完全没有消费它。
2. **license 发生在 attempt 之前。** 第 130 行先执行 `lmutil`，第 131 行才创建 attempt。license 失败时 namespace 仍 fresh，可反复查询，违反“一次 attempt 覆盖首次外部工具访问”。
3. **partial attempt 没有原子/终态保证。** LOCK/ATTEMPT/WORK 一次 `mkdir`，随后写 attempt、封 attempt，第 134 行才令 `WORK_ACTIVE=1`。这之前任一步失败，trap 不封 partial ATTEMPT/WORK，也不生成终态 quarantine。
4. **缺 M1893R2 要求的 DRC=0 门。** Tcl 会输出 `constraint_design_rules_posthold.rpt`，runner 却既不要求文件存在，也不解析 max-cap/max-transition/max-fanout/min-pulse/min-period 零违例。setup/hold MET 仍可带 DRC violation 发布 RAW_PASS。
5. **发布没有后验。** `mv -T -n` 在目标竞态出现时可成功返回但不移动；runner 随即关闭 trap、删除 lock、打印成功，从不证明 RESULT 存在、WORK 消失或 RESULT 双封仍有效。

## 修复门

必须新建 additive successor runner：先原子消费并封 attempt，再做 license；所有 namespace 写入后立即进入可封存 trap 状态；消费独立双封 release 并按 JSON 字段验证 schema/PASS/零严重度/different-author/所有 SHA；为 DRC 生成并检查 machine summary；发布后重验 canonical 双封与 WORK 不存在，之后才能清 trap。修后仍须另做不同作者 runner review，才能授权唯一一次两轴 DC。

本次唯一授权是继续写 successor；`M1896` 本身的 license、attempt、DC、PT、Formality 与 release 均为 0。
