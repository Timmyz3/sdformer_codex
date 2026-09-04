# M528 r2 已消耗失败链独立 fail-closed 锤审

## 裁决

**84/100，P0=1、P1=2、P2=1。M528 r2 已永久消耗，不得重跑、修补原件、删除 attempt、作为数字证据或作为新准入权限。** 本次只读锤审没有运行 production/EDA/GPU，没有修改 r2 或 docs/359。

允许 author 起草一条最小 r3 恢复链，但在“新源码静态锤审 → 独立 schema smoke 及锤审 → 新的双封 admission”三道门都通过前，不授权任何新 production。

## 失败事实与封条

- attempt sentinel 记录 CONSUMED_AT_FIRST_CPU_PRODUCTION_LAUNCH，时间为 2026-08-27T18:12:45+08:00；内封、外封和 identity 全部通过。
- quarantine 记录 FAILED_OR_INCOMPLETE_DO_NOT_CITE / exit code 1；内封、外封全部通过。
- 三次资源快照都满足 r2 门：commit headroom 为 56,773,692–57,006,352 KiB，MemAvailable 约 412.9–413.1M KiB，SwapFree 57,275,900 KiB，OOM 三项为 0。这不是资源失败。
- Python 在建立 process pool 前于 analyzer 第 435 行抛出 KeyError: 'area_um2'；production stdout 为空，没有 cycle/traffic/capacity 结果。
- canonical results/m528_h67_single_port_same_ledger_recompute_r2_20260827 不存在；因此没有 raw result，更没有 paper-admitted 结果。

## 真实 schema 与根因定性

冻结 mapping SHA 仍为 68017f...be4d，它的真实结构是：

    generated_view_inventory.logical_shape = "128x128b 1RW SP"
    generated_view_inventory.slow.area_um2 = 8758.3606
    generated_view_inventory.fast.{cycle_ns, access_ns, ...}

顶层 generated_view_inventory.area_um2 从不存在。而冻结 analyzer 读取 inventory["area_um2"] * 9，所以失败是确定的 schema/API 集成 bug。正确的 slow-corner 几何是 8758.3606 * 9 = 78825.2454 um2，恰与 governing contract 的冻结显式标量一致。

定性如下：

1. **不是环境漂移。** mapping、analyzer、runner、execution contract、admission 和 Python 全部与 admission/attempt 钉死的 SHA 一致；runner 资源门也通过。
2. **主因是 analyzer bug。** 代码用了不存在的 JSON 指针，且没有先给出可解释的 schema assertion。
3. **同时是契约身份/验证缺口。** governing contract 同时冻结了 mapping SHA 和 78825.2454，却没有冻结生成该标量的 JSON pointer/corner/macro count；runner 在消耗 attempt 前也没有 schema smoke。“文件精确 SHA”不等于“消费者与 schema 相容”。

## 为什么 source review 没有发现

r1 静态评审核对了 SRAM mapping 的封条、容量算术和 analyzer 整体 SHA，但没有做“代码指针 ↔ 真实 JSON 树”的字段级走读，也没有非 production schema smoke。r2 评审又把“analyzer 与 r1 字节相同”当成语义安全的依据，实际上只是精确保留了 r1 的旧 bug。admission-only 评审只验证权限和 SHA 闭包，不是该缺口的最初产生点。

因此 P0 是“在一次性 attempt 前未证明所有非流式输入解引可达”，而不是偶发 runtime 错误。

## 最小 r3 恢复合同

r3 只允许修复 schema 解引与增加前置 smoke；冻结 row ledger、M468/M473/M504/M505 基线、row64/B8/128 B/cycle/CAM64 坐标、pipeline/cycle/traffic/capacity 算法、sample-major/operator-isolated 粒度、决策门和 claim boundary 必须字节或规范化语义相同。

1. **唯一数据修复：** 要么显式要求 mapping schema tsmc28_sram_macro_mapping_audit_v1，钉死 generated cell/logical shape/slow corner，从 generated_view_inventory.slow.area_um2 读取有限正数 8758.3606，并验证九宏总和 78825.2454；要么使用新 r3 contract 中冻结的 explicit geometry {macro_count:9, per_macro_area_um2:8758.3606, total_area_um2:78825.2454}。不允许 dict.get 降级、多路 fallback 或不明 corner 的顶层字段。
2. **新 preflight-only 路径：** analyzer 需有一个不建 process pool、不遍历 51.84M rows、不产生 production result 的 schema smoke mode。它必须覆盖 execution/governing schema、所有 frozen SHA/双封、M468/M473/M505 所需 key path、SRAM mapping key path/corner/geometry、输出目录无覆盖，并输出唯一 PASS token。
3. **attempt 之前重复 smoke：** r3 runner 必须在创建 attempt_consumed 前运行已钉 SHA 的 preflight-only 模式并检查唯一 PASS。smoke 失败只能进入新 r3 pre-attempt quarantine，不得消耗 attempt。资源/EDA 冲突门仍需在 production attempt 前重新检查。
4. **身份隔离：** r3 必须有新 analyzer/runner/execution/canonical/attempt/quarantine/admission schema/status/SHA，r1/r2 admission 和 r2 attempt 不能通过 r3 谓词。r2 失败证据原样保留。

## 必要的新准入链

1. author 产出 r3 analyzer/runner/execution contract 与双封 handoff；源码 diff 只能是上述 schema/smoke 修复及 revision 身份。
2. 新的独立 source-only 静态锤审必须逐个验证 analyzer 访问的真实 JSON pointer，并证明 cycle worker/pipeline/aggregation 规范化 diff 为空。
3. 静态评审可另授权一次 schema-smoke-only 执行；它不计 production，不建 pool、不生成周期数。smoke 回执必须双封并由不同评审者复核 PASS token、无 result 和负向 malformed-schema 用例。
4. root 只能在静态锤审和 smoke 锤审都 P0/P1=0 后创建全新双封 r3 production admission；admission 必须钉死两份审阅的 outer seal，并只授权一次 CPU production。
5. r3 runner 在现场资源门后、消耗 attempt 前再跑一次同一 smoke。只有此次 PASS 才可消耗 r3 attempt 并启动 row replay。
6. raw production 仍需新的独立 result hammer；它通过前不得启动 RTL，不得引用 1.746753x/1.741232x 为 r3 结果。

## P0/P1/P2

- **P0-01：** one-shot attempt 在 schema 消费关系未验证时被消耗；r2 永久 NO-GO。
- **P1-01：** governing/execution contract 未钉死 area 的 JSON pointer/corner/macro-count 生成关系。
- **P1-02：** runner 缺少 attempt 之前的可执行 schema smoke。
- **P2-01：** r1/r2 static review 将“相同 SHA”过度解读为“相同语义正确性”；后续清单必须增加 consumer-path 走读。

## Claim boundary

本锤审只准入“起草 r3”，不准入 r3 production、RTL、VCS、Synopsys PPA、energy、full-network/system speedup 或 DATE headline。r2 没有可引用的性能结果。
