# M592｜M586/M579 r2 source-static hammer

日期：2026-08-28  
结论：**FAIL，93/100，P0=0，P1=2，P2=1。不得创建 execution release，不得启动 80-record 正式 CPU replay。**

## 一、裁决

M586 已经关闭 M584 的两个计算正确性 P0：冻结的 Python 3.10/NumPy 环境实际可 spawn，M43/M504/M505 可导入并执行八行 M505 dead-write-only recurrence；r1 的 partition-major 数组也确实在进入 `pipeline_cycles` 前转置成 M528 的 `[sample,operator,chunk,partition]` C-order。独立静态重验同时确认 80/80 packed payload、10x4 cohort、三个 accuracy scope、九行容量账和 claim boundary 均保持一致。

但是 immutable future runner 的发布状态机仍没有达到 request 所要求的“任意失败无窗口、长跑前后同一 identity”。两个 P1 都发生在正式计算之外，却都可能让一次长跑留下不可审计的 attempt/result 身份，因此本轮必须 fail-closed。

## 二、评分

| 维度 | 得分 | 满分 | 判断 |
|---|---:|---:|---|
| 冻结 runtime、SHA 与实际 spawn preflight | 20 | 20 | 精确 Python/NumPy、spawn、八行 recurrence 全过 |
| M528 task order、M505 recurrence 与公平成本 | 25 | 25 | chunk-major、20,304 tasks/operator、DMA/tail/commit/8 blocks 闭合 |
| 80 payload、M255 三口径与 M528 容量 | 25 | 25 | 严格解析、64 帧退化/no-Pareto 和 213,376 B 均闭合 |
| attempt、terminal identity 与原子发布 | 13 | 20 | 初末 contract SHA 未比较；trap 前有 attempt 残留窗口 |
| claim boundary 与 source-only 授权 | 10 | 10 | 三类 ratio 不相乘，system/RTL/PPA/energy/headline 均 false |
| **总分** | **93** | **100** | **FAIL_SOURCE_STATIC** |

## 三、P1 findings

### P1-1｜长跑开始与 terminal 的 execution-contract identity 未证明相同

runner 在正式 attempt 前调用 `--validate-contract-only`，但没有保存当时的 contract SHA。长跑结束后 `terminal_rehash()` 重新读取当前路径并把**当时**的 SHA 写进 receipt（analyzer 450–479 行）；没有任何断言把它与开始时的 SHA 比较。R1 结果中的 `identity` 则来自开始时读取的 contract。

因此，如果 execution contract 在长跑中被替换为另一份内部一致且可通过语义检查的合同，terminal 可对新合同/新 inputs 通过，而 staging result 仍描述旧合同/旧 inputs。runner 内完全没有 `CONTRACT_SHA` 起始快照；这不是“重新哈希”可以自动关闭的同一性证明。

要求：在任何 attempt 之前计算并原子写入 `contract_sha256_start`；production、terminal 和 rename 前均要求同一 SHA，terminal analyzer 接受并核对 `--expected-contract-sha256`，receipt 同时写 start/end。execution release 还必须锁定该 exact contract SHA。

### P1-2｜attempt 在安装 cleanup trap 前已创建，失败 quarantine 也没有 seal

runner 118 行先 `mkdir "$ATTEMPT_DIR"`，119–120 行再写 marker，直到 137 行才安装 `trap cleanup`。在 mkdir 后、trap 前发生 signal、磁盘错误或 marker 写失败，会把 canonical `.attempt` 留在正式坐标，既不 consumed 也不 quarantined。即使进入 cleanup，函数只 `mv` staging/attempt（123–135 行），没有为失败工件生成 member/outer seal。

这违反 request 的“任意失败 quarantine 均无窗口”，也没有关闭 M584 对 sealed quarantine/attempt 的要求。

要求：在任何 canonical state mutation 前安装 trap；用显式状态标记覆盖“尚未创建 attempt”的分支；所有失败目录通过 NOREPLACE 移入唯一 quarantine，并至少封存 attempt marker、stdout/stderr、失败阶段、contract start SHA 与 member/outer seal。修改 runner 后必须重新冻结 SHA 并做 fresh static hammer。

## 四、P2 finding

### P2-1｜terminal 的 required-input key set 依赖未来合同评审，analyzer 本身没有精确集合断言

source contract 列出了 r1 base、M43、M504、M505、r2 analyzer/runner、M247/M255/M528/docs359 等未来必需 identity；但 `validate_execution_contract()` 只遍历 `contract["inputs"]` 中实际声明的成员，没有断言其 key set 等于 source contract 的冻结集合。若未来 execution author 漏掉 `r2_runner` 等 key，terminal receipt 的“all declared inputs”仍可能为真，但不是“all required inputs”。

这可以由 fresh execution-candidate review 临时挡住，但更稳妥的修复是 analyzer 内冻结 exact required key set，并逐个路径/SHA 断言；尤其 runner 必须在长跑后显式 rehash，而不能只依赖未来合同自觉声明。

## 五、通过的独立机械检查

- request、source analyzer、runner、source contract 与 author handoff 全部只读审查；正式 CPU/GPU/EDA/remote 均未运行。
- exact runner `--preflight-only` 实际 PASS：Python 3.10.16、Python SHA、NumPy 2.0.1 路径/SHA、spawn child、M43/M504/M505 import、八行 recurrence（6 ideal issue、8 liveness cycle）。
- preflight 前后正式 `result/attempt` 快照均为 0；`formal_trace_records_processed=0`。
- r2 analyzer/runner/source-contract SHA 分别为 `70eb0746...d4b471`、`8e0efbb6...ecd45e`、`319d1c89...fa44ec`；source-contract member/outer seal 均通过。
- M504 是 hard-coded direct path/SHA，并在每个 worker 调用 r1 initializer 前复核；r1 base、M43、M505 也由冻结源码 SHA 绑定。
- frozen r1 AST 中 outer loop 为 partition、inner loop 为 row chunk，且每 task 只有一次 `simulate_liveness_task(tile, False)`；r2 对所有 seven cost arrays 做 `(432,47).T.reshape(-1)`，故 `pipeline_cycles` 实际接收 chunk-major。
- 顺序 anchor `[0,47,94,141]`，末 source index 20,303；432x47=20,304 tasks/operator，末 chunk 56 rows。
- 80/80 packed payload 重哈希通过；PAFT/control 各 40 records、10 samples x 4 operators、cohort/operator 对齐。
- 每 record shape/output-shape `[10,1,768,15,20]`、2,304,000 elements；positive/negative/numeric 三个 288,000-byte plane，offset 288,000/576,000，总 864,000 bytes；packing、negative=0、timestep support sum、basename uniqueness 均检查。
- M255 strict parse 同时保留：valid825 `1.4776177363→1.4691506688`（单 seed，+0.5730%）；相同十帧 `1.0035290166→1.0011476289`（5 win/5 loss）；完整 64 帧 `1.3518846194→1.3656589993`（PAFT 退化 1.0189020312%）。`accuracy_performance_pareto=false`、无共同 evaluator runtime SHA 明示。
- M528 hammer/JSON/CSV 均 strict/field check；候选容量九行合计 213,376 B，预算 245,760 B，余量 32,384 B；macro integration/PPA/energy 仍为 OPEN_NOT_ADMITTED。
- arithmetic-work、local-cycle、PAFT/control activity increment 分字段且无相乘路径；numeric Conv/Acc24、RTL、VCS、Synopsys PPA、energy、system speedup、headline 均未准入。
- docs/359 SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 六、允许的下一步

本轮不允许 formal CPU run。root 只能修 runner/必要的 analyzer 与 source contract，产生全新 source tuple 和双封，再请求 fresh 独立 source-static hammer。达到 score>=95、P0=0、P1=0 后，才可另建 `launch_now=false` execution candidate；source hammer 仍不能直接授权 production。
