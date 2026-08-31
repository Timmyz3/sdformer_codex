# M706｜M698 Table-A registry r10 fresh data-quality review

## Answer-first decision

**GO_REGISTRY_SCAFFOLDING_AND_PRODUCTION_ZERO_ONLY**。

M698 r10 可以作为后续原生新思证据的 fail-closed 注册表骨架，fresh review 为
`96/100`，`P0=0, P1=0, P2=1`。这个 GO 只准入两件事：

1. r10 structural registry scaffolding；
2. canonical production-zero（production runs / authority / evidence bundles /
   eligible rows 均为 0）。

它**不准入任何 Table-A PPA 行、面积、频率、功耗、能量、系统倍速或 analytical
range**。未来真实 native run 仍必须由 additive revision 钉死新的 fresh-review SHA；
r10 本身不得原地填充 authority allowlist。

## Dataset and grain

- 审查对象：M698 r10 extractor、builder、canonical config、contract、作者测试和作者
  handoff 双封。
- 预期粒度：每个 Table-A configuration 至多一个经 code-pinned authority 授权的
  production row。
- 当前实际粒度：0 production row；synthetic `b0_dense96_fixed_t10` 仅用于 structural
  grammar 测试，不能进入 production map。
- 比较基线：M695 对 r9 的 `P0=0, P1=5, P2=1` fresh failure root。

## Evidence integrity

- 作者 handoff 内部 `SHA256SUMS` 全部通过。
- 作者 handoff manifest SHA：
  `af345ac1d4e94d88508461992a08061818b71afa8dec300e51ce1172ec4f1370`。
- 外层 seal 内容精确绑定上述 manifest SHA。
- extractor / builder / tests / config / contract SHA 分别为：
  `66b5b988...f5585f85d` / `81fdc6e2...260849e` /
  `ea4bddf6...5dd3e45` / `6d9dedb3...ab2e5a3` /
  `e00172b2...eba6226`，均与 contract/author audit 一致。
- `docs/359` 只读复核 SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## M695 five-gap closure map

| M695 缺口 | r10 显式数据质量条件 | 既有 negative fixture | fresh 裁决 |
|---|---|---|---|
| P1-01 工具/DB 可冒充 | 工具 family/path/realpath/dev/inode/build-ID/snapshot/proc/version 交叉；五类 native hash/build-ID 互异；DB 需 native-read status、库名、cell/opcond/voltage/unit/fingerprint | tests 04、06 | CLOSED_FOR_SCAFFOLDING |
| P1-02 argv0 与实际 executable 解耦 | 八 step 强制 snapshot path = argv0，SHA 回到 logical tool；simv 另绑 exact simv SHA | test 05 | CLOSED_FOR_SCAFFOLDING |
| P1-03 wire stub/RTL=netlist/十算子范围注水 | native 模式拒绝 RTL==netlist、行为 wire stub、缺 mapped-cell census；十算子正 seq/comb/leaf 且逐层与 top 求和；Formality exact RTL/netlist、compare point≥100、unmatched=0 | tests 07、08 | CLOSED_FOR_SCAFFOLDING |
| P1-04 单 TC/自报覆盖率 | annotated rows 必须显式且唯一；分母来自 scope census；net source 必须出现在真实 SAIF TC；native TC distinct≥100；net/pin coverage≥95% | test 09 | CLOSED_FOR_SCAFFOLDING |
| P1-05 SRAM 宏面积/功耗缺失或拆分错误 | netlist/DC/PTPX 三份报告必须按同一有序 8 weight + 8 state + 1 parent 实例集合交叉；DC total=logic+macro；PTPX per-instance total=internal+switching+leakage，17 项求和回到 SRAM total | tests 10、11 | CLOSED_FOR_SCAFFOLDING |

## Core quality checks

- **Completeness：PASS。** M695 五类 P1 各有显式代码条件和至少一个既有 negative
  fixture；12/12 tests 通过。
- **Uniqueness：PASS。** 工具、step、operator 与 macro 集合均拒绝重复/缺行；
  configuration 由 row map 和 design identity 共同限定。
- **Validity：PASS。** native/synthetic evidence class 分列；整数 census、覆盖率阈值、
  有限且正的 area/power，以及面积/功耗等式都有 fail-closed 条件。
- **Consistency：PASS。** manifest、extension、RTL/netlist、SAIF、DC、PTPX、SRAM
  compiler diagnostics 通过 SHA/identity/root/equation 多路交叉。
- **Authority integrity：PASS。** `PINNED_PRODUCTION_AUTHORITIES = {}`；任意非空
  production bundle 在 structural parsing 前即因未 code-pin 的 authority 被拒绝；任意
  自写 clean review 不能成为 authority。
- **Volume/shape：PASS。** canonical builder 输出严格为
  `production_runs=0 authority=0 bundles=0 eligible=0`，headline/analytical=false。
- **Timeliness：N/A。** 本包是方法学注册表而非随时间更新的数据集；日期与 sealed
  predecessor 身份一致。

## P2 limitation

唯一 P2 是**当前没有真实 native production run 可供数据内容复核**。这不是 r10
scaffolding 的缺陷，但它限定了本次 verdict：12 项 synthetic/negative tests 只能证明
schema、拒绝路径和 fail-closed authority 模型，不能证明未来某个 DC/PTPX/SAIF 数字
真实、完整或可发论文。真实 run 必须在 additive authority-pinning revision 中重新做：

- raw native evidence 双封；
- fresh receipt-blind review；
- exact review SHA code-pin；
- canonical 与 production registry 分离重建；
- 仍不得由 structural parser 自己授予 production authority。

## Scope controls

- 最终 verdict 只使用 Python 作者测试、包内既有 negative fixtures、静态源码条件和
  canonical builder；范围调整前的未封存探索性夹具已撤去，未作为证据。
- 最终复核未访问 `/proc`，未把任何自建工具身份模拟纳入判定。
- 未运行 EDA、GPU、remote、training、capture 或性能任务。
- 未修改 M695、M691/r9 或 `docs/359`。
