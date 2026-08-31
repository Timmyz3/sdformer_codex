# M556｜M552/M545/M542/M534 PBR4 pre-RTL CPU contract r3 fresh static hammer

日期：2026-08-28  
模式：fresh independent、read-only static hammer；零 candidate CPU/analyzer、runner、RTL、EDA、训练、GPU、远端与 result  
裁决：`FAIL_SOURCE_ONLY_R3_PRE_RTL_CPU_CONTRACT__TWO_P1_REPAIRS_REQUIRED__NO_CPU_RUNNER_OR_RTL`  
评分：**82/100**；P0/P1/P2 = **0/2/0**

## 1. 结论

r3 已实质关闭 M549 的四架构普通 service/flush 分支：SC8/ISO8 partial flush、OSG serviceable、PBR4
ingress-vs-drain、context release 与 six-slice RMW 均已可独立复算。四个 no-newline golden 的长度与 SHA
全部匹配；exact T10、typed sign、FINAL_OUTPUT stall、`239,636 B` logical-only 边界、fixed strongest-A1
及未来 result/attempt absence 也均通过。

但它仍不能 source-only PASS，也不能进入 runner author admission。两个 P1 都是闭环问题，而不是性能好坏：

1. 公共 priority row 8 把 `block transition`、`time transition` 与 `directory-clear edge` 放在一个 guard 下，
   没有子优先级或互斥状态。在 layer/time 的最后 output block，block-close 与 time-epoch 前置条件可以同时
   成立；directory clear 又是收费的 1024-word 序列。不同实现可产生不同 cycle、owner 与 ready 对齐。现有
   四份 golden 只覆盖 resident-hit 两 lane，不覆盖末 block/time。
2. future authorization 的字节身份仍形成环：它规定 authorization 必须在 final-release review 双封之后生成，
   且 authorization 绑定 runner/review SHA；同时又规定已审的 runner source/shell 必须冻结这个后来才生成的
   authorization SHA。修改 runner 去加入该 SHA 会使 authorization 中绑定的 runner/review identity 失效。

因此本 hammer **不授权 CPU runner、CPU launch 或 RTL**。合法下一步只有 author 一个仍为
`run_authorized=false` 的 r4 contract/schema，关闭两处，再做新的 fresh independent hammer。

## 2. P1 findings

### M556-R3-P1-01｜末 block/time/directory-clear transition 未唯一排序

`cycle_machine_common.priority_table[8]` 的 guard 是 “block or time close predicate is true”，action 是执行
“one explicit block/time transition or directory-clear edge”。这不是一个唯一 action：

- r4 要求每个 current block 完整 commit 后做 block transition；最后 block 完成时又满足
  `epoch_transition_requires_all_output_blocks_committed`；
- r2/r3 要求 time epoch 闭合并执行 1024 个 charged directory-zero writes；
- 合同没有冻结 last-block retire、clear-start、clear-word-i、clear-end、time-retire 与 next-time owner-load
  的互斥 guard、顺序、state delta 和 primary class；
- 四份 minimal golden 均未走到末 block/time，不能消除该分支。

必须增加 terminal ordered submachine，并至少增加一份 last-block/time golden/hash。否则 runner 选择会改变
`block_transition_drain`、`time_epoch_directory_clear`、ready transcript 对齐和 A1-STRONG。

### M556-R3-P1-02｜future authorization 与 runner SHA 构成身份环

`future_runner_schema.future_launch_authorization.non_circular_identity_rule` 同时要求：

- final-release hammer 已存在并双封后，才生成独立 authorization；
- authorization 绑定 runner Python/shell 及 contract/runner/final-release review identities；
- runner source/shell 又必须冻结该 authorization 文件 SHA。

这在字节层不可构造。runner 若先审，就不知道后生 authorization SHA；authorization 生成后再修改 runner，
会使 runner hammer/final-release/authorization 绑定的旧 SHA 全失效。

可行修复是单向链：post-review authorization 使用 canonical path、独立双封并绑定 immutable runner/review；
runner 运行时重算 authorization member/seal 与其内部全部 identity，但不把 authorization SHA 嵌回 runner。
另一种方案是新增更晚的 immutable launcher wrapper 并单独审查，不能复用当前环状 schema。

## 3. 已通过的关键检查

### 身份、JSON 与零运行

- contract、handoff、request、M549 review、r2 contract 与 M534 r2/r3/r4 JSON 均 duplicate-key/non-finite
  fail-closed 通过；contract/handoff/request/M549/M534 双封均通过；六个 normative member SHA 精确匹配。
- `docs/359` SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；本 review 未修改它。
- future Python/shell、final authorization、canonical result、attempt marker、runner hammer 与 final-release
  hammer 均不存在；`run_authorized=false`。

### 四架构与 service

- SC8：最低 remaining lane 冻结 Cin tile，stable greedy、weight-bank/phase-bank 唯一；1--7 lane 在本 bundle
  扫描尾立即 flush。
- ISO8：只看 head 和紧邻 valid lane；不满足 full identity/tile/bank 条件时 singleton 当场 flush。
- OSG：`FULL || selected PRESSURE || CLOSE || BLOCK_DRAIN` serviceable 唯一；movable ingress、pressure、
  full、close、drain 顺序已冻结。
- PBR4：movable ingress 先于 pressure drain；ingress 空后 partial context 按 phase/index tail drain；explicit
  epoch retire 后下一 bundle 才可 accept。
- OSG/PBR4 context release 均是 slice5 后的独立收费 cycle；same-edge retire/replace 禁止。SC8/ISO8 的
  group-done/next-lock 与 bundle-retire/next-accept 也禁止同 edge。
- 一个 round 冻结一个 Cin tile、每 weight bank 最多一个 contributor；GROUP_LOCK 收费；六 slice 按
  issue1/L4/O8/1RW 执行；每 selected destination 恰好六 psum read + 六 write。

### golden 与整数复算

| point | cycles | recomputed SHA | result |
|---|---:|---|---|
| A1-SC8 | 18 | `69f86a715ea5c2644aaa30136e3105ac6f91d27b325dd7a7eae42ee736aec152` | match |
| A1-ISO8 | 18 | `89d7a3ee74d6a9b599bd1ecac47481796674a09194245fd6bbae1bdb7abb73ee` | match |
| A1-OSG | 22 | `88b397ce590ba252fa21b2ee6fe5f3a47aa3a3a40f86be460a7a5671713119dd` | match |
| PBR4 | 21 | `f8bbfb3c638bae1e3163ad541217601759bfe44046278c9ac6cdac85aa8cebdc` | match |

- raw M511：`69,624,000 bit/sample/T10`、`696,240,000 bit/S10/T10`、`87,030,000 B`；
- block replay：`92,688,000 bit/sample/T10`、`926,880,000 bit/S10/T10`；
- replay logical/padded：`115,860,000 / 115,860,800 B`；dense destinations=`11,040,000`；
- psum window=`42,393,600 B=0x0286E000`；`237,568+2,068=239,636 B`，headroom=`6,124 B`；
- numeric `1` 是 `+1`；独立 `source_sign_bit=0`；bit 1 malformed；product sign 只来自 signed INT8 weight；
- 三个 A1 必须完整 S10x4xT10 后固定一个 A1-STRONG，禁止 per-sample/layer/time oracle；四点不可由 CLI
  或 runner 增删、改名或改 transition。

## 4. 授权矩阵

- r3 contract static source-only PASS：**false**；
- runner source author admission / CPU analyzer / CPU launch：**false**；
- RTL / VCS / iverilog / Verilator / DC / PT / PTPX / Formality / training / GPU / remote：**false**；
- cycle / traffic / energy / PPA / system / paper headline：**false**；
- 唯一合法下一步：source-only r4 contract/schema repair，双封后重新 fresh hammer。

本 hammer 只新增本 review 目录及其双封文件；未修改被审 contract/handoff/request、normative imports、任何
runner、RTL、result、`docs/524` 或 `docs/359`。
