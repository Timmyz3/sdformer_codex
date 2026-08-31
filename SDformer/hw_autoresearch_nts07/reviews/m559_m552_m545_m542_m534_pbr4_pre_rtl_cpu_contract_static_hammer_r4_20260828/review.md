# M562｜M559 PBR4 pre-RTL CPU contract r4 fresh static hammer

日期：2026-08-28  
模式：fresh independent、read-only static hammer；零 candidate CPU/analyzer、runner/authorization/wrapper、RTL、EDA、训练、GPU、远端、result 与 attempt  
裁决：`PASS_SOURCE_ONLY_R4_CONTRACT__RUNNER_SOURCE_ADMISSION_ONLY__NO_EXECUTION`  
评分：**100/100**；P0/P1/P2 = **0/0/0**

## 1. 裁决

M556 的两项 P1 已在 r4 的冻结字节中关闭。

第一，r3 common priority row 8 已被明确删除并由单一、四架构共用的 terminal FSM 替换。普通 non-last
block 只允许先 retire、后 owner load；last block 只允许 retire 后依次执行 clear start、word 0--1023、clear
end、time retire，再进入唯一的 time/layer/sample/cohort 分支。15 个 terminal row 均为 1 charged cycle、一个
完整 state delta 和一个 primary class；同一 legal prior state 下只有一个 guard 为真。`POINT_COMPLETE` 是停机
状态，合同明确禁止再取下一 cycle，不构成可执行 prior state。

第二，future identity 已成为可拓扑排序的 N0--N9 DAG。immutable runner 不冻结后生 authorization SHA；
canonical authorization 等 final-release review 后才生成、独立 member/outer 双封且只绑定早先字节；后生
wrapper 冻结 authorization triple 和 runner SHA，但只冻结 wrapper review 的 canonical path；最终独立
wrapper review 绑定 wrapper self-SHA，且之后不存在新的 author permit。没有 backward/self hash edge。

因此本 hammer 允许的下一步仅为：**另开 immutable runner source admission**。本 PASS 不授权 CPU
execution、authorization/wrapper authoring、RTL 或任何性能/流量/能量/PPA/system/paper headline。

## 2. terminal FSM 独立复算

- `T00` 与 `T02` 由 `output_block<last` / `==last` 互斥；`T04` 与 `T05` 由 clear index `<1023` /
  `==1023 && count==1023` 互斥；`T07` 与 `T09` 由 time `<9` / `==9` 互斥；其余行由不同
  `terminal_state` 区分。
- non-last 路径严格为 `NONLAST_BLOCK_RETIRE`、`NEXT_BLOCK_OWNER_LOAD`，共 2 cycles。
- last/nonfinal-time 路径严格为 `1 retire + 1 clear-start + 1024 zero writes + 1 clear-end + 1
  time-retire + 1 next-time-owner = 1029 cycles`。
- time9 且非末 layer 为 1030 cycles；末 layer 且非末 sample 为 1031 cycles；最终 cohort 也是 1031
  cycles，最后一拍分别唯一落到 next-layer、next-sample 或 point-complete。
- word clear 从 prior `(index,count)=(0,0)` 开始；`T04` 接受 index 0--1022 共 1023 次，`T05` 只接受
  index 1023 一次；start/word0、双 word、word1023/end 均不能融合。
- common priority 2 显式排除 terminal-owned directory clear；terminal scope 又禁止四个架构覆盖、绕过或
  增加 terminal edge。四架构共享同一事件次序、state delta、收费与 class。

精确 UTF-8/no-newline 复算：

| golden | bytes | cycles | recomputed SHA256 | 结果 |
|---|---:|---:|---|---|
| COMMON_NONLAST_BLOCK | 218 | 2 | `dc68fdfc65716ec084377bb1bda5ed454504fe35f9d0acdbd8f094cc86bab628` | match |
| COMMON_LAST_BLOCK_TIME | 349 | 1029 | `46526954f88c08a91f082713d0f1248bdec23137fdb372f697601953257fa819` | match |

四个 imported resident-hit golden 也保持不变：SC8/ISO8/OSG/PBR4 分别为 18/18/22/21 cycles，四个
SHA 均与 M556 已冻结值一致。

## 3. future identity DAG 独立审计

可构造顺序为：contract/static review → immutable runner/static review → candidate review → final-release
review → canonical 双封 authorization → post-auth wrapper → 独立 wrapper static/release review → one shot。

- N2 只知道 N6/N8 canonical path，不知道其后生 hash；N6 只绑定 N0--N5；N7 才冻结 N6 triple 与 N2
  SHA；N8 再绑定 N7 self-SHA。因此不存在 runner↔authorization 或 wrapper↔review hash 环。
- authorization exact key set、禁止 self/wrapper/future-hash key、`100/0/0 + launch_now + absent
  result/attempt` closed predicate及 member+outer double seal 均已冻结。
- wrapper 必须按 canonical 双封 review 中的 `wrapper_sha256` 自检；wrapper review 是 terminal release，
  之后禁止任何 author-generated launch file。
- wrapper 与 runner 都必须递归重算 authorization 及其绑定的 earlier bytes；direct runner、修改 wrapper、
  手写 score/launch JSON、CLI/environment 改 architecture/transition/resource/gate 均在 result/attempt 前拒绝。

## 4. 保持不变的合同边界

- literal T10：raw M511 为 `696,240,000 bit/S10`；block replay 为 `92,688,000 bit/sample`、
  `926,880,000 bit/S10`。
- numeric `1` 是 typed `+1`；独立 `source_sign_bit=0`；bit1 malformed；符号只来自 signed INT8 weight。
- FINAL_OUTPUT stall 的 internal read/request/accept/retire/clear/cursor/owner/architecture delta 全为 0。
- 四点仍仅为 `A1-SC8/A1-ISO8/A1-OSG/PBR4`；三个 A1 完成 S10x4xT10 后才按固定 tie order 选一个
  A1-STRONG，禁止 per-sample/layer/time oracle。
- 同一 service/resource/GO gate 原样递归导入；logical-only 为 `237,568 + 2,068 = 239,636 B <=
  245,760 B`，headroom 6,124 B；foundry/CACTI/mapped-PPA 均 false。

## 5. 身份与零运行

r4 contract、author handoff、request、M556、r3/r2 contracts/reviews 和 M534 r2/r3/r4 imports 的严格 JSON、
member manifest 与 outer seal 均通过；所有冻结 SHA 匹配。future runner、runner/candidate/final reviews、
authorization triple、post-auth wrapper/review、canonical result 与 attempt marker 均不存在。

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

本 hammer 只新增本 review 目录及双封；没有修改被审文件和 normative imports，没有运行 candidate CPU
analyzer、runner、RTL、EDA、训练/GPU/远端，也没有建立 result 或 attempt。
