# M549｜M545/M542/M534 PBR4 pre-RTL CPU contract r2 fresh static hammer

日期：2026-08-27  
被审合同：`contracts/m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r2_20260827.json`  
模式：fresh independent、只读静态审计；零 candidate CPU/analyzer、RTL、EDA、训练、GPU 与远端运行  
裁决：`FAIL_SOURCE_ONLY_R2_PRE_RTL_CPU_CONTRACT__REPAIR_REQUIRED__NO_RUN_OR_RTL`  
评分：**90/100**；P0/P1/P2 = **0/1/1**

## 1. 结论

r2 已经关闭 r1 的 exact-T10、typed source/sign、FINAL_OUTPUT stall atomicity 与 `docs/524` author-task
provenance 缺口。独立整数复算与文件身份审计均通过：raw M511 为
`696,240,000 bit / 87,030,000 B`；block-outer replay 为 `92,688,000 bit/sample`、
`926,880,000 bit/S10`；四层逐 time、逐 sample 的 logical/padded byte 与 dense-destination 账一致。
numeric activity `1` 与独立 `source_sign_bit=0` 已分型，product sign 只来自 signed-INT8 weight。
六个 M534 r4/r3/r2 规范成员、contract/handoff/request 双封、`docs/359` 与 `docs/524` SHA 均匹配。

但合同尚不能 source-only PASS。它已经声明“runner 不得选择 scheduler”，却仍没有定义 ordinary legal
workload 下的唯一 context service/flush transition：

- `A1-OSG` 使用未定义的 `when serviceable`；等待更多同 destination contributor、在当前 contributor 后
  立即 drain、context 满时 drain 或 frontier 时 drain 都符合现有文字，却产生不同 group/RMW/cycle；
- `PBR4` 同时允许 ingress lane 搬入与 resident context partial-RMW drain，却没有给两类动作的唯一优先级、
  同 edge retire/replace 规则和 context 何时成为 serviceable；
- `A1-SC8/A1-ISO8` 没有冻结 partial/tail group 的唯一 flush guard。

M534 r2 的规范输入只写了 stable-lane 搬运、context drain 与 six-slice service，r3/r4 主要补 packing、backing、
frontier 和 dense output；六个递归输入没有消除上述普通状态分支。`unspecified_tie_or_transition=FAIL_CLOSED`
能阻止 runner 猜测，但会让合法 trace 在普通状态直接失败，不能替代 executable transition table。该选择会改变
`A1-STRONG` 身份以及 PBR4 的 group/RMW/cycle/traffic，直接作用于 `1.30x/1.10x` 决策门，因此为 P1。

本 hammer **不授权** runner source admission、CPU launch 或 RTL。合法下一步只有：author 一个仍为
`run_authorized=false` 的修复合同，逐 architecture 冻结 cycle-level guard/priority/action/state-delta/
primary-cycle-class 表，重新双封并接受新的 fresh static hammer。

## 2. P1 finding

### M549-R2-P1-01｜四架构仍缺唯一的 service/flush transition

合同 `architecture_algorithms` 已比 r1 明显进步，但以下关键谓词仍未定义：

1. `A1-OSG.scheduling` 写“when serviceable, choose phase ascending then context index ascending”，没有定义
   `serviceable`。同一 context 含 1--7 个 contributor 且 canonical transcript 后续仍可能到达相同 destination
   时，立即 drain 与等待继续 join 都合法；二者会改变 group utilization、partial RMW、port conflict 与周期。
2. `PBR4.scheduling` 要求 stable-lane 搬入 context，并允许 excess ingress 等 context partial-RMW 后继续搬入；
   但没有规定有可搬 lane、已有 resident context、pending write/link response 同时存在时，哪一个 transition
   先执行，也没有冻结 same-edge context retire-and-replace 是否允许。
3. `A1-SC8` 与 `A1-ISO8` 定义了 descriptor 可否合并，却没有定义不足 K8 的 partial group 在 bundle tail、
   event boundary、frontier 或 block drain 中哪个 guard 唯一 flush。

这些不是性能无关的 coding detail。不同选择会改变至少：group sequence、group count、RMW sequence、context
occupancy、psum read/write、link arbitration、total cycles，并可能改变 fixed strongest-A1。仅声明 runner 不得
选择或 unspecified 时 fail closed，不能使一个合法普通状态获得唯一 next state。

修复必须给每个 architecture 一张完整、确定性的 transition-priority table，至少冻结：

- 每个 cycle 的 ordered guard、唯一 action、所有 state delta 与 primary cycle class；
- group fill/eligible/partial-tail flush 条件；
- ingress move、context service、pending write、external response 与 frontier/output 的优先级；
- same-edge retire/replace、blocked transition 与 fault-drain 规则；
- 对一个最小合法 transcript 的逐 cycle golden schedule/hash，使独立实现能得到同一 group/RMW/commit 序列。

## 3. P2 finding

### M549-R2-P2-01｜future launch exact-key schema 未绑定 attempt marker 与实际审查身份

合同要求 canonical result 与 attempt marker 在 launch 前都不存在，并要求新的 independent release；但
`future_runner_schema.future_closed_authorization.required_exact_key_set` 只含 `result_path_absent`，没有
`attempt_marker_absent`，也没有绑定 source-contract hammer、future runner static hammer 与最终 release 的
exact SHA/outer seal。一个手写 `score_0_to_100=100` 的 JSON 在字段层面可以满足当前 schema，而不证明分数来自
哪份双封 review。

这不会扩大当前授权，因为当前 `run_authorized=false` 且 runner 不存在；但下一版 future schema 应至少加入：
`attempt_marker_absent`、本 contract hammer 的 review/member/outer identity、future runner static-review
identity、final launch-release identity，并把这些文件纳入 runner preflight 的 exact hash 检查。

## 4. 已通过的静态检查

### 4.1 JSON、身份、seal 与零运行边界

- contract、handoff、future schema、request、r1 review 与六个 M534 JSON 均以 duplicate-key/non-finite
  fail-closed parser 通过；
- contract SHA=`afdadd302cffdffedb34e0679c177424aa8fc9d7023e4fd9eecf7c4f5ff9bc63`；
  member sidecar file SHA=`57b5ecced98b433f1b0b2bc17da6faa81882ff3cc218833cc359a5ccb39951bf`；
  outer sidecar file SHA=`3e95bbf5820c816aa941194ec1b314975d9b2a3cbf944f90e2ad63611c927104`；
- handoff exact member set、member manifest 与 outer seal 通过；request member/outer seal 通过；
- M534 r4/r3/r2 README+JSON 六个 exact SHA 与三套 member/outer seal 全部通过；
- r1 contract、r1 review、M511 contract、M523 contract/RTL 等全部 `frozen_predecessors` 匹配；
- `docs/359` SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；
- `docs/524` 在 pre-author 与 handoff 均为
  `4f3ffe1c70027ef07efb3908c88876ec48cabc37ad8778150a6922f16c7abe0d`，且 claim 明确只覆盖本 author task；
- future Python/shell runner、canonical result、attempt marker 与 M511 payload directory 均不存在；
- `run_authorized=false`；当前 CPU/analyzer/RTL/VCS/iverilog/Verilator/DC/PT/PTPX/Formality/训练/GPU/
  远端/result-directory count 均为整数零；future CPU count=1，所有 future non-CPU count=0。

### 4.2 exact T10 与容量算术

| layer | raw bit/sample/T10 | replay bit/sample/T10 | logical B/sample/T10 | padded B/sample/T10 | dense dest/sample/T10 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4,608,000 | 18,432,000 | 2,304,000 | 2,304,000 | 48,000 |
| 1 | 9,240,000 | 18,480,000 | 2,310,000 | 2,310,080 | 96,000 |
| 2 | 18,528,000 | 18,528,000 | 2,316,000 | 2,316,000 | 192,000 |
| 3 | 37,248,000 | 37,248,000 | 4,656,000 | 4,656,000 | 768,000 |
| **sum/sample** | **69,624,000** | **92,688,000** | **11,586,000** | **11,586,080** | **1,104,000** |
| **S10** | **696,240,000** | **926,880,000** | **115,860,000** | **115,860,800** | **11,040,000** |

- canonical loop 唯一为 `sample -> layer -> time 0..9 -> output_block -> ordinal -> kernel`；M511 record
  binding 不能替代 time loop；四架构都必须完整执行 T10；
- ready transcript 对每个 sample/layer/time 独立同值复位；下一 time 复用 address 前要求 beat、command、
  directory 与 epoch state 全部退休；
- directory 连续覆盖 `[0,4800)`、`[4800,14400)`、`[14400,33600)`、`[33600,110400)`；
  `110,400*384=42,393,600 B=0x0286E000`，last byte=`0x4286DFFF`；
- modeled array `237,568 B` + soft-state `2,068 B` = modeled-logical `239,636 B`，低于
  `245,760 B` `6,124 B`；foundry/CACTI/mapped PPA 均为 false。

### 4.3 typed source、输出 stall 与公平门

- accepted numeric activity 是 `{0,1}`，numeric `1` 是合法隐式 `+1`；独立 `source_sign_bit=0` 必须，
  bit 1 malformed；product sign 只来自 exact signed-INT8 weight；三个 reference domain 禁止复用字段；
- `out_valid&&!sink_ready` 时 command/address/beat/data、output/psum read、新 request、accept/retire、directory
  clear、cursor、owner 与所有 architecture state delta 均为零；唯一 increment 是
  `final_output_sink_stall_cycles`，且进入 cycle conservation；
- 三种 A1 在 candidate 不可见时完整跑完 S10x4xT10 并双封；只固定一个 complete-sum-cycle 最强 A1，
  tie order 固定，禁止 per-sample/layer/time oracle；
- 四点共享 M218 `6x16/L4/O8/FIFO4/Acc24`、modeled `239,636 B` coordinate、source replay、external
  link、dense three-beat output、traffic 与 symbolic-energy 账；
- gate 仍为 function/conservation mismatch 全零、ratio-of-sums `>=1.30x`、每 sample `>=1.10x`、weight
  traffic 不增、无隐藏资源、OSG sequence 不等价；失败时只有 psum read+write `>=30%` 才能作 support；
- S10 明确不是 multi-sequence、full-network、system speedup、paper headline 或 PPA evidence。

## 5. 最终授权矩阵

- r2 contract static source-only PASS：**false**；
- runner source author admission / CPU launch / CPU analyzer：**false**；
- RTL / VCS / iverilog / Verilator / DC / PT / PTPX / Formality / training / GPU / remote：**false**；
- cycle / traffic / energy / PPA / system / paper headline claim：**false**；
- 合法下一步：只允许 author 一个修复后的 `run_authorized=false` contract，加入唯一 transition-priority
  table，双封后重新进行 fresh independent static hammer。

本 hammer 只新增本 review 目录及双封文件；未修改被审合同、handoff、M534 r2/r3/r4、任何 runner、RTL、
result、`docs/524` 或 `docs/359`。
