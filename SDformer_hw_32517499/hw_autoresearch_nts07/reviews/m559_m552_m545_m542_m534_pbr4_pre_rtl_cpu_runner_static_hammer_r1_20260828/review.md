# M574｜M565/M559 immutable CPU analyzer/runner source fresh static hammer

日期：2026-08-28  
模式：fresh independent、read-only source hammer；零正式 analyzer/runner、真实 trace/weight、result/attempt、authorization/wrapper、RTL、EDA、训练、GPU 与远端  
裁决：`FAIL_IMMUTABLE_RUNNER_SOURCE__NO_LAUNCH_ADMISSION__REPAIR_REQUIRED`  
评分：**24/100**；P0/P1/P2 = **3/2/1**

## 1. 裁决

M565 的身份、语法和一部分小 golden 是干净的，但 production analyzer 并没有实现 M559 r4 递归导入的
cycle machine、数值/traffic ledger 和全部 GO 门。若按当前字节运行，它可以在没有读取 decoder weight、没有
计算 Acc24/commit data、没有执行外存与 SRAM 状态机的情况下生成 `PASS_CPU_GO`。因此本 review **禁止**
N4 launch-candidate authoring、authorization、wrapper、正式 CPU run 与 RTL 晋级。

本裁决不是因为倍率偏低，而是因为未来倍率的分子/分母尚不是冻结合同定义的 cycle 数。修复必须生成新版本
analyzer/runner/source contract/request；不得修改或复用本 r1 source identity，也不得建立当前 canonical result 或
attempt。

## 2. P0 findings

### P0-01｜逐 cycle 资源/协议状态机被静态计数器替代，A1/PBR4 坐标不成立

M559 r3/r4 要求同一 priority-0..9 machine、M218 L4/O8/1RW/RAW、weight refill、psum
resident/backing、directory、restore/writeback/final-output 与共同 ready transcript 逐 cycle 执行。当前源码却：

- `service_group()` 对每组无条件一次性收 `12 productive + 2 weight_L4_wait`，没有 O8、L4 response identity、
  psum 1RW、pending RAW 或同拍冲突状态（analyzer 305--323）；
- `model_row()` 把所有 dense destination 无条件当成 `6-cycle zero-build + 32 + 3`，没有 resident/backed
  分支、restore/writeback、shared-link arbitration 或 sink stall（455--482）；
- 27 个 primary class 中只有 8 个在 production AST 中存在显式 `charge()`；其余 19 个，包括全部 refill、
  restore、writeback、`psum_1RW_conflict`、`O8_full`、`final_output_sink_stall` 和 directory RMW，永远不能收费；
- frozen xorshift ready transcript 只被 static self-test 调用一次，production 不使用；公共 terminal FSM 也只按
  2/1029/1030/1031 总数批量加账，没有 state、clear index/hash、owner 或 close predicate；
- A1-OSG 的 persistent `feed()` 不收合同要求的每 bundle `BUNDLE_RETIRE`；context 用 insertion-ordered
  destination dict 代替固定 index，释放/复用后 `next(iter(bank))` 不再等价于最低 context index
  （331--393）。

因此 SC8/ISO8/OSG/PBR4 的整数 totals 不是同资源 cycle simulation；即使四点共享部分漏项，也不能以
“公共项抵消”补救，因为 context/order 会改变 refill、resident eviction、RMW conflict 和 traffic。

### P0-02｜数值、commit、traffic 和 GO/support 门均可产生假 PASS

decoder weight package 只在 preflight 做目录 seal 检查，production 从不打开任何 weight member。源码没有
signed INT8 product、Acc24 wrap、psum data、dense output data 或 commit sequence：

- `commit_hash` 初始化后从未 `update`，1600 行都会得到同一个空 SHA；
- `functional_mismatches` 与 `source_time_output_cycle_mismatches` 被 literal 写成 0，
  `output_value_evidence` 只是“future release required”字符串（485--496）；
- mandatory per-row occupancy/refill/backing/output stall/commit ledger 与 aggregate all-traffic 不存在；
- `go` 只检查 ratio-of-sums `>=1.30` 和十个 sample `>=1.10`（686--687）。它不检查 mismatch=0、PBR4
  weight active reads/refill bytes不增加、hidden state、group/RMW/commit 与 OSG 非等价，也不实现 psum/data
  `>=30%` support-only 门；结果甚至把 `group_rmw_commit_equivalence_kills_novelty=true` 与可能的
  `cpu_go=true` 同时写出（688--706）。

这违反 M559 `mandatory_outputs_and_gates`，足以把不正确或与 OSG 等价的候选升成 GO，属于发布阻断。

### P0-03｜N0--N9 runtime rehash 与 A1 freeze 未闭合

AUTH key set本身与 future schema 的 40 keys 完全一致，但 production preflight 并未消费全部 binding：

- `future_runner_schema_path/sha256`、四个 `contract_static_*` 字段以及 auth 的 `contract_path` 从未比较；
  auth `schema/status` 也只要求 key 存在，不要求 canonical value；
- `verify_review()` 只调用 runner-static、launch-candidate、final-release，没有验证 M562 contract-static；
  production 也不验证四个 resident-hit 与两个 terminal golden；
- descriptor 的 authorization/review path 被钉住，但 `wrapper_path` 未钉 canonical path，也未与 wrapper-review
  中的 canonical source path比较；相同 SHA 的任意副本可通过当前 parent/cmdline 检查（520--548）；
- A1-only receipt 虽写 manifest+outer seal，却在 PBR4 可见前只裸读 `selection.json`，没有
  `verify_directory()` 或冻结 receipt hash；长运行中 receipt mutation 不会按合同 fail/quarantine
  （663--676）。

这不满足“runner independently rehashes every authorization-bound earlier byte”和“A1 receipt double-sealed
before PBR4”的 closed identity，不能生成 launch admission。

## 3. P1/P2 findings

### P1-01｜exact cohort/ledger 常量被冻结但没有被执行核验

`EXPECTED_RAW_BITS=696240000`、`EXPECTED_REPLAY_BITS=926880000`、
`EXPECTED_DENSE_DESTINATIONS=11040000` 在 production AST 中 load 次数均为 0。`record_map()` 只要求 40 个
`(sample_id,module_index)` key，没有核验 schema、checkpoint/sequence、shape、layout、T10 byte length、layer
identity或总 replay/dense counts。sealed verifier directory 也只验 member bytes，不解析其 PASS/identity。
短读会失败，但错 shape、额外数据或错误冻结元数据不一定失败。

### P1-02｜post-attempt failure closure 存在 try 之外窗口

`attempt.mkdir()`、attempt receipt/write seal 与 `staging.mkdir()` 位于 `try` 之前（635--647）。这些步骤任一
异常会留下未封/半封 attempt 或 staging，且没有 `FAILED_OR_INCOMPLETE` 双封隔离。异常路径若目标 quarantine
已存在也只保留 staging，不满足 contract 的 mandatory quarantine。应从 attempt creation 前建立统一 trap/FSM，
并让每个 post-attempt exit 都产生可复验 failure receipt。

### P2-01｜内置 golden coverage 不能证明生产语义

static self-test 的 two-lane 检查只覆盖 SC8/ISO8 的总 cycle=18，不比较其 event/hash，并完全不覆盖
OSG=22、PBR4=21；terminal 只 hash 两个 byte string，不走 production terminal path。它可保留作 smoke test，
但不得作为四架构/terminal implementation admission。

## 4. 通过的静态项

- source contract、handoff、request、M559 r4、M562 PASS、r3/r2 和 M534 r2/r3/r4 imports 的 manifest、
  outer seal 与冻结 member SHA 全部匹配；六个 M534 normative member 独立复算全部 match；
- Python 源码 SHA=`3c5233772db02cb520f4cdfa7831f10a087555a2990530f1857936338e7c8e95`，
  在本机 `/usr/bin/python3` 3.6.8 原生 AST parse 通过；shell SHA=
  `54e5b1d066ff984b1f18fae329c42da16e88aa0606cc281a31df0a4725bd32f8`，`bash -n` 通过；
- shell 静态上只接受按固定顺序的六个 named argument/value pair；Python strict JSON 可拒 duplicate/nonfinite，
  directory/member/outer seal 基本逻辑成立；bitpack scan 是 seek+chunk、little-bit-first，block loop 会重放；
- independent ledger 复算 raw=`696240000`、replay=`926880000`；terminal string SHA 与合同一致，tail count
  为 nonlast=2、last/nonfinal-time=1029、time9/nonfinal-layer=1030、末 layer/非末 sample=1031、final=1031；
- canonical result、attempt、authorization、wrapper、candidate/final/wrapper reviews 均 absent；
  `docs/359_DATE终局冻结_20260813.md` SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 5. 唯一允许的下一步

只允许 source repair authoring，至少需要：

1. 一个真正逐 cycle 的 common FSM，并实例化四个 architecture table、M218 response/O8/1RW/RAW、weight
   refill、psum resident/backing、directory、shared link、dense output ready/stall 与 r4 terminal states；
2. 真实读取冻结 decoder INT8 weights，执行 contributor-order Acc24，生成并比较 group/RMW/commit/data hash，
   完整输出 per-row/aggregate traffic/conservation；
3. GO 必须逐条合取全部 mismatch、speed、每样本、weight/refill、hidden-state 和 OSG non-equivalence 门，
   support-only 单独判定；
4. 补齐 N0--N8 每个 auth binding/golden/canonical wrapper path 的 runtime rehash，并在 PBR4 开始前重验
   A1 receipt；统一 post-attempt failure trap；
5. 对新 source 重新做 fresh static hammer。当前 r1 不得 author N4 或正式运行。

本 hammer 未运行正式 analyzer/runner，未读取真实 trace/weight，未创建 result/attempt/authorization/wrapper，
未运行 RTL/EDA/训练/GPU/远端，也未修改被审 source、normative imports 或 `docs/359`。
