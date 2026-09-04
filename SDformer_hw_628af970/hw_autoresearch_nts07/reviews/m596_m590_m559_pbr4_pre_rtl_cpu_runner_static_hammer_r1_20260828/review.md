# M596｜M590 / M559 PBR4 r6 immutable CPU source fresh static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_SOURCE_HAMMER__NO_FORMAL_CPU_GPU_EDA_REMOTE_RUN`  
裁决：`FAIL_SOURCE_STATIC__NO_LAUNCH_CANDIDATE__R7_REPAIR_REQUIRED`  
评分：**56 / 100**，`P0/P1/P2 = 3/2/1`

## 1. 范围与裁决

本次完整读取 M591 request、M590 r3 source contract、author handoff/future schema、r6 analyzer/runner、
M588 FAIL 与递归冻结的 M559/M552/M545/M542 合同和 goldens。只执行 Python 3.6 AST/compile、
`bash -n`、内置 static self-test、production-bound 六组 synthetic golden 和临时 fault injection；没有运行
正式 analyzer/runner，没有读取真实 M511 或 decoder weight payload，没有建立 canonical result/attempt/launch，
也没有运行 RTL/VCS/DC/PT/PTPX/Formality、训练、GPU 或远端。

r6 对 r5 有实质进步：`low3!=000` ready 已修正；四个 resident-hit event transcript 与两个 terminal
compact golden 都能由 production classes 精确重放；candidate descriptor 已显式携带 `+1/sign=0`；direct
reference 使用独立 mmap/offset/signed-decode/kernel loop；大部分 mandatory traffic counter 与 696.24M/
926.88M/11.04M/1600 aggregate assertion 已落地；shell 会在 exec 前校验 analyzer SHA；普通 post-rename
异常也会隔离 canonical output。

但 source 仍可在 descriptor 身份错误、terminal 状态错误、隐藏端口/容量超限或 output protocol 错误时得到
零 mismatch，并进入 `PASS_CPU_GO`/`PASS_CPU_SUPPORT_ONLY`。因此 P0/P1 不为零，本 review 不得复制或
重命名成 future canonical N2 PASS，也不允许 launch-candidate authoring 或正式 CPU replay。

## 2. P0 findings

### M596-P0-01｜terminal/common FSM 仍只对上 compact 拍数，没有执行冻结状态与全局优先级语义

1. `terminal_tail()` 的 `state`、clear counter 与 committed bitmap 都是一次函数调用内的局部对象；
   `simulate_row()` 又为每个 sample/layer/time/architecture 重建 `CycleLedger`。time/layer/sample/cohort receipt
   没有跨 row 保存和核对，owner 也只是外层 loop 参数。因而 T10/T11/T12/T13/T14 的“验证十个 time、
   四个 layer、十个 sample receipt 后再迁移 owner”没有被执行。
2. 冻结 T05 要求 word1023 接受后把 state 从 `DIRECTORY_CLEAR_WORD` 置成 `DIRECTORY_CLEAR_END`，T06 再从
   `DIRECTORY_CLEAR_END` 迁移到 `TIME_RETIRE`。r6 在 1024 次 Python loop 后 state 仍是
   `DIRECTORY_CLEAR_WORD`，随后 1286 行直接调用
   `edge("DIRECTORY_CLEAR_WORD", "TIME_RETIRE", ..., "DIRECTORY_CLEAR_END")`。compact label/count 是
   1029，但冻结的 prior-state/next-state 语义被绕过。
3. 没有 cycle-at-a-time common priority scheduler。weight refill、restore、writeback 分别占用三个不同
   `resource_owners` 名字，而冻结坐标只有一个 external link；busy `acquire()` 会先收费一拍再立即抛
   `resource re-entry`，不会保持 payload 并在下一拍重试。临时 fault test 对已占用 directory port 得到
   `cycles=1` 后直接 `ContractFailure`，证明 priority-2 stall/hold/retry 未实现。

影响：六个最小 golden 都 PASS 仍不足以证明正式 1600 行运行在 M559 冻结的同一资源/同一状态机 cycle
coordinate 上；绝对周期与倍率不可准入。

### M596-P0-02｜descriptor 与 output protocol miter 仍存在可复现 false-pass

1. `FrontierTracker` 只按 source ordinal 保存 remaining count，不保存 accepted descriptor identity multiset。
   `descriptor_accept_sha256` 与 `descriptor_retire_sha256` 虽写入 row，却从未比较或进入 exact/GO gate。
   临时 synthetic fault 先 accept 两个同 ordinal、不同 destination 的 descriptor，再 retire 第一个两次；结果
   `accepted=2, retired=2, protocol_mismatches=0, frontier_closed=true`，只有两个 hash 不同，而 r6 仍不报错。
2. `transition_mismatches` 除初始化和输出外没有任何 increment；
   `source_time_output_cycle_mismatches` 仅为 `conservation_mismatches + transition_mismatches`，没有独立的
   source/time/output/cycle oracle。
3. final-output hold check 是先 `held=command` 再比较 `command!=held`；address check 是把 address 赋成同一公式
   后再与该公式比较。两者是恒假式。没有独立 accepted owner/address/beat/beats-remaining/data hash、cursor
   model 或可注入 duplicate/wrong-owner/wrong-beat transition，因此错误 output FSM 可保持 mismatch=0。

积极项：independent reference 确实不调用 `event_taps()`/`WeightSet.get()`，signed INT8、typed `+1/sign=0`、
per-contributor Acc24 wrap 与最终数值比较均存在。但独立 weight/data oracle 不能替代 descriptor/protocol oracle。

影响：同 ordinal descriptor 替换、错序/重复 retire、output owner/address/beat/stall-side-effect 错误仍可满足
`exact_gate`，构成直接的 `PASS_CPU_GO` false-positive。

### M596-P0-03｜hidden-resource/capacity gate 与 mandatory aggregate ledger 未闭合

1. `hidden_resource_gate` 只有 `239636<=245760` 常量和 `final_state_empty`，没有把 row 中已经统计的
   `max_occupancy_ingress/contexts/context_slots/O8/FIFO4/pending_write/resident_destinations` 与冻结容量逐项比较。
   这些 counter 超限不会影响 GO/support。
2. 冻结资源是一个 `128 B/cycle` external link；r6 分成 `weight_refill_link`、`restore_link`、
   `writeback_link`，final output 又没有占同一 owner。也没有执行性的“无 candidate-only port/bandwidth/
   lookahead/prefetch/oracle”检查；把 modeled byte 常量抄进 row 不能证明隐藏资源为零。
3. aggregate 确实断言总 400 rows/architecture、926.88M replay、11.04M commit，但缺少冻结 r2 导入的
   per-sample-layer-T10 exact ledger/every-layer ratio 输出；common hash gate 也未覆盖 descriptor-retire、terminal
   transition 或 accepted-output protocol identity。
4. `go`/`support_only` 直接合取上述不完整 `hidden_resource_gate` 与 mismatch，因此 occupancy 超限、单一外链
   被拆端口或 protocol hash 不一致时仍可能为真。

影响：同资源和完整 traffic/conservation 仍未成为 decision predicate，正式运行会有不可引用的假准入风险。

## 3. P1 findings

### M596-P1-01｜direct-runner wrapper attestation 仍可由假 parent cmdline 冒充

shell 的 analyzer prehash 已通过临时篡改测试（退出 66），四阶段 review 也检查 schema/status/100/0/0 与
md/json/manifest/outer。剩余问题在 `verify_wrapper_descriptor()`：它只要求 parent cmdline 的任意 argv 成员
包含 canonical wrapper path。任意父进程可把该路径作为无关 dummy argument，自己填写当前 PID/starttime 和
read-only descriptor，再直接执行 runner；源码没有验证 wrapper path 是父进程实际执行的 script argv 位置/
解释器对象。`verify_review()` 还允许 review JSON 任意额外 key 和两种 p0/p1 字段位置，不是 exact stage key
set。N8 的“只能由 reviewed wrapper 调用”尚未 fail-closed。

### M596-P1-02｜dangling canonical symlink 绕过 absent/quarantine predicate

`preflight()`、`failure_close()` 都用 `Path.exists()` 判断 canonical output。临时 fault test 将 canonical output
建成指向不存在目标的 symlink；`failure_close()` 返回成功，`exists=false`，但 `is_symlink=true` 且
`lexists=true`，canonical 路径仍在。需要对 result/attempt/staging/quarantine 全部使用
`os.path.lexists()`/显式拒绝 symlink，并让任何 post-attempt failure 最终断言 canonical path neither exists
nor is symlink。

普通 post-rename directory 异常路径已通过：canonical 被移到唯一 quarantine，attempt 和 quarantine 均重封。

## 4. P2 finding

### M596-P2-01｜内置 self-test 没有覆盖刚暴露的 false-pass

`static_self_test()` 对四架构只检查 `len(production_resident_hit_trace)==18/18/22/21`；精确 event compare 是
preflight 的另一个 helper，后者本次独立调用已 PASS。self-test 没有 descriptor duplicate/replace、terminal
T05 prior-state、single-link contention、occupancy overflow、output hold/owner/beat 或 dangling-symlink fault。
建议把这些小型 fault tests 收入下一版 source 自检，但不能用自检替代 fresh hammer。

## 5. 通过项、身份与零运行

- source contract、M591 handoff/request、M588、M559/M552/M545/M542 递归合同/review 的 member manifest 与
  outer seals均匹配；analyzer SHA=`5550dfb032ad2c43752137c3c1038a97228a2d5265697c6d89d54425d904ccf1`，
  runner SHA=`2c4c49a25266a5d5edf5c38d3193278bc55df958a0ce92bbc7049105864f2b01`。
- `/usr/bin/python3` 3.6.8 compile/AST、`bash -n`、内置 self-test PASS；production-bound six-golden helper
  精确 PASS，resident-hit 为 SC8/ISO8/OSG/PBR4=`18/18/22/21`。
- low3-ready、typed descriptor、独立 direct reference、signed INT8/Acc24、执行型总 cohort assertion、runner
  pre-exec analyzer hash 和普通 post-publish quarantine 是本轮确认关闭的修复。
- canonical M590 result、attempt、authorization、wrapper、candidate/final/wrapper reviews仍 absent。
- `docs/359_DATE终局冻结_20260813.md` SHA256 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 6. 唯一允许的下一步

只允许新建 r7 source identity，不得覆盖 M590 r6：

1. 建立跨完整 architecture point 持久的 terminal owner/state/receipt FSM；显式实现 T05
   `WORD→CLEAR_END` 和 T06 `CLEAR_END→TIME_RETIRE`，并用一个 cycle-at-a-time common priority machine 和
   单一 external-link owner 执行 stall/hold/retry；
2. 为 accepted descriptor 建独立 identity multiset/sequence oracle并与 retire 比较；为 output 建独立
   owner/address/beat/hold/retire/cursor oracle，所有 mismatch 实际 increment 且进入 exact gate；
3. 将 occupancy 上限、单一端口/链路、全部 mandatory per-layer/T10 ledger 与 common protocol/terminal hash
   纳入 hidden-resource、GO 和 support predicates；
4. wrapper attestation 验证真实父进程执行关系；所有 absent/failure path 使用 lexists + no-symlink；
5. 新 source 再做 fresh independent hammer。当前 M596 不授权 launch-candidate、正式 CPU、RTL 或任何
   performance/traffic claim。
