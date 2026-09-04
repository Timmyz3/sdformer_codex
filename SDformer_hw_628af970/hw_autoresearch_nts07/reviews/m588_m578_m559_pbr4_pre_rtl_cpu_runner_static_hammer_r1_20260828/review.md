# M588｜M578 / M559 PBR4 r5 immutable CPU source fresh static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_SOURCE_HAMMER__NO_FORMAL_CPU_GPU_EDA_REMOTE_RUN`  
裁决：`FAIL_SOURCE_STATIC__NO_LAUNCH_CANDIDATE__REPAIR_REQUIRED`  
评分：**42 / 100**，`P0/P1/P2 = 3/2/1`

## 1. 范围与裁决

本次完整读取并复算 M578 request、r2 source contract、author handoff/future schema、r5 analyzer/runner，
递归核对 M559 r4、M552 r3、M545 r2、M542 r1、M562 PASS 与 M574 FAIL。只执行 Python 3.6
AST/compile、`bash -n`、内置 synthetic static self-test，以及不读取真实 payload 的四架构最小 resident-hit
golden。没有运行正式 analyzer/runner，没有读取 M511 或 decoder weight payload，没有创建 result/attempt，
没有运行 RTL/VCS/DC/PT/PTPX/Formality、训练、GPU 或远端。

r5 相比 M565/r4 有实质修复：它打开 signed-INT8 weight、逐 contributor 做 Acc24 wrap，建立 psum/
directory/backing/output data，四个 architecture 的最小 resident-hit cycle 数也能得到冻结的
`18/18/22/21`。但是冻结合同要求的不只是“局部 cycle 数能对上”；当前 source 仍可在错误 ready
transcript、共享错误的 reference、缺失守恒账本或硬编码 mismatch 下生成 `PASS_CPU_GO`。因此 P0/P1
不为零，禁止进入 launch-candidate authoring 或正式 CPU 重放。

## 2. P0 findings

### M588-P0-01｜common cycle / terminal FSM 未执行冻结 transcript

1. `CycleLedger.sink_ready` 在 analyzer 256--257 行实现为 `bool(ready_state & 1)`。递归冻结的 M542
   sink transcript 是“低三位仅在 `000` 时 not-ready”，即应为 `bool(ready_state & 7)`。独立 synthetic
   展开在状态 `0x9bfc2d14`、`0xedff3346`、`0x7754284a` 等点已经出现合同 ready=1、r5 ready=0；
   dense-output stall 和所有后续绝对 cycle 坐标因此改变。
2. r5 的 `terminal_tail()` 只顺序调用 `step()`：没有 `terminal_state`、block owner、committed-block
   bitmap、clear index/write-count/hash，也没有 terminal-owned directory-port acquire/release。它可以复现
   2/1029 等总数，却不能执行 M559 T00--T14 的 close predicate、错序/重复/漏边 sticky mismatch。
3. `verify_goldens()` 仅重哈希 contract 内的六个静态字符串，未把 production architecture machine 的
   event sequence 与 golden 比较。独立最小展开虽然 cycle 数为 SC8/ISO8/OSG/PBR4=
   `18/18/22/21`，但 production labels 是泛化的 `GROUP_LOCK`、`INGRESS_MOVE`、`RETIRE_SLICE5`；
   冻结 ISO8 需要 `ADJACENT_PAIR_GROUP_LOCK`，OSG/PBR4 还要求 lane/context/完成标签。production
   `cycle_sequence_sha256` 没有任何 golden oracle，因此实现漂移不会 fail。
4. 外链和端口 `acquire()` 都在一个同步 Python 调用内立即 release；`atomic_ingress_backpressure`、
   link stalls、O8/RAW/phase/psum/directory conflict 没有一个显式的 common priority-state scheduler。
   在当前合法调用拓扑中多个冲突 predicate 结构上恒假，不能证明 M559 priority 0--9 的 shared-resource
   arbitration，而只是把长操作串行批量收费。

影响：四点 totals 不是递归冻结的同一 ready/terminal/common-FSM cycle coordinate；即使未来 ratio 看起来
很好，也不能作为 M559 的可执行周期结果。

### M588-P0-02｜功能 reference 与候选共享错误域，typed/source/协议 mismatch 仍可被硬编码掩盖

1. `ReferenceAccumulator` 与 candidate 共用同一个 `event_taps()` descriptor 流和同一个
   `WeightSet.get()` 地址函数。它只独立保存第二份 vector；若 descriptor geometry、kernel/weight 地址或
   typed-source 编码一起错，reference 与 candidate 会同错并保持 mismatch=0。这不是冻结合同要求的独立
   direct-convolution/reference domain。
2. descriptor identity 只含 output block、ordinal、channel、kernel、destination，没有 M545 明确要求的
   `numeric_activity=1` 与独立 `source_sign_bit=0` typed fields；源码也没有任何 malformed sign path。
3. 行结果的 `source_time_output_cycle_mismatches` 在 1115 行仍被 literal 写成 0。没有比较四点的 source/
   descriptor accept/retire/frontier/weight sequence、dense commit count/order、output-data hash，也没有
   terminal transition mismatch。`protocol_mismatches` 只覆盖“没有 pending descriptor 却 retire”这一种错误。
4. final output 没有显式 owner/address/beat/beats-remaining command state，因此无法计算 frozen protocol 中
   duplicate、wrong address/owner/beat、stall-side-effect 和 cursor-advance mismatch。

积极项是 signed byte 解码、`+1` 乘 weight、每次加法 `wrap24()`、384-B padded output、commit/data SHA
均已真实存在；但这些不能替代独立 oracle 和完整 mismatch。

影响：共享 geometry/address bug、typed-source bug、output protocol/terminal bug 可在
`exact_gate = (mismatch_total == 0)` 下假 PASS。

### M588-P0-03｜mandatory traffic/conservation 与 GO/support 合取不完整

M542/M545/M552 要求每个 sample-layer-time-architecture 行输出 source bits/active reads/logical+padded
bytes、frontier、descriptor accept/retire、group histogram、refill requests/beats/bytes、psum/backing traffic、
dense commit、directory、occupancy/capacity 与全部 conservation mismatch。r5 每行只有 primary counters、
若干 sparse `events` 和 hashes：没有 source-scan byte ledger、group utilization、refill request/beat、persistent
command/beat、commit count、resident hit/miss、最大 ingress/context/O8/FIFO/pending occupancy 或 capacity
predicate。

具体 false-pass 路径：

- `EXPECTED_REPLAY_BITS=926880000` 和 `EXPECTED_DENSE_DESTINATIONS=11040000` 只被抄进 result，production
  没有和执行计数比较；raw bits 只依赖外部 verifier receipt，`EXPECTED_ROWS=1600` 是唯一完整 population
  assertion。
- 没有验证四架构 source common-ledger equality、descriptor/commit conservation、`resource_owners`/contexts/
  output owner 终态或 239636<=245760 的运行 gate。
- `go` 只合取不完整的 mismatch、aggregate/sample speed、weight bytes/reads 与 OSG non-equivalence；冻结要求的
  “no hidden state/port/bandwidth/lookahead/oracle”和全部 conservation 不在 predicate 中。
- `support_only` 同样基于不完整 exact gate 和仅 psum event counters，不能证明“同一完整 T10 ledger 下
  >=30%”。

影响：账本漏项、隐藏状态、回放/commit 数错误时仍可能发布 `PASS_CPU_GO` 或
`PASS_CPU_SUPPORT_ONLY`，属于 launch 阻断。

## 3. P1 findings

### M588-P1-01｜N0--N9 identity bootstrap 仍有可替换/弱验证窗口

- shell runner 只检查 analyzer 是普通文件，然后直接 `exec /usr/bin/python3`；它不在 exec 前核对冻结
  analyzer SHA。analyzer 的“自哈希等于 authorization”发生在被执行之后，被替换的 analyzer 可以直接删掉
  该检查。后续 wrapper 必须独立冻结并预哈希 analyzer+shell 两个 source，而不能只信 analyzer self-check。
- `verify_review()` 对 contract/source/candidate/final review 只要求 `score=100,p0=p1=0` 和 auth hash binding，
  不验证各 review 的 canonical schema/status/launch predicate/exact key set。一个被 authorization 绑定但语义
  不对应阶段的 100 分 JSON 仍能通过。
- source-static 的 future canonical path 是
  `reviews/m578_m559_pbr4_pre_rtl_cpu_runner_static_hammer_r1_20260828`；本 M588 失败证据不填充该路径，
  后续不得把本 review 重命名或复制成 PASS N2。

### M588-P1-02｜成功 publish 后的异常不能闭合 canonical output

统一 `try` 已覆盖 attempt 创建，pre-attempt failure 也确实零落盘；staging 在 publish 前双封并验证。这些修复
成立。但 `os.replace(staging, output)` 之后若最终 `verify_directory(output)` 抛异常，`failure_close()` 只处理
attempt 和仍存在的 staging，从不隔离/移走 canonical output。更直接地，`weights.close()` 位于 `finally`；若
它在成功 rename 后抛异常，该异常不会重新进入同一个 `except`，会留下 canonical result + consumed attempt，
同时 runner 以 failure 退出。故“任一 post-attempt failure 都是 attempt+quarantine 双封且 canonical result
absent”尚未成立。

## 4. P2 finding

### M588-P2-01｜内置 static self-test 的 primary closure 是恒真式

1582--1584 行检查 `name in terminal.classes or name in PRIMARY_CLASSES`；循环变量本就来自
`PRIMARY_CLASSES`，所以第二项恒真。内置测试没有覆盖 27 类 reachability、四 architecture production event
SHA、正确 low-three-bit ready、terminal state/index/hash、reference independence、traffic conservation 或
failure injection。它只能保留作 smoke test，不能作为 source admission。

## 5. 通过项与独立检查

- M578 request/handoff/source contract、M559/M552/M545/M542 contract、M562 与 M574 review 的 member/
  manifest/outer seals均匹配；analyzer SHA=`51458810a498b7fa335672a2221e496cc8a7ade254ed01eb1567e9d6751fe0b6`，
  runner SHA=`a2e22f102797827c236a5fe588571347958269e6381c2d236abdd486da61b447`。
- `/usr/bin/python3` 3.6.8 native compile/AST PASS，`bash -n` PASS，内置 self-test 输出
  `PASS M578 M559 repaired immutable analyzer static self-test`。
- independent synthetic resident-hit expansion 的 cycle 数为 SC8=18、ISO8=18、OSG=22、PBR4=21；
  xorshift GF(2) jump、Acc24 boundary wrap、M523 9-tap phase-major smoke test通过。
- M511/weight schema、member size/hash、strict duplicate/nonfinite JSON、目录 member/outer seal、六对 CLI
  surface、canonical input/result path、wrapper PID/starttime/cmdline 的 source checks均存在。
- canonical result、attempt、authorization、wrapper、candidate/final/wrapper reviews仍 absent；
  `docs/359_DATE终局冻结_20260813.md` SHA仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 6. 唯一允许的下一步

只允许产生新的 source identity（不得覆盖 r5）：

1. 改为 `ready = (state & 7) != 0`，建立可执行的 common priority/terminal state，并让 production synthetic
   trace逐字对上六个 golden，包含 terminal clear index/hash与 owner/bitmap mismatch；
2. 建立不共享 `event_taps/WeightSet.get` 的 direct reference，descriptor hash显式加入 typed `value=1/sign=0`，
   计算 source/time/output/cycle/transition/protocol mismatch；
3. 补齐 mandatory per-row/all-traffic/occupancy/capacity/conservation，所有 696.24M/926.88M/11.04M/1600
   常量必须由执行计数断言；GO/support逐条合取；
4. runner/wrapper 在 exec 前预哈希 analyzer+shell，review stage schema/status精确验证；把 publish、final verify
   和 `weights.close()` 纳入能隔离 canonical output 的 failure FSM；
5. 对新 source 再做 fresh hammer。当前 M588 不授权 launch-candidate、正式 CPU、RTL 或任何性能/traffic claim。

