# M1854｜M1845 C2 fresh-mapped production-energy 唯一失败独立审阅

结论：**审计 PASS（99/100），M1845 生产准入 FAIL_CLOSED；P0=0、P1=1、P2=0。M1845 已消费且 `automatic_retry=false`，不得重跑；不得产生或引用 C2 mapped power/energy 数字。**

## 执行与封存

- attempt latch 与 ordinary failure quarantine 均通过目录内层 manifest 和外层 seal 的独立校验。
- M1849 exact launch release 及 M1853 different-author release audit 均双封一致；唯一执行的固定预算为 `license=1 / VCS compile=2 / simv=10 / SAIF=10 / PTPX=10`，禁止自动重试和复用旧 `simv`。
- 实际消费数量严格为 `license=1 / VCS compile=2 / simv=1 / SAIF=0 / PTPX=0`，失败相位为 `SIM_k8_0`，异常类型为 `RuntimeError`。
- K8 与 K1x8 两个映射 VCS compile 均完成并生成各自 `simv`。仅 K8/case0 被执行；canonical result 不存在。
- raw build 只保留在 `private_build.unsealed_do_not_cite`。本审阅只对其中两份 compile log、K8/case0 runtime log 和 assertion report 做只读取证并冻结其当时 SHA；没有把 private build 封成可引用结果。
- 本审阅没有启动或查询 VCS、simv、SAIF、PTPX、DC、PT、Formality、license、GPU 或远端，也没有修改 RTL、TB、runner、docs/359 或 `ucli.key`。

## 故障事实

K8/case0 在 30,000 ps 已打印 `M979_SAIF_WINDOW_START` 并执行 UCLI `power -enable`，随后 `m1831_c2_registered_public_fault_production_assertions.sv:48` 因聚合向量

`{protocol_error, numeric_overflow, stale_response_seen, endpoint_fault[7:0]}`

含 X/Z 而 `$fatal`。因此：

- SAIF 窗口只被打开，没有正常关闭，也没有生成 SAIF 文件；
- source/source-packet cover 各命中一次，但 endpoint、result/commit 与 done cover 均为 0；
- 没有完成任何 production workload，也没有得到数值结果、功能 PASS、功耗或能量结果；
- 这不是缺少 PASS token 或 parser-only 的问题，而是 gate-level fault boundary 在真实 workload 开始阶段不满足二值性。

## 最窄可信 X/Z 边界

**直接证据边界**只能落到上述 11-bit 聚合向量，M1845 日志和 assertion report 没有逐位打印，故不能直接声称具体是哪一个 public 输出或哪一个 endpoint 位。

**静态源码推断**更支持 public mapped 输出侧：TB-only memory model 的 `endpoint_protocol_fault_q` 在 reset 分支明确写 `0`，之后所有主动写入均为常量 `1`，其余路径保持原值；M979 reset 覆盖多个有效时钟沿。按标准 SystemVerilog 语义，该状态在 reset 后不应自行产生 X。相反，三个 public 输出来自映射网表，可能由组合锥或寄存器 D 锥传播未知。因此最可信的候选集合是 `{protocol_error, numeric_overflow, stale_response_seen}`。

这仍是 **inference，不是 admission-grade localization**：层级 tap 的编译/可见性没有逐位运行时证据，三个 public 输出也没有逐位采样。不得用此前不同 netlist 的诊断替代 M1845 定位，更不得直接删除 `endpoint_fault` 检查来制造功耗 PASS。

## 论文边界与后续

M1845 不产生 mapped functional、production SAIF、PTPX、component power 或 component energy 准入；也不影响已经独立闭合的 C2 面积/时序/等带宽周期证据。

唯一合法后续是在**新 additive diagnostic namespace**中先逐位打印三项 public 输出和八项 endpoint fault，并保持原聚合 fatal 不被绕过。若证明确为 public mapped 输出，再修复其 reset/unknown 传播根因并重新走 different-author source review 与 exact launch release。M1845 本身永久 `FAILED_DO_NOT_CITE`、不得重试。
