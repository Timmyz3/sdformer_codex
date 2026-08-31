# M519 registered-release VCS receipt-blind 独立打铁复审 r2

## 裁决

**97/100，P0=0，P1=4。M519 registered-release 的 K1 / K8 / K1x8 定向 RTL 功能正确性可以准入；DC 仍未授权，必须先统一独立 review 路径，再另建并双封 post-VCS DC launch-admission。**

本复审没有运行 VCS、simv、DC、Formality、PT 或 PTPX，也没有修改结果、runner、RTL/SVA/TB/filelist/Tcl。裁决只适用于已封存的两套 directed VCS 测试和其中的 component cycle rows；不构成 complete FC2、无组合环、PPA、功耗、全系统加速或论文 headline 证据。

## 双封、拓扑和身份

- 正向 canonical 的 `SHA256SUMS` 与外层 seal 均通过；清单精确覆盖除两份 seal 文件外的 191 个 regular file。四个 VCS 生成 symlink 都是相对路径，解析后仍在 canonical 内。
- wrong-runner-SHA negative preflight 的双封通过，清单精确覆盖四个 regular file。子进程以 exit 3 在 runner-SHA 门处退出，stderr 为 `M519 caller must pin the independently reviewed VCS runner SHA`，记录的 VCS invocation 为 0。
- negative receipt 中 `positive_canonical_result_absent=true` 是 preflight 发生时的历史事实；正向目录随后按合同创建，不把该字段误读成当前文件系统断言。
- 正、负 receipt 均可解析且所有数值有限。当前文件重新核验 `input_sha256.txt` 全部通过；r3 contract、VCS runner、r3 static hammer、旧 failure hammer、negative preflight 和 docs/359 身份闭合。
- docs/359 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。

## Receipt-blind 日志复算

两套测试均从各自 `sim.log` 直接读取得到 `compile.rc=0`、`sim.rc=0`；编译和运行日志都标识 `Synopsys VCS V-2023.12-SP1_Full64`。

主测试 PASS 行独立给出：clean 10、reset 2、protocol attack 4、numeric/tuple/weight mismatch 全 0、same-edge release violation 0；request/response-injection/result/raw stall 分别为 1363/3143/47/4509，next-cycle slot/context reuse 分别为 7850/322。

equal-bandwidth PASS 行独立给出：clean 10、reset 2、protocol attack 4、numeric/tuple/weight mismatch 全 0；request/result/raw stall 分别为 375/45/1165。两份 `sim.log` 和 assertion report 都没有 assertion-failure、offending、fatal、watchdog 或非零 mismatch 签名。

从日志直接复算的三轴周期如下，两个测试对同一 K1x8 路径给出完全相同周期：

| B | events | K1 cycles | K8 cycles | K1x8 cycles |
|---:|---:|---:|---:|---:|
| 1 | 20 | 259 | 51 | 53 |
| 2 | 41 | 737 | 131 | 133 |
| 4 | 90 | 3153 | 486 | 499 |
| 8 | 110 | 7569 | 1231 | 1246 |
| 1 | 0 | 14 | 14 | 14 |

这些是 component directed cycles；不得外推成 H67 全网倍速。

## Retained cover 与两项 invalid cover

主测试仍保留并命中 candidate service 的 same-cycle distinct release、next-cycle slot/context reissue、result stall 和 done；关键 match 数为 5607、4144、180、24、5。M499 adapter 保留并命中 pending-request stall 1261、out-of-order response 3163、cutthrough response 7464、protocol attack 4。八个 baseline lane 的所有 required service cover 均非零。

equal-bandwidth 测试的 top、K8 service、M490 adapter 以及八个 baseline lane 的全部 required cover 均非零。

旧失败定位的两项 cover 没有被冒充：

- 主测试 candidate service `cp_protocol_fault_rise` 仍为 0 match，runner 不再把它当 required gate，receipt 也没有宣称它命中。
- 主测试 candidate M499 `cp_retire_then_slot_reuse` 仍为 0 match，runner 不再把它当 required gate，receipt 也没有宣称它命中。
- equal-bandwidth 中 `M490.cp_retire_then_slot_reuse=134` 属于另一 adapter 实例和另一测试合同，不能替代或洗白 M499 的 0 match；本裁决没有这样使用它。

因此 r3 修复保持为 runner cover-domain 纠偏，没有放松数值、事务、stall、protocol attack 或 registered-release next-cycle reuse 的功能门。

## 旧输出未复用

旧 r1 失败目录的当前 tree fingerprint 仍为封存的 `8e66bed...`。r1 与正向 r2 之间 regular-file inode 交集为 0；新主编译时间为 13:59:54、新 equal-bandwidth 编译时间为 14:00:08，compile/sim 日志含新 r2 输出路径和新 VCS 进程归档号。主 `assert.report` 因同 seed 和确定性 cover 计数而与旧诊断 hash 相同，但 inode、mtime 不同，且新 compile/sim 证据证明它是重新产生，不是 hardlink/copy。equal-bandwidth 也在新目录完整编译与仿真。

## 准入边界和下一门

本复审授权：

- M519 registered-release K1/K8/K1x8 在这两套 directed VCS/SVA 负载下的 RTL 功能正确性；
- 上表同一 M519 身份下的 component cycle rows。

本复审不授权：

- DC、Formality、PT、PTPX 或功耗；
- combinational-loop-free、complete FC2、paper PPA ready；
- 系统倍速、能效或 DATE headline。

当前 `m519_fc2_registered_release_dc_launch_admission_r1_20260827.json` 仍是 blocked，要求的新 r2 launch-admission 不存在，也没有 M519 DC canonical。此外，本任务指定的实际 review 路径是 `reviews/m519_registered_release_vcs_receipt_hammer_r2_20260827/`，而当前 DC runner 硬编码读取 `reviews/m519_registered_release_vcs_hammer_r2_20260827/`。现有 DC runner 因此不能消费本 review。

DC 的合法下一步必须是：先统一 review 路径并对改后的 DC launch chain 重新静态复审，再另建一份新的独立 DC launch-admission，绑定本 review、正向 VCS receipt/outer seal、最终 DC runner/Tcl 和 docs/359；只有它可以授权一次 DC。不得通过复制本 review 到另一个目录来绕过身份修复。

## P1

1. `authorized_vcs_invocations=1` 指一次 outer campaign；该 campaign 内实际包含两次全新 compile/sim。语义已由 r3 static contract 定义，但后续表格宜写 `one campaign / two tests`。
2. VCS 动态输出文件名没有预先白名单；完成后的全 regular-file manifest 与 confined symlink 已关闭实际拓扑。
3. assertion pass 由 rc=0、无失败签名和 required cover 非零共同建立；report 没有单独的 aggregate assertion-fail counter。
4. 本任务要求的 `vcs_receipt_hammer` 路径与当前 DC runner 硬编码的 `vcs_hammer` 路径不同；DC 必须先修复路径、重做静态准入，不能直接启动。
