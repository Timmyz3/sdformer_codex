# M518 r11 matched Fixed-T10 ATLIF VCS receipt-blind 独立打铁 r1

日期：2026-08-27  
状态：`PASS_DIRECTED_FIXED_T10_VCS_BEHAVIOR__DC_PPA_SYSTEM_HEADLINE_NOT_ADMITTED`  
评分：**98/100**；P0/P1/P2 = **0/0/2**。

## 裁决

`results/m518_matched_fixed_t10_atlif_vcs_r11_exact_20260827` 可以准入为 **M518 matched Fixed-T10 ATLIF 的定向 RTL/VCS 行为证据**。本评审不采信作者 receipt 为事实来源，而是从冻结合同、r11 静态准入、原始 `compile.rc`、`sim.rc`、`compile.log`、`sim.log`、`assert.report`、TB/SVA 源码以及 seal 独立重建结论：

1. r11 author request、独立静态评审、one-shot launch admission、正向结果和 wrong-TB 负控的 member manifest / outer seal 全部通过；runner、fixed wrapper、admission、RTL、SVA、TB、filelist 与工具身份一致；
2. `compile.rc=0`、`sim.rc=0`，完整终端 PASS 恰好一次；未出现 assertion failure、`Offending`、error、fatal、watchdog、timeout 或 unknown 签名；
3. 冻结源码含 **51 个 assertion label、25 个 cover label**；运行报告含恰好 25 个 required cover，全部非零，且没有 runtime-disabled assertion；
4. `sealed_V01_V20` 定向 campaign 的独立数值 oracle、Q24 饱和边界、tuple 唯一性、FIFO/raw/result/context 守恒和故障隔离闭合；作者 receipt 所列 `numeric_mismatches=0` 可由 TB 的 fail-fast closure 与唯一 PASS 独立推出；
5. 周期锚点成立：每 tile 17 个 issue cycle，clean `N=1` 为 29 cycles，clean `N=4` 为 80 cycles；源码闭合还要求 extreme `N=2` 为 46、equality `N=1` 为 29；
6. wrong-TB-SHA 负控以 rc=10 在工具前失败，负控目录无 compile、simv、正向 receipt 或 `RUN_COMPLETE`；正向 `RUN_COMPLETE` 的纳秒时间戳晚于两套 manifest 与 outer seal，且冻结 runner 控制流只允许在四次 seal 自校验后创建它。

因此准入的是 **固定 seed、定向 stimulus 下的 RTL 功能、数值、协议、守恒和 cycle observation**。未准入 DC/Formality/STA/PTPX、PPA、功耗、能量、面积归一吞吐、完整网络/真实 trace、系统加速或 DATE headline。

## 身份与封存

- 合同 SHA256：`f0b8b2379138fa52d4abfe0b82884e8bfaf10d7a83ae7f1bc04badb071903690`
- r11 runner SHA256：`4e50a78cae0a4a05cad50865468e8321897d7ce74d851212551d5ccfa4d660a8`
- fixed launch wrapper SHA256：`798f433ff0ee790058b86b781e01de9fd021c0947cdf49c8bfcc0e95480c3650`
- r11 admission SHA256：`3b72f8b0da705b2f81b16b37ea9a643ea32f68322ad3237cfad9f92c198f6676`
- r11 admission member-manifest / outer-seal-file SHA256：`8e68dfb280a5edbf696ea4a045ddbf329f73cbdf373d9c6be615158a613c3a70` / `0dd5af5ee3330cbb9785bcdee2b7c2b95e96cdf7730971cebcc30ba8d2439e9f`
- r11 static review JSON / member-manifest / outer-seal-file SHA256：`13ddb58395083412338a6b314dc1c2c3b5c798a4305624d9d21a3dd13a4ce687` / `a2a1780b26ddd297f3f60e3930b1f0a708947933f6e02c393e174fb22dfe2e41` / `02c7323887d6f295f5ba584eb23e03f1154cdac7c94e9c5d99037da791b2508c`
- 结果 `RUN_MANIFEST.sha256` / outer-seal-file SHA256：`ecf79dec2f19a3d9d2c75507ee496905fd6e03d2044e764a39f06064259ce9ed` / `aaf12be32a72b7305af39c3e4e49022908ed45dc934c06716cdbf356a5dfd3d5`
- 结果 `SHA256SUMS` / outer-seal-file SHA256：`ecf79dec2f19a3d9d2c75507ee496905fd6e03d2044e764a39f06064259ce9ed` / `5c26dbdbdabd18c6c9e3e99279fcd7410c21bd43f00a88a41c6bc0f20e855384`
- wrong-TB 负控 member-manifest / outer-seal-file SHA256：`6b704017b5ec3b77952a5d935b652f92b71f5cd8a8e7596545ff6acd0b525079` / `7ff29f66ca3b9c21311e777f8ac68049de13df7a25e2423131910cefa221d5e9`
- 工具：`Synopsys VCS V-2023.12-SP1_Full64`
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；本评审未修改该文件。

r11 admission 的 `authorized_invocations` 是严格 JSON integer `1`，`vcs_authorized=true`、`dc_authorized=false`，并精确绑定上述 runner、wrapper 和 canonical result path。结果中的 42 条冻结输入均 independently observed=expected，零 mismatch。

## 原始执行证据

| 项 | 独立重算 |
|---|---:|
| compile rc | 0 |
| sim rc | 0 |
| 完整 PASS signature | 1 |
| compile warning / error | 0 / 0 |
| sim/assert bad signature | 0 |
| assertion labels / cover labels | 51 / 25 |
| assert-report cover rows | 25 |
| required cover 为 0 | 0 |
| runtime-disabled assertions | 0 |
| `RUN_FAILED_OR_INCOMPLETE` | absent |

`compile.log` 明确解析冻结的 RTL、SVA、TB，并以 `tb_m518_matched_fixed_t10_atlif` 为 top；`sim.log` 的编译器与 runtime 均为 `V-2023.12-SP1_Full64`。唯一终端行为为：

`PASS M518 matched Fixed T10 ATLIF sealed_V01_V20 clean_N1=29 clean_N4=80 ... slot_tuples_per_tile=1600 multiplier_slots=96 issue_cycles=17 vcs_only=true dc=false ... speedup=false ppa=false headline=false`

## 25/25 required cover

| cover | match | cover | match |
|---|---:|---|---:|
| `cp_first_issue` | 88 | `cp_first_close` | 87 |
| `cp_tail_close` | 83 | `cp_close_stall` | 802 |
| `cp_phase12_stall` | 1 | `cp_phase16_stall` | 246 |
| `cp_result_stall` | 1575 | `cp_fifo_full` | 1252 |
| `cp_full_pop_push` | 150 | `cp_raw_backpressure` | 1995 |
| `cp_release_wait` | 1374 | `cp_release` | 27 |
| `cp_context_retire` | 26 | `cp_fault` | 238 |
| `cp_zero_tile_fault` | 9 | `cp_config_frame_fault` | 221 |
| `cp_raw_frame_fault` | 8 | `cp_fault_with_pop_push` | 1 |
| `cp_dual_ready_oldest_bank1` | 1 | `cp_beat0` | 80 |
| `cp_beat1` | 77 | `cp_beat2` | 77 |
| `cp_beat3` | 77 | `cp_beat4` | 76 |
| `cp_reset_recovery` | 249 |  |  |

最低的四个 required cover（phase12 stall、fault+pop+push、dual-ready oldest-bank1）仍各命中一次，不是零覆盖 PASS。

## V01–V20、数值与守恒审计

冻结 TB 用独立整数 oracle 解码 100 个 signed INT8 weight、10 个 signed Q24 bias、一个 signed Q24 threshold 和 160 个 signed INT8 input；每个 row/lane 累加后做 Q24 saturation，再与 threshold 比较。所有结果以 tag、beat、valid bits 和 48-bit data 逐项 `!==` 比较，任一 mismatch 立即 `$fatal`。每个 context 释放前还要求 `expected_read==expected_write` 且 `numeric_mismatches==0`。

campaign 的 closure 同时要求：

- clean cycle：`N1=29`、`N4=80`、extreme `N2=46`、equality `N1=29`，与 `17*N+12` 一致；
- 4 个固定 seed 随机 context、2 个 rail context / 6 个 Q24 边界点；
- 每 tile 17 个 issue、1600 个唯一 tuple、96 multiplier slots，tail cycle 仅 64 active；
- config/raw/tile/stage1/product/result/context 账本精确守恒；
- 5 个 config frame attack、216 个 padding-bit attack、7 个 raw frame attack、1 个 zero-tile attack、1 个 fault-edge pop/push、9 个 reset attack、5 个 release-state attack；
- FIFO-full、same-cycle pop/push、phase12/phase16 close stall、raw/result backpressure、oldest-bank1 选择、sticky quarantine 与 reset recovery 均被命中。

上述门都在唯一 PASS 之前；任何一项不满足都会走 `$fatal`，所以从封存源码身份、rc=0、无 fatal 与唯一 PASS 可以独立准入定向数值/守恒行为。它不是形式化证明，也不是 checkpoint/full-network equivalence。

## 负控与完成顺序

wrong-TB-SHA 负控只有五个封存文件，`negative_preflight.rc=10`，唯一 mismatch 是 TB expected 全零、observed 为冻结 TB SHA。目录中不存在 `compile.log`、`simv`、作者 receipt 或 `RUN_COMPLETE`。冻结 runner 将该负控放在 `PREFLIGHT_COMPLETE` 和首次 VCS identity query 之前，因此它确实是 pre-tool fail，而不是工具失败后包装成负控。

纳秒时间戳顺序为：

`PREFLIGHT_COMPLETE` 17:38:08.848 → VCS ID 17:38:09.159 → compile rc 17:38:19.881 → sim rc 17:38:23.549 → artifacts complete 17:38:23.613 → 两套 manifest/seal 17:38:23.619–626 → `RUN_COMPLETE` 17:38:23.635。

`RUN_COMPLETE` 是 seal 自检之后生成的 terminal sentinel，按设计不属于前一刻冻结的 member manifest；其顺序由 exact-SHA runner 控制流和纳秒 mtime 双重佐证。

## P0/P1/P2

### P0 = 0

未发现 seal 破损、身份漂移、RC/PASS 矛盾、required cover 为零、数值或守恒错误、负控越过工具门、错误/unknown 签名或 claim 越界。

### P1 = 0

对本次 **定向 VCS 行为准入** 未发现高风险缺口。

### P2 = 2

1. VCS 的 `assert.report` 列出 25 个 cover 的 attempts/matches，但不列 51 个 assertion 各自的 attempt/non-vacuity 统计。51 个 label、SVA 编译、零 disable 与零 failure 成立；这仍只支持定向 campaign 内“未观察到 assertion failure”，不支持形式化或全面 non-vacuity claim。
2. V01–V20 以一个 fail-fast campaign 和一个终端 closure signature 汇总，没有逐 V 编号的独立 runtime receipt。冻结源码与 47 个 `$fatal` oracle 使假 PASS 风险可控，但后续若调试回归，逐 phase 小结会提高可诊断性；不要求重跑本次 VCS。

## Claim boundary 与下一步

准入：matched Fixed-T10 ATLIF 的定向 RTL compile/simulation、V01–V20 固定 campaign、独立整数数值 oracle、协议/故障隔离、守恒、required cover，以及 `N1=29`、`N4=80`、17 issue cycles/tile 等 RTL cycle anchors。

不准入：production/full-network equivalence、真实 checkpoint/trace、DC、Formality、STA、PT/PTPX、area、power、energy、PPA、throughput/mm²、系统加速或 headline。

允许的后续是为 matched Fixed-T10 与 rank-3 对照建立新的、双封的 DC static admission；不得把本次 VCS receipt 直接升级成 PPA 或系统性能数字。
