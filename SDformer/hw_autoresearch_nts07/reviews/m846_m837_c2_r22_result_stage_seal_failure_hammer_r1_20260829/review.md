# M846 — M837/C2 R22 RESULT_STAGE_SEAL 失败独立打铁

## 裁决

**PASS100（仅针对失败分类与证据边界）。** M837 R22 的唯一正式 attempt 已经原子发布并永久消费；canonical result 不存在；双封 failure quarantine 完整。失败发生在 `RESULT_STAGE_SEAL`，不是 VCS 编译或仿真阶段。

这不是一个可引用的 VCS 成功结果。未封存 work 树中可以观察到两条 VCS 路径均 `compile.rc=0`、`sim.rc=0` 并出现精确 PASS token，但该树没有 `SHA256SUMS`/outer seal，也没有原子发布为 canonical result。因此这些日志只允许用于工程故障定位，禁止用于论文性能、功能准入或 RTL validation 表。

## 事实分层

### 可正式引用的失败事实

- attempt 目录存在且双封校验通过，身份绑定 runner `a771246...`、contract `6349b11...`、candidate `a0cc6dd...`、release `21a943c...` 和 final-hammer outer seal `53e9230...`。
- quarantine 双封校验通过；`failure.json` 明确 `attempt_consumed=true`、phase=`RESULT_STAGE_SEAL`、return code=1、`paper_performance_citable=false`、`vcs_complete=false`。
- canonical result `results/m837_c2_r22_unicode_channel_split_vcs_r1_20260829` 不存在。
- 同一 M837 release 的 `max_attempts=1` 已消费，不得重跑、续跑、补封或改名为成功。
- docs/359 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

### 只允许用于诊断的未封存观察

- ATTACK：编译/仿真 rc 均为 0；精确 PASS 行报告 12 类攻击、12 个 reset case、same-cycle reuse=1、side-effect violations=0。
- EQUAL_BANDWIDTH：编译/仿真 rc 均为 0；精确 PASS 行报告 5 个 exact-cycle case，K8/K1x8 为 `51/53, 131/133, 486/499, 1231/1246, 14/14`，numeric/tuple/weight mismatch 均为 0。
- work 中生成的 pending receipt 自称 `vcs_validated=true`，但它自身位于未封存、未发布的 work 树，不能越过 sealed failure receipt 的 claim boundary。
- 上述日志在审计时仍存在，哈希已记录在 `review.json`；由于原 work 树可变，这些哈希只是审计时观察，不构成原 attempt 的追补封存。

## 符号链接根因

work 树中共有 4 个 symlink：attack/equalbw 各一个 `csrc/_*_archive_1.so`，以及各一个 `simv.vdb/.../assert.verilog.shape.xml`。四个链接都解析到同一 work 树内的常规文件；相同链接形态在大量既有 VCS run 中可复现。输入 filelist 的 RTL/TB/SVA 成员均为常规文件。

因此这是 **VCS 正常编译/覆盖率产物与“整个 work 树禁止任何 symlink”的封存策略不兼容**，不是输入污染。冻结 guard 的 `seal_directory()` 在遍历时无条件执行 `require(not member.is_symlink(), "symlink in seal target")`，于是 full-work-tree seal 必然失败。

## 孤儿与留存

没有发现命令行或 cwd 绑定 M837/1011579 的 VCS、vcs1 或 simv 孤儿。系统上另有与本项目无关的长期 simv，未误杀。原 work/log 保留原位，未修改、未删除。

## 最小 successor 修复

新 successor 必须使用新的 runner/contract/release/final-hammer/attempt/result 身份和一次性授权，重新运行；不得把本次失败 work 事后补封。

只修改结果出版边界：

1. VCS 仍在私有 work 树生成 `simv/csrc/simv.vdb`，但这些工具产物不进入 canonical result。
2. 仿真结束后，以 `O_NOFOLLOW`/常规文件检查从 work 中复制精确白名单：launch identity、两路 compile/sim rc、compile/sim log、assert report，再生成 receipt 与 RUN_COMPLETE。
3. 在独立 private result stage 中检查 exact member set、双封并 `renameat2(RENAME_NOREPLACE)` 原子发布。
4. 冻结 M803 RTL/TB/SVA、种子、PASS regex、五组 exact cycles 与 mismatch gates；不要借此修改性能口径。

这样修复的是 artifact packaging，不是硬件结果；后续 fresh result hammer 通过前，功能与性能仍禁止引用。
