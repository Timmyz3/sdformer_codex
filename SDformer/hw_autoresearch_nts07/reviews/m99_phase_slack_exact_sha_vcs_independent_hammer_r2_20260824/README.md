# M99 phase-slack exact-SHA VCS 独立打铁 r2

日期：2026-08-24  
评审边界：只读检查生产 contract/filelist/launcher/source/data 和封存 VCS run；独立重算原始输入。评审未运行 producer、launcher 或 VCS，也未修改任何生产文件。

## 结论

评分 **93/100**，`P0=0 / P1=1 / P2=4`。

**GO：M99 获得限定范围的 VCS 功能准入，并可进入同一 TSMC28、3.000 ns、ideal-clock、ZeroWireload 的 M99/M97（M85）DC A/B。**

这次 r2 已关闭 r1 的关键阻塞：loaded old phase 上同时出现 load/lookup 时，协议改为只禁止双重接收；动态 directed 场景确实观察到 `load_ready=0`、`lookup_ready=1`，并返回旧 phase 的合法 output。内部 parser SVA 同时覆盖 start、逐 entry 前进、entry 127 完成、cursor delta、metadata capture 稳定、poison 单调和 parse 期间 datapath blocking。

封存 run 的四个 compile/sim RC 全为 0，directed 与 actual-record 两条 PASS 唯一且精确，assert report 与 sim log cover 完全一致，未发现 assertion disable、compile error/warning 或 simulation fatal/error signature。contract、六项 filelist 输入、metadata 和两个外部二进制输入均通过 exact-SHA preflight。

当前准入只证明 M99 在 audit latency 对齐后，对冻结 M85 的 ready、bank-row address、signed output 和错误行为保持等价。它**不**证明 DC/PPA、SRAM-inclusive timing/area、能耗、Fmax、模块或系统加速，也不把 128-cycle metadata audit 变成当前 M86 的 zero-incremental parser cycle。

## 独立重算

实际数据不依赖 PASS 行重算得到：

| 项目 | 独立结果 | 封存结果 |
|---|---:|---:|
| phases | 1,728 | 1,728 |
| entries / outputs | 221,184 / 221,184 | 221,184 / 221,184 |
| code 0/1/2/3/4 | 52,248 / 128,893 / 37,144 / 2,898 / 1 | 相同 |
| beats | 835,383 | 835,383 |
| bank address checks | 835,383 | 835,383 |
| II checks | 219,456 | 219,456 |
| parser cycles | 221,568 | 221,568 |
| masked nonzero words | 733,459 | 733,459 |

算术关系为：

- `entries = outputs = 1,728 x 128 = 221,184`；
- `beats = 52,248x3 + 128,893x4 + 37,144x4 + 2,898x5 + 1x1 = 835,383`；
- `II checks = 1,728 x 127 = 219,456`；
- `parser cycles = (1,728 actual + 3 poison) x 128 = 221,568`；
- actual campaign 无 output stall，因此每个 beat 恰有一次独立 address check。

directed campaign 的 `128 entries / 436 beats / 640 parser cycles / 10 stalls` 也闭合。640 cycles 来自五次完整 128-entry audit；第六次 early-lookup attack 只命中 parser first-entry cover，随后 reset，因此 directed cover 为 first `6`、middle/final `5/5`。

## exact-SHA 与封存证据

| artifact | SHA256 / result |
|---|---|
| contract | `a89fde382fb19b639523a0b2d0b4500b498794a09ec960a529c25c390324c420` |
| filelist | `12bcb401f2779407fed42577476c8c456eaff85f742daca31f259205a0ab1975` |
| launcher（当前观察值） | `836fbb8ced08039a5147e99cfda2ece314eb7f146efd263c4fc1db1e62df2009` |
| records / offsets / metadata | `6de1521b...` / `1cddfc80...` / `52b700b1...` |
| preflight | 9/9 expected=observed |
| compile/sim RC | directed `0/0`，actual `0/0` |
| output manifest | 7/7 `sha256sum -c` 通过 |
| assertion disabled | 0 |

directed cover 的 loaded lookup priority、simultaneous unloaded、lookup stall 均为 1；escape 为 28。actual cover 的 parser first/middle/final 各为 1,731，width9/10/11 分别精确等于实际 code population，escape 为 1。actual 中 loaded-priority、simultaneous 和 stall 为 0，由 directed campaign 补齐。

## Open findings

### M99-R2-P1-01 — launcher 没有被 contract 或封存 run 反向绑定

当前 launcher 静态检查良好，观察 SHA 为 `836fbb8c...`，但它既不在 contract `frozen_sources`，也不在 run 的 `preflight_sha_checks.txt` 或 `input_sha256.txt`。因此可以证明当前 launcher 能验证并产生这种封存结构，却不能从封存 run 自身密码学证明当时执行的恰是这些 launcher bytes。

影响：这是 producer provenance 缺口，不改变已经精确绑定的编译输入、数据、PASS/cover 和结果内容，所以不否定本次限定 VCS 功能准入；但同样的缺口不得带入 DC/PPA receipt。

必改：下一次 Synopsys launcher 必须在 contract/preflight/input receipt 中冻结自身以及 Tcl、SDC、filelist、RTL 和 DB，并记录实际 invocation/corner。

### M99-R2-P2-01 — output manifest 不是完整运行封印

七个关键 compile/sim/assert/RUN_COMPLETE 文件被绑定，但 manifest 未包含 launcher、input/preflight manifest、四个 RC、两个 disablelog 和两个 simv。建议 final receipt 覆盖所有轻量控制/证据文件；simv 可选择记录 SHA 而不必复制。

### M99-R2-P2-02 — actual-record campaign 没有 backpressure 多样性

actual replay 固定 `output_ready=1`，因此 835,383 个 beat/address check 的 cycle-aligned differential 很强，但没有实际数据上的 ready/valid 扰动。directed 只提供一个 synthetic image 上的 10 个 stall cycle。建议后续加固定种子的随机 burst stall，并保持 PASS/cover 精确冻结。

### M99-R2-P2-03 — actual escape 样本只有一个

实际 221,184 个 entry 中 code4/escape 只有 1 个；directed 提供 28 个合成 escape。escape 数据通路已覆盖，但对真实 phase/base 分布的普适性仍薄。建议扩充 trace 或增加多 seed 合法 metadata campaign。

### M99-R2-P2-04 — poison contract 只冻结三类攻击

三类 poison 均被端到端测试，SVA 证明 poison 单调与阻塞；但没有分别冻结 code6/code7、mid/final reserved code、独立 fetch/cursor overflow、held second load、capture 后 live-input mutation 和 parse 中 reset-abort。它们不在本次 frozen contract 的 admission 条件内，因此列为 P2 而不是阻塞项。

## DC A/B 准入边界

允许启动下一里程碑，但必须：

1. M99 与冻结 M97/M85 使用相同 TSMC28 setup/min DB、`3.000 ns`、operating condition、ideal clock 和 ZeroWireload；
2. 冻结并校验 contract、launcher、Tcl、SDC、filelist、RTL、DB 和 VCS completion；
3. fail-closed 检查 link/unresolved reference、setup/hold、constraint、macro count 和 operator-family counts；
4. 对 M99 增量设置 `32-bit add <= 16`、`32-bit compare <= 16`、`4-bit compare <= 8`，并执行 `area <= 1.35 x sealed M97` gate；
5. receipt 明确 `logic_only/pre_macro=true`，`SRAM/energy/system_speedup/headline/paper_ppa_ready=false`。

本评审的 GO 是“去跑并比较 DC”，不是预先承认 DC 通过，更不是性能 headline。

机器结论见 `m99_phase_slack_exact_sha_vcs_independent_hammer_review_r2.json`，原始输入复算和封存校验见 `audit_m99_sealed_vcs_r2.py`、`m99_sealed_vcs_independent_audit_r2.json` 与日志。
