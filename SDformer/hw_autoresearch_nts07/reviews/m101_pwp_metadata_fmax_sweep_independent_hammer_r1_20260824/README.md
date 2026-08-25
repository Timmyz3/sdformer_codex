# M101 PWP metadata 16-point Synopsys sweep independent hammer review

## Verdict

**72/100，conditional pass：当前 16 点 DC 本体可复算，但 production postrun seal 不满足 fail-closed，修复前不应把 `PASS_SCOPED_SAME_FUNCTION_GRID` 当作可引用的封存状态。**

独立解析所有 16 个点后，当前磁盘证据支持以下窄口径：

- M85 在冻结网格中 3.750 ns 失败（setup WNS -0.3402 ns），4.000 ns 首次通过（报告为 `MET 0.0000 ns`）。
- M99 在网格下界 2.750 ns 通过（setup WNS +0.0009 ns）；由于没有更短周期点，它的边界仍被网格截断。
- 两个“最短通过网格点”的目标频率比为 `4.000 / 2.750 = 1.454545x`。
- 对应两个点的 standard-cell area 为 28,928.718041 与 13,832.784035 um2，M99 面积分数 0.478168、降低 52.1832%。
- 在八个相同周期上重新计算，M99/M85 面积分数为 0.389874–0.492332；面积优势不是由单个分母巧合造成。
- 16/16 的目录名、`point_identity`、report top、`report_clocks` 周期、corner、backend、setup/hold、constraint sections 和 mapped artifacts 在当前磁盘上都能对上；当前 production receipt 的逐点 slack/area/pass 也能复算。

这只能称为 **same-recipe, logic-only, pre-macro mapped target-grid closure ratio**。它不是连续 Fmax、不是模块吞吐 speedup、更不是 full-network/system speedup 或 paper-ready PPA。

## Scorecard

| Dimension | Score | Review |
|---|---:|---|
| 当前 16 点证据真实性与可复算性 | 19/20 | 当前目录逐点身份和数字一致；production auditor nominal replay 与封存 receipt byte-identical。 |
| sweep 与 timing 方法 | 16/20 | 同 recipe、同 library/corner/constraint；但两个前沿点几乎零裕量，M99 又位于网格下界。 |
| 面积比较与分母 | 18/20 | fastest-point 面积分数 0.478168；matched-period 全表仍为 0.390–0.492。 |
| 功能同一性口径 | 12/20 | 有 latency-aligned directed + 1728 frozen-phase differential VCS，但不是全输入/全状态 formal refinement，且 M99 增加 128-edge phase audit。 |
| 封存、审计器和 claim boundary | 7/20 | 文本 claim boundary 大体克制；但审计器能被错误点身份、缺 netlist、被改合同阈值同时绕过。 |
| **Total** | **72/100** | **当前数值可保留为候选证据；production PASS seal 必须重做。** |

## P0

**0 个。** 没发现当前 production 16 点被替换、top/clock 不符或数值算错；`docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## P1

### P1-1 — production auditor 对身份、周期、netlist 和合同阈值 fail-open

可复现负例同时做了四件事：

1. 将真实 `m99_3p000ns` 报告放在 `m99_2p750ns` 路径下；其中 `point_identity.txt` 明写 `clock_period_ns=3.000`。
2. fixture 中不提供任何 mapped Verilog/DDC/SDC。
3. 将合同的最小频率比改为 99×。
4. 将合同的 `all_points_exact_input_identity` 改成 `false`。

production auditor 仍以 rc=0 输出：

```text
M101 status=PASS_SCOPED_SAME_FUNCTION_GRID m85=4.0ns m99=2.75ns ratio=1.454545x area_fraction=0.478094
```

根因是审计器只根据目录名赋值 `period_ns`，只计算 `point_identity.txt` 的 SHA 而不解析其内容；它不读 `report_clocks` 或 report top，不验证 RTL/filelist/TCL/SDC/library identity，不要求 launcher 明确要求的三个 mapped artifacts，并把 1.25、0.5、3.0 等门槛硬编码而不是从合同读取。合同要求的 `all_points_exact_input_identity` 甚至没有出现在 receipt acceptance gates 中。

最小修复：逐点解析并核对 design key/top/period/filelist SHA，交叉核对 `report_clocks` 与所有 report 的 Design 字段；校验完整冻结输入 SHA、拒绝 symlink、要求 mapped V/DDC/SDC；所有 threshold 和 required gate 必须由合同驱动，合同 gate 集和 receipt gate 集必须 exact match。

### P1-2 — durable manifest 没有封住 DC run evidence

`SHA256SUMS.complete_r1.txt` 只覆盖 production auditor、contract、顶层 backend marker 和 receipt，没有覆盖任一点的 DC log、point identity、reports、mapped netlist/DDC/SDC。Receipt 每点只留 point identity、QoR、setup、hold 四个 SHA，也没有把这些 SHA 与 prelaunch frozen identity 串起来。

因此当前文件虽经独立核对一致，现有 seal 本身不能证明此后一字未改。最小修复是对 16 个点的全部 launcher-required evidence 建 canonical manifest，并让 receipt、review 和最终封存清单共同绑定该 manifest SHA。

## P2

### P2-1 — `same-function` 应降为 workload-conditioned、latency-aligned differential equivalence

M99 的 VCS 合同很强：directed differential，加上 1728 phases / 221,184 entries 的真实记录，与 M85 在 latency alignment 后 cycle-identical。可是 M99 还引入 128-edge serial audit，当前没有跨两 RTL 的全输入/全状态 sequential formal/refinement proof。

在补 formal 之前，建议正文固定写作：

> latency-aligned functional equivalence on the frozen 1,728-phase trace plus directed protocol tests

不要只写无条件的 “same-function”。

### P2-2 — 前沿点 margin 太薄，M99 没有下侧 bracket

M85 4.000 ns 的最差 setup slack 显示 `MET 0.0000 ns`；M99 2.750 ns 仅 +0.0009 ns。`MET` 状态说明这不是简单把负数四舍五入成零，现有点按 DC 定义可以算 pass，但它们不构成稳健 continuous-Fmax 数字。

建议在 M85 3.75–4.00 ns、M99 2.25–2.75 ns 内做 0.05/0.10 ns 局部扫描，并同时给出：最短 MET 点、前一失败点、保守 guardband 点。M101 当前 1.454545× 必须始终带 “frozen-grid target closure ratio”。

### P2-3 — 仍是 logic-only / ideal-clock / ZeroWireload / no-macro

当前比较没有 CTS、route parasitic、SRAM macro、真实 address-to-data path、SAIF/PTPX 或 workload power。三个 M82 状态输出在 postcompile `check_design` 中仍是 unconnected（两边一致，不影响这次 A/B 公平性，但说明它是被裁剪的 logic island）。

所以 52.18% 只能叫 mapped standard-cell area reduction；不能升级为 macro-aware area、energy、PPA 或系统优势。

## Claim boundary after hammering

现在可以保留：

- “在冻结 0.25 ns 网格和相同 TSMC28 pre-macro DC recipe 下，当前证据显示 M99 closes 2.750 ns，而 M85 first closes 4.000 ns；两点目标频率比 1.454545×。”
- “fastest-point standard-cell area fraction 为 0.478168；相同周期逐点比较为 0.389874–0.492332。”

暂缓：

- fail-closed sealed M101 PASS（先修 P1-1/P1-2）；
- continuous/post-layout Fmax；
- 不带 trace/latency 限定的 same-function；
- 模块或系统吞吐 speedup、与 M88 cycle ratio 相乘、PPA/power/energy、DATE headline。

## Artifacts

- `m101_pwp_metadata_fmax_sweep_independent_audit.json`：16 点逐点独立解析、hash、指标、production nominal replay 和 hostile auditor attack。
- `audit_m101_independent.py`：只读 production 输入；负例全部在临时目录生成，不修改 production。
- `manifest.sha256`：本 review 的封存清单。

独立复跑：

```bash
python3 reviews/m101_pwp_metadata_fmax_sweep_independent_hammer_r1_20260824/audit_m101_independent.py \
  --output reviews/m101_pwp_metadata_fmax_sweep_independent_hammer_r1_20260824/m101_pwp_metadata_fmax_sweep_independent_audit.json
```
