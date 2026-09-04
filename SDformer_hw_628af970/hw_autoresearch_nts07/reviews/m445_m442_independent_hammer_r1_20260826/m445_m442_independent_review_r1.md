# M445：M442 完全独立打铁评审

## 结论

评分 **94/100**，`P0=0 / P1=2 / P2=2`。

严格范围内 `GO`：M442 确实把冻结 M430 catalog 的完整静态 codec population 送过了合法 M433 RTL。M445 没有把候选 receipt 当 oracle，而是从四个 H67 ep35 INT8 weight bin 与冻结 train-only catalog 独立重算全部 `4×432×32×8=442,368` blocks、每 block 96 lanes，共 `42,467,328` lanes；所有 metadata、low8/high4、signed12、narrow 决策和 stimulus 字段逐条检查为 0 mismatch。

严格范围外 `NO-GO`：这不是 `127,277,168` 次 runtime repeated output-block issue 的 VCS replay，不验证 downstream `old_psum + PWP + correction`，不产生 RTL cycle/system speedup，也不能把 M430 `1.435375301×` 升级为 RTL 测得加速。

## 独立重算

- M430a、M430b、M442a、M442b 的内封和外封全部通过；全部 exact-SHA 输入通过。
- 完整人口：`442,368` blocks；`42,467,328` lanes。
- Narrow：`70,503` blocks；wide：`371,865` blocks。
- 全局范围：`[-1089, 1059]`；signed12 violation `0`。
- Stimulus metadata mismatch `0`，payload mismatch `0`，逐 lane reconstruction mismatch `0`，narrow high-side nonzero `0`。
- 独立 codec global SHA：`4938438e4bde7c8831deb4ed8661450261ff534113ff73dfb5045fd9612d1ba7`，与冻结 M430 identity 一致。
- Stimulus SHA：`6afd66512fc8b6fe2b4a7f759bca1299bd0cd825a51d7a5923ebadb84e4d3c1a`。

## Raw Synopsys VCS/SVA

- VCS `V-2023.12-SP1`，compile/sim rc 均为 `0`，无 compile warning/error marker。
- Accepted/retired `442,368` blocks，retired `42,467,328` lanes。
- Metadata/arithmetic/X/protocol/SVA failure 均为 `0`。
- 同拍 pop+push `442,367`；II=1 request cover `442,259`；stall cycles `108`；最大 scoreboard depth `1`。
- `cp_long_stall=0`、`cp_protocol_fault=0`：M442 自身没有长 stall 或攻击覆盖，必须与已独立通过的 M434 directed evidence 组合表述。
- SVA 的 `cp_narrow=70,489`、`cp_wide=371,771` 是 `##1` 邻接 cover，不是人口计数；人口必须使用 accepted/debug counter 的 `70,503 / 371,865`。

## TB 审计与缺口

1. `P1`：候选 TB 的 `expected_data` 来自同一 stimulus line 的 `scan_expected`，不是在 VCS 内从 low/high 独立重构。因此 M442b 单独看存在 self-consistent generator defect 风险。当前冻结里程碑由 M445 的另一套完整 source reconstruction 与 exact-SHA 双封消除了这个风险；后续 TB 应从输入内联生成 expected，或使用独立 gold stream。
2. `P1`：静态 block 只证明 codec contents/adapter arithmetic。runtime issue 数约为静态 block 的 `287.72×` 重复，runtime index、phase timestamp、persistent old PSUM 和 correction 都未进入 M442。
3. `P2`：M442 的 stall 仅为分散的单周期 stall；长 stall 与 fail-closed attack 仍引用 M434，不得移花接木。
4. `P2`：tag 是全局唯一、center/block 均逐条核对；tile 只编码 operator parity，但全局 tag 已消除 op0/op2 与 op1/op3 的身份歧义。若未来移除全局 tag，需要显式 operator/partition metadata。

## 下一项 P0

在任何 RTL-cycle admission 前，接入 runtime issue/index stream 和 downstream accumulator，逐项证明 `persistent old_psum + exact M433 PWP delta + correction`，并把 phase timestamp/cycle 与冻结 M430 replay 对齐。无需机械地在 VCS 搬运 1.27 亿份重复 payload；可以让 442,368 个已验证静态 codec entry 被完整的 runtime index/timestamp trace 引用，但每次引用、correction 和累加结果必须穷尽核对。

`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
