# M535 纸面加速指标独立打铁复审 r1

日期：2026-08-27 CST  
模式：receipt-blind、只读复算；未运行 HDL/EDA/训练/远端任务  
裁决：**PASS_WITH_TWO_P2_CLARIFICATIONS**  
评分：**97/100，P0=0，P1=0，P2=2**

## 1. 结论

M535 的关键整数、倍率、容量余量和三层表准入边界均与冻结源收据一致，没有发现会改变论文结论的分母、数值或标签错误：

- M528 只准入四层 H67 bottleneck Conv、单序列、51.84M source-row 的 exact CPU same-ledger 局部候选；不得写成 RTL/PPA/energy/system headline。
- M472 只准入官方 Prosperity artifact 在 H67 support-tile workload 上的 external product-vs-bit mapping；不得写成 ours RTL、same-resource、monolithic Conv 或 full-network 结果。
- C2 的 `4.764209x` 只属于 120-record H67 FC2 always-ready standalone frontend 的 K8/K1 低带宽轴；K8/K1x8 必须并列，且目前只准入同一 M519 identity 的 directed component cycles，不是 complete FC2、PPA 或系统结果。
- Table A admitted system headline 为空、lossy headline 为空的裁定正确；`1.794--1.823x` analytical envelope 没有被提升为实测系统倍率。

## 2. 独立整数复算

所有比值均直接由源 JSON 的整数计算，没有从四舍五入后的 M535 表格反推。

| 指标 | 源整数 | 独立复算 | M535 | 裁定 |
|---|---:|---:|---:|---|
| M528 / M468 strongest-zero | `760350133 / 435293339` | `1.746753430104750x` | `1.7467534301047505x` | 一致 |
| M528 / same-coordinate bit | `757946784 / 435293339` | `1.741232213066325x` | `1.741232213066325x` | 一致 |
| M504 all-write / M528 dead-write | `456016645 / 435293339` | `1.047607680024711x` | `1.047607680024711x` | 一致；仅 liveness 消融 |
| M472 official product / bit | `556188432 / 226140006` | `2.459487119673995x` | `2.459487119673995x` | 一致；external mapping only |
| C2 K8 / K1 frontend | `429716335 / 90196785` | `4.764209001462747x` | `4.764209001462746x` | 一致；浮点末位差异不影响值 |
| C2 K8/K1 logic area | `20587.39208 / 20436.696076` | `1.007373794836484x` | M532 `+0.737%` | 一致；logic-only、pre-macro |
| C2 logic area overhead | `(20587.39208-20436.696076)/20436.696076` | `0.737379483648398%` | `0.7373794836484038%` | 一致 |
| M528 macro-rounded margin | `245760 - 213376` | `32384 B` | `32384 B` | 一致；占预算 `13.177083%` |

M519 同一 identity 下，K8 相对 K1x8 的 active directed component rows 也逐行复算一致：

| B | events | K1x8 cycles | K8 cycles | K1x8/K8 |
|---:|---:|---:|---:|---:|
| 1 | 20 | 53 | 51 | `1.039215686274510x` |
| 2 | 41 | 133 | 131 | `1.015267175572519x` |
| 4 | 90 | 499 | 486 | `1.026748971193416x` |
| 8 | 110 | 1246 | 1231 | `1.012185215272136x` |
| 1 | 0 | 14 | 14 | `1.000000000000000x` |

因此 M535/M532 的 `约 1.01--1.04x` 对**非零 directed shapes**成立；包含零事件行时完整范围是 `1.00--1.0392x`。这些行是 component directed VCS，不是冻结 120-record FC2 trace 的 K1x8 系统对照。

M528 的 logical parent-access traffic 也与源账本一致：

- total parent read+write：`52,428,622,464 -> 30,457,108,224 B`，减少 `41.907479554869%`；
- parent write：`31,456,014,336 -> 11,459,751,552 B`，减少 `63.568965128284%`。

M535 将其限定为 logical parent scratch access、不是物理 SRAM/DRAM energy，边界正确。

## 3. Claim admission 复核

### M528

源收据明确给出：`exact_cpu_cycle_recompute=true`，而 `date_headline/energy/rtl/vcs/synopsys_ppa/system_speedup=false`；scope 是 `one-sequence ten-sample four-bottleneck-Conv CPU cycle/traffic/capacity recompute`。M535 的 machine JSON 和推荐英文句均保留这些边界，PASS。

### M472

源 receipt 与独立 hammer 都准入 `2.459487119673995x`，但要求标记为 official Prosperity product-vs-bit on H67 support tiles，并禁止 same-resource、monolithic Conv、full-network、energy/PPA/headline 外推。M535 将其放在 Table C external mapping、`m472_ours=false`，未提升为 H67 自研分子，PASS。

### C2

M216 源 receipt 的 `4.764209x` 明确是 120 records、5.58M tokens、143.895M events 的 standalone sparse frontend，且 `complete_fc2/physical_speedup/system_speedup/headline=false`。M519 源 receipt只准入 directed RTL functional/component cycle rows，`dc/power/paper_ppa_ready/system_speedup/headline=false`。M535 的 `c2_4p764_equal_service=false` 和“双表后再看 matched area/energy”的裁定一致，PASS。

### Table A/B/C

- Table A：无 admitted system 行，正确。
- Table B：允许 M528 exact CPU local、C2 narrow frontend/direct VCS companion，并要求 evidence tag，正确。
- Table C：外部原论文和 M472 mapping 隔离，正确。
- 有损表：PAFT valid825 未过冻结 `Delta AEE <= 0.02` 门，因此无 admitted lossy headline，正确。

## 4. 身份、封存与仓库卫生

- M535 inner manifest 与 outer seal：PASS。
- M532 inner manifest 与 outer seal：PASS。
- M528 r4 result inner manifest 与 outer seal：PASS。
- M519 r2 VCS result inner manifest 与 outer seal：PASS。
- M472 producer 历史目录本身没有 outer seal；其 admission 与 independent hammer 目录均已通过 inner/outer seal，且独立 hammer 重算值一致。
- M535、M532、M528、M472、M216、M519 关键 JSON：均可解析。
- `git diff --check`：PASS。
- `docs/359_DATE终局冻结_20260813.md`：SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- 本复审只新增本目录；未修改 M535、M532、源收据或 docs/359。

## 5. P0/P1/P2

### P0

无。

### P1

无。

### P2-01｜M535 machine JSON 的直接源身份不够完整

M535 的 `source_sha256` 只固定 M528 与 M532，没有直接固定用于 M472、C2 K1/K8 和 C2 K8/K1x8 复算的 producer receipt SHA。数值没有错误，M532 也列出了路径，但审计链多了一跳。

最小修订：后续 r2 machine JSON 直接加入 M472 receipt、M216 replay、M216 logic-area recovery、M519 VCS receipt 及其独立 hammer outer-seal SHA。此项不影响当前结论。

### P2-02｜C2 companion range 需要在论文表头写出 active directed scope

`1.01--1.04x` 是 M519 非零 directed `B={1,2,4,8}` component rows 的范围；零事件行是 `1.0x`，且该 suite 不是 M216 的冻结 120-record trace。若仅写“K1x8 companion”容易让读者误以为两列来自相同 workload。

最小修订：表头写 `[directed VCS, active shapes]`，零事件另列 `1.0x`；直到冻结 FC2 trace 的 matched rerun 完成前，不把该范围称为 representative trace aggregate。

## 6. 最终评分

| 维度 | 得分 |
|---|---:|
| 关键整数、分母与倍率复算 | 30/30 |
| scope 与 claim admission | 25/25 |
| 源身份与封存链 | 18/20 |
| 三层表安全性与论文措辞 | 19/20 |
| JSON、diff 与 docs/359 卫生 | 5/5 |
| **总分** | **97/100** |

最终裁决：**M535 可作为当前纸面指标准入审计使用；关键值与边界通过。两项 P2 应在下一版机器身份和论文表头中修订，但不阻塞当前局部表分享，也不授权任何系统 headline。**
