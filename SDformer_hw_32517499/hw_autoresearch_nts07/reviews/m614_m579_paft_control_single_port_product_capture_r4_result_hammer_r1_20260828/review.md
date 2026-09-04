# M614｜M579 PAFT/control single-port product-capture r4 result hammer

## 裁决

**PASS，100/100；P0/P1/P2 = 0/0/0。** 正式 result 与 consumed attempt 的闭集成员、member manifest、outer seal、terminal receipt、冻结身份、20-row CSV、JSON 聚合、两臂 conservation、capacity/accuracy disclosure 与 canonical success state 全部通过 fresh independent read-only result hammer。

本评审没有再次运行 formal 80-record CPU、GPU、EDA 或 remote，没有调用 runner `--execute`，没有修改 result、consumed attempt、contract、M611 或 `docs/359`。通过本 M614 双封后，result 只可在下述窄 claim boundary 内引用。

## Result、attempt 与 terminal 身份

- result JSON SHA `bfda1d067565f9bcea3361322a0de31fe706fa93db0bd7185c9e659c1559e1ce`；CSV SHA `aa437580649a1b0f6ec85fe8dc84b2c0c5d6f6cbdc67dc050bcc3cf4bc79b47a`。
- result manifest SHA `57f84f673178588ef76afaa33727b57ce38a9d8aad5efb8d28f309587ec79bd8`；outer-seal-file SHA `8d4b926382b690f407a394a651c259a89e16bc1d63108c292e33876f8e06ef98`。
- consumed attempt manifest SHA `5bf9e3c3bd73327478c82232faf443dcdf2feef96bfb576606071c87814116fe`；outer-seal-file SHA `06bb9b6f147e8aede5ea2f07af6091fd017da421c8dac1702a702a68a9b98b78`。
- 两个 package 都是 closed member set，所有成员均为非 symlink 普通文件。result 的 6 个成员与 consumed attempt 的 5 个成员逐个重哈通过；result/attempt 的 production stdout 精确相同，production/terminal stderr 全为空。
- stdout 包含且仅包含连续 `1/80` 至 `80/80` progress 和两个最终 PASS 行。terminal receipt 精确绑定 result/CSV SHA，并记录 15 inputs、80 payloads、task order 与零 terminal stderr。
- contract start = terminal = live `29a471dc...011bb0`；runner start = terminal = live `8c0fcbea...ad53fe`；analyzer result identity = terminal = live `ba8fc032...b115195`。consumed marker 与 completion marker 绑定同一 contract/runner。
- M610 contract/release 与 M611 PASS100 review 的双封仍有效；`docs/359` 保持 `dedde7ce...bdfc4`。

## 15-input / 80-payload 独立重哈

result `.identity` 与 production contract `.inputs` canonical object 完全相等，精确 15 keys；15 个 live paths 均存在、非 symlink、普通文件且 SHA 匹配。

PAFT/control trace manifests 各有 40 records，均形成唯一完整的 10 samples × 4 operators 坐标。80 个 packed payload 均重新检查 safe basename、size、regular/no-symlink 与 SHA；80/80 通过，未运行 record analyzer。

## CSV、JSON 与 conservation

CSV header 精确，20 个 `(arm, sample_id)` 坐标唯一且完整。每行四个 cycle 整数和四个 ratio 与 JSON `per_sample` 精确一致；以下公式逐行重算通过：

- `local_cycle_speedup_vs_bit = bit_cycles / single_port_product_cycles`
- `local_cycle_speedup_vs_strong_zero = strong_zero_cycles / single_port_product_cycles`
- `single_port_tax_vs_ideal = single_port_product_cycles / ideal_product_ceiling_cycles - 1`

每臂 10 行的四个 cycle sum 与 JSON aggregate 完全相等；四类 ratio 的 arithmetic mean、geometric mean、population CV、min/max/count 均独立复算一致。每臂 4 个 operator support counters 对所有公共 aggregate 字段逐项求和一致。

两臂均通过：`residual + exact_parent = product_issues`、`macro_reads + forwarded = parent_edges`、`macro_writes + dead_write_elisions = active_rows`、bit/product issues 的 8-block scaling，以及 `10×4×432×3000 = 51,840,000` rows。

## 核心 work / cycle 独立复算

| 指标 | control | PAFT | PAFT 相对 control |
|---|---:|---:|---:|
| input NNZ | 78,759,612 | 67,844,260 | −13.859072845610266% |
| product issues，8 blocks | 299,969,072 | 261,778,936 | −12.731357851452096% |
| bit cycles | 650,033,368 | 563,851,729 | −13.25803308607998% |
| single-port candidate cycles | 374,674,056 | 337,471,557 | −9.929296785897556% |
| bit-work / product-work | 2.1004728647492033× | 2.073329841939613× | — |
| local cycle speedup vs bit | 1.734930288314385× | 1.6708125982895796× | — |

同一局部 cycle/resource coordinate 下，`control_candidate_cycles / paft_candidate_cycles = 1.11023891711265×`。这是 PAFT 训练活动造成的四层 support-trace 局部 cycle 差异，不是 decoder、RTL 或系统 throughput 结果，也不得与其他 ratio 相乘。

每个 sample 的 local-cycle-vs-bit 最小值分别为 control `1.6716332608463256×`、PAFT `1.5856163161986794×`，均超过冻结 1.5× local gate。

## Accuracy、capacity 与 canonical state

- M255 accuracy 精确继承：valid825 单 seed PAFT +0.5730215096601543%；十帧 5 win/5 loss；完整 64 帧 `zurich_city_09_a` PAFT **退化 1.0189020311889285%**。无 multi-seed significance、无双臂 same-evaluator-runtime SHA 绑定、无 accuracy-performance Pareto。
- M528 capacity ledger 精确 9 rows，213,376 B / 245,760 B，margin 32,384 B；这是容量坐标，不是 integrated macro PPA/energy。
- canonical result 和 consumed attempt 是非 symlink 目录；unconsumed attempt 在 `lexists` 口径下 absent；quarantine staging/final 与 PID staging 命中为零。既有 consumed attempt 保持 one-shot 防重入。

sealed result 没有包含 root 紧前 live resource/cgroup/collision check 的独立 receipt，且冻结 runner 不执行这些 gates；因此 M614 不准入 live-host resource-admission claim。这一边界不改变 result 的 sealed identity、逐行 arithmetic 或局部 cycle 复算裁决。

## Claim boundary

本 PASS 准入：精确 paired support trace、support/parent/residual conservation、两臂 arithmetic-work reduction、同一冻结 local CPU cycle/resource coordinate 内的 local cycle ratios，以及上述 PAFT/control 活动差异。

本 PASS 不准入：numeric Conv/ACC24 equivalence、decoder complete、accuracy-performance Pareto、RTL、VCS、Synopsys/integrated macro PPA、energy、live-host resource compliance、end-to-end/system speedup、headline，或任何 ratio multiplication。
