# M716｜M714 PCTDA pre-run receipt-blind 静态打铁 r1

## 裁决

**52/100，当前 M714 不允许上 A800。** 这是静态 pre-run FAIL，不是对 PCTDA 机制本身的 NO-GO：signed-INT8 distributed arithmetic（DA）代数、同组等模式合并和 11-bit subset table 都成立，值得修完执行链后采一次 S10；但当前脚本没有不可变的 M714 运行身份、没有合同要求的四次 GPU 空闲门、会无条件覆盖 M366 状态为 PASS，而且输出没有原子提交与 seal。现在运行会得到“有数字但不能准入”的目录，还可能与 A800 上的训练/验证冲突。

本评审没有读取任何 M714 运行 receipt，没有 import/执行作者 M714/M366 模块，没有使用 GPU、VCS 或新思 EDA，也没有修改作者脚本。独立复算见 `recompute_m716_static.py` 与其冻结 stdout。

| 维度 | 得分 |
|---|---:|
| signed DA / pattern 数学 | 18/20 |
| trace 捕获与数值身份 | 13/20 |
| cycle、资源与公平分母 | 9/25 |
| fail-closed 执行与 receipt | 3/20 |
| claim 纪律与复现性 | 9/15 |
| **总分** | **52/100** |

## 成立的部分

1. **signed INT8 DA 公式正确。** M714 对 bit 7 使用 `-128`、对 bit 0–6 使用 `2^b`。独立穷举全部 256 个 8-bit code，重构 mismatch 为 0。该标量恒等式对任意整数权重线性成立，所以把十个 source 划成两个五元组不会破坏乘加结果。
2. **同 pattern 合并是 exact。** 在同一 temporal group、bit plane、16-lane tile 内，相同 5-bit address 查询的是同一个十输出 subset vector；相同 pattern 的 lane mask 互斥，广播不改变每 lane 的值。零 address 对应零向量，可无损 elide。两个 group 使用不同权重表，M714 分开收费是正确的。
3. **位宽与基础容量账正确。** 五个 signed INT8 权重子集范围是 `[-640,635]`，signed 11-bit 足够；两组、32 entries、十输出的表为 `2×32×10×11=7040 bit=880 B`。一个 128×128 macro 可按 64×110 容纳一个配置，或按 128×110 容纳两个配置。
4. **25-bit accumulator 不是 bug。** M518 RTL 本身就是 25-bit guard accumulator 后饱和到 Q24。保守地把全部 bit-plane 绝对贡献和 Q24 bias 相加，上界 `8,715,008 < 2^24`，signed25 可容纳。缺口是实际 checkpoint 事件 miter，不是这一位宽。
5. **Fixed 分母锚点真实。** M518 directed VCS 已闭合每 tile 17 issue cycles，clean service 为 `17N+12`，即 N1=29、N4=80。这个事实可作为 denominator anchor，但不能自动证明 M714 的 candidate latency。

## P0：上 A800 前必须修

### P0-1｜没有不可变的 M714 launch authority，也没有 GPU idle gate

M714 第 203–213 行允许任意 `--m366-contract`，只调用 M366 的内部校验；它没有 pin canonical M366 contract SHA，也没有要求调用者提供经独立审阅的 M714 expected SHA。M366 第 72–93 行只校验“所传合同内部”的 path/SHA 自洽，因此另一个 schema-compatible 合同可替换 checkpoint/config/workload 后仍通过。

更直接的阻塞是：M366 第 591–605 行创建输出目录后直接进入 CUDA。合同第 79–80 行明确要求四次连续 idle check，且训练/eval/valid/profile 存在时禁止启动；实际 execute 内没有实现。

修复门：新增 M714-r2 immutable contract，固定 M714 SHA、canonical M366 contract SHA、M366 SHA 和全部身份；再由一个独立审阅、caller-pinned SHA 的 one-shot runner 在 launch 前执行四次连续 GPU/process 空闲检查并留回执。

### P0-2｜失败的 M366 可以被覆盖成 M714 PASS

M714 第 294–340 行读入 M366 JSON 后，无条件把 status 改成 `PASS_M714...`，并把 `pctda_s10_pattern_capture` 与 `pctda_issue_opportunity` 设为 true。脚本没有要求：10 samples、45 T10 sites、450 calls、零 nonfinite、零 signed-Q8 range violation、零 bound violation、零 integer mismatch。

这不是形式问题。pattern hook 统计的是 M366 clamp 到 `[-128,127]` 之后的 `x_q`；如果 unclamped 输入越界，仍会生成整齐的 pattern 数字，而冻结合同明确规定 saturation 只可用于 diagnostic、不得 promotion。

修复门：M714 PASS 前逐项检查 canonical M366 identity、population、numeric gate 与下节所有守恒式；任一失败可保留 diagnostic payload，但 status 必须 FAIL/NO_GO，所有 M714 opportunity admission 为 false。

### P0-3｜没有可恢复、可验证的完成态

M366 第 594–596 行先创建 canonical output directory；M714 第 342–346 行直接写最终 JSON 与 payload path。异常会留下 partial directory，下一次又因“拒绝覆盖”无法重试。两层都没有 attempt token、staging、failure quarantine、atomic rename、member manifest、outer seal 或终态自检。

修复门：一次性 attempt 消耗；同文件系统 staging；失败目录 quarantine；成功后原子 rename；生成并回验 `SHA256SUMS`、outer seal；最后才写 `RUN_COMPLETE`。

## P1：数字口径必须先收紧

### P1-1｜代数 smoke test 不是真实输出 equivalence

M714 第 56–81 行的 256 个随机向量可验证公式，却没有把冻结 45 张 weight table、bias、Q24 saturation、threshold event、lane-mask broadcast、tag/output conservation 放进 DA miter。脚本真正采集时只数 address，没有计算 PCTDA 输出。

在 claim 上只能写“pattern opportunity”。升成 exact accelerator 前，需要在真实 S10 上做 dense M518-order 与 PCTDA 的逐 site、逐 tile、逐 lane、逐十输出 miter，并验证 signed25 prefix、安全饱和、event bit 与所有账本为 0 mismatch。

### P1-2｜cycle 是 ideal-resource lower bound，不是 conservative executable schedule

当前公式是每 group、bit plane 收费 `ceil(unique/P)`，并给每 bit plane 至少一拍。它没有收费：16-lane address 形成/去重、P 个 mask 选择、1RW macro read response、P-way 110-bit broadcast、160 个 accumulator update path、控制 tail 和 candidate-specific commit pipeline。

因此 `cycles` 可作为明确假设下的**架构下界**，不能叫 conservative executable cycle，更不能在 RTL 前用 `warm_full_service_gate_ge_1p25` 晋级。真正可执行周期需 cycle-accurate port schedule 或新 RTL 证明。

### P1-3｜warm/cold 把两种配置模式混在了一起

M518 TB 的计时从第一个 config accept 开始，所以 `17N+12` 的 12-cycle intercept 已跨过五个 config beats。M714 的 fixed/candidate cold 又分别 `+5`，重复计费。

此外，两个合法模式必须分开：

- **build from weights**：M518 五个 256-bit beat 携带 1064-bit payload，然后额外 64-cycle table build；外部流量不是 880 B table；
- **direct table load**：7040-bit table 需要至少 28 个 256-bit beat，不需要 64-cycle build。

现有“5 beats + 64 build + 880 external bytes”同时取了两种模式的字段。修复时必须定义配置寿命和计时端点，再给对应 Fixed 分母。

### P1-4｜45 配置驻留的物理税被隐藏

M714 只输出 all-45 macro count，却用 active macro + accumulator 的 `<=24 KiB` 做 gate。独立重算如下：

| Vector read ports | active macro+acc | all-45 physical capacity | all-45 macro area |
|---:|---:|---:|---:|
| 1 | 2,548 B | 46 KiB | 201,442 µm² |
| 2 | 4,596 B | 92 KiB | 402,885 µm² |
| 4 | 8,692 B | 184 KiB | 805,769 µm² |
| 8 | 16,884 B | 368 KiB | 1,611,538 µm² |

所以只能二选一：每 call 重建/装载并完整收费 cycles/energy，或把 45-config resident 的物理容量、macro area、leakage、ports 全部放进 DSE。active `<=24 KiB` 不能证明 resident-45 合格。

### P1-5｜tp/area 只能是隔离的 diagnostic

`66,778.235814 µm²` 来自 incomplete/quarantined Fixed DC；candidate 侧只计 table macro，未计 pattern detector、selector、broadcast、accumulator/control，也没有 macro latency/energy。因此 `optimistic_throughput_per_area_upper` 和 1.25 area budget 不能参与任何 GO gate。等 matched Fixed 与完整 PCTDA candidate 都完成 DC、macro timing 绑定后再重建。

## P2：收据守恒与措辞

修订版应 fail-closed assert：

- `sites=45`、`calls=450`；
- `tile_bitplanes=8×tiles`；
- `sum(unique_histogram)=2×tile_bitplanes`；
- `sum(k×hist[k])=distinct_nonzero_group_addresses`；
- `distinct<=nonzero<=32×tile_bitplanes`；
- 每个 port 下 `coalesced<=uncoalesced`，且 per-site 汇总严格等于 aggregate；
- `chunk_columns % 16 == 0`，并证明只有 call 最后一个 chunk 可 padding，否则跨 chunk tile identity 会变化。

另外，`pure_python_da_selftest` 注释称 exhaustive-pattern，实际是 deterministic randomized 256 vectors；应改名，或纳入本评审使用的 256-code scalar exhaustion。

## A800 重新授权的最小条件

以下五项全部满足后，允许**恰好一次**冻结 H67 ep35/no-running S10 A800 capture；在此之前不运行当前 M714：

1. M714-r2 合同 pin 当前或修订后脚本 SHA、canonical M366 contract SHA 和全部输入；
2. one-shot runner 自身 SHA 由独立 admission pin，且 launch 前四次 idle/process check 通过；
3. M366 identity、population、numeric 和 counter gates 是 M714 PASS 的前置条件；
4. staging/quarantine/atomic commit/双层 seal/terminal verify 闭合；
5. cycle 输出改名 ideal-resource lower bound，并把 build-from-weights 与 direct-table-load 两种 cold path 拆开。

修复只涉及 CPU/static 脚本和执行合同，不需要先占 A800，也不需要新开 RTL/DC。

## Claim boundary

本评审准入：signed-INT8 PCTDA 静态代数；同组 equal-address 合并 exactness；11-bit table 和独立存储算术；M518 `17N+12` directed denominator fact。

不准入：当前 M714 的 A800 执行、任何 M714 result receipt、真实 checkpoint PCTDA output equivalence、executable/warm/cold speedup、RTL/VCS/DC/Formality/STA/power/energy/PPA、full-network/system speedup或论文 headline。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
