# M140 sparse-mask K4 descriptorizer DSE：独立打铁复核

## 结论

**96/100；P0=0、P1=0、P2=3。M140 是数值、身份和结论均可信的 negative-value screen。**

独立复核从冻结的 20 条 heldout raw mask 重新构建了 69,120 个 descriptor，没有调用 production M140 analyzer 或 M122 `FoldSchedule`。三条 two-bank recurrence 全字段复现 production result；M132 dualrow512 baseline 也完整逐字段一致。Production analyzer 另行重跑，输出与封存 result byte-identical。

最强安全结论是：在 frozen heldout、same-clock service-island cycle model 中，K4x1 把 producer fill payload 减少 46.9313%，却只把 candidate cycles 从 245,485,910 减到 244,697,828，即 `1.003220633×`；K4x2 upper bound 也仅到 244,663,654，即 `1.003360761×`。因此不值得把 wide K4 descriptorizer 作为下一性能 RTL，应该先研究 PWP 与 correction service 的合法重叠。

## 独立复算

| 模型 | Candidate cycles | 相对 raw speedup | Cycle count reduction |
|---|---:|---:|---:|
| M132 raw-source fill baseline | 245,485,910 | 1.000000000× | — |
| sparse-mask K4x1 | 244,697,828 | 1.003220633× | 0.321029% |
| sparse-mask K4x2 upper bound | 244,663,654 | 1.003360761× | 0.334950% |

K4x2 相对 K4x1 只再减少 34,174 cycles，增量为 `1.000139677×`，也就是 0.013968% speedup uplift。

需要严格区分三个分母：

- fill payload：188,148,490 → 99,847,888，减少 46.931337%；
- descriptor fill cycles（含每 descriptor 一个固定 edge）：188,217,610 → 99,917,008，减少 46.914102%；
- 最终 candidate cycles：245,485,910 → 244,697,828，减少 0.321029%，对应 `1.003220633×`。

Contract 的 0.322% 是 speedup uplift，而严格的 cycle-count reduction 是 0.3210%；不影响决策，但论文表格应标明定义。

## 为什么 fill 减 46.93%，周期只减 0.322%

Candidate recurrence 的守恒式直接给出原因：

```text
raw    = 119,447,791 PWP
       + 124,730,596 correction
       +     827,363 service idle
       +     480,160 commit/flush
       = 245,485,910 cycles

K4x1   = 119,447,791 PWP
       + 124,730,596 correction
       +      39,281 service idle
       +     480,160 commit/flush
       = 244,697,828 cycles

K4x2   = 119,447,791 PWP
       + 124,730,596 correction
       +       5,107 service idle
       +     480,160 commit/flush
       = 244,663,654 cycles
```

Producer fill 在 ping-pong bank 上与 service 链并行，不是直接串加到 candidate cycles。K4x1 节省 88,300,602 个 descriptor-fill cycles，但 producer-bank stall 同时增加 87,512,088；约 99.107% 的 fill 节省被 bank 等待吸收，只有 788,082 cycles、即 fill 节省的 0.8925%，真正落到关键时间线。

另一个视角更直接：raw 模型可被 descriptorizer 消除的 service idle 总共只有 827,363 cycles。即使把它全消掉，理论 lower bound 仍是 244,658,547 cycles，raw 的极限也只有 `1.003381705×`。K4x1 已消除 95.25% 的这部分 idle，K4x2 已消除 99.38%；继续加 descriptorizer lane 基本没有空间。

## 下一瓶颈

K4x1 中 99.9839% 的时间来自不可被本次改动触及的 PWP、correction、commit 和 flush：

- correction service：124,730,596 tokens，占 50.973%；
- PWP service：119,447,791 tokens，占 48.814%；
- commit/flush：480,160 cycles，占 0.196%；
- 剩余 service idle：39,281 cycles，占 0.016%。

所以 production decision 正确：下一步应做 dependency-correct 的 per-descriptor PWP/correction overlap recurrence，再确认两条链是否拥有真正独立的 SRAM port、buffer、控制器和写回资源。只有模型与资源同时闭合后才能报 overlap ratio；不能直接用全局 `sum → max` 的理想值当 speedup。

## Identity 与可重复性

- M140 direct frozen identity：4/4 一致，包括 M132 script/result、M132 correction overlay 和新增的 M109 result pin。
- M132 transitive identity：12/12 一致。
- M40 每条 heldout packed payload 由冻结 M105 decoder 在读取时按 manifest SHA 再校验；M41 weight payload 同样按冻结 SHA 校验。
- Production rerun SHA256 为 `c0f33cb...6508`，与 sealed result byte-identical。
- 负测将 M109 result 做 JSON-valid SHA 漂移，M140 在输出前拒绝：`frozen input identity drift: m109_result`。
- 负测经 loader 注入 M122 result 传递漂移，M140 在输出前拒绝：`M132 transitive input identity drift: m122_result`。

M132 评审曾指出的 M109-result 漏 pin 已在 M140 中真正关闭，没有新的 transitive identity P0/P1。

## P0

**0 个。** 没有 exact-work、M132 baseline、full recurrence、candidate cycles、ratio、直接或传递 SHA mismatch。

## P1

**0 个。** 该里程碑明确是 negative-value cycle screen；它没有把理想 descriptorizer、上游 mask 或 inherited dualrow512 假设冒充成硬件实现。

## P2

### P2-1 — 三种 reduction 的分母必须继续分开

46.93% 是 fill payload reduction，46.914% 是 descriptor-fill cycle reduction，0.3210% 才是 candidate-cycle reduction。正文或表格应同时写 metric name 与 denominator。

### P2-2 — K4x2 只是乐观 upper bound

当前没有 two-lane mask ingest、ordered merge、backpressure、RTL、timing 或 area。这个缺口不会推翻“不做 K4x2”的负结论，因为真实开销只会让它更差；但不得把 1.00336× 当实现结果。

### P2-3 — PWP/correction overlap 仍只是下一瓶颈假设

需要新的逐 descriptor dependency recurrence 与独立硬件资源证明，不能把全局 PWP/correction token 数直接变成 headline speedup。

## 安全引用口径

> Independent heldout trace reconstruction and a separate two-bank recurrence exactly reproduce the M132 baseline (245,485,910 cycles), K4x1 (244,697,828) and K4x2 upper bound (244,663,654). K4x1 removes 46.93% of producer-fill payload but only 0.3210% of modeled candidate cycles (`1.003220633×`) because nearly all fill work is hidden behind serialized PWP and correction service. This is a same-clock service-island cycle-model negative screen, not RTL, physical or system speedup evidence.

## Artifacts

- `independent_recompute_m140.py`：独立 raw heldout reconstruction 与 recurrence。
- `independent_result/m140_independent_recompute.json`：独立全字段数值回执。
- `production_rerun/`：production byte-identical 重跑。
- `audit_m140_independent.py`：identity、负测、metric 与瓶颈机器审计。
- `m140_identity_negative_tests.json`：direct/transitive fail-closed 负测。
- `m140_independent_audit.json`：评分与最终机器结论。
- `manifest.sha256`：review artifact 封存清单。

仅写入本 review 目录；production、contract、result 和 `docs/359` 均未修改。`docs/359` SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
