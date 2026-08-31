# Motion 非 Conv 硬件—算法反哺独立裁决 r1

## 结论先行

当前应把 **Motion/H67 作为唯一主动开发主线**，但只能把 Local5 冻结为对照，不能宣称已经淘汰。可用 valid825 收据中，Motion/H67 ep35 的 AEE 为 1.32967764，Local5 ep44 hardware-order AEE 为 1.28042357；低者更好，Motion 当前绝对差 0.04925407、相对差 3.8467%。同时 Local5 attention 仍缺非零全系统周期，尚不能做同资源硬件比较。

非 Conv 优先级是：

1. **立即做 ATLIF 算法闭环，不再扩写通用 RTL。** ATLIF 占当前 620,302,905-cycle compute envelope 的 20.6384%，完全免费 Amdahl 上限 1.2601×。M26 rank2 lower-bound 若被训练、数值和端口共同闭合，替换后才有条件得到 1.1139× 包络；M37 已证明 exact signed INT8 CSD/reconstruct standalone RTL 的 VCS、DC/STA、Formality，但没有证明该 rank2 训练候选、真实 checkpoint descriptor/payload 或系统周期。
2. **对 FFN 做一个受限的 stage-2 成组剪枝训练与地址 trace，不先做完整 fusion RTL。** FFN 占 25.7590%，完全免费上限 1.3470×，理论头部最大；但 resident fusion 在当前不含 memory cycle 的包络中可承认收益仍是 1.0000×。只有 `fc1 output channel` 与对应 `fc2 input column` 同时按 16/32-channel group 删除，才能形成新增的结构化算术/权重收益。
3. **冻结 RQTB core；若 attention 还做一项，只做 Q/K/projection shared group pruning census。** attention 全类完全免费也只有 1.0547×；RQTB core 自身完全免费仅 1.0050×，当前 Fixed→RQTB 放入包络只为 1.00091×。Q/K/projection 占 29,072,080 cycles，远大于 3,090,731-cycle core，但目前没有 checkpoint-bound 结构 mask 或精度闭环。
4. **冻结 prediction 性能线。** M62 已有正/负向 VCS、3 ns pre-macro DC 和 RTL↔netlist Formality；其局部 M60 比率 3.7359× event / 3.0864× event+commit 仍不是系统倍率。预测头在当前包络只占 0.0437%，完全免费也只有 1.00044×。

独立决策就绪度评分为 **7.6/10，P0=0、P1=7、P2=4**。热点、Amdahl、模块级功能证据可信；物理共存、训练后精度和 Motion-vs-Local5 同资源证据尚未闭合。结论等级：`PASS_PRIORITY_DECISION_MOTION_ONLY_RETIREMENT_NOT_ADMITTED`。

## RQTB 与修正后 Conv 模块能否共存

| 层级 | 裁决 | 证据边界 |
|---|---|---|
| 网络功能 | **能** | RQTB 服务 attention；M150/M152/M154 路径针对四层 bottleneck Conv3x3，算子结果域不重叠，可顺序执行。 |
| 向量身份 | **M154 standalone 已修复** | M150/M152 correction 证明 774,144 个跨 destination vector pair 中 identical=0，禁止 single-vector multicast。M154 改为四个独立 768-bit vector、共 3,072-bit result，并通过 VCS。 |
| standalone 逻辑时序 | **能** | M154 logic-only DC：13,282.668 µm²，3 ns setup slack +1.6514 ns、hold +0.0002 ns；但 98,304 resident bits 被排除。 |
| 物理共存/性能 | **不能 admission** | M154 没有 weight-loading RTL、真实 checkpoint payload replay、四个 SRAM macro、四路 accumulator RMW/commit；RQTB 与 Conv 若共享 SRAM 也没有端口/容量/功耗账本。 |
| 论文 cycle simulator | **可以保守建模** | 不需要先造总体调度器。将两类模块按网络顺序串行，分别计入有限队列、vector-load、bank-read、accumulator commit 和 stall；禁止未经证明的 overlap credit。 |

M154 改变了 correction P0 的状态：**接口与独立 vector-supplier 功能洞已闭合**，但没有恢复 M150/M152 的 75,032,786-cycle、1.80535758× hardware/system/headline admission。四路 update-port sensitivity 仍显示：4/2/1 ports 对应机会比 1.805×/1.129×/0.634×；因此 accumulator 与 memory port 才是下一物理门，不是再做 descriptor DSE。

## 非 Conv 定向优化与 Amdahl

| 类别 | 当前 cycles | 占比 | 完全免费上限 | 8 月底裁决 | 算法反哺 |
|---|---:|---:|---:|---|---|
| ATLIF | 128,020,500 | 20.6384% | 1.2601× | **做闭环** | rank2/rank3 参数化、CSD 项数约束、q24→q8 舍入/饱和与 tile-resident intermediate 联合训练；导出真实 descriptor/payload 给 M37 miter。 |
| FFN fc1+fc2 | 159,784,111 | 25.7590% | 1.3470× | **做 bounded pilot** | 优先 stage 2；共享 pair mask 同时删 fc1 output 与 fc2 input column，对齐 16/32 channel；另统计 T10×spatial 全零 sn2 channel group，不能与已有 scalar activity 重复计数。 |
| attention 全类 | 32,162,811 | 5.1850% | 1.0547× | **冻结 core；仅轻量 census** | 若仍投算法资源，只测 Q/K/proj shared head/channel group pruning；先给 weight-zero/mask census，再决定是否训练。 |
| RQTB core | 3,090,731 | 0.4983% | 1.00501× | **冻结** | 保留正确性与共存接口，不再以 core 内局部倍率为性能主线。 |
| prediction | 271,156 | 0.0437% | 1.00044× | **冻结** | 仅在已有流水线顺手时补 quantized-head valid825；不再做性能 DSE/PPA。 |

敏感性而非预测：若减少 50% FFN 算术，当前 envelope 上限为 1.14784×；若同时采用尚未 admission 的 ATLIF rank2 replacement，则组合 envelope 为 1.30047×。这两个数字不能相乘、不能写摘要，也不包含 memory、stall 或精度代价。

## 各线的证据缺口

### ATLIF

M37 的证据很强但范围窄：117,600 direct product miters、39,200 output-bit miters、65,536 个 signed input/coefficient pair 全覆盖且 mismatch=0；TSMC28 3 ns logic-only DC cell area 63,114.408 µm²，setup +0.4173 ns、hold +0.0104 ns；Formality 5,276 passing、0 fail/abort/unverified/unmatched。它证明的是 exact CSD/reconstruct operator，不是 trained H67 low-rank admission。

8 月底必须补的是：冻结 checkpoint 的 rank/CSD census、训练后的 valid825 AEE/event-rate guardrail、真实 descriptor/payload integer replay，以及 resident intermediate/port recurrence。没有这些，不再写新 ATLIF RTL。

### FFN

12 个 `fc1 -> sn2 ATLIF -> fc2` pair 在十样本中均 10/10 有序出现；stage 2 六对占 FFN 周期 56.17%。`fc2` input activity 仅 4.12935%，但这些 scalar zero 已被 activity-weighted model 计入，不能再次作为 fusion savings。

8 月底只做两个能判生死的产物：

- checkpoint-bound paired 16/32-channel mask 与训练后 valid825；
- `fc1 output -> sn2 -> fc2 input` 的 address/residency/cycle trace。

若结构化 FFN work reduction 小于预注册阈值或 AEE/event guardrail 失败，冻结 FFN RTL；若通过，再实现最小 resident bridge。当前 34,146 B 串行/64,452 B 双缓冲只是 bit-tight 下界，不是 macro 面积。

### Attention

RQTB core 已没有足够 Amdahl 空间。Q/K/projection 的 29,072,080 cycles 占全网 4.6868%，完全免费上限也只有 1.04917×。因此不做新的 RQTB 数据通路；只允许一天级别的 shared Q/K group mask census。没有结构化删除和 valid825 收益即冻结 attention 性能线。

### Prediction

M62 已经是成熟 standalone module：正向 VCS 241 groups/787 events，负向 VCS 6 个 full-8 legal group 与 5 类 accepted-then-fail-closed attack，DC cell area 35,459.172 µm²、setup +0.6523 ns、hold +0.0101 ns，Formality 3,995 passing/0 failing。缺少 macro/power、quantized valid825 和系统集成，但由于完美免费仅 1.00044×，这些都不是 8 月底性能关键路径。

## Motion-only 淘汰 Local5 的证据门

当前结论是 **0/5，不能淘汰**；但可把 Local5 转成只读基线，停止新增 Local5 RTL。五个门必须全部通过：

1. **同一算法评价合同。** 两边在同一 825 样本、相同 preprocessing、相同 deploy quantization 下运行；运行前冻结 AEE 非劣 margin。当前 raw AEE 是 Motion 1.32968、Local5 1.28042，Motion 尚落后 3.8467%。
2. **完整同资源周期账本。** Local5 attention 必须是非零实数项；两边使用同一 frequency、lane/bank/macro 容量、load/commit/stall recurrence，named operator 覆盖 100%，不可把 missing 当 zero。
3. **Conv correction 物理闭环。** M154 加入真实 checkpoint vector replay、weight loading、4-bank macro 与四路 accumulator commit 后，重新得到 corrected Conv cycles；当前 M150/M152 ratio 不算通过。
4. **跨 sequence 稳健性。** 在相同 DSEC sequences/event-density buckets 上报告每 sequence cycles、energy、AEE；Motion/Local5 throughput speedup 的 paired 95% CI 下界必须高于预注册的“值得淘汰”阈值，而不是只看十样本均值。
5. **效率与面积无隐藏转嫁。** macro-aware PT/SAIF/PTPX 或等价 cycle-energy model 中，Motion 在 throughput/energy/EDP 至少两个主指标胜出，第三项和 area 不越过预注册 guardrail。

建议的操作规则是：现在停止 Local5 新功能开发、保留 checkpoint/trace/valid825 与最小 simulator adapter；只有 5/5 后才在论文和代码层面正式淘汰。这样主力集中于 Motion，又不会因为当前 accuracy deficit 和 Local5 missing-attention 而失去可复核对照。

## 评分与问题分级

| 维度 | 分数/10 |
|---|---:|
| 数据身份 | 9.0 |
| 热点/Amdahl 账本 | 9.5 |
| standalone RTL/VCS/DC/FM 证据 | 8.5 |
| 物理共存与 macro | 4.0 |
| 算法反哺可执行性 | 8.0 |
| Motion-vs-Local5 决策证据 | 3.0 |
| 声明卫生 | 10.0 |
| 加权综合 | **7.6** |

P1 共 7 项：M154 real-payload/load；M154 macro；四路 accumulator commit；ATLIF trained numeric/valid825；FFN structural mask/address trace；Local5 full-system attention/cycles；Motion-vs-Local5 paired macro-aware comparison。

P2 共 4 项：M152 zero-conflict 只对冻结数据成立；H67 ep35 与 Local5 ep44 checkpoint epoch 不同；ordered s10 与 profile100/valid825 population 不同；M62 quantized-head valid825 尚未跑。

没有修改 production 或 `docs/359`；本审计快照中 `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复核

```bash
python3 results/motion_nonconv_hardware_algorithm_feedback_review_r1_20260824/validate_review.py
sha256sum -c results/motion_nonconv_hardware_algorithm_feedback_review_r1_20260824/source_manifest.sha256
sha256sum -c results/motion_nonconv_hardware_algorithm_feedback_review_r1_20260824/manifest.sha256
```

机器可读裁决见 `motion_nonconv_hardware_algorithm_feedback_review_r1.json`，优先级与淘汰门分别见两个 CSV。
