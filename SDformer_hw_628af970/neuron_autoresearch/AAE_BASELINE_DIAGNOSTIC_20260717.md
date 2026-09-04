# AAE Baseline Diagnostic (2026-07-17)

## Verified Definition Mismatch

- Legacy local `AAE` is the 2-D direction angle between `(u,v)` vectors.
- DSEC benchmark `AE` follows Barron and uses normalized space-time vectors `(u,v,1)`.
- The paper's `4.871` is official DSEC hidden-test AE. Historical local valid825
  runs used cropped geometry, while the current re-audit uses full 480x640;
  both use the local validation population and are not the official hidden test.
- Therefore legacy valid825 `AAE` and paper/test `AE` are not directly comparable.

## Same-Checkpoint Valid825 Audit

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D |
|---|---:|---:|---:|---:|
| NB0 | 59 | 1.4872 | 9.9300 | 9.2506 |
| H67 Motion-XOR | 19 | 1.4671 | 9.4155 | 8.7949 |
| H81 no-motion | 19 | 1.4813 | 9.4636 | 8.8450 |

## Reporting Rule

Use legacy AAE-2D only to compare historical local runs. Report DSEC/Barron AE-3D for benchmark-facing tables, and label valid825 separately from official test.

## Full-Resolution Re-audit (2026-08-05)

### Answer first

The local NB0 angular errors cannot establish either superiority or inferiority
to the paper rows because the validation population and aggregation are not
identical. The apparent gap came from comparing different rows and protocols:

| source | population | geometry | AEE/EPE | outlier field | angular metric |
|---|---|---:|---:|---:|---:|
| SDformerFlow-v2 paper Table I | official hidden DSEC test, seven sequences | 480x640 | 1.602 | 10.051% | official AE 4.871 |
| SDformerFlow paper Table IV, PSN+SPE+QK s10-c2 | authors' validation split, full-resolution test | 480x640 | 1.61 | 8.91% | reported AAE 7.23 |
| local NB0 ep29 | local valid825 held-out frames | 480x640 | 1.4454 | legacy prediction-magnitude outlier 7.93%, not standard DSEC Fl | legacy AAE-2D 6.5128; Barron AE-3D 6.1803 |
| local H67 ep30 | same local valid825 | 480x640 | 1.3387 | legacy prediction-magnitude outlier 6.47%, not standard DSEC Fl | legacy AAE-2D 6.0147; Barron AE-3D 5.7558 |

The local values are numerically lower than the paper validation row under a
non-identical local-validation protocol; this is not a controlled reproduction
or a superiority claim. In particular, the historical local outlier field used
prediction magnitude, whereas standard DSEC Fl-all uses GT magnitude. The
official `4.871` remains a test-server result that can only be reproduced by an
official submission, not by extending local valid825 training until its number
happens to match.

### Metric and aggregation reconciliation

1. The released `flow_supervised.py::AAE` computes the 2-D direction angle
   between `(u,v)` vectors. `AAE_Benchmark` computes the Middlebury/DSEC angle
   between normalized `(u,v,1)` vectors.
2. The local evaluator first computes a masked mean per frame and then averages
   825 frame means with equal weight. It does not pool all valid pixels before
   averaging. New standard `spike_profile.json` artifacts serialize this as a
   fail-closed `metric_contract`, including the local-validation population marker;
   they cannot be relabeled as official hidden-test AE and do not reproduce the
   DSEC server's hidden-test aggregate.
3. Local `valid_split_seq.csv` contains 825 held-out frames from 18 sequences
   also represented in `train_split_seq.csv`. The official test result uses
   seven different hidden test sequences.
4. The DSEC leaderboard's seven published SDformerFlow sequence AEs have a
   simple arithmetic mean of `4.9919`, while its reported all-sequence AE is
   `4.871`. This proves that the official aggregate is not the same unweighted
   sequence mean; it is consistent with a different pixel/frame pooling
   contract.

Primary sources: the
[SDformerFlow-v2 paper](https://arxiv.org/abs/2409.04082) and the
[official DSEC optical-flow benchmark](https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/).
The code evidence is `third_party/SDformerFlow/loss/flow_supervised.py`,
`third_party/SDformerFlow/eval_DSEC_flow_SNN.py`, and the active DSEC sequence
lists.

Executable metric regression (rechecked 2026-08-06):
`PYTHONPATH=third_party/SDformerFlow python -m unittest
third_party.SDformerFlow.tests.test_aae_metrics -v` passes all four tests,
covering the Barron `(u,v,1)` formula, per-batch masks, separation from the
legacy 2-D direction metric, and GT-magnitude DSEC Fl-all.

### Convergence evidence

| model | late interval | train loss change | AEE change | legacy AAE change | AE-3D change | conclusion |
|---|---|---:|---:|---:|---:|---|
| NB0 | ep24 -> ep29 | 1.1547 -> 1.1399 (-1.28%) | 1.5304 -> 1.4454 (-5.56%) | 6.5414 -> 6.5128 (-0.44%) | 6.2591 -> 6.1803 (-1.26%) | AEE not plateaued; angle nearly plateaued |
| H67 | ep25 -> ep30 | approximately 1.1197 -> 1.1025 (-1.54%) | 1.3726 -> 1.3387 (-2.47%) | 6.0102 -> 6.0147 (+0.08%) | 5.7661 -> 5.7558 (-0.18%) | AEE still improves; angle plateaued |

The paper specifies 80 cropped epochs plus 30 full-resolution epochs at an
initial learning rate of `1e-3`; the released YAML contains 60 cropped epochs
and `1e-4`. Local NB0 begins from the released ep59 lineage. This makes
under-training relative to the paper protocol plausible, but it cannot explain
the official-test/local-validation gap by itself, especially because angular
error is already nearly flat and the local/paper validation protocols are not
identical.

### Decision rule

- Do not use `4.871` as the local early-stop target.
- Keep local AEE primary, Barron AE-3D secondary, and legacy AAE-2D only for
  historical continuity.
- Decide any `+10` convergence extension only after Local-5 has the same
  ep9/14/19/24/29 curve; if run, extend NB0 and the selected candidate with the
  same budget.
- Freeze the final checkpoint before official DSEC test submission. If an
  official submission is unavailable, use the already frozen MVSEC
  train-to-test protocol as external validation and label it separately.

## Executable Metric Receipt (2026-08-05)

- `AAE_METRIC_TEST_RECEIPT_20260805.json` schema v2 records a fresh `8/8` PASS from
  the metric and multi-aggregation tests in the production conda environment.
- The receipt SHA-binds `loss/flow_supervised.py`, `eval_DSEC_flow_SNN.py`,
  `utils/metric_aggregation.py`, and both test files; it also fail-closes the
  2-D legacy definition, Barron `(u,v,1)` definition, GT-magnitude DSEC Fl-all,
  three aggregation modes, and eval batch size 1.
- The DATE closure auditor recomputes all five source hashes before accepting
  this receipt, so later metric/evaluator edits invalidate the result.

## 2026-08-05 多聚合审计升级

- 生产 evaluator 现在于同一 forward/mask 上同时累积 frame-equal、pixel-global 和
  sequence-balanced 的 AEE、AAE-2D、Barron/Middlebury AE-3D与标准DSEC Fl-all。旧
  `AEE_outliers`保留为prediction-magnitude legacy字段，排名不变。
- 本地 `valid_split_seq.csv` 当前为825帧、18个 subsequences，SHA256
  `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0`。每个新 profile 将绑定
  该文件的实际路径与 SHA，不再只用“valid825”文字标签代替 population 证据。
- 该升级能回答“聚合方式造成多少 AAE 差距”，但仍不能把本地18段 validation 变成
  论文七序列 hidden test population。最终表中必须继续分开 local-valid 和 official-test。

## 2026-08-05 fullres 同口径端点结论

- 公式回归重新执行：metric侧`4/4 PASS`、多聚合侧`4/4 PASS`；新增标准DSEC Fl-all使用GT flow
  magnitude，历史prediction-magnitude outlier不再用于论文比较。receipt已重新绑定当前源码SHA。
- 同一local DSEC valid825、480x640、crop null、T2x15x15、BN no_running、batch1下，H67 ep30为
  `AEE/AAE-2D/AE-3D=1.33874/6.01474/5.75580`，NB0 ep29为
  `1.44535/6.51280/6.18034`。H67分别改善`7.376%/7.647%/6.869%`，并将spikes从
  `126.1156G`降至`81.3086G`（`-35.529%`）。
- 所以AAE公式没有阻止H67在同协议上超过baseline；无法复现official-test `4.871`是population/
  聚合可比性问题，不能反过来否定同local-valid上的算法增益。机器收据见
  `neuron_autoresearch/H67_NB0_FULLRES_HEAD_TO_HEAD_20260805.json`。
- H67 ep25到ep30 AEE仍改善2.47%，NB0 ep24到ep29 AEE仍改善5.56%；端点 dominance 已成立，
  但两者收敛签核仍等待equal+10，不能混为一个结论。

## 2026-08-05 三聚合结果的当前证据边界

- 现有 H67 ep30 与 NB0 ep29 fullres `spike_profile.json` 生成于多聚合升级之前，只保存 production frame-equal 指标；它们没有 `metric_aggregation_audit`，因此当前不能定量声称 pixel-global 或 sequence-balanced 能解释多少 official-test 差距。
- equal+10 流水线在独立目录重新评估 H67 labels 30/35/40 与 NB0 labels 29/34/39。profile reuse 合同强制要求三种聚合、825 frames、18 sequences、validation-list SHA、checkpoint path/SHA 与 config path/SHA；旧 schema 无法通过并会触发重建。
- H67/NB0 staged model 分别与 ep30/ep29 source hardlink；optimizer/scheduler/scaler 状态连续，旧 milestones 清空而 LR 保持不变。历史 state 不含 RNG state，因此该实验是 audited state continuation，不声明逐 bit RNG 复现。
- 在新 source-point profile 实际产生之前，AAE 结论仅限于：公式已通过回归、同 frame-equal/local-valid 下 H67 优于 NB0、official hidden-test population 不可由本地 valid825 替代。

## 2026-08-06 标准DSEC Fl与跨协议措辞修正

- 历史`AEE_outliers`实现以预测流幅值作为5%阈值分母，不是标准DSEC Fl-all；旧值继续保留用于历史
  连续性，但禁止与论文8.91%/10.051%横向比较。
- 新增`DSEC_Fl`，定义为`EPE>3px && EPE>0.05*|GT flow|`，以百分数输出；未来Local5、H67、NB0
  新profile同时保存frame-equal、pixel-global和sequence-balanced三种结果。
- 本地NB0数值只能表述为“under a non-identical local-validation protocol numerically lower”，不能
  写成“复现或超过论文validation”。AAE-2D `6.5128`与official AE-3D `4.871`尤其不能跨公式比较。

## 2026-08-06 SHA绑定机器诊断

- 新增 `NB0_AAE_GAP_DIAGNOSTIC_20260806.json/.md`。生成器逐份绑定 NB0 ep19/24/29、H67
  ep20/25/30 profile SHA，并重新验证指标/评估器/聚合器源码 SHA 和 `8/8 PASS` 指标回归。
- NB0 ep24->29 的 AEE 改善 `5.558%`，故 AEE 尚不能签收敛；AAE-2D/AE-3D 只改善
  `0.437%/1.258%`，角度已接近平台。H67 ep25->30 的对应变化为
  `2.468%/-0.075%/0.179%`，同样是 AEE 边界最优、角度近平台。
- 机器结论固定为：`formula_bug=false`、`NB0_AEE_undertraining_plausible=true`、
  `NB0_angle_gap_explained_by_undertraining_alone=false`。最终 closure 已要求该收据 PASS；equal+10
  会以新三聚合 schema 重评源点和续训点，旧 profile 缺失字段不能用于最终签核。
