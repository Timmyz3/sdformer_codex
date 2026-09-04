# SDFormerFlow 神经元替换与融合实验矩阵

本文档是神经元算子优化实验的索引表。核心约束：

- **不把实验代码写进 `third_party/SDformerFlow` baseline 文件夹**；
- 每个实验在 SDformer repo 内有独立目录；
- 每个实验目录保存该实验所有代码改动文件；
- 训练/推理入口可以继续使用 baseline，也可以放进实验目录；
- 运行时通过实验 overlay 目录优先 import 修改文件，baseline 中未修改的文件继续从 `third_party/SDformerFlow` 调用。

## 0. 总体目录

当前已经在 repo 根目录生成：

```text
neuron_experiments/
    README.md
    _templates/
        overlay_runner_notes.md
        configs/
            smoke_template.yml
            subset_template.yml
            full_template.yml
    E0_psn_baseline/
        README.md
        configs/
            smoke.yml
            subset.yml
            full.yml
        overlay/
        results/
    E1_exp_sn/
        README.md
        entrypoints/
            train.py
            eval.py
        configs/
            smoke.yml
            subset.yml
            full.yml
        overlay/
            models/
                __init__.py
                STSwinNet_SNN/
                    Spiking_modules.py
                    experimental_neurons/
                        __init__.py
                        base.py
                        factory.py
                        single/
                            __init__.py
                            sn.py
        results/
    E2_exp_atlif/
    E3_exp_lmh/
    E4_exp_tslif/
    E5_exp_tsn/
    F1_fused_adaptive_psn/
    F2_fused_lmh_atlif/
    F3_fused_adaptive_tslif/
    F4_fused_lmh_tslif/
    F5_fused_signed_hybrid/
```

`_templates/` 只做模板，不作为实际训练时的代码来源。每个实验以自己的 `overlay/` 为准，保证实验可复现。

如需重新生成这些实验目录，运行：

```bash
python tools/scaffold_neuron_experiments.py
```

重新生成会覆盖各实验的入口、配置、overlay 和结果模板。正式跑完的结果文件建议先备份或提交后再重新生成。

## 1. Overlay 调用机制

有两种入口模式。后续优先使用 **模式 B：实验自带入口**，因为它把配置、代码改动、启动逻辑都封在同一个实验目录里。

### 模式 A：baseline 入口 + 实验 overlay

baseline 原本调用链：

```text
third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py
  -> configs/parser.py
  -> models/STSwinNet_SNN/Spiking_modules.py
  -> models/STSwinNet_SNN/Spiking_submodules.py
```

实验 overlay 后的调用链：

```text
tools/run_sdformerflow_overlay.py
  -> 设置 sys.path:
       1. neuron_experiments/<ID>/overlay
       2. third_party/SDformerFlow
       3. repo root
  -> runpy 执行 baseline 入口:
       third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py
  -> baseline 入口继续正常 import
  -> 如果某个模块在 overlay 中存在，优先使用 overlay 版本
  -> 如果 overlay 中不存在，回落到 third_party/SDformerFlow baseline 版本
```

也就是说，入口仍是 baseline 的训练/推理脚本，但被改动的 Python 文件来自当前实验目录。

`overlay/models/__init__.py` 需要使用 namespace 扩展，保证 `models` 包既能读 overlay 文件，也能回落 baseline 文件：

```python
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
```

### 模式 B：实验自带入口 + 实验 overlay

实验目录中保存入口文件：

```text
neuron_experiments/<ID>/entrypoints/train.py
neuron_experiments/<ID>/entrypoints/eval.py
```

调用链：

```text
neuron_experiments/<ID>/entrypoints/train.py
  -> 设置 sys.path:
       1. neuron_experiments/<ID>/overlay
       2. third_party/SDformerFlow
       3. repo root
  -> 切换 workdir 到 third_party/SDformerFlow
  -> 调用 baseline 的训练逻辑，或运行该实验复制后的入口逻辑
  -> 未修改模块继续从 third_party/SDformerFlow 读取
```

模式 B 的优势：

- 每个实验目录自包含，后续复现实验时不需要记复杂命令；
- 如果某个实验需要改 train loop，例如 AT-LIF activity regularizer，可以只改该实验的 `entrypoints/train.py`；
- baseline 入口保持完全不动；
- 相比把入口复制到 `overlay/`，放在 `entrypoints/` 更清楚，因为它是启动脚本，不是被 baseline import 的模块。

## 2. 不允许直接改动的 baseline 文件

以下文件作为 baseline 骨架，不把实验代码写进去：

```text
third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py
third_party/SDformerFlow/eval_DSEC_flow_SNN.py
third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py
third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py
third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py
third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py
```

如果某个实验需要修改这些文件，做法不是改 baseline，而是在该实验目录复制同路径文件到 `overlay/`：

```text
neuron_experiments/E3_exp_lmh/overlay/models/STSwinNet_SNN/Spiking_modules.py
```

运行 E3 时使用这份 overlay 文件；运行其他实验时不受影响。

## 3. 运行配置层级

每个实验都有独立配置文件，不放进 `third_party/SDformerFlow/configs/`。

| 层级 | 用途 | 配置位置 | 基础来源 |
|---|---|---|---|
| smoke | 1 条 train + 1 条 valid，验证前向/反向/保存日志 | `neuron_experiments/<ID>/configs/smoke.yml` | 复制自 `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml`，并覆写为实验目录内的小 split |
| subset | 小规模 DSEC subset，观察 loss、AEE、firing rate | `neuron_experiments/<ID>/configs/subset.yml` | 复制自 `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml` |
| full | 正式训练或较长训练 | `neuron_experiments/<ID>/configs/full.yml` | 复制自 full/fastsafe 配置 |

所有配置只改这些字段：

```yaml
experiment: <experiment_id>

spiking_neuron:
    neuron_type: <target_neuron_type>
    num_steps: 10
    v_th: 0.1
    v_reset: Null
    surrogate_fun: surrogate.ATan()
    tau: 2.
    detach_reset: True
    spike_norm: "BN"

experimental_neuron:
    enabled: True
    # 当前实验专属参数

# 仅 smoke 配置额外包含：
data:
    sequence_list_overrides:
        train: /abs/path/to/neuron_experiments/_templates/smoke_train_split_seq.csv
        valid: /abs/path/to/neuron_experiments/_templates/smoke_valid_split_seq.csv
test:
    sample: 1
```

## 4. 通用运行命令

推荐使用实验自带入口。

模式 B，训练 smoke：

```bash
python neuron_experiments/E1_exp_sn/entrypoints/train.py \
  --config neuron_experiments/E1_exp_sn/configs/smoke.yml
```

模式 B，训练 subset：

```bash
python neuron_experiments/E1_exp_sn/entrypoints/train.py \
  --config neuron_experiments/E1_exp_sn/configs/subset.yml
```

模式 B，推理/验证：

```bash
python neuron_experiments/E1_exp_sn/entrypoints/eval.py \
  --config neuron_experiments/E1_exp_sn/configs/eval.yml
```

模式 A，baseline 入口 + overlay launcher，训练 smoke：

```bash
python tools/run_sdformerflow_overlay.py \
  --overlay neuron_experiments/E1_exp_sn/overlay \
  --entry third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py \
  --workdir third_party/SDformerFlow \
  -- \
  --config ../../neuron_experiments/E1_exp_sn/configs/smoke.yml
```

模式 A，训练 subset：

```bash
python tools/run_sdformerflow_overlay.py \
  --overlay neuron_experiments/E1_exp_sn/overlay \
  --entry third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py \
  --workdir third_party/SDformerFlow \
  -- \
  --config ../../neuron_experiments/E1_exp_sn/configs/subset.yml
```

模式 A，推理/验证：

```bash
python tools/run_sdformerflow_overlay.py \
  --overlay neuron_experiments/E1_exp_sn/overlay \
  --entry third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --workdir third_party/SDformerFlow \
  -- \
  --config ../../neuron_experiments/E1_exp_sn/configs/eval.yml
```

`--workdir third_party/SDformerFlow` 是为了保持 baseline 中相对数据路径的行为不变。

运行训练前需要确认 baseline 依赖已经安装，至少包括：

```text
mlflow
spikingjelly==0.0.0.0.14
```

当 `SDFORMER_USE_MLFLOW=0` 时，实验入口会容忍环境缺少 `mlflow`；但 `spikingjelly` 是模型核心依赖，必须安装。

## 5. 每个实验目录的最低内容

每个实验至少包含：

```text
neuron_experiments/<ID>/
    README.md
    entrypoints/
        train.py
        eval.py
    configs/
        smoke.yml
        subset.yml
        full.yml
    overlay/
        models/
            __init__.py
            STSwinNet_SNN/
                Spiking_modules.py
                experimental_neurons/
                    __init__.py
                    base.py
                    factory.py
                    ...
    results/
        metrics.md
        run_commands.md
```

如果某个实验不需要改 `Spiking_swin_transformer3D.py`，就不要复制它。原则是：**overlay 里只放本实验确实改过或新增的文件**。

`entrypoints/train.py` 和 `entrypoints/eval.py` 属于实验启动逻辑，也算该实验代码的一部分。它们可以有两种写法：

```text
轻量 wrapper：只设置路径、workdir，然后 runpy 执行 baseline 入口
复制改造入口：当实验需要改 train/eval loop 时，在本实验入口里保存改造后的逻辑
```

第一轮神经元替换优先使用轻量 wrapper；第二轮如果加入 activity regularizer、特殊 spike 统计、额外日志，再使用复制改造入口。

## 6. 当前已有 repo 侧候选文件

这些文件已经在 SDformer repo 中，用于独立验证候选神经元的 shape/backward，不属于 baseline 改动：

| 文件 | 作用 |
|---|---|
| `src/models/modules/spiking_neurons/candidates/common.py` | 候选神经元公共 surrogate/helper |
| `src/models/modules/spiking_neurons/candidates/sn.py` | Simple SN |
| `src/models/modules/spiking_neurons/candidates/atlif.py` | AT-LIF |
| `src/models/modules/spiking_neurons/candidates/lmh.py` | LMH |
| `src/models/modules/spiking_neurons/candidates/tslif.py` | TS-LIF |
| `src/models/modules/spiking_neurons/candidates/tsn.py` | TSN |
| `tests/test_candidate_spiking_neurons.py` | 候选神经元单测 |

后续接入训练时，从这些候选实现复制/适配到对应实验的 `overlay/models/STSwinNet_SNN/experimental_neurons/`。

## 7. 单独替换实验表

| 实验 ID | 实验名 | `neuron_type` | 实验目录 | 实验入口 | overlay 调用模块 | overlay 需要存放的代码文件 | 配置文件 |
|---|---|---|---|---|---|---|---|
| E0 | PSN baseline | `psn` | `neuron_experiments/E0_psn_baseline/` | `entrypoints/train.py`, `entrypoints/eval.py` | 无实验神经元，使用 baseline PSN | 通常 `overlay/` 为空；只放配置和结果 | `configs/smoke.yml`, `configs/subset.yml`, `configs/full.yml` |
| E1 | Simple SN | `exp_sn` | `neuron_experiments/E1_exp_sn/` | 同左 | `experimental_neurons/single/sn.py::SNNode` | `models/__init__.py`, `Spiking_modules.py`, `experimental_neurons/base.py`, `factory.py`, `single/sn.py` | 同左 |
| E2 | AT-LIF | `exp_atlif` | `neuron_experiments/E2_exp_atlif/` | 同左 | `experimental_neurons/single/atlif.py::ATLIFNode` | E1 公共入口文件 + `single/atlif.py` + 可选 `stats.py` | 同左 |
| E3 | LMH | `exp_lmh` | `neuron_experiments/E3_exp_lmh/` | 同左 | `experimental_neurons/single/lmh.py::LMHNode` | E1 公共入口文件 + `single/lmh.py` | 同左 |
| E4 | TS-LIF | `exp_tslif` | `neuron_experiments/E4_exp_tslif/` | 同左 | `experimental_neurons/single/tslif.py::TSLIFNode` | E1 公共入口文件 + `single/tslif.py` | 同左 |
| E5 | TSN | `exp_tsn` | `neuron_experiments/E5_exp_tsn/` | 同左 | `experimental_neurons/single/tsn.py::TSNNode` | E1 公共入口文件 + `single/tsn.py` + `stats.py` | 同左 |

### E1 配置差异

```yaml
experiment: E1_exp_sn
spiking_neuron:
    neuron_type: exp_sn
```

### E2 配置差异

```yaml
experiment: E2_exp_atlif
spiking_neuron:
    neuron_type: exp_atlif
experimental_neuron:
    threshold_learnable: True
    threshold_min: 0.02
    threshold_max: 2.0
    activity_regularizer: 0.0
```

### E3 配置差异

```yaml
experiment: E3_exp_lmh
spiking_neuron:
    neuron_type: exp_lmh
experimental_neuron:
    levels: 2
    temporal_mix_init: identity
    detach_mem: True
```

### E4 配置差异

```yaml
experiment: E4_exp_tslif
spiking_neuron:
    neuron_type: exp_tslif
experimental_neuron:
    gamma: 0.5
    decay_factor: [0.8, 0.2, 0.3, 0.7]
    learnable_decay: True
```

### E5 配置差异

```yaml
experiment: E5_exp_tsn
spiking_neuron:
    neuron_type: exp_tsn
experimental_neuron:
    signed_spike: True
    fire_ratio: 1.0
```

TSN 的 activity rate 统一使用 `abs(spike).mean()`，不要和二值 spike 的 `spike.mean()` 混用。

## 8. 融合实验表

| 实验 ID | 实验名 | `neuron_type` | 实验目录 | 实验入口 | 融合内容 | overlay 调用模块 | overlay 需要存放的代码文件 |
|---|---|---|---|---|---|---|---|
| F1 | Adaptive PSN | `fused_adaptive_psn` | `neuron_experiments/F1_fused_adaptive_psn/` | `entrypoints/train.py`, `entrypoints/eval.py` | PSN 时间混合 + AT-LIF 阈值 | `experimental_neurons/fused/adaptive_psn.py::AdaptivePSNNode` | `Spiking_modules.py` + factory/base + `fused/adaptive_psn.py` + `stats.py` |
| F2 | LMH-AT-LIF | `fused_lmh_atlif` | `neuron_experiments/F2_fused_lmh_atlif/` | 同左 | LMH 时间混合 + 自适应阈值 LIF | `experimental_neurons/fused/lmh_atlif.py::LMHATLIFNode` | `Spiking_modules.py` + factory/base + `fused/lmh_atlif.py` + `stats.py` |
| F3 | Adaptive TS-LIF | `fused_adaptive_tslif` | `neuron_experiments/F3_fused_adaptive_tslif/` | 同左 | TS-LIF 双 compartment + 自适应阈值 | `experimental_neurons/fused/adaptive_tslif.py::AdaptiveTSLIFNode` | `Spiking_modules.py` + factory/base + `fused/adaptive_tslif.py` + `stats.py` |
| F4 | LMH-TS-LIF | `fused_lmh_tslif` | `neuron_experiments/F4_fused_lmh_tslif/` | 同左 | LMH 全局时间混合 + TS-LIF 局部时间动态 | `experimental_neurons/fused/lmh_tslif.py::LMHTSLIFNode` | `Spiking_modules.py` + factory/base + `fused/lmh_tslif.py` |
| F5 | Signed Hybrid | `fused_signed_hybrid` | `neuron_experiments/F5_fused_signed_hybrid/` | 同左 | binary spike + TSN signed spike gated mixture | `experimental_neurons/fused/signed_hybrid.py::SignedHybridNode` | `Spiking_modules.py` + factory/base + `fused/signed_hybrid.py` + `stats.py` |

### F1 配置差异

```yaml
experiment: F1_fused_adaptive_psn
spiking_neuron:
    neuron_type: fused_adaptive_psn
experimental_neuron:
    threshold_learnable: True
    threshold_min: 0.02
    threshold_max: 2.0
```

### F2 配置差异

```yaml
experiment: F2_fused_lmh_atlif
spiking_neuron:
    neuron_type: fused_lmh_atlif
experimental_neuron:
    levels: 2
    temporal_mix_init: identity
    threshold_learnable: True
    threshold_min: 0.02
    threshold_max: 2.0
```

### F3 配置差异

```yaml
experiment: F3_fused_adaptive_tslif
spiking_neuron:
    neuron_type: fused_adaptive_tslif
experimental_neuron:
    gamma: 0.5
    decay_factor: [0.8, 0.2, 0.3, 0.7]
    threshold_learnable: True
```

### F4 配置差异

```yaml
experiment: F4_fused_lmh_tslif
spiking_neuron:
    neuron_type: fused_lmh_tslif
experimental_neuron:
    levels: 2
    temporal_mix_init: identity
    gamma: 0.5
    decay_factor: [0.8, 0.2, 0.3, 0.7]
```

### F5 配置差异

```yaml
experiment: F5_fused_signed_hybrid
spiking_neuron:
    neuron_type: fused_signed_hybrid
experimental_neuron:
    gate_init: 0.0
    signed_spike: True
    binary_branch: exp_sn
    signed_branch: exp_tsn
```

## 9. 需要特殊 overlay 的 baseline 文件

大部分实验只需要 overlay `Spiking_modules.py` 和新增 `experimental_neurons/`。

少数情况可能需要 overlay 额外 baseline 文件：

| 文件 | 何时需要 overlay | 原因 |
|---|---|---|
| `models/STSwinNet_SNN/Spiking_swin_transformer3D.py` | 需要修改 attention scale 特判时 | 当前只对 `psn/glif` 设置 `scale=1`，新神经元可能也需要 |
| `train_flow_parallel_supervised_SNN.py` | 第二轮加入 activity regularizer 时 | 优先把改动保存在 `entrypoints/train.py`；只有必须被 import 时才放 overlay |
| `eval_DSEC_flow_SNN.py` | 需要推理时统计 signed spike 或新 firing rate 时 | 优先把改动保存在 `entrypoints/eval.py`；TSN/F5 的统计口径不同 |

这些文件依然不改 baseline，只复制到当前实验的 `overlay/` 中。

## 10. 统计口径

二值神经元：

```text
firing_rate = spike.mean()
```

三值神经元：

```text
activity_rate = spike.abs().mean()
positive_rate = (spike > 0).float().mean()
negative_rate = (spike < 0).float().mean()
```

E5/F5 必须使用三值统计口径。

## 11. 实验推进顺序

推荐顺序：

```text
E0 -> E1 -> E2 -> E3 -> E4 -> E5
F1 -> F2 -> F3 -> F4 -> F5
```

每个实验必须按：

```text
smoke pass -> subset pass -> full train
```

推进。

## 12. 结果记录表模板

每个实验的结果写在：

```text
neuron_experiments/<ID>/results/metrics.md
```

模板：

| 实验 ID | 配置文件 | overlay 目录 | commit/hash | 数据层级 | 是否通过 | train loss | valid AEE | firing/activity rate | 显存 | throughput | 备注 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| E1 | `configs/smoke.yml` | `overlay/` |  | smoke |  |  |  |  |  |  |  |
| E1 | `configs/subset.yml` | `overlay/` |  | subset |  |  |  |  |  |  |  |
| E1 | `configs/full.yml` | `overlay/` |  | full |  |  |  |  |  |  |  |

## 13. 最小实现检查清单

每新增一个实验神经元，至少完成：

```text
[ ] 新增 `neuron_experiments/<ID>/README.md`
[ ] 新增 `neuron_experiments/<ID>/entrypoints/train.py`
[ ] 新增 `neuron_experiments/<ID>/entrypoints/eval.py`
[ ] 新增 `neuron_experiments/<ID>/configs/smoke.yml`
[ ] 新增 `neuron_experiments/<ID>/configs/subset.yml`
[ ] 新增 `neuron_experiments/<ID>/configs/full.yml`
[ ] 新增 `neuron_experiments/<ID>/overlay/models/__init__.py`
[ ] 在该实验 overlay 中复制并修改 `models/STSwinNet_SNN/Spiking_modules.py`
[ ] 在该实验 overlay 中新增 `models/STSwinNet_SNN/experimental_neurons/factory.py`
[ ] 在该实验 overlay 中新增 single/fused neuron 模块
[ ] 确认 `third_party/SDformerFlow` 没有新增或修改实验代码
[ ] 跑候选神经元 shape/backward 单测
[ ] 用 overlay launcher 跑 DSEC single-sequence smoke
[ ] 记录命令、配置、日志路径和指标到 `results/`
```
