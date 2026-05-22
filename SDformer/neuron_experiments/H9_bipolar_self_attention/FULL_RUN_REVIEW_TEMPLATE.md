# 全量训练前 Review 模板

每个候选进入全量训练前必须单独复制本模板并填写。没有 review 的候选不启动全量。

## 1. 候选身份

- 实验名：
- 配置文件：
- 计划输出目录：
- 续训 checkpoint：
- 短测来源：
- 短测指标：
  - steps：
  - valid samples：
  - AEE：
  - AAE：
  - SOPs：
  - firing：

## 2. 神经元范式检查

- 所有替换节点是否都是 `PSN + ATLIF`：
- Q/K 是否为三值 `PSN+ATLIF`：
- 非 Q/K 的高 SOP/FFN/downsample 是否为二值 `official_atlif PSN+ATLIF`：
- 是否存在非 ATLIF 的实验性神经元混入：
- 是否存在 target-rate ATLIF 被误称为 official ATLIF：

## 3. 论文/开源范式依据

- ATLIF 依据：`optimization_sources/neuron_optimization/ATLIF_Activity-Pruning-SNN`
- 三值/注意力依据：
  - BSA：`Bipolar Self-attention for Spiking Transformers` 的 `sign(Q) @ sign(K)^T -> Shiftmax` 思路；
  - alpha-XNOR：二值/三值 spike 相似性打分扩展；
  - signed consensus：signed popcount gate，作为 BSA/三值发放兼容的硬件友好变体。
- 与开源/论文不一致处：
- 这些不一致是否是因为 SDFormerFlow 的 QKFormer 无独立 V 分支：

## 4. 接入方式检查

- 是否通过 `neuron_experiments/H9_bipolar_self_attention/overlay` 接入：
- 是否修改 third_party baseline：
- 是否只通过入口脚本/config 调用：
- 是否保存了独立 config 和日志路径：

## 5. 学习率与训练策略

- backbone LR：
- ATLIF neuron LR：
- ATLIF threshold 参数 LR：
- threshold_update base LR：
- activity_eta / threshold_lr_scale：
- 是否 AMP：
- batch/workers/pin_memory：
- 选择依据：

## 6. 风险判断

- 精度风险：
- AAE 风险：
- 稀疏不足风险：
- 硬件友好性风险：
- 需要额外 profile/check 的点：

## 7. 结论

- 是否允许进入全量：
- 若允许，推荐全量配置：
- 若不允许，下一步短测：
