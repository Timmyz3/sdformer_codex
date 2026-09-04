# 全量训练前 Review：h37_strict_bsa_qkv_sqrt_signv_neuronfast_reviewed_auto_full_20260521_023504

## 选择结论

- 选择方案：`h37_strict_bsa_qkv_sqrt_signv_neuronfast`
- 选择原因：H37 修正版注意力达到或接近 H36 fallback，优先选择论文范式更干净的 H37。
- 短测指标：AEE=1.5345, AAE=7.5801, SOPs=3.4978G, firing=0.08205
- 源配置：`neuron_experiments/H9_bipolar_self_attention/configs/h37_strict_bsa_qkv_sqrt_signv_neuronfast.yml`
- 全量配置：`neuron_experiments/H9_bipolar_self_attention/configs/h37_strict_bsa_qkv_sqrt_signv_neuronfast_reviewed_auto_full_20260521_023504.yml`

## 范式检查

- 神经元主线：Q/K 使用三值 PSN+ATLIF；高 SOP 替换层使用二值 official ATLIF。
- baseline 完整性：入口仍走 baseline 训练逻辑，改动通过 `neuron_experiments/H9_bipolar_self_attention/overlay` 注入。
- 外部 review 处理：H37 已新增严格 QKV-BSA、二元 alpha-XNOR、QKV-A2OS2A；旧 strict-BSA/alpha-XNOR/A2OS2A 只按 adapted/inspired 表述。
- 学习率策略：采用 H36/H37 短测中对应配置的 differential LR，backbone 小 LR，新神经元/阈值较大 LR。

## 风险

- 如果选择的是 H36 fallback，则注意力仍是 SDFormerFlow 适配范式，不应在论文中写作原版 alpha-XNOR/BSA/A2OS2A。
- 如果选择的是 H37 QKV 分支，则 V 是从 K copy 初始化后独立训练，属于 baseline QKFormer 的结构扩展，需在实验表里单独标注参数量增加。
