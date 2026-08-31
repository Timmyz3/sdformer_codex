# Prosperity 官方 Simulator 复现与 SDformer 适配合同

## 1. 结论

本报告真实调用官方 `Simulator.run_fc` CPU 路径，不再只导入 `Stats`。结果用于验证工具链和确定适配字段，不能当作 Motion/Local5 性能。

## 2. 官方 reference 结果

| 层 | density | product cycles | bit-sparse cycles | 官方周期加速 | g_wgt 读取降低 |
|---|---:|---:|---:|---:|---:|
| fc_q_enc_0 | 0.0835 | 41429 | 73883 | 1.783× | 56.52% |
| fc_o_enc_0 | 0.3622 | 18590 | 106832 | 5.747× | 89.64% |
| fc_2_enc_0 | 0.0381 | 33599 | 44939 | 1.338× | 32.19% |

## 3. CPU import 处理

官方 `simulator.py` 无条件导入 CUDA extension，但 `run_fc` CPU 路径不调用该扩展。本探针只在 `sys.modules` 注入空的 import shim；未修改官方仓库，且周期、product-sparsity 搜索、存储分账均执行官方 Python 源码。

## 4. SDformer 矩阵合同

真实适配输入必须是 `[time_steps, sequence_length, input_dim]` 的 0/1 张量，并绑定主线、sample、block、head、输出维度和语义。计数、密度或 histogram 不能替代逐元素矩阵。

当前状态：合同与校验器已完成；Motion/Local5 真实矩阵等待 fullres follower 增加导出。导出前，旧 `online-matcher oracle` 只能保留为本地解析下界。

## 5. 证据边界

- 本报告真实调用 Prosperity 官方 Simulator.run_fc CPU 周期路径。
- 官方 reference 数字只验证工具链，不代表 Motion/Local5。
- 本网络 profile 只有计数/直方图，不能重构 Prosperity 所需逐元素二值矩阵。
- Phi 未发现公开官方 simulator，不能用本探针冒充 Phi 结果。

## 6. 复现

```bash
/opt/conda/envs/sdformerflow/bin/python scripts/run_prosperity_official_probe.py
```
