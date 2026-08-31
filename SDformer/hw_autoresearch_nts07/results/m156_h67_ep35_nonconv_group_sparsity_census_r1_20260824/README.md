# M156 H67 ep35 非 Conv 成组稀疏快速审计

## 结论

冻结 checkpoint 中没有可直接给硬件跳过的成组稀疏：

- 12 个 FFN pair 的 16/32-channel 成组记录共 1,656 条，float 与 canonical INT8 的 exact-zero group 都是 0。
- 12 个 attention 模块的 shared Q/K/projection 记录共 414 条，exact-zero group 同样是 0。
- FFN 单个 INT8 权重虽然有约 0.85%–1.18% 量化为零，但没有任何整组同时满足 `fc1 output rows + fc2 matching input columns` 为零。

低能量组也没有形成免训练的明显长尾：选最小能量的 25% group 会删掉 24.55%/24.69% 的 paired weight energy；选 50% 则删掉 49.43%/49.60%。这基本等于按通道比例硬删，不能预期无损。

## 硬件与算法决策

- FFN 只有在算法侧训练 shared pair mask 后才值得写 skip RTL：删除 `fc1` 的 16/32 输出通道时，必须同时删除 `fc2` 对应输入列。应先在占 FFN 56.17% 的 stage 2 做 bounded pilot。
- 25% 成组删除若能通过训练保住精度，在当前 compute envelope 中只是 **1.06883× sensitivity**；50% 才是 **1.14784× sensitivity**。两者均非 hardware/system admission。
- attention 不应根据现有 checkpoint 新写 skip 数据路径；如果算法侧不训 shared Q/K/proj group mask，这条性能线继续冻结。

## 证据边界

本审计固定 checkpoint、M41 loader、FFN cycle ledger 和 `docs/359` SHA，并完整扫描冻结模型中的 12 个 FFN pair 与 12 个 attention 模块。它未修改 checkpoint，未训练，未跑 valid825，未生成 address-timed trace，也不接纳硬件或系统加速。

机器可读结果见 `m156_h67_ep35_nonconv_group_sparsity_census.json`，全 group 账本见 `group_ledger.csv`。

## 复跑

```bash
/opt/anaconda3/envs/pytorch310/bin/python \
  hw_autoresearch_nts07/system_simulator/scripts/analyze_m156_h67_ep35_nonconv_group_sparsity_census.py \
  --output <new-empty-output-directory>
```
