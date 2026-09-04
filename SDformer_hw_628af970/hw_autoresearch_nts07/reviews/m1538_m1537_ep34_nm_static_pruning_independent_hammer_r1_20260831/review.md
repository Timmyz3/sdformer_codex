# M1538｜M1537 ep34 N:M 静态剪枝独立打铁

裁决：**M1537 的“FP32 checkpoint、逐 tensor 连续存储顺序、oracle top-N”静态机会数字可复现；只准进入候选池。训练或硬件之前必须做 row-local r2。**评分 92/100，P0=0，P1=2。

## 可确认的事实

- checkpoint、M1531、M1512 与受保护 `docs/359` 的 SHA pin 全部匹配；M1537 内层 manifest 与外层 seal 通过。
- 新目录重跑的 JSON、Markdown、`SHA256SUMS` 与原结果逐字节一致。
- 独立用升序排序重算 5 类 × 6 种 pattern，30 组 weight/group/tail 计数和 oracle L1、L2-squared 删除比例均在 `1e-6` 内一致；selector metadata 等于 `ceil(log2(C(M,N)))`。
- `4:8` 删除的 L1 mass 是 `23.00%--25.00%`；`8:16` 是 `21.23%--23.01%`。这证明“删 50% count”仍会删掉显著权重质量，**不证明 2x、周期、流量、能量或精度**。
- machine-readable boundary 已把 accuracy/AEE/cycles/speedup/traffic/energy/RTL/headline 全设为 false，人类报告也明确重复。因此没有发现把 count/mass 冒充 cycle 或 speedup 的漏洞。

## 两个 P1

1. `patch_embed` 类混入 6 个 `spiking_neuron.weight` 的 `10x10` ATLIF 时间矩阵，共 `600/466872 = 0.128515%`。数值影响小，但不能让 patch N:M 训练脚本顺带剪 C3。r2 必须拆成 patch Conv 权重与 ATLIF temporal 权重。
2. tensor-flat 分组不是最终硬件 row-local 分组。patch 类在 `M=8` 有 `78/58356 = 0.133665%` group 跨 storage row，在 `M=16` 有 `90/29178 = 0.308452%`。其余四类本次 pattern 均为 0 crossing。r2 必须按实际 reduction/fetch row 重分组，并绑定转换后 layout SHA。

## 合法使用方式

M1537 可写成：“ep34 静态筛选表明，50% N:M oracle mask 会删除约 21%--25% L1 权重质量，因此无损 N:M 关闭，只保留 hardware-aware retraining 候选。”不能写“50% sparsity 带来 2x”，也不能进入性能主表。

顺序门保持为：可执行 row-local 静态审计 -> 新 checkpoint/mask 身份与 paired official AEE（overall Delta-AEE `<=0.02`、每序列 `<=0.03`）-> address-timed same-resource replay -> RTL/VCS/DC/PTPX。硬件门是局部周期 `>=1.15x`；或者周期退化 `<=5%` 且实测 weight bytes `>=30%`、memory energy `>=20%` 下降。metadata、selector/decompressor、tail、bank/port/conflict、psum 与 dense commit 必须全部收费。
