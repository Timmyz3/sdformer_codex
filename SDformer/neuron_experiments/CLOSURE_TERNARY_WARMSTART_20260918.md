# 结论封存：三元 warm start 线（t49-t52 / h12*）—— 2026-09-18

**裁决：`NO_GO_TERNARY_WARMSTART__PIVOT_BINARY_PRUNING_HW`。**
binary + 剪枝 + 硬件线为主；三元降级为从头训练的独立实验（不在 TCAS-II 主线上）。

## 1. 完整证据链（全部可复查）

| 实验 | 配置 | 结果 | 位置 |
|---|---|---|---|
| t49 三元 zero-shot（不训练直接评） | 三元转换 ep34 | **firing 46.5%、spikes 597G（基线 82G 的 7.3×）、AEE 20.19/84.4/82.9、outlier 99.8%** | `H9_bipolar_self_attention/results/t49_ternary_zeroshot_ep34/standard_valid825/epoch34/eval.log` |
| t49 三元 5-epoch 全量 | θ=1.0 + H9 缺陷 backward | val loss 7.72 → **8.41 发散** | `results/t49_ternary_ws4_ep5/train.log`（无 AEE 评估） |
| t52 系列 | **已退回 output_mode: binary** | — | `configs/generated/t52_*.yml` |
| h12cal（θ 重标定 v1） | 一次性分位θ + H12 backward | AEE 7.78、firing 0.83 饱和 | `H12_atlif_gradfix/results/h12cal_ternary_ws4_ep5/` |
| h12s0（+stage0-local5） | 同上 + stage0 换 local5 | AEE 7.78（与 h12cal 几乎相同 → 与 stage0 变量无关） | `results/h12s0_local5_ternary_ep5/` |
| h12cal2 probe（θ 迭代标定 v2） | 闭环 5 轮收敛（firing 0.32） | **AEE 8.39 更差** | `results/h12cal2_probe/` |
| h12base probe（**θ=1.0 原样**） | 只隔离 H12 backward，600 步 | **AEE 7.76、firing 0.060** | `results/h12base_probe/` |

## 2. 根因判定

1. **三元转换本身破坏网络**：zero-shot（不训练）firing 6%→46.5%（7.3×），AEE 20-84。正阈值不变（θ=1.0）的前提下发放率不该变——转换在时间混合器/center/装载映射某处不等价（**未定案，见 §4 遗留**）。
2. **不是 θ 标定的错**：v1（往小调）与 v2（迭代闭环往大调，firing 已收敛 0.32）都 ~8 → θ 重标定救不了坏转换。
3. **不是 H12 backward 的错**：h12base（θ=1.0 原样、静态阈值）下 H12 与 H9 行为等价（backward 修复只改 grad_thre，θ 静态时不生效）；单元测试 4 判据全过（`H12_atlif_gradfix/tests/test_gradfix.py`）。
4. **ep34 权重与 binary 语义共同训练适配**：负脉冲（{-1}×1.0，neg_thre=4.0）与任何 θ 重缩放都会打碎协同；残差路径上不同尺度信号直接相加，5 epoch 无法恢复。
5. 另一条线已用脚投票（t52 退回 binary），本文档为正式封存。

## 3. 存活资产（全部有效，不受本裁决影响）

- **binary 主线**：H67 ep35（AEE 1.3287）/ **local5 ep44（1.2819，更优）**，硬件定点版 1.2786-1.2804。
- **H12 修复 backward**：四个 Surrogate 的 grad_thre 恒负/缺恒等项缺陷已修复并有单元测试——对未来**从头训练**或 eta>0 场景仍然必要。
- **θ=1.0 免存储叙事**：硬件加分项不变。
- **motion S1 popcount 恒等式**：bit-exact 已证（9 组对拍），与三元无关，binary 线直接受益。
- **硬件调研 Top10**：全部结论不变（死神经元→LoAS 式剪枝、MM2IM、跳零、ICG、模式复用…）。
- 死神经元逐层取证（`/tmp/deadprobe_*.json`）——注意那是**三元模块**上的测量，binary 线需重新取证。

## 4. 遗留未定案（三元从头训练前必须回答）

zero-shot 转换的 46% 发放爆炸源头未定位。候选：ATLIFTernaryPSN 装载时时间混合器 W/bias 与 binary PSN 不等价、center 语义、neg_thre=4 的不对称效应。差分探针（同一输入过 binary PSN vs 转换后三元模块，对比 h_seq 分布）半天可定位。定位前不要投任何三元训练。

## 5. 新主线（binary + 剪枝 + 硬件）

1. binary 线死模块取证 → LoAS 式静默剪枝 mask → 剪枝 + 短 ft → valid825 AEE + SOPs。
2. S1 popcount 位等价重写（`h60_pc`）→ ckpt 直载 → bit-exact 确认。
3. 死神经元旁路 + ICG → 重跑 DC/PTPX（远程 Synopsys 交付包流程）。
4. MM2IM/跳零前置统计 → 定 RTL 投入优先级。
