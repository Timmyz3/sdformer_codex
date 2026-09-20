# 行动方案：死神经元 + 阈值卡死 + motion 简化 + 顶会 idea（2026-09-17）

来源：5 路并行取证与调研（死神经元取证 ✓、算法调研 ✓、硬件调研 ✓；motion 审计 ✗、TCAS-II 画像 ✗ 待补）。
配套文档：`H12_ATLIF_GRADFIX_TERNARY_DESIGN_20260917.md`、`H12_atlif_gradfix/README.md`。

## 0. 一句话裁决

三值线当前不是调参问题，而是"**尺度失配 + 梯度缺陷**"双问题。主线 = **per-layer 静态 2 的幂 θ 标定**（保 `threshold_eta=0` 免存储叙事）+ **H12 修复 backward 全量替换进队列**（正确性前提）；H12 自适应闭环（η>0）仅作对照消融；12 个 sn2_q 死参数从统计中剔除（零成本）。

## 1. 死神经元取证（决定性证据）

方法：探针脚本 `/tmp/dead_neuron_probe.py`（stub sys.modules 绕开版本漂移，env312），2 个 DSEC 验证 batch × 2 个 ckpt（t49_ternary_negscale/ns4/ep36、t51 ep35），收集全部 81 个 ATLIFTernaryPSN 模块的 r/pos_r/neg_r/thresh，另测 **θ×0.1 反事实**区分死因。逐层表：`/tmp/deadprobe_table_{t49ep36,t51ep35}.json`。

**死模块表（r<0.001，两 ckpt 一致，全部在 swin attention sn_q/sn_k）**：

| # | 模块 | r (t49/t51) | r_x01 反事实 | 死因 |
|---|------|------------|-------------|------|
| 1 | layers.1.swin_blocks.0.attn.sn_k | 0.000/0.000 | 0.992/0.995 | 尺度死（真死） |
| 2 | layers.2.swin_blocks.3.attn.sn_q | 8e-6/2e-5 | 1.000/1.000 | 尺度死 |
| 3 | layers.2.swin_blocks.3.attn.sn_k | 7e-6/2.4e-5 | 0.888/0.891 | 尺度死 |
| 4 | layers.1.swin_blocks.1.attn.sn_q | 6e-5/6.1e-5 | 0.981/0.980 | 尺度死 |
| 5 | layers.0.swin_blocks.0.attn.sn_q | 1.4e-4/1.2e-4 | 1.000/1.000 | 尺度死 |
| 6 | layers.1.swin_blocks.0.attn.sn_q | 3e-4/3.6e-4 | 0.963/0.968 | 尺度死 |

- **尺度死是主因**：θ×0.1 反事实下死模块全部复活（r 0.89–1.00）；**全网平均 r 0.03 → 0.90**——不止死模块，整个网络被 θ=1.0 压制。warm start 源 ckpt 是 v_th=0.1 二值谱系，θ=1.0 差 10×。注意 θ=0.1 也过饱和（r_x01≈1.0），**直接整体 ×0.1 不是解，必须逐层标定**。
- **结构死是次因**：12 个 `attn.sn2_q` target（Shiftmax overlay 不可达，所有 batch forward calls=0）——是死参数而非死神经元，但污染 installer 的 num_modules=105 / threshold_mean 统计。conv/FFN/decoder/patch_embed 路径**无一死模块**（r 0.01–0.19）。
- **θ 卡死 1.0 机理**：threshold_eta=0 刻意决策 + H9 Surrogate backward 四缺陷叠加（即便开 eta，缺陷版 Adam 也只会单调推高阈值）。队列实验全部还在用 H9 缺陷版。

## 2. 队列裁决

1. **立即停排 H9 缺陷 backward 版新实验**——缺陷梯度不仅卡死 θ，还向经发放神经元回传错误梯度，污染上游权重更新；t49/t51 已有数字全部在缺陷梯度下产生。
2. **H12 修复 backward 合入队列路径**（正确性前提，非可选）。
3. **主线 = per-layer 静态 2 的幂 θ 标定 + threshold_eta=0 不变**：~100 个 DSEC batch 前向（~10 min GPU）收集每模块膜电位分布，取 99% 分位 ÷ 三值步长后 round 到 2 的幂。2 的幂 → 硬件仅移位，保住免逐神经元阈值存储承诺。**禁止全网络统一 θ**（各层谱系不同，上表已证）。
4. **H12 自适应闭环（η>0）仅作对照消融**：自适应 θ 收敛依赖代理梯度质量，且收敛值若非 2 的幂需再量化。
5. **t51_lrprobe 结果只作 LR 安全边界参考**：两个探针都在"失配 + 错梯度"状态下跑，不能作为最终 LR；若 3e-4 发散则修复后 LR 上限取 1e-4 附近。**不要等它跑完才修。**
6. **零成本立做**：sn2_q 死参数从 installer 统计中剔除（install 路径已有 unreachable-skip，只差统计口径对齐）。

## 3. motion 注意力简化（审计 agent 失败，以下为保守结论）

- 当前算子份额 0.59%，任何 attention 侧简化都撞同一天花板。
- **推荐档 A（零风险）**：只清理 12 个不可达 sn2_q 死参数 + sn_q/sn_k 随 θ 标定复活，不动计算图——bit 级等价。
- **档 B（不推荐先做）**：去掉 sn2_q(sum(Q)) 动态幅度门。不等价（sn2_q 是数据相关乘性门），AEE 风险中-高且收益同样撞 0.59% 天花板。仅当档 A 后探针显示存活 sn2_q 门输出近常数时作消融。
- 完整算子级审计待补跑（bsa_attention.py ternary_alpha_xnor_shiftmax 路径逐项核算 + 数值对拍）。

## 4. 顶会 idea Top-N（全部经 WebSearch 核实出处；过滤标准：必须作用于 99.4% 主体算子）

| # | idea | 出处 | 落点 | 第一步 |
|---|------|------|------|--------|
| 1 | **per-layer 静态 θ 标定 + H12 backward** | Nat Commun 2024 (TIME 按层阈值重置, s41467-024-51110-5) 思想同源 | PSN/FFN 发放层 | 100-batch 膜电位收集 → θ 表 → 5 epoch 重训 |
| 2 | **Ternary Spike surrogate 公式核对** | AAAI 2024 (Guo et al.) | H12 backward | 逐条对照其梯度公式（恒等项/负支路符号），单元测试对拍 |
| 3 | **LoAS 式双稀疏 + silent-neuron 合法化** | MICRO 2024 (arXiv:2407.14073, RTL 开源) | PSN W_TxT / conv | 死神经元 r 表 → 剪枝 mask（先微调后整列置零），spMspM 列压缩仿真估 SOPs 削减 |
| 4 | **三值×spike 电路退化：主体算子全路径去乘法** | TESSA (MICRO 2022) 一脉电路基础 | conv/deconv PE + PSN | PE 改 sign-select-add，DC 重综合 + PTPX，对比 3902 μm² / 6.25 mW 基线 |
| 5 | **TMA 时间压缩 + DTSS 逐层时间步** | ICCV 2023 (2303.11629) / arXiv 2311.16456 | patch embed 前 T→少片聚合 | 插入式浅 MLP，warm start 兼容；DTSS 静态 per-layer T 适配 333MHz DC |
| 6 | **QKFormer 式 Spiking Patch Embed** | NeurIPS 2024 (2403.16552) | patch embed（RTL 缺口） | 建模替换 + 补 RTL，覆盖 PTPX 评估盲区 |
| 7 | **spike 门控 SRAM 读使能** | LoAS/SpinalFlow 共有思想 | conv/deconv 输入寄存器堆 | 读使能 AND 门 + bank 计数器，PTPX 量化动态功耗削减 |

**不投入**（attention 0.59% 天花板或已证伪）：attention 内部优化、BitStopper 类 attention 早决、token 级动态剪枝、XNOR-popcount 全栈二值（AAE 71.6）、官方 TSN（29.77）、三元扩 FFN（SOPs 变密）、neg_scale=30、Winograd、样本级早期退出、事件点云稀疏 conv、CIM、MS 多脉冲主攻（与三值线幅度假设冲突，降为备选）。

**硬件调研关键情报**：
- LoAS 的 **silent-neuron 消除**与"死神经元"问题正面撞上——可把死神经元反转为合法剪枝机会（先微调后整列置零 mask），且其 FTP 数据流与 PSN addmm(W_TxT) 全时间并行同构，有开源 RTL 可参照。
- 对比表口径：本项目综合级数据可与 LoAS/Prosperity/TESSA/SATA（均评估级）同口径粗比（最好把 LoAS 开源 RTL 在同 28nm 库重综合）；**ReckOn（ISSCC/JSSC 2023）是唯一 28nm 硅实测锚点**，单独一行引用；FireFly（FPGA 实测）不可直接比绝对值。

## 5. TCAS-II 最小可发表数据包

- **方法**：网络图 + θ 标定流程与最终 θ 表（2 的幂→移位说明）+ H12 梯度公式（引 Ternary Spike）+ 消融表（二值/三值 × 缺陷/修复 backward × 统一/per-layer θ）。
- **精度**：DSEC valid825 主表（AEE + 分位 outlier）vs EEMFlow+/EDCFlow/TMA/SDformerFlow；per-layer firing 修复前后表（死神经元归零证据，支撑 0.x spikes/interval 能效叙事）。
- **硬件**：28nm/3ns 声明；logic-only 3902 μm² ✓；macro-aware WNS −4.91ns **必须收敛**；PTPX 6.25 mW 需在完整网上重跑——**RTL 缺 patch embed/MLP-FFN/PSN 是最大工作量项，最先启动**；算子/SOPs 分解表（0.59% attention 是叙事素材）；三值 PE 数据通路图。
- **对比表**：同口径 28nm 综合级（LoAS/Prosperity/TESSA/SATA）+ ReckOn 硅实测锚点行；能效 SOPs/J + 每帧能耗，标注口径。

## 6. 一周执行排序

| 天 | 行动 | 产出 |
|---|------|------|
| D1 | ① sn2_q 死参数剔出统计；② H12 backward 对照 Ternary Spike 公式对拍；③ 标定脚本就绪 | 净化统计 + 对拍通过 |
| D2 | ① 100-batch 膜电位标定（~10 min GPU）；② per-layer 2 的幂 θ 表；③ 收 t51_lrprobe 结果做 LR sanity | θ 表 + LR 决策 |
| D3 | 主实验：per-layer θ + H12 backward + warm start 5 epoch（≈6.9h 后台） | run 启动 |
| D4 | 检查死模块归零、全网 r 回 0.1–0.5 健康带、AEE vs 1.3287 | 死神经元修复判定 |
| D5 | 消融（η>0 短跑或档 B 探针）；**RTL 补全开工（patch embed/MLP-FFN/PSN，最大工作量项）** | 消融数据 |
| D6 | 三值 PE sign-select-add 改造 + DC 重综合 + PTPX；WNS 收敛尝试 | 新面积/功耗 |
| D7 | LoAS RTL clone + 剪枝仿真；TCAS-II 数据包 gap list | Top3 硬件初步数字 |

**协调注意**：t49/t51 队列由另一并行工作流驱动（/root/t49_ternary_retrain.py）。D1-D3 的队列变更（H12 接入、θ 标定）需与该工作流对齐，避免同一 GPU 上的实验互相踩踏。
