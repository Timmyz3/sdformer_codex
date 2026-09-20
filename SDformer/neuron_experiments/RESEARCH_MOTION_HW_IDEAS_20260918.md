# 调研报告：motion 注意力改造 + 硬件 idea 总排行（2026-09-18）

来源：6 路并行调研（motion 代码级审计+GPU 对拍 / 体系结构 / EDA / 电路期刊 / 硅实测 / 有损手段，全部 WebSearch 核实出处）+ 综合裁决。
配套：`ACTION_PLAN_DEADNEURON_MOTION_IDEAS_20260917.md`、`H12_ATLIF_GRADFIX_TERNARY_DESIGN_20260917.md`。

## 0. 一页结论

1. **motion 注意力的"繁杂"是写法繁杂，不是数学繁杂**：实际生效的只有 3 个量（same_nonzero popcount、α0·same_zero、motion |ΔK| popcount），SC 融合/opposite/single_active/K_mag 全是 ×0 死算。**S1 popcount 恒等式重写已在 GPU 上对拍证明逐位等价（9 组配置 bitexact=True）**，前端算子减 ~3×，全部落进 AND/XOR/popcount/shift 域，零 AEE 风险，ckpt 直载。
2. **99.4% 主体的最大杠杆是"复用 + 跳过 + 时间轴"，不是 attention、不是位宽**。三元权重 {-θ,0,+θ} 把 pattern 组合空间压到极小、θ 取 2 的幂把乘法变移位——本项目在复用类（Prosperity/Phi/LoAS）和静默剪枝上比原论文场景更占便宜。
3. **一周内可零重训产出 TCAS-II 有损 story 前三组数字**：死神经元静态旁路+ICG、S1 位等价重写、TP-Spikformer training-free token 早停。
4. **最高审稿风险是 NYCU ISCAS'25 Spiking Transformer 加速器**（同 28nm 同网络族，38.3 TSOPS/W），必须引用并以"事件光流任务 + 三元神经元 + 精度闭环"三点差异化。

## 1. motion 注意力改造方案

### 现状判定（代码级，行号已核）

实配（H67 ep35 及全部 motion 变体）用 `mode: h60`（`tx_sc_k_mag_no_carrier_shiftmax`），`bipolar_mu=0` 下退化为纯 TX 分数 + motion XOR。h60 前向（bsa_attention.py:6092-6147，窗 M=450, D=32, T=2）逐项：

| 项 | 位置 | 判定 |
|---|---|---|
| TX 分数：same_nonzero 掩码 + α0·same_zero | :1755-1770 | **生效**（α0=1/64） |
| opposite 项 | :1757,1768 | **死代码**（计算后 ×mismatch=0.0） |
| single_active 项 | :1758-1763,1769 | **死代码**（XOR 掩码全算后 ×0.0） |
| motion \|ΔK\| popcount | :1771-1774,1787-1814 | **生效**（α=0.125/0.25，反极性计 2） |
| K_mag 校正 | :1776-1779 | 短路跳过 |
| SC 分数 signed consensus | :3424,1593-1666 | **死代码**（:6101 ×bipolar_mu=0.0） |
| center 行均值 / Q7 量化 / Shiftmax / gate×n_tokens / attn=K×gate | :6120-6141 | 生效 |

### S1（第一优先，已证 bit-exact）：popcount 恒等式重写

事件编码 2-bit（active/sign）后，TX + motion 分数折叠为单一整数式：

```
64·D·s = 65·o − q − k + opp + 16·m + D
o   = popcount(act_q·act_k)                    # same_nonzero
opp = popcount(act_q·act_k·(s_q⊕s_k))          # 反极性重叠
q,k = popcount(act_q), popcount(act_k)
m   = popcount(act_t⊕act_p) + 2·popcount(act_t·act_p·(s_t⊕s_p))   # motion |ΔK|
```

- **等价性已对拍证明**：`/tmp/motion_parity.py` 用仓库真函数（`_tx_sc_fusion_score_pair`、`_apply_hardware_score_quant`、`shiftmax`、gate 量化）对拍，H67 实配下 attn **逐位相等**，3 seeds × 3 发放密度（5%/25%/60%）× 9 组全过。推导关键：same_zero = D−q−k+o+**opp**（反极性通道既非 same_nonzero 也非 same_zero）。
- **算子变化**：每 token 从 ~20 次浮点掩码/比较 + 4FMA×32 + 32 次死算乘加 → **4 次 popcount32 + ~6 次 AND/XOR + 常数移位 + 4 次加减**；死分支（SC 全路径、opposite、single_active，约 40% 前端算力）整体删除。
- **实现**：新增 mode `h60_pc` 替换 `_ternary_alpha_xnor_token_scores` 调用点，~80 行，ckpt 直载，一次 DSEC valid 确认。

### S2（观察先行）：行相对 2-level gate

对拍发现 shiftmax gate 行相对值 r=gate/row_max 高度集中（合成数据 p10=0.864、p90=0.932）。若真实 DSEC 激活同样集中 → gate 二值化为 {0.5,1}，**attn 变成 K 或 K>>1，乘法彻底消失**，gate 退化为 1 个 sign 位——与三元 2-bit 通路统一的终点形态。非精确等价（行均值尺度 ~1/0.89 可被 BN 吸收），先在 S1 之后的真实激活上统计 r 分布再决定。

### S3（兜底）：若 gate 几乎行常数 → QKFormer 纯 carrier（K×sn2_q(sum(Q))），gate 整体退场。

### 移植候选（S1/S2 之后再排期）

| 候选 | 出处 | 要点 | 精度代价 |
|---|---|---|---|
| 纯二值 α-XNOR gate | Xiao et al., CVPR 2025 "Rethinking Spiking Self-Attention…α-XNOR" | `α·D+(1−α)·popcount(Q⊕K)`，掩码逻辑减 3/4；与 S1 同一条 popcount 通路 | gate 极性盲，对光流方向可能有害，需 ablation |
| Spike 位置编码 + 地址求交 SMAM | NJUST arXiv 2501.07825 / 中山大学 IEEE 10877394（同一工作） | multiplication-free 地址比较器阵列，FPGA 307.2 GSOP/s、25.6 GSOP/W | 地址化无损，量化另计；RTL 增 ESS SRAM |
| EDCFlow 时序差分 bias | CVPR 2025 | 差分=减法、注入=加法，无乘法，但跨窗布线成本高 | 中性偏正 |

**推荐顺序：S1（立即）→ S2 统计（S1 后一天）→ 视结果决定 S2/S3/移植。**

## 2. 硬件 idea 总排行 Top 10（全部作用于 99.4% 主体算子）

| # | idea | 出处 | 落点 | 数字 | 精度代价 | 成本 |
|---|------|------|------|------|---------|------|
| 1 | **死神经元静态旁路 + 细粒度 ICG** | LoAS MICRO'24 静默剪枝 × ISCAS'21 0.34pJ/SOP 锚点 | PSN addmm 行 / conv 累加行 | LoAS +20% 性能叠加；ICG 省 20-40% 动态功耗 | 零 | **最低**：探针 JSON 已给逐层清单，生成旁路配置+重跑 DC/PTPX |
| 2 | **MM2IM：deconv 重写 col2im 矩阵乘** | arXiv 2507.07683（IEEE 收录） | decoder deconv 栈 | TCONV 2.7×，端到端 2.4×/1.7× | 零（数学等价） | 低-中 |
| 3 | **COMPASS 投机 + 稀疏位置编码跳零** | MICRO 2024 SJTU × NJUST 2501.07825 | conv 输入行全零跳整行 | COMPASS 26.7× 加速/能耗 −386.7×；NJUST 13.24× 吞吐 | 零（带回滚/结构性零） | 低-中 |
| 4 | **MI-TRQR 时间冗余掩码** | NeurIPS 2025（开源） | conv/deconv/PSN 时间维 | CIFAR10-DVS 精度 +1.7%、能耗 −37.5% | **负（涨点）** | 低 |
| 5 | **TP-Spikformer token 剪枝 + block 级早停** | ICLR 2026 arXiv 2603.00527 | Swin 各 stage 主体 | 留 53% token → OPs −47% 近无损 | 可调 | 中（training-free 版一天） |
| 6 | **Prosperity + Phi 模式复用** | HPCA 2025 / ISCA 2025（开源） | PE 内模式→部分和查找表 | 7.4×/8.0×；Phi 3.45×/4.93× | 零 | 中 |
| 7 | **时间轴裁剪（STI-SNN 减 T + SEENN 早退）** | arXiv 2506.08842 / NeurIPS 2023 | PSN W_TxT 维度 + 全部逐时间步 conv（唯一同时压两者的手段） | 能耗减半、层流水 9.9× | 回归需重验 | 中 |
| 8 | **Nebula multi-skipping** | ISSCC 2025（28nm 硅实测） | 按窗口稀疏度跳整层 | 109.8 TOPS/W | 实验定阈值 | 中 |
| 9 | **ESDA 稀疏 token dataflow** | ACM FPGA'24（开源） | **patch embed 新写 RTL 直接按稀疏 token 实现** | 10% 非零输入 4.5-11× | 零 | 中 |
| 10 | **SPEAR/SCA 通道结构化剪枝** | NeurIPS 2025 / ICML 2024 | conv/FFN 通道 | SynOps 剩 54.8% 精度 +0.07% | 近无损 | 中 |

**辅助锚点**：TCAD'25 三元 spike 残差加速器（28nm, 0.63mm², 0.39mJ/帧, 6 timesteps——spike 级三元 ≠ 已证伪的权重全栈三元，神经元三元化的直接可引锚点）；FPGA Spikformer 开源 RTL（补 patch embed/MLP-FFN 抄 conv+BN 融合）；Voyager 方法学（iso-resource + 布线密度折算，对比表必用）。

**28nm 硅实测对比表主线**：ReckOn 5.3 pJ/SOP（0.45mm²）/ ANP-I 1.5 pJ/SOP（1.628mm²）/ NYCU ISCAS'25 38.3 TSOPS/W（198K gates, 139KB SRAM）。FireFly（FPGA）不入主表。

## 3. 有损手段专节

- **零成本先行**：① 死神经元静态旁路（证据在手：6-7 全死 + 21-27 个 r<0.01，`/tmp/deadprobe_*.json`）；② TP-Spikformer training-free 评测（不重训）。
- **重训类**：MI-TRQR / 减 T / SPEAR——统一挂**特征级 KD 补偿**（教师=H67 ep35，对齐中间脉冲率/特征图），普遍回血 1-2%；TCAS-II 叙事="有损手段 + 损失控制闭环"。
- **硬件使能统一收敛到一套基础设施**：per-stage 活动/token 位图寄存器 + PE 行时钟门控 + 权重裁剪加载（SpikeX 能耗分解：内存占 62%），#5/#8/#9/#10 共用，不为每个手段各做控制器。

## 4. 与三无线协同

- 三值化 + θ 2 的幂标定是**所有候选的共同放大器**：pattern 组合空间 3^k（Prosperity/Phi 复用率更高）、乘法变移位（S1 的位运算域、A3 的 α、OzMAC 同构）。
- 补 RTL：patch embed 按 ESDA 稀疏 token 接口新写；MLP-FFN 按 FPGA Spikformer conv+BN 融合；PSN 按 STI-SNN 层流水 + 参数化 T′（时间剪枝预留）+ spike-gated enable。
- 28nm DC 收敛：#1 旁路直接砍面积/路径；SPEAR"变小"型利好；OzMAC 延迟 2.38× 与 Nebula bypass mux 是唯二可能恶化 WNS 的，需时序预算前置评估。

## 5. 一周执行排序

| 天 | 行动 |
|---|------|
| D1-2 | S1 popcount 重写落地（`h60_pc` mode）+ ckpt 直载 + valid825 确认 bit-exact；并行生成死神经元旁路配置 |
| D2-3 | TP-Spikformer training-free 评测（有损 story 第一组数字）；DC/PTPX 重跑验证旁路+ICG 面积/功耗 |
| D3-4 | MM2IM 前置统计（decoder 无效 MAC 占比）+ COMPASS 前置统计（输入行全零率）→ 定 #2/#3 优先级 |
| D4-5 | S2 gate 分布统计（真实激活 r 的 p10/p90）；集中则上 2-level gate 跑一次 eval |
| D5-7 | Voyager 口径对比表初稿（ReckOn/ANP-I/NYCU 三主线）+ NYCU 差异化段落；下周排 MI-TRQR 重训与 KD 脚本 |

## 6. 不要做清单

- attention 内部一切再优化（H5x/H6x 加项、K_mag、方向通道、LSH 桶化、Winograd、popcount 微架构）——0.59% 天花板
- 权重全栈三元替换 / strict BSA / 三元扩 FFN——已证伪（AAE 71.6 / SOPs 变密）
- TSN 全局替换——已证伪（AEE 29.77）
- S3 shift-L1 gate 作 drop-in——对拍 attn 相对误差 1.29，等价性破坏过大
- OzMAC 整体优先投入——延迟 2.38× 与 3ns 预算冲突，无 28nm 数据
- SRAM-CIM / 模拟 CIM / 模拟 LIF / CNTFET / RFET / 异步架构——与 28nm 数字 DC 流程不兼容
- TRIP ROI / NeuCODEX / BLADE / NeuroFlex / ToMe / DT-SNN / 值预测跳过——dense 任务罚 AEE / 范式冲突 / 无顶会数字
- Tianmouc（传感芯片）、TrueNorth/Loihi（工艺不可比）、片上学习类（口径失真）——仅 background
