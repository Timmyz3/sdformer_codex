# SDFormerFlow 神经元替换方案全面研究报告

日期: 2026-05-09 | 基线: PSN, AEE=1.5848, SOPs=3.6219G

---

## 一、神经元基础设施全景

项目内神经元分为三层：

### Layer 1: 上游神经元（实际被模型使用）
定义于 `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py`

| 神经元 | 来源 | 核心机制 | 输出类型 |
|--------|------|---------|---------|
| **PSN** | SDFormerFlow原生 | 可学习[T,T]权重矩阵一次性并行时间混合 | 二进制(0/1) |
| GatedLIF | SDFormerFlow原生 | 3门控(alpha/beta/gamma)可学习LIF | 二进制 |
| SLTTLIF | SDFormerFlow原生 | LIF+detached-v(节省显存) | 二进制 |
| IFNode/LIFNode/PLIF | SpikingJelly | 标准IF/LIF/带可学习tau的LIF | 二进制 |

### Layer 2: 候选神经元（独立于SpikingJelly，用于消融实验）
定义于 `src/models/modules/spiking_neurons/candidates/`

| 神经元 | 来源论文 | 核心机制 | 关键参数 |
|--------|---------|---------|---------|
| **SNNode** | 无(脚手架) | 标准LIF: mem=mem*decay+x, 硬重置 | decay, v_th |
| **ATLIFNode** | Activity-Pruning-SNN | 可学习阈值+活动依赖更新 | thresh(可学习), tau, activity_scale |
| **TSLIFNode** | TS-LIF (ICLR 2025) | 双室(v_short/v_long)交叉耦合 | decay[4], alpha_s, alpha_l |
| **LMHNode** | LM-HT (NeurIPS 2024) | 多层级阈值+可学习时间掩码 | mask[T,T], alpha, L(层级) |
| **TSNNode** | Ternary-Spike (AAAI 2024) | 三值发放{-1,0,+1} | decay, v_th, fire_ratio |

### Layer 3: 硬件包装器（装饰任意基础神经元）
定义于 `src/models/modules/spiking_neurons/`

| 包装器 | 硬件映射 | 机制 |
|--------|---------|------|
| **HardwareSparseNeuron (GTCN)** | 时钟门控+ATLIF自适应阈值 | gate_logit→STE硬门控+EMA发放率追踪 |
| **FusedSparseNeuron (FSN)** | N比较器+门控 | GTCN+多层量化(2/3/4级)+有符号模式 |
| **RefractoryNeuron** | 2bit饱和计数器+AND门 | 发放后2步不应期强制静默 |
| **ATLIFThresholdNeuron** | 1可学习标量 | PSN包装器+输入代理ATLIF窗口 |

### 神经元在模型中的放置模式

上游模型的`Spiking_neuron`工厂在每层中实例化神经元：
```
SEW模式: Conv → BN → SN   (编码器、投影、patch embedding)
        Linear → BN → SN  (注意力Q/K/V、MLP fc)
MS模式:  SN → Conv → BN   (膜电位捷径)
```

模型中共有约105个`Spiking_neuron`实例，分布在4个Swin阶段（深度[2,2,6,2]）。

---

## 二、用户已完成的所有实验

### 2.1 全面替换实验 (E0-E6)

| 实验 | 神经元 | 来源论文 | 关键改动 | AEE | AAE | 发放率 | SOPs | vs基线SOPs | 结论 |
|------|--------|---------|---------|-----|-----|--------|------|-----------|------|
| **E0** | PSN | SDFormerFlow基线 | 无(基线) | 1.5848 | 7.5012 | 0.08496 | **3.6219G** | 1.00x | 当前最佳 |
| E1 | SNNode | 无(脚手架) | 标准LIF替换 | 仅烟雾 | — | — | — | — | 脚手架搭建 |
| E2a | ATLIF早期 | Activity-Pruning-SNN | 错误的替代函数+无threshold_update | 4.0057 | 21.49 | 0.3856 | 16.44G | 4.54x | 破损实现 |
| E2b | ATLIF修正 | Activity-Pruning-SNN | 部分修正但训练规模错误 | 8.6602 | 67.89 | 0.3788 | 16.15G | 4.46x | 仍破损 |
| E2c | ATLIF官方副本低SOP | Activity-Pruning-SNN | official Surrogate + threshold_update + eta=1e-3/lrs=1000 | 3.7574 | 18.62 | 0.06730 | 2.8692G | 0.79x | **SOP减少21%但AEE+137%** |
| E2d | ATLIF全预训练 | Activity-Pruning-SNN | 从PSN epoch59初始化 + 弱惩罚 | 2.5128 | 12.54 | 0.12212 | 5.2062G | 1.44x | 精度第二但SOP反而更多 |
| E2e | ATLIF Plan A | Activity-Pruning-SNN | 保守lr=1e-5 + 极弱惩罚 | 5.6600 | 27.66 | 0.1610 | 6.8619G | 1.89x | 阈值未增长，效果全败 |
| E2f | ATLIF冻结阈值 | Activity-Pruning-SNN | 冻结54.9M参数仅训105个阈值 | 2.5837 | 13.59 | 0.11563 | 4.9292G | 1.36x | 微幅改进但不如PSN |
| **E3** | LMHT | LM-HT (NeurIPS 2024) | 官方LMHTNeuron + L=2 + 无推理重参数化 | 2.5621 | 9.6492 | 0.22770 | 9.7070G | 2.68x | 多层级输出反增SOPs |
| **E4** | TS-LIF | TS-LIF (ICLR 2025) | 官方双室+标量alpha近似+交叉重置 | 2.1816 | 9.8193 | 0.09417 | 4.0146G | 1.11x | **全面替换中最佳平衡** |
| E4b | TS-LIF官方风格 | TS-LIF (ICLR 2025) | Adam+分组lr+梯度裁剪 | 6.9871 | 83.87 | 0.05075 | 2.1633G | 0.60x | 稀疏但精度崩溃 |
| E5b | TSN | Ternary-Spike (AAAI 2024) | 官方三值发放函数 | 29.7720 | 98.37 | 0.60730 | 25.8892G | 7.15x | 灾难性失败 |
| **E6a** | NASN | NASN (arXiv 2604) | 量化发放[alpha,alpha+D]/N + D=4,N=4 | 2.1676 | 8.3613 | 0.78138 | 33.3102G | 9.20x | **精度不错但SOPs爆炸** |

### 2.2 部分插入实验 (G1)

| 实验 | 机制 | 目标节点 | AEE | SOPs | 结论 |
|------|------|---------|-----|------|------|
| **G1** | HardSparseGate (标量STE门控) | 6个layer0 Swin节点 | **1.6056** | **2.7134G** | **最佳结果: -25% SOPs, +1.3% AEE** |

G1的6个精确目标节点:
1. `layers.0.swin_blocks.0.attn.proj_sn`
2. `layers.0.swin_blocks.0.mlp.sn1`
3. `layers.0.swin_blocks.0.mlp.sn2`
4. `layers.0.swin_blocks.1.attn.proj_sn`
5. `layers.0.swin_blocks.1.mlp.sn1`
6. `layers.0.swin_blocks.1.mlp.sn2`

关键训练技巧：冻结骨干+仅训练门控(BN设为eval模式)、门控从关闭状态初始化(init_logit=-2.0)、lr=0.01(比正常高100倍)、reg_lambda=0.02保持关门压力。

### 2.3 融合神经元实验 (F1-F5，仅烟雾测试)

| 实验 | 融合内容 | 烟雾训练损失 | 烟雾验证损失 | 烟雾质量 |
|------|---------|------------|------------|---------|
| F1 | PSN时间混合+可学习阈值 | 8.1597 | 6.2164 | 中等 |
| F2 | LMHT时间掩码+ATLIF递归膜 | 6.9589 | 6.4206 | 较好 |
| F3 | TS-LIF双态+可学习阈值 | 6.0360 | 6.3523 | **烟雾最佳** |
| F4 | LMHT掩码+TS-LIF双态 | 14.8995 | 15.3773 | **烟雾最差** |
| F5 | 二进制/三值软混合 | 9.0083 | 6.8135 | 中等 |

### 2.4 活跃实验 (A1-A9，neuron_autoresearch)

| 实验 | 机制 | 目标范围 | 状态 | 预期 |
|------|------|---------|------|------|
| A1 FSN on G1 | FusedSparseNeuron(2级有符号)替换G1的6个门 | layer0 6节点 | 配置完成未跑 | SOPs<2.5G, AEE<1.75 |
| **A5 Refractory** | RefractoryNeuron(2步不应期)包裹所有PSN | 全encoder proj+mlp | **训练完成60epochs待profiling** | SOPs=2.9-3.3G |
| A6 Bipolar Attn | FSN有符号三值用于注意力Q/K | 各stage attn.proj_sn | 配置完成未跑 | 注意力稀疏化 |
| A8 Dual Sparse | 发放率MSE+L1权重双惩罚 | 损失项叠加 | 配置完成未跑 | 额外10-15% SOP减少 |
| A9 ATLIF Threshold | ATLIFThresholdNeuron包装PSN(layer0) | layer0 6节点 | 配置完成但未接入entrypoint | 自适应阈值剪枝 |

---

## 三、原始论文算法对比

| 特性 | PSN | LM-HT | ATLIF | TSN | TS-LIF |
|------|-----|-------|-------|-----|--------|
| **核心创新** | 并行GEMM替代时间循环 | 多级阈值发放 | 可学习自适应阈值 | 三值发放+死区 | 双室动力学 |
| **输出类型** | 二进制(0/1) | 多级(0,th,2th,4th) | 缩放(0或thresh) | 三值(-1,0,+1) | 双通道加权和 |
| **可学习参数** | weight[O], bias | mask[T,T], alpha | thresh | V_th(可选) | decay[4], kk, yy, alpha_s/l |
| **时间实现** | 并行(O(1)) | 串行(O(T)) | 串行(O(T)) | 串行(O(T)) | 串行双室 |
| **替代梯度** | ZIF三角窗 | 矩形窗口 | 三角窗+阈值梯度 | STE(clamp) | ArcTan |
| **训练技巧** | TET损失 | 混合ANN-SNN, 直接推理重参数化 | threshold_update外部调用 | 层间KD | 分组优化器 |
| **硬件友好度** | 高(带状矩阵, 二进制) | 中(非二进制输出) | 中(可调稀疏但非二进制) | 中(有符号算术) | 低(参数多, 双室) |
| **集成复杂度** | 低 | 中 | 中 | 中 | 高 |
| **SDFormerFlow适配度** | 原生支持 | 需改注意力机制 | 需改训练循环 | 需改注意力(有符号) | 需大量改动 |

---

## 四、失败根因分析

### 4.1 为什么全面替换全部失败

**核心原因：PSN的时间混合是SDFormerFlow训练出的最优解。** PSN通过可学习的[T,T]权重矩阵一次性对所有时间步做混合，这既是计算效率最高的方式，也是模型学到的信息处理方式。任何替换都是在破坏已优化好的特征路径。

| 失败类型 | 具体案例 | 根因 |
|---------|---------|------|
| **精度崩溃** | E2a/E2b/E2e/E5b | 错误的替代函数或训练规模, 破坏了预训练特征 |
| **SOPs爆炸** | E3(LMHT)/E6(NASN) | 替代神经元的输出空间更大(多级/量化), 自然产生更多非零输出 |
| **过度剪枝** | E2c(ATLIF低SOP) | threshold_update机制有效但太激进, 在不该剪的地方也剪了 |
| **范式不匹配** | E5b(TSN)/H2(AT-PSN) | 三值{-1,0,+1}在此设置下85%活动来自负尖峰, 无法稀疏化 |
| **约束近似** | E4(TS-LIF) | 标量alpha代替通道级alpha, 限制了双室机制的潜力 |

### 4.2 为什么G1成功

1. **敏感性分析驱动**：先测哪些节点的零化对精度影响最小, 再下手
2. **最小扰动**：只改6/105个节点, 其余99个保持PSN
3. **HardSparseGate是最简单的机制**：一个标量STE门控, 不改变PSN内部动力学
4. **训练策略正确**：冻结骨干+BN设为eval+从关闭状态开始+高lr快速收敛

### 4.3 ATLIF的微妙之处

E2的6个分支揭示了关键细节：
- **threshold_lr_scale必须匹配优化器lr**：SDFormerFlow的AdamW lr=1e-4只有官方SGD lr=0.1的1/1000, 所以需要lrs=1000来补偿
- **AMP与threshold_update不兼容**：FP16下阈值梯度产生NaN, 必须关闭AMP或添加sanitize
- **activity_eta的甜点区间很窄**：1e-4过度剪枝, 3e-5剪枝不足, 中间值难以找到

---

## 五、融合可能性分析

### 5.1 已验证可行的组合方向

| 方向 | 基础 | 融合内容 | 理论优势 | 风险 |
|------|------|---------|---------|------|
| **G1 + ATLIF阈值** | G1局部插入 | ATLIF自适应阈值替代固定门控 | 动态调节稀疏度, 比固定关闭更灵活 | E2的过度剪枝教训 |
| **G1 + Refractory** | G1 6节点 | 叠加不应期机制 | 双重稀疏(G1关节点+不应期减发放) | 两者叠加可能过度稀疏 |
| **E4 + 通道感知alpha** | TS-LIF全面替换 | 改标量alpha为通道级张量 | 恢复官方TS-LIF的表达能力 | 需要知道每个层的特征维度 |
| **F3 + 训练** | 融合自适应TS-LIF | 烟雾最佳, 需全训练验证 | 双室+可学习阈值=稀疏+精度 | 未知, 基础是融合作品 |

### 5.2 有潜力的新融合方向

| 方向 | 融合内容 | 参考工作 | 理论依据 |
|------|---------|---------|---------|
| **PSN+LMHT多级** | PSN并行混合后接多级阈值量化 | PSN + LM-HT | PSN的并行效率 + LMHT的信息密度 |
| **ATLIF阈值+PSN** | PSN保持核心, ATLIF阈值作为外挂自适应剪枝 | ATLIF + G1策略 | G1证明了外挂比替换好 |
| **TS-LIF部分+PSN** | 仅在高发放层(>20%)用TS-LIF双室, 其余保持PSN | TS-LIF + G1节点选择 | 避免全面替换的精度损失 |
| **FSN混合模式** | 不同层用不同FSN模式(深层用三值, 浅层用2级量化) | FSN + 敏感性分析 | 按层功能定制稀疏策略 |

---

## 六、迭代改进路线图

### 阶段一：验证已知最有希望的方案（立即可做）

| 优先级 | 实验 | 内容 | 预期 |
|--------|------|------|------|
| P0 | **Profile A5** | 对A5 Refractory训练好的60epoch模型做profiling | 获得第一个refractory完整数据 |
| P1 | 跑A1 | FSN(2级有符号)替换G1的6个节点, 20epoch gate-only训练 | 验证FSN是否比G1的简单门控更好 |
| P2 | 跑A6 | FSN有符号用于注意力Q/K投影 | 验证注意力稀疏化 |
| P3 | 跑A8 | 双稀疏正则化叠加H1基 | 验证损失项叠加效果 |

### 阶段二：融合探索

| 优先级 | 实验 | 内容 | 预期 |
|--------|------|------|------|
| P4 | G1+ATLIF阈值 | 在G1的6个节点上改用ATLIF自适应阈值替代固定STE门控 | 动态稀疏调度 |
| P5 | G1+Refractory | G1的6门+FSN的refractory模式 | 双重稀疏 |
| P6 | E4部分插入 | TS-LIF仅替换发放率>20%的层(约30个节点) | 比全面替换更好 |
| P7 | F3全训练 | 融合自适应TS-LIF做全训练 | 验证融合方案 |

### 阶段三：新方案设计

| 优先级 | 实验 | 内容 | 预期 |
|--------|------|------|------|
| P8 | PSN+ATLIF阈值外挂 | ATLIFThresholdNeuron包装所有PSN的proj+mlp | 自适应全局剪枝 |
| P9 | 分层FSN | 浅层2级量化+深层三值有符号 | 硬件定制稀疏 |

---

## 七、对当前autoresearch的启示

1. **不要碰全面替换**：15个全面替换实验0成功, 这是已验证的死路
2. **坚持部分插入策略**：G1证明了选对节点+最小改动是正确的
3. **包装优于替换**：A系列实验的正确设计模式——在PSN外面加东西, 不动PSN内部
4. **三值尖峰在此场景不适用**：E5b/H2都因为负尖峰计入发放而SOPs暴增, 除非改变SOPs计算方式
5. **ATLIF阈值机制是可用的**：但需要谨慎的eta/lrs调度和限幅防止过度剪枝
6. **A5 Refractory应优先profile**：它已经跑完了60epochs训练, 是最快能得到答案的实验
7. **sparsity preprocessing仍然有价值**：作为神经元稀疏的补充, 在数据进入模型前就减少信息量

---

## 八、综合方案对比总表

| 方案 | 参考工作 | 实现改进 | vs基线预期 | 风险 | 已验证? | 推荐优先级 |
|------|---------|---------|-----------|------|---------|-----------|
| G1 局部STE门控 | 无(自研) | 6个layer0节点关断 | -25% SOPs, +1.3% AEE | 低 | ✅ 已验证 | **最佳基线** |
| A1 FSN on G1 | FSN+G1 | G1节点升级为FSN(2级有符号) | -30% SOPs, +3% AEE | 中 | ⬜ 待跑 | P1 |
| A5 Refractory | 不应期神经元 | 全encoder PSN外挂2步不应期 | -10~15% SOPs, +2% AEE | 低 | ⬜ 训练完成待profile | **P0** |
| A6 Bipolar Attn | BSA+FSN | 注意力Q/K投影改用三值FSN | -20% SOPs, +5% AEE | 高(三值历史差) | ⬜ 待跑 | P2 |
| A8 Dual Sparse | 双稀疏正则化 | 发放率MSE+L1权重惩罚 | 额外-10% SOPs | 低(仅损失项) | ⬜ 待跑 | P3 |
| G1+ATLIF阈值 | G1+ATLIF | STE门控→ATLIF自适应阈值 | -30% SOPs, +5% AEE | 中(eta需调) | ⬜ 新方案 | P4 |
| G1+Refractory | G1+A5 | 双重稀疏叠加 | -35% SOPs, +5% AEE | 中(可能过度) | ⬜ 新方案 | P5 |
| E4部分插入 | TS-LIF+G1策略 | TS-LIF仅替换高发放层(>20%) | -5% SOPs, +10% AEE | 中 | ⬜ 新方案 | P6 |
| H3 ATLIF-PSN | ATLIF+PSN | PSN混合+ATLIF阈值(仅attn Q/K) | -5% SOPs, +3% AEE | 低 | ⬜ 仅烟雾 | P6 |
| F3全训练 | TS-LIF+ATLIF | 双室+可学习阈值, 全训练 | 未知 | 高(融合未验证) | ⬜ 仅烟雾 | P7 |
| PSN+ATLIF外挂 | ATLIF+G1策略 | ATLIFThresholdNeuron包装proj+mlp | -15% SOPs, +5% AEE | 中 | ⬜ A9待接入 | P8 |
| 分层FSN | FSN+敏感性 | 浅层2级+深层三值 | -25% SOPs, +5% AEE | 中 | ⬜ 新方案 | P9 |
| 稀疏预处理+训练 | 自研 | 时间步预算+token剪枝, 训练适配 | -15% SOPs, +10% AEE | 中(需训练) | ⬜ 基础设施已就绪 | P9 |
| E4+通道alpha | TS-LIF | 标量alpha→通道级张量(已知特征维度处) | -5% SOPs, +5% AEE | 低(仅改alpha维度) | ⬜ 代码改动 | P9 |
| F2全训练 | LMHT+ATLIF | 时间掩码+递归膜(烟雾较好) | 未知 | 高 | ⬜ 仅烟雾 | P10 |
| LMHT推理重参数化 | LM-HT | 实现L*T扩展用于SDFormerFlow | -30% SOPs? | 高(范式壁垒) | ⬜ 未尝试 | P10 |

---

## 九、2024-2026 顶刊外部论文调研

### 9.1 脉冲Transformer注意力效率

| 论文 | 会议 | 核心机制 | 与SDFormerFlow关联度 | 可集成性 |
|------|------|---------|-------------------|---------|
| **SpiLiFormer** | ICCV 2025 | 侧抑制(Lateral Inhibition): FF-LiDiff+FB-LiDiff模块抑制不相关token注意力, +1.6% ImageNet, 仅39%参数量达SOTA | ⭐⭐⭐ 高 | 注意力Q/K投影处可插入侧抑制 |
| **QKFormer** | NeurIPS 2024 Spotlight | Q-K注意力, 层次化脉冲Transformer, 85.65% ImageNet, 线性复杂度 | ⭐⭐⭐ 高 | 可替换SDFormerFlow的Swin注意力为Q-K注意力 |
| **STAtten** | CVPR 2025 | 块级时空注意力, 即插即用, O(TND²)复杂度 | ⭐⭐⭐ 高 | 直接替换现有窗口注意力模块 |
| **A²OS²A** | CVPR 2025 | 精准纯加法脉冲自注意力, 混合二值/ReLU/三值脉冲神经元, 无softmax/scale | ⭐⭐ 中 | 注意力机制改动较大 |
| **Spike-Driven Transformer** | NeurIPS 2023→ICLR 2024 | SDSA: 仅mask+加法, 零乘法, 87.2×更低注意力能耗 | ⭐⭐ 中 | 需改造Swin为Linear Attention |
| **FWformer** | Front. Neurosci. 2025 | 傅里叶/小波替代自注意力, O(NlogN), 20-25%更少能耗 | ⭐ 低 | 范式差异大 |

### 9.2 Token剪枝与自适应计算

| 论文 | 会议 | 核心机制 | 与SDFormerFlow关联度 | 可集成性 |
|------|------|---------|-------------------|---------|
| **TP-Spikformer** | ICLR 2025 | **免训练**token剪枝: 时空信息保留准则+块级早停, 适用于分类/检测/分割/跟踪 | ⭐⭐⭐ 高 | **最直接可集成**: 在预处理阶段剪枝token |
| **STAS** | arXiv Aug 2025 | 时空自适应计算时间: 2D token剪枝(空间+时间), 30-46%能耗节省 | ⭐⭐⭐ 高 | 时间步预算+空间token剪枝的统一框架 |
| **AT-SNN** | arXiv Aug 2024 | 自适应Token: ACT+token相似度合并, 减少42.4% token | ⭐⭐ 中 | token合并机制可用于preprocessing |
| **Bishop** | ISCA 2025 | 误差约束TTB剪枝+BSA训练, 5.91×加速, 6.11×能效 | ⭐⭐⭐ 高 | **硬件联合设计**, BSA训练方法可借鉴 |
| **ST Spiking Feature Pruning** | IEEE TCDS 2025 | 无参数剪枝(仅加法+排序), Softmatch补偿, 训练181h→128h | ⭐⭐ 中 | 思路可借鉴 |

### 9.3 自适应阈值与替代梯度

| 论文 | 会议 | 核心机制 | 与SDFormerFlow关联度 | 可集成性 |
|------|------|---------|-------------------|---------|
| **DS-ATGO** | AAAI 2026 | 双阶段: 前向自适应阈值(MPD驱动, 逐时间步) + 后向动态替代梯度 | ⭐⭐⭐ 高 | **改进ATLIF**: MPD驱动阈值替代固定eta |
| **AT-LIF + ASG-S** | KBS Jun 2025 | 可学习自适应阈值LIF + 可学习替代梯度缩放因子 | ⭐⭐⭐ 高 | 可学习阈值+自适应SG, 直接改进ATLIFNode |
| **AdaLi** | Front. Neurosci. Mar 2026 | 轻量自适应替代梯度, 按epoch自动调整梯度边界 | ⭐⭐ 中 | 训练技巧, 非架构改进 |
| **MPD-AGL** | IJCAI 2025 | 膜电位分布驱动的动态SG, 增加梯度可用区间内的神经元比例 | ⭐⭐ 中 | 改进SpikeFn的训练效果 |

### 9.4 事件光流专用

| 论文 | 会议 | 核心机制 | 与SDFormerFlow关联度 | 可集成性 |
|------|------|---------|-------------------|---------|
| **ST-FlowNet** | Neural Networks Oct 2025 | ConvGRU跨模态特征增强+时间对齐, 参数自由的BISNN训练策略 | ⭐⭐⭐ 高 | BISNN训练策略可借鉴 |
| **SENECA benchmark** | Neural Networks Aug 2025 | 首次ANN vs SNN公平硬件对比(event光流), SNN: 44.9ms/927μJ, ANN: 71.8ms/1232μJ | ⭐ 低 | 基准参考 |
| **SDformerFlow** | ICPR 2024 | 你已在用的基线论文 | — | — |

### 9.5 硬件高效剪枝

| 论文 | 会议 | 核心机制 | 与SDFormerFlow关联度 | 可集成性 |
|------|------|---------|-------------------|---------|
| **Phi** | ISCA 2025 | 两级稀疏层级(向量模式+元素), 3.45×加速, 4.93×能效 | ⭐⭐⭐ 高 | 硬件架构参考 |
| **QP-SNN** | ICLR 2025 | 联合量化+结构化剪枝, SVS剪枝准则(基于尖峰活动奇异值) | ⭐⭐ 中 | SVS剪枝准则可用于节点选择 |
| **SpQuant-SNN** | Front. Neurosci. 2024 | 三值膜电位+空间通道动态剪枝, 13×显存减少, 4.7× FLOPs减少 | ⭐⭐⭐ 高 | 三值膜电位可应用于FSN |
| **SpikeFit** | EurIPS Workshop 2025 | 聚类感知训练+FSC剪枝(基于Fisher信息), 真North/Loihi目标 | ⭐⭐ 中 | FSC用于节点敏感性分析 |

### 9.6 新神经元模型

| 论文 | 会议 | 核心机制 | 与SDFormerFlow关联度 | 可集成性 |
|------|------|---------|-------------------|---------|
| **MSF Neuron** | Nature Comms 2025 | 多突触发放: 每轴突多个突触不同阈值, 同时编码空间强度+时间动态, 泛化LIF和ReLU | ⭐⭐ 中 | 新范式, 与PSN差异大 |
| **ADA-TLIF (All-in-One)** | arXiv 2025 | 多级脉冲神经元, 1时间步推理, 2-3×能量节省 | ⭐⭐ 中 | 多级输出可参考 |
| **DLIF** | Applied Intelligence 2025 | 动态LIF: 非线性自反馈+动态阈值调节+自调节发放率 | ⭐⭐ 中 | 动态阈值机制可借鉴 |

---

## 十、最新外部方案对现有工作的改进机会

### 10.1 可直接落地的高价值方案

| 优先级 | 外部论文 | 集成方案 | 预期收益 | 改动量 |
|--------|---------|---------|---------|--------|
| **P0** | TP-Spikformer (ICLR 25) | 免训练token剪枝→数据预处理层 | 减少20-40% token, 零额外训练 | 低(仅预处理) |
| **P0** | STAS (arXiv 25) | 时空自适应计算→统一时间步预算+空间剪枝框架 | 30-46% 能耗节省 | 中(需训练) |
| **P1** | SpiLiFormer (ICCV 25) | 侧抑制→注意力层Q/K投影 | 提升注意力质量, 减少无效token | 中(需改注意力模块) |
| **P1** | DS-ATGO (AAAI 26) | MPD驱动自适应阈值→改进ATLIF机制 | 替代固定eta, 动态调阈值 | 中(需改threshold_update) |
| **P1** | Bishop (ISCA 25) | BSA训练+TTB剪枝→稀疏感知训练 | 结构化稀疏, 硬件友好 | 高(需改训练循环) |
| **P2** | AT-LIF+ASG-S (KBS 25) | 可学习阈值+自适应SG→改进SpikeFn | 提升替代梯度质量 | 低(仅改SpikeFn) |
| **P2** | QKFormer (NeurIPS 24) | Q-K注意力→替换Swin窗口注意力 | 线性复杂度注意力 | 高(架构改动大) |
| **P3** | SpQuant-SNN (Front. Neurosci. 24) | 三值膜电位+动态剪枝→FSN扩展 | 13×显存减少 | 中 |

### 10.2 融合思路

1. **TP-Spikformer免训练剪枝 + G1部分插入**: 先用免训练方法在数据预处理层做token筛选（零成本），再在选出的高价值token上用G1门控做神经元级稀疏。两个层面互补。

2. **DS-ATGO MPD驱动阈值 + ATLIF threshold_update**: 用MPD(膜电位分布)作为阈值更新的驱动信号，替代固定eta参数。这样阈值增长速率自适应每层的实际活动状态，解决E2中"eta难调"的问题。

3. **SpiLiFormer侧抑制 + A6 Bipolar Attn**: 在注意力Q/K投影处同时应用侧抑制（抑制不相关token）和FSN三值有符号（增强token判别力）。两者都在注意力机制上工作但角度互补。

4. **Bishop BSA训练 + 稀疏预处理**: 训练时用BSA(Bundle Sparsity-Aware)损失函数引导模型学习结构化稀疏，推理时用TTB(Token-Time Bundle)剪枝执行。这比随机稀疏更硬件友好。

### 10.3 新增实验方案

| 方案编号 | 方案名称 | 参考论文 | 实现内容 | 预期 | 优先级 |
|---------|---------|---------|---------|------|--------|
| E7 | 免训练Token剪枝 | TP-Spikformer | 在数据预处理中实现时空信息保留token筛选, eval-only | -15% SOPs, +5% AEE | **立即可做** |
| E8 | MPD自适应阈值 | DS-ATGO | 用膜电位分布驱动ATLIF阈值更新, 替代固定eta+threshold_lr_scale | ATLIF不再过度剪枝 | 需接入MPD |
| E9 | 侧抑制注意力 | SpiLiFormer | 在SDFormerFlow的attn.proj_sn处加FF-LiDiff旁路 | +稀疏+精度 | 需改模型 |
| E10 | BSA稀疏训练 | Bishop | 在损失函数中加bundle sparsity项, 引导结构化稀疏 | 硬件友好稀疏 | 仅改损失 |
| E11 | STAS统一框架 | STAS | 实现2D(空间+时间)自适应token剪枝 | -30% SOPs | 需训练 |
| E12 | Q-K注意力 | QKFormer | 用Q-K注意力替换Swin窗口注意力 | 线性复杂度 | 架构改动大 |

---

## 十一、神经元专项深度调研（PSN外挂包装器范式）

以下论文均按"能否作为PSN外部包装器"标准筛选。核心原则：**不动PSN内部的[T,T]时间混合矩阵，只在PSN外面加机制。**

### 11.1 即插即用包装器（零/极少参数）

| 论文 | 发表 | 机制 | PSN包装可行性 | 硬件代价 |
|------|------|------|-------------|---------|
| **AHSAR** (Homeostatic Spark) | arXiv Dec 2025 | 零参数！每层维护一个homeostatic状态变量, 映射发放率偏差到自适应阈值缩放因子, 跨层扩散防不均衡 | ⭐⭐⭐ 最易: 在PSN输出后乘一个阈值缩放因子, 完全不动PSN内部 | 零额外参数, 仅需一个标量状态变量 |
| **RPLIF** (Refractory Period LIF) | arXiv Sep 2025 | 发放触发阈值动态: spike后暂时抬高阈值使神经元不应 | ⭐⭐⭐ 等同A5: 在PSN输出端加hard/soft不应期 | 2bit计数器+AND门, 与A5完全相同 |
| **TM-OTTA-SNN** | arXiv May 2025 | 测试时阈值在线调制, 基于发放率归一化 | ⭐⭐ 可在PSN输出后加test-time calibration | 推理时可启用 |
| **Rhythm-SNN** | Nature Comms Sep 2025 | 异质振荡信号调节神经元只在ON相发放, 跳过OFF相, 10-100×能耗降低 | ⭐⭐ 可包装为时间步级的发放门控 | 群体级节律门控 |

### 11.2 自适应阈值包装器

| 论文 | 发表 | 机制 | PSN包装可行性 | 与现有ATLIF对比 |
|------|------|------|-------------|---------------|
| **AT-LIF / Activity Pruning** | NeurIPS 2025 | 可学习自适应阈值, 发放后阈值上升抑制后续发放. 证明过度正则化→鞍点陷阱. ImageNet上0.06平均发放率 | ⭐⭐⭐ 最相关: 阈值更新机制可直接用于PSN包装器, 替代E2的固定eta | E2用的是2024版ATLIF, 这是NeurIPS 2025改进版, 理论更完备 |
| **DS-ATGO** | AAAI 2026 | 膜电位分布(MPD)驱动的前向自适应阈值 + 后向动态替代梯度. 逐时间步调阈值 | ⭐⭐⭐ MPD可替换E2的固定eta调度. PSN虽然无膜电位, 但可用输入代理(同A9的window_proxy) | 比固定eta更智能: 阈值增长速度由实际活动分布决定 |
| **ST-Thresholds** | Neurocomputing Sep 2025 | LTF(可学习时间因子)+ALSF(自适应空间因子), 将固定阈值变为时空自适应 | ⭐⭐ LTF可应用于PSN的时间维度输出, ALSF可应用于空间维度 | 代码开源: github.com/gzxdu/ST-Thresholds-SNN |
| **SpQuant-SNN** | Front. Neurosci. 2024 | 逐层可学习阈值+SGP(Separate Gradient Path)+GPW(Gradient Penalty Window), 剪枝负膜电位神经元 | ⭐⭐⭐ SGP/GPW可改进SpikeFn的替代梯度质量 | 膜电位量化为三值, 13×显存减少 |

### 11.3 门控机制包装器

| 论文 | 发表 | 机制 | PSN包装可行性 | 与G1对比 |
|------|------|------|-------------|---------|
| **DGN** (Dynamic Gated Neuron) | arXiv Sep 2025 | 膜电导随神经元活动动态演化, 功能类似LSTM遗忘门, 增强随机稳定性 | ⭐⭐ 门控作用于PSN输出: gate=σ(conductance), output=PSN(x)*gate | G1是固定STE门控, DGN是动态门控 |
| **LT-Gate** | arXiv Oct 2025 | 双时间尺度膜室(快τ+慢τ), 可学习γ∈[0,1]软门控混合. 方差追踪正则化 | ⭐⭐ 可在PSN时间混合后用γ控制快慢通道权重 | G1是硬门控(开/关), LT-Gate是软门控(连续混合) |
| **GLIF** (Gated LIF) | NeurIPS 2022 | 可学习门控因子融合多种生物特征, 扩大表示空间. 77.35% CIFAR-100 | ⭐ 需改神经元内部 | 基础性工作, DGN/LT-Gate的前身 |

### 11.4 多级/量化输出包装器

| 论文 | 发表 | 机制 | PSN包装可行性 | 风险评估 |
|------|------|------|-------------|---------|
| **QB-LIF** | arXiv Apr 2026 | 量化burst-LIF: 将膜电位饱和均匀量化为多级输出, 可学习scale, 推理时可吸收进权重 | ⭐⭐⭐ PSN输出后可接量化器: quantize(PSN(x), scale). 推理时scale吸收到下一层权重 | E3(LMHT)多级输出失败, 但QB-LIF的吸收策略可能不同 |
| **三元脉冲学习** | AAAI 2024 | 三值{-1,0,+1}输出, 每层可学习发放幅度 | ⭐⭐ PSN后接三值量化器 | E5b/H2三值均失败, 风险高 |
| **自适应比特分配** | Neural Networks 2026 | 逐层学习时间长度+权重量化比特+发放比特, 联合优化 | ⭐⭐ 可作为全局优化框架 | 改动大 |
| **T8HWQ** | Front. Neurosci. 2025 | 三值{−1,0,+1}权重(首层)+8bit(后续层), 通道自适应阈值三值发放 | ⭐ 需改权重 | 20% FPGA LUT节省 |

### 11.5 树突/多室包装器

| 论文 | 发表 | 机制 | PSN包装可行性 | 备注 |
|------|------|------|-------------|------|
| **MSF Neuron** | Nature Comms Aug 2025 | 单轴突建立多个突触(不同阈值)到同一突触后神经元. 同时编码空间强度+时间动态. 泛化LIF和ReLU | ⭐⭐ PSN输出分多路经不同阈值→合并. 但本质改变了发放语义 | Nature Comms级别, 理论完备 |
| **DendSN** | arXiv Dec 2024 | 多非线性树突分支, Triton GPU加速, 树突分支门控用于增量学习 | ⭐ 需改神经元结构 | 更鲁棒, 抗噪声 |
| **TC-LIF** | AAAI 2024 | 双室(体细胞+树突)用于长序列时序建模. 梯度传播理论分析 | ⭐⭐ 可参考E4(TS-LIF)的双室包装模式 | 已有代码 |
| **MMDEND** | ACL 2025 | 多分支多室并行脉冲树突神经元, 比例可调, SSF发放机制 | ⭐ 复杂度过高 |

### 11.6 训练技巧（不改架构）

| 论文 | 发表 | 机制 | 可集成性 |
|------|------|------|---------|
| **OSBC** (Optimal Spiking Brain Compression) | arXiv Jun 2025 | 一次性后训练剪枝+量化, 损失函数最小化膜电位误差(非输入电流), 膜电位损失与输出保真度相关0.98 | ⭐⭐⭐ 可用于G1后对模型做进一步压缩 |
| **活动剪枝(AT-LIF)理论** | NeurIPS 2025 | 过度正则化→鞍点, 证明需要平衡稀疏正则化与表示学习 | ⭐⭐⭐ 直接指导eta/activity_eta调参 |
| **Operational Manifolds** | Front. Neurosci. Feb 2026 | (τ,Vth)空间分析, 找发放平衡区, 复合效率评分(精度×SOP成本) | ⭐⭐ 可用于系统性选择PSN的超参数 |

---

## 十二、PSN包装器范式下的新实验方案

基于以上调研，所有方案都遵循"PSN保持核心, 外部包装"原则：

### 12.1 高优先级（改动小, 风险低, 收益明确）

| 编号 | 方案 | 参考论文 | 包装方式 | 预期SOPs | 预期AEE | 改动量 |
|------|------|---------|---------|---------|---------|--------|
| **N1** | AHSAR零参数自适应发放率 | AHSAR (Dec 2025) | PSN输出后乘homeostatic阈值缩放因子, 每层一个标量状态 | -10~15% | +2~5% | **极低**(仅加标量) |
| **N2** | RPLIF发放触发不应期(改进A5) | RPLIF (Sep 2025) | 在A5 RefractoryNeuron基础上加spike-triggered阈值动态 | -15~20% | +3~8% | 低(改不应期逻辑) |
| **N3** | AT-LIF NeurIPS改进版阈值 | Activity Pruning (NeurIPS 2025) | 升级E2的ATLIF阈值机制: MPD驱动+鞍点理论指导的eta调度 | -20~30% | +5~15% | 中(改threshold_update) |
| **N4** | SGP替代梯度改进 | SpQuant-SNN (2024) | 替换SpikeFn: 阈值和膜电位用分离梯度路径+梯度惩罚窗口 | 训练更稳定 | ±0% | 低(仅改SpikeFn) |
| **N5** | DGN动态门控(升级G1) | DGN (Sep 2025) | G1的6个节点: 固定STE门→电导动态门控gate=σ(c(t)), c(t)随活动演化 | -25~30% | +2~5% | 中(改门控逻辑) |

### 12.2 中优先级（改动中, 需验证兼容性）

| 编号 | 方案 | 参考论文 | 包装方式 | 预期SOPs | 预期AEE | 改动量 |
|------|------|---------|---------|---------|---------|--------|
| **N6** | QB-LIF量化burst | QB-LIF (Apr 2026) | PSN输出后接: round(clamp(PSN(x)/scale, 0, L)) * scale, 推理时吸收scale | -20~30% | +5~10% | 中(需改下一层输入) |
| **N7** | LT-Gate软门控 | LT-Gate (Oct 2025) | PSN的[T,T]混合改为快慢双通道, γ软混合, 方差追踪正则化 | -15~25% | +3~8% | 中(改PSN输出混合) |
| **N8** | MSF多阈值包装 | MSF Neuron (Nature 2025) | PSN输出→K路不同阈值量化→合并, 每路对应不同"突触" | -20~30% | +5~10% | 中(多路输出) |
| **N9** | OSBC一次性后训练压缩 | OSBC (Jun 2025) | 在G1训练好的模型上做膜电位损失最小化的一次性剪枝 | -10~20% | +2~5% | 低(后训练) |

### 12.3 长线探索（复杂度高, 但潜力大）

| 编号 | 方案 | 参考论文 | 包装方式 |
|------|------|---------|---------|
| **N10** | Rhythm-SNN节律门控 | Nature Comms (Sep 2025) | 时间步级: 只在ON相让PSN发放, OFF相强制静默 |
| **N11** | 自适应比特分配联合优化 | Neural Networks (Apr 2026) | 逐层学习最优发放比特数+时间步数+权重量化比特 |
| **N12** | DendSN多树突分支 | DendSN (Dec 2024) | PSN输出进多个非线性树突分支, 门控选通 |

---

## 十三、总结：PSN包装器方案的优先级矩阵

按**收益×可行性/风险**排序：

```
高收益                         N3(AT-LIF改进)    N8(MSF多阈值)
  │               N5(DGN门控)  
  │  N2(RPLIF)    N7(LT-Gate)    N6(QB-LIF)
  │  N1(AHSAR)    N4(SGP改进)    N9(OSBC)
低收益                         N10(Rhythm)
  └──────────────────────────────────────────
     低风险                          高风险
             
★ 最推荐首试: N1(AHSAR, 零参数, 极低风险) → N5(DGN门控, 升级G1) → N3(AT-LIF改进, 解决E2痛点)
```

---

## 十四、Token Mixing 替换QKV注意力：专项调研与实验设计

### 14.1 为什么在SNN里QKV注意力是冗余的

SDFormerFlow中发现的层firing分析显示：**大量attention层(attn_sn)的firing rate为0%**。这意味着：
1. 注意力计算产生的是全零输出 → 注意力对模型forward无贡献
2. QKV三个投影（3×Linear + 3×BN + 3×SpikingNeuron）产生的计算热量被浪费
3. 脉冲二值性使softmax的"概率分布"语义失去意义

### 14.2 顶刊相关工作

| 论文 | 会议 | 核心方案 | 关键指标 |
|------|------|---------|---------|
| **STMixer** | NeurIPS 2024 | Conv+FC token mixing, 纯异步事件驱动, 无softmax | 与Spikformer持平, T=1-4 |
| **SWformer** | ECCV 2024 | 频率感知Token Mixer(FATM): 小波+Conv+逐点Conv | 50%+能耗降低, 21%参数减少, +2.4% ImageNet |
| **FWformer** | Front. Neurosci. 2025 | 傅里叶/小波基替代自注意力, O(N²d)→O(NlogN) | 20-25%能耗降低, 19-70%推理加速 |
| **SMixer** | ICLR 2026 | Spiking Mixer + 时空剪枝(STP) | 训练加速+事件驱动 |
| **Max-Former** | NeurIPS 2025 | Max-Pool高通滤波+深度可分离Conv恢复高频 | 82.39% ImageNet, +7.58% vs Spikformer |
| **ASFNOformer** | Electronics 2025 | FFT over H,W,T + 块对角权重MLP | CIFAR10-DVS SOTA |

### 14.3 四种Token Mixer设计方案

所有方案都是`Spiking_BN_WindowAttention3D`的drop-in替换。

| Mixer | 机制 | 参数量 vs QKV Attention | 硬件友好度 | 设计参考 |
|-------|------|------------------------|-----------|---------|
| **Identity** | 无token mixing, 仅保留proj层 | ~67%减少(去掉QKV投影) | ⭐⭐⭐ 最高 | 基准测试 |
| **Conv** | 深度可分离Conv(空间)+1×1 Conv(通道) | ~70%减少 | ⭐⭐⭐ 加法为主 | ConvMixer, STMixer |
| **MLP** | Transpose→Token-MLP→Transpose→Channel-MLP | ~60%减少(减少hidden dim) | ⭐⭐ FC层较多 | MLP-Mixer, SMixer |
| **Pool** | AvgPool(空间mixing)+proj | ~80%减少 | ⭐⭐⭐ 极简 | PoolFormer |

### 14.4 实验状态

| 实验 | 配置 | 状态 | 预期 |
|------|------|------|------|
| Smoke-Identity | 5 epoch, IdentityTokenMixer, 从PSN epoch59初始化 | 🔄 运行中 | 验证pipeline, 观察AEE降幅 |
| Smoke-Conv | 5 epoch, ConvTokenMixer | ⏳ 排队 | 空间conv mixing |
| Smoke-MLP | 5 epoch, MLPTokenMixer | ⏳ 排队 | MLP-Mixer风格 |
| Smoke-Pool | 5 epoch, PoolTokenMixer | ⏳ 排队 | 最简方案 |

### 14.5 Identity Token Mixer 初期实验结果

**配置**: IdentityTokenMixer (无QKV注意力, 仅proj层), 从头训练, 5 epochs, batch=1

| 指标 | PSN Baseline (epoch 59) | Identity Mixer (epoch 4) | 变化 |
|------|------------------------|------------------------|------|
| Train Loss | ~1.0 (约) | 1.99 | — |
| AEE | 1.5848 | 2.1029 | +32.7% |
| SOPs | 3.6219G | **2.9495G** | **-18.6%** ✓ |
| Firing rate | 0.08496 | 0.06919 | -18.6% ✓ |
| AAE | 7.5012 | 9.1866 | +22.5% |

**关键发现**: 
1. ✅ **QKV注意力在SNN Transformer中是可移除的** — 去掉所有QKV投影和attention计算后, 模型仍能有效学习光流
2. ✅ **SOPs立即降低18%** — 即使只训练5个epoch, 能耗已经低于完全训练的PSN baseline
3. ✅ **训练loss持续下降** — epoch 0: 4.28 → epoch 4: 1.99, 说明还有改进空间
4. 🔄 正在测试: ConvTokenMixer, PoolTokenMixer, MLPTokenMixer 是否能进一步改善AEE
