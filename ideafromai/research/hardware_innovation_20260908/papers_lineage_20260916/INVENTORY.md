# /home/zhumd/work/papers 精读清单（2026-09-16）

深度：机制页 + 加速从哪来，不是摘要。身份：AT-LIF `{0,θ}` 吸入 W。Codex 已抄过 Prosperity/Phi/Gustav/BitL 半套/ELSA 概念。

| # | 论文 | 会 | 加速从哪来 | 本网照抄预期 | 失败后适配 |
|---|---|---|---|---|---|
| 96 | Prosperity 产品稀疏 | HPCA’25 | 公共 1-子集复用，SpikeBERT 产品密度 1.23% | 已抄：森林 −8%，Phi 叠上去更慢 | 停窄 K16；联合节点未完 |
| 98 | Phi 层次模式 | ISCA’25 | 离线 pattern×W 查表 | 已抄：93% 组已是 0/onehot | 只在复杂组才查 |
| 93 | Bishop bundling | ISCA’25 | 时序打包 + 误差约束剪枝 | 部分 A | 物理事务目标剪枝未闭 |
| 101 | FireFly-T 双引擎 | IEEE TC’26 | 稀疏 conv + 二值注意力 overlay | overlay A；同刊 ESTU | 不是双 last-use |
| 678 | GustavSNN CPTB/NRV | HPCA’26 | 列并行 tick-batch + 全零 NRV | 局部 8–12%，生产边界未闭 | **P1a 真源写出→NRV→门** |
| 686 | ELSA 弹性推理 | ISCA’26 | token 一经产出立刻前传；首响 O(L) | 空间双 context 慢 74–94% | 按 **区域** 弹性出流，不要整层同步 |
| 693 | BitL 横/纵查找 | MICRO’25 | 关键路径是最密的那一行 1，不是平均稀疏 | T10 半套拍数不变 | 对 **Y24/Q2 位平面** 做完整 A* 换向 |
| 649 | **Zhang 28nm 事件光流** | **CICC’26** | **深 U-Net 级跳过 + MaxPool/ReLU 推测 + 变位宽压缩**；ops 0.20×、延迟 0.19× | **尚未当主线抄** | 见 COPY-1 |
| 707 | **C-STEP** | **DATE’26** | 时间早退 + 早期静默通道剪枝 + 邻 token 公共脉冲 | 分类 softmax，**不能直接抄到稠密 OF** | 见 COPY-2 |
| 714 | **Scrooge** | **DATE’26** | 注意力攒够分数就停；1.7× | SSA 会撞 ESTU | 「足够统计量就停」接到 **decoder/PSN 组** |
| 717 | SATA | DATE’26 | TopK QK 重排提高局部性 1.5–1.8× | CIM+注意力 | 调度 A；禁 CIM 标题 |
| 703 | STELLAR | HPCA’24 | Few-spike 短窗 + stRS；硬件最高 7.1× | 换神经元/短 T | 训练学生；不要当无损 |
| 684 | DJP 动态联合剪枝 | DAC’25 | CIFAR SOP **126×**（算法压缩） | 分类剪枝，不是 OF 硅 | SOP 模型可借；数字不可写进信件 |
| 709/719 | Jung SSA 结构稀疏 | TVLSI’26 / 2025 | 整行整列无脉冲可跳；吞吐 +24% | SSA | ESTU/FireFly 类 |
| 695 | Bootes 谱聚类重排 | MICRO’25 | 行重排降片外 1.4–2.3× | r0 稀疏 W 的 B 复用 | 预处理税；决策树先判断值不值 |
| 705 | ROZK | 2026 | NoC + 局部跳零 | 通用 DNN | 低优先 |
| 699 | HiPACK | 2025 | sub-8b 直接卷积 SIMD | 连续卷积 | r0 Q2 段 |
| 697 | MCBP | 2025 | LLM bit-slice 稀疏 | LLM | 位片 A |
| 712 | MD-SNN | 2026 | 膜感知蒸馏量化 | 算法 | 训练 |
| 681 | Unicorn | 2022 | 多核神经形态 | 旧平台 | 停标题 |

**成倍加速的共同思路（不是「更聪明的加法器」）：整段工作取消。** 深层级、剩余时间步、剩余 token、静默通道、相似输入窗。GeMM 跳零在本网发放率 4% 上已经吃过，再抄只能再挤几个百分点。
