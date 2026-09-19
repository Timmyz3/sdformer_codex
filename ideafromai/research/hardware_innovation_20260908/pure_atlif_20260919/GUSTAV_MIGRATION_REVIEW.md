# 去 PSN、采用递推 AT-LIF 后的 GustavSNN 迁移审阅

2026-09-19。范围仅限 GustavSNN 全文、官方 AT-LIF 源码及当前 H9 的相关执行边界；未运行 GPU、未修改生产代码。**去掉 dense T×T PSN，确实使神经元更接近 Gustav 的逐 tick 局部状态模型；但不是现有网络已经能直接套用其完整无全局膜访存数据流。动态 BN、窗口／注意力依赖和真实时间轴布局仍须处理。**

**原文究竟实现什么。** [GustavSNN 官方 HPCA 页面](https://2026.hpca-conf.org/details/hpca-2026-main-conference/83/GustavSNN-Unleashing-the-Power-of-Gustavson-s-Algorithm-on-SNN-Acceleration-with-Col)与 [IEEE 论文页面](https://ieeexplore.ieee.org/document/11408587/)核对了题名、作者和 DOI `10.1109/HPCA68181.2026.11408587`。详细机制来自本地 [14 页 PDF](../literature/GustavSNN_HPCA2026_public_mirror.pdf)及[文本](../literature/GustavSNN_HPCA2026_public_mirror.txt)：

- §II-A Eq.(1)，PDF p.2／文本约 178–194 行：输入积分为 `v_pre[t]=λ·v_post[t−1]+Σw·s[t]`；到阈值发放 1，然后 `v_post=v_pre−Vθ`，即论文定义的 **soft reset**。本次没有取得作者 RTL，不能把论文公式称作逐位 RTL 独立验证。
- §IV-C Algorithm 1、§V-B Fig.12，PDF p.4、p.7／文本约 366–447、703–757 行：输出通道 m 映射 tile，空间列分区 k 映射 PE 并行；每个 PE 在当前 tick 扫完该分区全部输入源、提交神经元输出，再进入下一 tick；同一组 P 个局部膜状态保留到该列批全部 T 完成，然后换列批并从 tick 0 开始。
- **CPTB 的并行维是空间列／输出通道，不是同时展开所有时间步。** tick-batch 表示一层的时间批处理；论文明确区分 PTB 的 time batch 和 FTP 时间并行。§V-B 说本作实现 LIF，其他逐 tick 神经元可替换 PE 内逻辑；这不是已经实现 PSN 的证据。全文未见 PSN 实现。非因果 dense PSN 会改变逐 tick 可提交条件与状态量，不能由“支持任意神经元”的架构表述推导零代价接入。

**官方 AT-LIF 与 Gustav 的差异。** [官方 layers.py](https://raw.githubusercontent.com/putshua/Activity-Pruning-SNN/main/models/submodules/layers.py) 的 `ATLIF.forward` 为每次调用先令 mem=0，再逐 t 做 `mem=tau·mem+x[t]`、`y=θ·H(mem−θ)`、`mem=(1−y/θ.detach())·mem`。其前向是 **hard reset 到 0**；分母 θ 被 detach，spike 本身没有 detach，复刻训练时不能悄悄改成 detach-reset。θ 是共享的可训练阈值，推理冻结时是静态常数，不是必然每 tick 自适应的 θ[t]。默认 tau=1；[官方 main.py](https://raw.githubusercontent.com/putshua/Activity-Pruning-SNN/main/main.py) 约 600–611 行对 CIFAR／VGG 分支显式用 0.9，ImageNet ResNet 分支沿用默认。这里 tau 是直接衰减乘数，不能按另一框架的膜时间常数解释。

| 去 PSN 后的接口 | 可直接借入的部分 | 仍需做的部分 |
|---|---|---|
| 二电平输出到 Conv／Linear | GP 稀疏权重累加；P-bit NRV 及逐 tick 空行跳过；权重行 tile、共享双端口权重缓存；L1D／merger、局部累加和 fetch/execute 双缓冲 | 静态 θ 在合法线性消费者的输入列／通道折入权重，覆盖所有分支；bias 不乘 θ；量化和幅值范围仍要验证。 |
| 递推神经元 | 每 PE 保存 P 个膜状态，在 tick 完成时更新、比较、发放的 CPTB 调度 | 把论文 soft reset 改为官方 hard reset；匹配 tau、比较精度、膜位宽、初始化和调用边界。NRV 跳过零输入行不等于可跳过衰减、bias 或整个神经元 tick。 |
| 原 dense T×T PSN | 移除未来 tick 对当前神经元门的 dense 依赖及对应矩阵乘法 | 这是新模型，原学到的 T×T 权重／时间 bias 不能自动变成递推状态；需要同协议恢复精度，不能用算子省量代替 AEE。 |
| 动态 BN | GP 仍可生产 BN 前的线性结果；θ-fold 恒等式仍成立 | 当前 multi-step BN 用 batch 统计，跨其时间／批次／空间归约域；局部列批完成不能立即得到归一化值。需全域统计＋保存／重算或另一经验证方案。固定 BN 可编译为仿射，但不能默认其精度已恢复。 |
| T2 Swin／h60 注意力 | Q/K 的线性投影可用 GP，已有专用支持位／motion 算子可保留 | 当前 motion XOR 读成对时间 K，score 中心化／Shiftmax 还跨整个 2×15×15 token 窗口；窗口重排、shift、跨窗口批次 BN、门控载荷与 residual 仍有缓冲／同步需求。去 PSN 不会把这些变成逐 tick 独立 SpMM。 |

**额外发现：本地 T2 首轴不应直接当成真实相邻时间。** 上游 `SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:100` 的 `window_partition_v2` 先排列为 `[B,Dgroup,Hgroup,Wgroup,localT,...]`，然后直接 `contiguous().view(windowT,-1,Wh,Ww,C)`，没有先把 localT 移到首轴。本次 AST 抽取并执行原函数，只用 NumPy 包装替代 Tensor 的 `view/permute/contiguous`，没有手改重排表达式；D=4、H=W=C=1、windowT=2、输入标签 `4*b+t`，结果如下：

| 输入 batch | 原函数输出的首轴矩阵 | 沿首轴递推的 lane |
|---|---|---|
| B=1 | `[[0,1],[2,3]]` | `(0,2)`、`(1,3)`，非相邻时间 |
| B=2 | `[[0,1,2,3],[4,5,6,7]]` | `(0,4)`、`(1,5)`、`(2,6)`、`(3,7)`，跨不同样本 |

可复现的原表达式等价 NumPy 最小片段（不需 torch/GPU）：

```python
import numpy as np
for B in (1, 2):
    x = np.arange(B * 4).reshape(B, 4, 1, 1, 1)
    y = x.reshape(B, 2, 2, 1, 1, 1, 1, 1)
    y = y.transpose(0, 1, 3, 5, 2, 4, 6, 7).copy().reshape(2, -1, 1, 1, 1)
    print(B, y[:, :, 0, 0, 0].T.tolist())
```

这是**已有冻结张量行为**，不是本次 AT-LIF 改造引入：本地 git `7979e8c0`（2026-08-17）已有同表达式；`open_fusion_execution/default_bn/onepass/aee_check/diagnose_flip.py:76–85` 已专门捕获 actual window function 的 reshape／padding／roll，而不假定时间顺序。但这些证据不证明原作者有意让“时间”跨样本；当前 BSA 注释所写的自然时间对语义也不能推翻实际索引。`SSA` 使用配套 `view → window_reverse` 把相同展开序列还原，因此纯 identity 往返可以成立，仍不保证中间 T2 神经元轴就是局部时间。本地 H9 overlay 未找到对此函数的替换；远端实际载入版本仍应核对。

补核（root，2026-09-20）：远端本轮实际仓库 `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:100–113` 也是上述表达式，未见远端局部修正。

因此隔离“只去 PSN”臂宜保留既有布局并准确称为**沿现有张量首轴递推**；若要称生理／事件时间上的 causal AT-LIF，需要另验真实 tick identity，并把布局修正列为独立变更。T10-only 臂保留旧 T2 部分，有助于隔离神经元替换，不代表已经消除整网该布局语义。即使修好布局，注意力神经元一次调用长度为 2、外部部分层长度为 10；官方每调用 mem=0 的语义也不等于整网一个跨所有窗口持续的 T10 膜状态。本文不修改分窗函数或其逆变换。

**迁移判断。** Gustav 的 GP+CPTB+NRV+局部状态是可复用强基线，官方递推 AT-LIF 只要求替换小的 neuron update 逻辑，神经元层面比 dense PSN 更合适；上述全网依赖使“直接套完整 Gustav、自动没有宽状态/全局等待”仍不成立。本文只给机制与源代码判断，不把原论文性能倍率迁移到本网络，也不据此预言恢复精度。
