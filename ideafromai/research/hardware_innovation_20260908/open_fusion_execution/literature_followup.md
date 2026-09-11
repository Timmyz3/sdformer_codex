# 本轮试验产生的近邻补读：空间/通道剪枝

2026-09-12。针对正在执行的相位H8剪枝补最近邻；不等全文综述闭合才试验，也不把没有搜到完全相同的接口当作首次证明。

| 工作 | 这次核了什么 | 对当前融合的影响 | 尚未尝试 |
|---|---|---|---|
| [MICRO 2019：Boosting the Performance of CNN Accelerators with Dynamic Fine-Grained Channel Gating](https://zhouyuan1119.github.io/papers/cgnet-micro2019.pdf) | 作者PDF首页、方法/架构相关文本 | 原目录把它写成NeurIPS论文题名；已在汇总层分开。部分和决策、通道分支、规则性与取数布局均应借入，不能卖普通空间-通道门控 | 完整原作者条件分支硬件在本模型迁移 |
| [NeurIPS 2019：Channel Gating Neural Networks](https://proceedings.neurips.cc/paper/2019/file/68b1fbe7f16e4ae3024973f12f3cb313-Paper.pdf)；[作者代码](https://github.com/cornell-zhang/dnn-gating) | 官方论文摘要/首页与仓库README | 同族另文；空间/通道条件稀疏不是本地新原语 | 完整训练与原条件执行控制 |
| [WACV 2024：SPSRC](https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Towards_Better_Structured_Pruning_Saliency_by_Reorganizing_Convolution_WACV_2024_paper.pdf)；[作者代码](https://github.com/AlexSunNik/SPSRC) | 官方摘要、方法入口、作者README | 卷积重组后得到空间saliency已有。应作为比简单幅值更强的剪枝对照；本地`gate+PED`误差评分本身不够撑标题 | 原`conv_to_mat`及谱/核/Frobenius范数的同预算迁移 |
| [CVPR 2017：Spatially Adaptive Computation Time](https://openaccess.thecvf.com/content_cvpr_2017/papers/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.pdf) | 官方原文摘要/方法入口 | 空间位置选择计算量、保留对齐已有；不能把棋盘少算作为新颖性来源 | 与本网络非因果T10、真实残差的训练适配 |

当前实际候选比这些先验多问的一步是：消费者的固定stride与物理H8广播域，能否反向限定**整条T10生产任务**可删的集合，并在保留原始I24的情况下得到比普通全空间减宽更好的AEE/服务交换。它现在有真实执行与免训AEE，但尚无标题级优势。前向范围、后继义务和相同普通剪枝控制继续保留。

检索为本轮有界补查。上面没有标“全文精读”的条目不提升阅读等级。作者开源代码不等于提供全部ASIC RTL；没有借用其加速比、面积或功耗作为本地数据。
