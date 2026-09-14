# 本轮打包 RTL 的定向先验补读

2026-09-14，root 在实现并行期间补读。目的是把强底座带完整，不因找到先验而取消试验；也不把经典子字并行改名当 X。此前 UCNN/Phi/SmartExchange/LoAS/Prosperity 的实际章节与代码范围沿用[来源表](../r0_stream_fusion_20260914/novelty/source_table.csv)，不是本轮又读了同样多篇。

| 工作 | 本次实际阅读 | 对当前实现的约束 |
|---|---|---|
| Bit Fusion，ISCA 2018 | [作者预印本 v2](https://arxiv.org/html/1712.01507) §II–III：融合单元、供数、符号处理、混合位宽和移位归约；工件站本次 502 | BitBrick 组成不同宽度乘法、位宽对应供数和宽部分和已有完整设计。当前 RTL 只借入精度匹配的思路，未实现其阵列或 ISA。 |
| Envision，ISSCC 2017 | [原会议三页 digest](https://reconfigdeeplearning.wordpress.com/wp-content/uploads/2017/02/isscc2017-14-5digest.pdf)，正文及图注；论文镜像，不是代码 | 原作已有可重构子字累加器和寄存器、横向特征 FIFO 复用、稀疏标志控制访存与 MAC。因此“打包＋卷积窗口＋零跳过”整体仍有直接先验，不能单凭这三者的组合立 X。 |
| BitBlade，DAC 2019 | [作者所在学校的出版记录/摘要](https://snu.elsevierpure.com/en/publications/bitblade-area-and-energy-efficient-precision-scalable-neural-netw/)；本次未完整读电路章节 | 位宽扩展的移位归约控制开销已有针对性设计。没有在本轮完整复现，也没有据摘要宣称其缺少某个缓存/控制分支。 |
| BitL，MICRO 2025 | [正式 DOI](https://doi.org/10.1145/3725843.3756044) 搜索条目可读，正文请求 403 | 保留近期 bit-serial/parallel 和关键路径方向的全文补缺；不凭搜索片段做细节差分或将其算作全文阅读。 |

当前双位置接口的可检验问题较窄：固定 Q1g 的累加上界容许每位置用 signed13 保存；将相邻两个位置放进同一个 26 位 bank 字后，同一个时间事件是否能共同更新两个独立部分和，而且在完整 Q2 与真实 I24 消费者上仍少拍。两臂须给同样 208 位向量口、520B latent 和 carry 分割权限。这个问题适配本网，但适配性不是新颖性证明。

若有收益，先保留可运行执行底座；还需证明特有的表示/训练或消费者约束相对通用子字控制多做了什么。若收益小，只停止把当前打包控制作主标题，保留它作为之后共同控制，不恢复更弱分母。
