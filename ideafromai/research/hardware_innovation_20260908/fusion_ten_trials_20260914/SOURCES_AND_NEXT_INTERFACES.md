# 本批来源与尚未实施的接口

这十项以实际RTL筛选为主。下表区分借入机制与完整论文复现：本批没有把作者整芯片复制下来，也不把自写局部适配称为完整UCNN、LUT-DLA或GustavSNN。比较用本地最新native source、dualP、cachedOS和真实消费者，作者论文中的速度不借来作本地结果。

| 来源 | 读到的内容/公开范围 | 与当前试验的关系 |
|---|---|---|
| [UCNN，ISCA2018](https://arxiv.org/abs/1804.06508) | 作者预印本，weight repetition、dot-product factorization、activation-group reuse；本批复核摘要及作者PDF方法片段 | Q1完整列字典借用先归约再乘的A；当前32类完整向量接口不是整个UCNN复制 |
| [Distributed arithmetic电路文献](https://www.mdpi.com/2079-9292/8/1/108) | 出版者方法页，按输入位构造LUT地址和符号位处理 | Q2两组R4的普通DA强先验，表建立、读口、最短signed位宽都已在本批RTL计费 |
| [da4ml](https://arxiv.org/abs/2507.04535) | 作者预印本与既有本地常量编译调研 | 已有CSE/DA框架不能单独包装成X；本批没有声称运行其完整编译器 |
| [LUT-DLA，HPCA2025](https://arxiv.org/abs/2501.10658) | 作者预印本；本地此前有详细阅读 | AS2的原型、编码、查表有直接A；本次实际只测试K4+一残差及相同资源控制 |
| [Computation reuse via input similarity，ISCA2018](https://upcommons.upc.edu/entities/publication/f74c9548-1ae7-42b5-b59c-fc6832cfe280) | 作者机构文献页 | AS3不能把近似相邻输入复用当首创；额外差分权限必须给逐rank控制 |
| [Bishop，ISCA2025](https://arxiv.org/html/2505.12281v1) | 既有本地组稀疏文献 | AS1整组删除本身属于A；I24误差校准还须相对普通幅值剪枝胜出 |
| [MIDAP](https://openmidap.github.io/) | 作者项目/University Demonstration，完整栈计划2027H1公开 | 数据生命周期、缓冲与层流水是强A；目前不是已可下载的完整公开RTL |
| [DeVSA，Electronics2026](https://www.mdpi.com/2079-9292/15/6/1296) | 本次出版者摘要/检索片段，全文429 | 宽算术/脉冲复用的相关先验；不能据未取得正文宣布内部边界空白 |

另外定向补读了三个接口，**不计本批十项已测数**：

1. [R-Sparse，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/c0c165157df7e3082f9c6d70d3a4b6e9-Paper-Conference.pdf)：已读§3.2–3.5。它把大输入分量送原W，小输入分量送低秩近似，并给出列布局及按层预算搜索。作者代码链接在论文中。这对连续latent/PSN有参考价值；对当前二值g不能直接按幅值分流，对已R8的Q2也不能再次冒用原dense矩阵分母。未试接口是“当前连续消费者的互斥精细/近似两路”，而不是在R8旁无条件再加一条全量支路。
2. [D-com预印本](https://arxiv.org/abs/2510.13147)：已取得作者HTML、核摘要，正文尚未在本批完整精读。目标是运行时activation分解，摘要明确正面处理分解本身费用。当前T10/R8尺寸小，是否值得在线分解未知；没有据LLM结果宣称适配或杀掉该方向。
3. [ScanNow，ICCAD2025](https://researchportal.hkust.edu.hk/en/publications/scannow-a-scan-window-based-sparse-matrix-multiplication-accelera/)：作者机构方法摘要与官方会议目录定位，未取得完整实现。它按稀疏模式平衡输入/输出复用，提醒双context交错和窗口调度本身是A。本批双context RTL是自己的受限接口，不称复现ScanNow。

本次来源更新是有界检索，未补齐所有CICC/ISSCC/DATE目录。上述未试项只是后续接口备选，不影响本批十条的实际去留，也不能把读到摘要计为完成实验。
