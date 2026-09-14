# R8 完整消费者、825 与五条融合 RTL

2026-09-14。本批五个表示/执行接口均已实际实现并筛选；R8 从完整源卷积接到真实 FP32 identity、固定 BN 和 I24，并完成两套各含对照/候选的单指令整帧运行。新函数的 valid825 为 **1.327635**，同一 A800 环境原模型 NB0 为 **1.447937**。保留性能底座，当前没有标题级 X 或强接收结论。

- [阶段报告与去留](REPORT.md) · [同口径比较表](comparison.csv)
- [最终支持、时间差分、原生窗口 RTL](fusion_rtl/README.md)
- [跨 rank 同幅归组 RTL](partition_rtl/README.md) · [双位置打包 RTL](packed_rtl/README.md)
- [原生窗口＋真实消费者完整帧](consumer_rtl/REPORT.md) · [打包＋同一消费者完整帧](consumer_packed/REPORT.md)
- [质量与真实数据](data/README.md) · [825 配对验证](data/quality_validation.json)
- [独立新颖性判断](ROOT_NOVELTY_REVIEW.md) · [子字电路先验补读](PACKING_PRIORS.md)
- [原生消费者根审阅](REVIEW_CONSUMER_ROOT.md) · [归组根审阅](REVIEW_PARTITION_ROOT.md)
- [打包独立审阅](data/REVIEW_PACKED.md) · [打包完整消费者独立审阅](data/REVIEW_CONSUMER_PACKED.md) · [质量尾差与复测](data/QUALITY_REPORT.md)

这是本批已选融合的收口，不是宣称 idea 仓库所有论文/接口均已尝试。数据、编译器产物和模型大文件保留在本地对应路径，Git 记录源码、重建入口、结果与解释。生产 nts07、主稿、历史封存均未改动。本目录的百分比是隔离 Verilator 周期，尚无 VCS＋DC/PT＋Formality 闭环、PPA 或整网 FPS。
