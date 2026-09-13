# 稀疏与打包研究交付

- [中文机制报告](mechanism_report.md)：当前切口、九篇primary方法、两个候选、强控制及创新边界。
- [下一轮RTL工单](rtl_workorders.md)：完整K864/N96/T10 r0 Conv2底座、lane-private增量、原生字闭包增量。
- [WINS方案独立评审](review_decomposition_d2.md)：对联合幅值类/消费者mask的整字反例。
- [初始四项假说](hypotheses_initial.md)：阅读前独立提出，保留被降级路线。
- [来源CSV](source_master.csv) · [来源JSON](source_master.json)：title、venue、year、primaryURL、codeURL、read_scope。
- `sources/`：九篇primary的PDF及本地文本；只计正文方法读过的九篇，SpiDR最终venue未核实。

本轮未跑训练、CPU机制筛选、RTL或EDA。优先把普通N:M完整原生供数做成强A；独立lane队列不算创新，X需来自mask选择与实际端口执行的可隔离净增量。固定fill1:4的已有负结果没有换名重推。
