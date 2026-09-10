# C1 / C2 同负载重构复核

先读 [一页净收益表（PDF）](net_benefit.pdf) 或 [完整说明](report.html)。两条线均未完成生产重构；本包把完整算子与真正时间并行的强对照补齐，当前新增共享均没有时延净收益。仅面向 TCAS-II，未改生产 RTL、主稿或旧封存证据。

- C1：完整 M3000/K6912/N768，五轴×四档系数缓存×两种端口，共40点CPU模型；全部保留 θW、完整K、有限父/Y状态、共同值散播和最终提交。当前融合未胜出；不开展融合RTL。[周期表](c1_cycle_table.csv)、[算法独立审阅](c1_independent_review.md)、[排程独立补审](c1_schedule_independent_addendum.md)。
- C2：32个捕获位置的完整T10/C384/H96；生产原SHA、真正八bank时间并行、相同结构加共享三轴，Verilator加共同的行为I/O。共享多215周期（+0.23454%），总加法少8.9552%。[实现与边界](c2/README_R4.md)、[独立审阅](c2_parallel_independent_review.md)。
- 五项未被一般性证伪的思想分别说明先验、迁移条件、旧负例范围和下一项区分性测量：[半页逐项复审](five_ideas_reassessment.md)。独立概念差异5–6/10，不是录用概率。

## 复现边界

使用 `/opt/anaconda3/bin/python3.12`、G++8.5和Verilator4.028。C1入口依次为 `c1_prepare.py`、`c1_run.py`、`c1_cache_run.py`、`c1_audit_run.py`；数值、资源与阶段修订见各plan和attempt快照。C2入口为 `c2/run_r4.py`；重跑应使用新的attempt目录，保留既有失败与成功收据。不要覆盖本封存包来“更新”结果。

C1诊断使用冻结模块的精确连续θ和确定性诊断W；C2使用非单位θ=9/8的有理数诊断。两者都没有以真实训练W验证冻结FP32算术或AEE。当前C2不是完整LoAS实现，BN2/shortcut、完整生产/压缩及全帧尚未闭合。CPU周期、Verilator条件计时、ASIC实测不可混写；没有VCS＋DC/PT＋Formality同工作负载闭环或PPA准入。

`SHA256SUMS`封存本包文件；`artifact-qa.json`记录呈现核查，`completion-integrity.json`记录生产只读与主收据完整性。具体执行进展与缺口见 `WORKLOG.md`。
