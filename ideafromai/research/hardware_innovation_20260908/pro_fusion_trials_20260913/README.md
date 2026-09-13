# Pro 建议与点名先验的实际 RTL 融合筛选

2026-09-13。本阶段已交付三个可运行 SystemVerilog 执行器、真实输入/权重的 Verilator 结果及交叉独立审阅。**不是只做论文卡片；也没有把这批筛选写成 ASIC 加速或新标题。**

- [本阶段结果与下一步](REPORT.md)
- [全部点名论文、完整 A 与未试项](NAMED_LITERATURE.md)
- [Kronecker：真实连续矩阵与两级收缩](kron/REPORT.md)
- [T10 区间：端点/prefix/RNE 完整执行](interval/REPORT.md)
- [共同活动打包：U16 与昂贵 r0 N96 实测](packing/README.md)
- [ELSA/Phi/剪枝打包先验补缺](literature/REPORT.md)
- 独立代码审阅：[Kronecker](interval/REVIEW_KRON.md)、[端点](kron/REVIEW_INTERVAL.md)、[打包](literature/REVIEW_PACKING.md)；[新颖性评阅（含自审）](kron/NOVELTY_REVIEW.md)
- 网页Pro协作：[共享对话逐项审阅](pro_assist/SHARED_DISCUSSION_REVIEW.md)、[连接核查](pro_assist/CONNECTION_OPTIONS.md)、[下一次Pro问题包](pro_assist/NEXT_PRO_REVIEW.md)

身份保持 AT-LIF `{0,θ}`，静态 θ 可折权；连续 PED 独立计费。每项明确学生快照、量化/舍入与入口范围。质量门按用户最新指令与原 SDformerFlow NB0 比较；本轮没有新训练或 AEE，近似候选不继承历史 AEE。没有修改 nts07、main.tex，也没有 EDA/PPA 或 hash 流程。
