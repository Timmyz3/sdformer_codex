# r0 分解与有限域执行研究

- [主报告](REPORT.md)：WINS有限字母表为第一候选，整数桶与Kronecker为两个挑战者；明确A/B/X、算子、误差、变换税与首RTL范围。
- [结构化来源CSV](source_table.csv) / [JSON](source_table.json)：10篇方法/硬件段精读，1篇FINEA摘要碰撞，逐项说明venue/year/code与未复现范围。
- [检索前四假说](independent_hypotheses.md)：受本地历史启发，不伪称盲法。
- [穷举脚本](verify_winograd_alphabet.py) / [65,536输入结果](winograd_binary_exhaustive.json)：纯数学核验，无目标模型或硬件性能实验。
- [对sparse lane-private K4的独立半页审阅](review_sparse_lane_private.md)。

本轮没有训练、RTL仿真、EDA或性能PASS；primary PDF与提取文本仅作本地可查阅读档案。`sources/hybridnet.*`下载但未精读，不计技术来源数。
