# 新 dense 的固定低 RF 强对照

**这一固定编译点不满足原 512 指令 ROM：完整排程需要 524 字，多 12 字。** 算术和两级 RF 时序已核对；没有把程序截断、扩大 ROM，或写成通过原 RTL。

本项保持新 stage320 dense 的 As、最终 RNE/sat、literal 阈值与真实输入，逐行 signed CSD 展开、两条累加链交织。十个输入始终留在 RF0–9，链占 RF10/11，行和占 RF12，原门指令使用 RF95。13 个工作 RF 足够算这个 dense 函数，因此原公共 CSE 的 61 个峰值活 RF 不能当作 dense 的状态下界。

| 项目 | 实测 |
|---|---:|
| 每行 CSD 项 | 54、50、45、45、53、50、45、51、48、51 |
| 加减指令 | 482 |
| LOAD / gate / commit | 10 / 10 / 1 |
| 不计 NOP 的基础程序 | 503 字 |
| RAW / 排空 NOP | 20 / 1 |
| 完整程序 / 原 ROM | 524 / 512 字 |
| 工作 RF / 独立门 RF | 13 / 1 |

596 个边界和随机标量向量通过原 `validate_program` 的逻辑标签、两槽读就绪和精确最终门检查。真实 corner/interior 的 193,920 个输入值，分别核对了 193,920 个完整矩阵积整数与 193,920 个最终门，全部零差；两窗发放数仍为 4,165 / 6,625。最终 RNE/sat 采用与公共 dense 编译相同的精确门前像，没有改变模型函数。

这一结果只停止固定逐行两链 CSD 布局。跨行填空、低状态行 CSE、重算与其他普通编译接口本轮未试，不能据此证明 lifting 的状态优势已经胜过所有普通控制。也没有可报告的 RTL 周期或完整消费者加速比。后续 source/consumer 交织可以保留这一资源约束记录，同时仍须面对已准入的公共 CSE 与其他被明确选定的普通对照。

程序完整保存在 [program_not_admitted.json](program_not_admitted.json) 和相同字段编码的 [program_not_admitted.txt](program_not_admitted.txt)，结果见 [results.json](results.json)。从本目录执行 `../../../../psn/cmvm_20260909/.venv/bin/python run.py` 即可复跑，无需 GPU 或 EDA。该环境为 Python 3.12，使用既有源编译模块的原数值辅助函数。
