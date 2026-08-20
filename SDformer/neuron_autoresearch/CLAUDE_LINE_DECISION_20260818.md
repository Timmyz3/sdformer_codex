# 线路锁定决策（用户指令，2026-08-18）

## 决策

1. **H82/H86 线停止**：H86 训练已停止（epoch 0，status.log 已记录 ABANDONED）。H86 合同保持冻结原状（OPERATOR_FROZEN_WAITING_FOR_H82_GPU），不删除、不改动，作为存档。
2. **算法锁定 Motion（H67）与 Local5 两条线**，DATE 主线在两者中选一。
3. **创新重点**：两条线的创新分 4+ 推进（用户确认 4+ 在其他线上可行）。
4. **实验补齐**：以 Motion 或 Local5 为主线发 DATE 所缺的实验，直接推进。

## 现状锚点（2026-08-18）

| 线 | valid825 AEE | 硬件创新分 | 硬件综合 |
|---|---:|---:|---:|
| Motion/H67 ep35 | 1.3297 | 3.2 | 3.1 BR |
| Local5 ep44 | 1.2819 | 3.1 | 3.1 BR |

- MVSEC day2-scratch 已完成：H67 唯一过四序列门
- 已知完整度缺口（docs/439）：full-encoder Amdahl、多样本真实 INT8 全通道、12-block 同窗、目标库 DC/SAIF（新思机器）
- AAE 差距：本地三聚合不能闭合 4.871，剩余项是官方 DSEC 提交 writer

## 执行

- Round2 创新攻击：Motion 跨窗 RQTB quotient（进行中）；Local5 独立攻击（启动中）
- 算法缺口审计+补实验：独立 agent（启动中）
