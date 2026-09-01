# M1696｜M1695 C1 fast-min hold closure source 独立 hammer

## 裁决

**PASS 99/100，P0=0、P1=0、P2=1**：仅授权不同作者编写 M1697 one-shot release；当前仍不授权 DC。

## 已闭合

- 设计数据库输入只有冻结 M1665 DDC；M1678 只作为双段 Formality PASS 与独立 PT fast-min 失败/29.174 ps 校准证据，所有读取文件均由 exact SHA 绑定。
- standard-cell 与 9 个 SRAM macro 分别唯一执行 slow→fast `set_min_library`。
- `0.081 ns` 只包围一次 `set_fix_hold` 与一次 `compile -incremental_mapping -only_hold_time`；随后立即恢复 `0.050 ns`，最终 SDC/报告由 runner 重新验证 `3.000/0.200/0.050 ns` 且禁止 0.081 泄漏。
- 禁止 false/multicycle/min/max delay、disable timing、case analysis、第二次 compile 或其他 EDA。
- future positive gate 要求 setup/hold WNS≥0、TNS=0、violations=0、9 macros、area≤168188.4885824 µm²、DRC=0，以及非空 DDC/SVF/SDC/mapped Verilog。
- `/tmp/date_dual_synopsys_same_uid_eda_queue.lock` 在资源探测前独占，并保持到结果 seal/publish；锁后共 3 次 ancestry-aware collision scan，其中包含立即锁后和 launch 前。
- attempt 由唯一 `mkdir` 在 dc_shell 前消费；失败封 quarantine，禁止 retry。

## 回归

双 Python 输出逐字节一致；每端作者测试 16/16；独立新增 24 类 DDC/RTL、library、uncertainty、restore、compile、exception、macro/area、M1678、lock/collision、attempt/retry、artifact/DRC/review-order 突变，全部拒绝。未运行 EDA、未创建 attempt/result/release。

## P2

M1695 只是 DC candidate flow。即便将来 DC 正结果，也必须重新完成 M1665→M1695 gate-to-gate Formality、独立 PrimeTime slowmax/fastmin（reported hold 0.050 ns）和不同作者结果 hammer，才能进入论文 PPA。
