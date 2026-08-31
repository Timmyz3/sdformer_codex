# M470 独立打铁审计 r1

结论：**数据与算术 PASS；RTL 轴 KILL。** 评分 **78/100**。M470 可保留为四个冻结 H67 bottleneck Conv3x3 的 CPU 周期 DSE 负证据，不准入性能主张，不启动 RTL/Synopsys。

## 独立复算结果

- 双层生产者封存通过：result SHA-256 `7817460e...9ad165`，seal 文件 SHA-256 `e4697fb4...8c8a8`；`docs/359` 仍为 `dedde7ce...bdfc4`。
- r1 admission 仅写错 producer result SHA，已经 fail-closed 撤销；r2 正确绑定未改动的 producer result/seal。
- 共复算 2,978 项检查，0 mismatch；JSON/CSV 的 147 个 DSE 粒度键唯一且一一对应。
- 147/147 点均满足：`spill write = reload read = operator_window_boundary_count × 5,472,000 B`，cycle component sum 精确相加，logical 与 macro-rounded 容量均不超过 245,760 B，`m40_payload_reads=0`。
- 128 B/cycle 的 stored-PWP 最优点为 P=5、4 banks：892,869,158 cycles；同资源 strong-zero 为 1,148,674,816 cycles，即 1.286498x。
- 请求集合 P={1,2,4,8} 的最优可行点为 P=4：964,742,918 / 1,218,613,216 cycles，即 1.263148x；stored-PWP P=8 在 4/8 bank 下均不通过 macro 240 KiB 门。

## 打铁裁定

内部同 schedule 的 1.286498x 是自洽的，但不是可用的竞争优势：同一候选 892,869,158 cycles 相对更强的冻结 zero 基线 742,148,386 cycles 仅 **0.831195x**，相对 M468R3 stored 872,452,768 cycles 仅 **0.977134x**。因此：

- `performance_admitted=false`
- `rtl_nominated=false`
- `decision=KILL_M470_RTL_AXIS_KEEP_CPU_DSE_AS_NEGATIVE_EVIDENCE`

边界：只覆盖四个冻结 H67 bottleneck Conv3x3，是 CPU cycle-simulator estimate；不是 RTL/Synopsys、能量、全网或系统倍速。

复跑：`/opt/anaconda3/envs/pytorch310/bin/python hw_autoresearch_nts07/reviews/m470_independent_hammer_r1_20260826/audit_m470_independent.py`
