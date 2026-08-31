# M376：M368/M363 q128 DC 独立打铁

结论：**92/100，P0/P1/P2 = 0/0/5。全并行 q128 performance mainline NO-GO；这不是对所有 q128 实现的永久否定。**

M376 只读复核了 M363 author VCS、M364 independent VCS、M368 q128 DC 和 M329 q16 DC 的 manifest/seal。两组 DC 双层 seal 均逐文件通过；M368 日志无 `Error:`、`Fatal:`、`ELAB-312`、`TIM-209`、`OPT-150` 或 unresolved design。约束中没有 functional `disable_timing`、multicycle 或 case analysis；唯一 false path 是异步 `reset_n`。max/min delay、max capacitance、max transition、max fanout 五类均 clean。

q128 在 3 ns TSMC28 HPC+ pre-macro DC 下为 22,828.931733 µm²、22,348 cells、3,181 sequential cells、26 logic levels；setup/hold 分别为 +1.5158/+0.0250 ns。mapped Verilog、SDC、DDC、SVF 均非空且在 M368 seal 内。q16 同口径为 1,997.981971 µm²、2,394 cells、156 sequential cells、34 levels；setup/hold 为 +1.1141/+0.0252 ns。

独立比值：q128/q16 cell area = 11.425995×，cell count = 9.335004×，sequential cells = 20.391026×。两者 directed module-boundary 都是 II=1，所以原始 matcher issue/area 比值确为 0.0875197×，即低 91.248%。但 q128 搜索 128 个 center，q16 搜索 16 个；按 catalog capacity 归一化后是 0.7001579×，仍低 29.984%，却不能再称为 91.25% 的等功能密度损失。

为什么仍然 NO-GO 当前全并行 q128 主线：已封存的 k16/q128 exact-work opportunity 是 2.043940×，k16/q32 是 1.692877×；q128 相对 q32 只提供 1.207377× 的 work-bound 上限（额外减少 17.176% candidate work），尚未扣除 matcher、PWP 存储、DMA、队列和集成成本。面对 11.426× q16 area 和 20.391× FF，这不足以继续把 fully-parallel q128 当性能主线。active line 应维持 q32/O4 executable scheduler。

五个 P2：M368 原 seal 未覆盖 shell runner；q16 面积在 M368 中硬编码且只冻结 RUN_COMPLETE；raw density 未做 catalog-capacity 归一化；当前仍是 ideal-clock/ZeroWireload/0-macro、total area undefined；没有 Formality、PrimeTime、SAIF/PTPX 或 complete finite-queue executor。因此本结果不是 physical timing、paper PPA、cycle/system speedup 或 DATE headline 准入。时分复用、分层或 workload-specific q128 仍可在新证据下重新评估。
