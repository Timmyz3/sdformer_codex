# M1088：M1080 C2 mapped-gate 失败独立审计

结论：**M1080 已消费、失败并保持 DO NOT RETRY。** fresh DC 与 mapped compile 均完成，但唯一运行的 K1 case0 没有完成；五例中 0 例通过，不存在 SAIF、PTPX、功耗或生产 PPA 结果。

## 精确边界

- 失败包递归封印完整；DC `rc=0`，mapped VCS compile/link `rc=0`。
- logic-only、pre-macro、ideal-clock、ZeroWireload DC 的诊断值为 `124351.163170 um^2`；3 ns setup 最差路径 `+0.0007 ns MET`，precompile `TIM-209=0 / OPT-150=0`。hold 未闭合（QoR 报 29111 条 hold violation）。由于后续 mapped 功能门失败，这些数只能用于失败诊断，禁止进入论文 PPA 表。
- case0 进程本身以 `rc=0` 退出，是因为 TB 的 `$fatal` 后执行 `$finish`；runner 的 anchor/PASS gate 正确把它判为 `return_code=3` 的 `FRESH_MAPPED_VCS_CASE0` 失败，不能用 shell rc 冒充通过。

## 首个可证明的停滞

fresh transcript 在 edge 3 打印 `M979_SAIF_WINDOW_START`，证明 header 被接受；之后没有 result、token_done 或 PASS token，直到 `300015000 ps` 的局部 watchdog。因 quarantine 中波形数为 0，M1080 证据只能把首个停滞界定为 **header accept 之后、token_done 之前**，不能直接定位到某个内部信号或宣称本次重新观察到 X。

它与 M1046 的外部症状和 watchdog 时间完全相同。M1046 的独立短探针曾把首个 X 定位在首个 raw packet 后、首个 memory request 前的 25--28 ns，并看到 service/memory-adapter/core-adapter fault 与 `mem_req_valid/ready` 等被污染；该结论是强先验，但不是 M1080 fresh run 的新波形证据。

## M1058 reset payload 是否保留

保留。source diff 只给 K1 service FIFO 新增 4 项 payload reset；mapped netlist仍含精确的 96 个 `fifo_tag`、12 个 `fifo_block`、12 个 `fifo_bank_id`、48 个 `fifo_channel` 寄存器，共 168 bit。代表性 D 锥由 reset 派生选择网络控制，而不是被 DC 删除。因此不能把本次失败归因于“M1058 reset 语句完全未综合进去”。

但该修复只覆盖 service FIFO payload，不能证明所有跨模块有效位、fault/hold 状态和被解码 payload 均完成 X 隔离。fresh log 也没有足够观测去判断 token 是卡在 raw accept、frontend group、memory request、response还是 completion。

## 根因分类与最小 additive 后继

当前证据排除：许可证/compile/link 失败、header 不接受、只需加大 watchdog、M1058 新增 payload reset 被整体优化掉。复位在 4 个完整时钟周期保持有效，并在负边沿释放，**reset timing 不是首选解释**；本次为零延迟 mapped 仿真且无 SDF，亦没有证据支持 SDF race。

最小后继分类为 **remaining unreset control/payload or valid-bit X-isolation gap**，其次才是 synthesis-specific X reconvergence/optimization。不要重跑 M1080。新 namespace 先做一个 observation-only、短窗 gate case0：逐拍 fail-closed 记录 `raw_accept`、frontend group accept、service request accept、adapter bank request、response、result/done，以及这些 valid/fault/ready 的 `$isunknown`；首个 X 或首个永久不前进级一旦锁定，只对该级补完整 reset或 valid-bit 隔离，再建立新的 RTL→DC→mapped gate 封印。禁止 initreg 作为生产修复，也禁止先跑五例或 PTPX。

`docs/359_DATE终局冻结_20260813.md` SHA 保持 `dedde7ce...`。
