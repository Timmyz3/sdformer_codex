# M1050：M1046 C2 mapped-gate watchdog 独立失败审计

结论：**M1046 已消费且失败，DO NOT RETRY。** 完成 gate case 为 0，生产 SAIF 为 0，不存在功耗或能量结果。

## 根因

失败不是许可证、编译、plusarg、UCLI、SAIF 命令或工作量过长。tiny UCLI-power preflight 通过，K1 mapped netlist 编译/链接成功，case0 在 edge 3 接受 header。去掉 UCLI 和 SAIF 后，quarantine `simv` 仍在 300015 ns 触发同一 watchdog。

短时信号探针显示，raw packet 在 22 ns 合法且 `raw_ready=1`；首个 raw accept 后、首个 memory request 前的 25–28 ns，K1 mapped netlist 的 service、memory-adapter 和 core-adapter fault state 开始被 X 污染，随后 `mem_req_valid/ready`、`result_valid`、`protocol_error` 扩散为 X。request/response/result 计数始终为 0。

使用完全相同的 cell model、mapped K1 netlist、memory model 和 TB，仅在诊断编译加入 `+vcs+initreg+random` 后，全 0、全 1、随机 seed 1/7/29 五种二进制初态全部通过：20 events、259 cycles，numeric/tuple/weight/unknown/protocol mismatch 均为 0。这把根因闭合为 gate-level 未初始化状态的 X propagation，而不是功能协议死锁。诊断不生成 SAIF，也不构成 canonical 结果。

## 额外发现

M867 的十个 clean case 只覆盖 K8 和等带宽 K1×8。M979 `expected_cycle()` 对 K1 返回 -1，因此当前生产 TB 没有 K1 cycle anchor；259 cycles 只能作为本次诊断观察，不能直接升为 admitted reference。

## 最小 additive repair 门

1. 新 runner、新 namespace；禁止复用或重试 M1046，也禁止只放大 watchdog。
2. tiny preflight 和三个 mapped axis 都必须 compile-time 启用 `+vcs+initreg+random`。
3. 消费新 attempt 前，15 个 case 必须在全 0、全 1和至少三个冻结随机 seed 下全部功能通过。
4. 先独立跑 RTL K1 五 case，冻结 cycle tuple；后继 mapped K1 必须逐 case exact equality，禁止保留 `-1` bypass。K8/K1×8 继续绑定 M867 anchors。
5. canonical SAIF 使用一个冻结的非均匀 seed，活动窗口仍从 accepted header 后开始；完成独立 source/release hammer 后才允许一次新尝试。

更强但工程量更大的替代方案，是给被优化掉 reset 的状态补完整 reset 或可证明的 valid-bit X isolation，重新综合后再跑 mapped-gate。
