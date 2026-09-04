# Bitmap-Preserving Bypass 与 45-Head 长尾消除

## 1. 本轮结论

本轮在完整 C0 片上 Builder 中实现了 Bitmap-Preserving Bypass（BPB）。当格式为 FADC24 且 term fanout 大于 21 时，canonical workspace 不再把 162-bit destination bitmap 展开为逐 token 事件，再由 Serializer 重建相同 bitmap；它直接传递原始 canonical bitmap。小扇出 FADC、IPD32W 和 RAW41 路径不变。

BPB 是语义保持的数据表示旁路，不是近似裁剪。四 stage 全部 45 个真实 head 的 861 个回放 word 全部与软件金参考一致，3226 个逻辑 destination 也保持不变 `[rtl]`。

## 2. 数据流变化

优化前：

```text
canonical bitmap
  -> segmented first-nonzero walker
  -> 逐 token destination
  -> Serializer bitmap assembler
  -> 21-byte FADC bitmap
```

优化后：

```text
FADC && fanout > 21
  -> 162-bit canonical bitmap 单次握手
  -> 21-byte Serializer
  -> 传输中逐 byte popcount 校验

其余 term
  -> segmented first-nonzero walker
  -> 逐 token list
```

Serializer 在原本必经的 21-byte 输出过程中累计每个 byte 的 popcount，并在最后一个 byte 检查其是否等于 descriptor fanout。错误 bitmap 只会污染私有 payload buffer，不会发出 `commit_begin`，因此仍满足 atomic commit。

## 3. 真实 45-Head 结果

证据范围是 `sample0/B0/window0` 的四 stage 全部 45 个 head，其中 44 个选择 IPD32W，1 个选择 FADC24。

| 指标 | BPB 前 | BPB 后 | 变化 |
|---|---:|---:|---:|
| 逻辑 destination | 3226 | 3226 | 不变 |
| destination 扫描/旁路握手 | 3226 | 2728 | -15.44% |
| C0 45-head latency 总和 | 14576 | 14078 | -3.42% |
| latency p99/max | 1561 | 1063 | -31.90% |
| S3/H4 FADC service | 1298 | 800 | -38.37% |
| 回放 word | 861 | 861 | 零失配 |

S3/H4 含 15 个 bitmap term、814 个逻辑事件。其中 bitmap term 承载 513 个事件，BPB 将 513 次 token 交付替换成 15 次 bitmap 交付，精确减少 498 拍。RTL 实测 S3/H4 也从 1561 拍降到 1063 拍，差值恰为 498 拍。

## 4. 面积代理与被否决实现

最初实现直接在 bitmap 接收周期计算 162-bit 组合 popcount。开放 Yosys 结构中，Serializer 从 492 增至 680 generic cells，增幅 38.21%。该实现会扩大接收关键路径，已被否决。

最终实现把 popcount 折叠进 21-byte 输出：

| 开放结构 | legacy | 最终 BPB | 变化 |
|---|---:|---:|---:|
| Serializer generic cells | 492 | 534 | +8.54% |
| 完整 C0 层次 generic cells | 3099 | 3181 | +2.65% |
| 完整 C0 `$mem_v2` | 13 | 13 | 不变 |

这些数字只用于 RTL 结构比较，不是目标库面积、频率或功耗。最终 PPA 必须由相同 SDC、相同 SRAM 宏假设和相同工艺库下的 DC/STA/SAIF 给出。

## 5. 验证闭环

- Serializer legacy/BPB 双模式：Icarus、Verilator+SVA、Yosys、Erie 全通过；
- 正向真实 FADC：bitmap 直通后输出 word 与金参考逐 word 相等；
- 负向合约：descriptor 声明 fanout=22、bitmap popcount=21，RTL 原子 abort，未产生 commit；
- 完整 C0：IPD/FADC/RAW 自动选择、commit、inspect、replay、release 全通过；
- 45-head：Icarus 与 Verilator+SVA 全通过，861 word 零失配；
- workspace：线性与 16-bit 分段 walker 双模式全流程通过。

## 6. 对架构贡献的意义

BPB 不应单独包装成整篇论文的主贡献。它应属于“表示保持的自适应三格式驻留数据流”中的跨层优化：前端 canonical bitmap、格式决策和后端 FADC 物理表示共同决定是否绕过事件展开。它和普通稀疏 skip 的区别是没有删除 silent token、没有改 Shiftmax 分母，也没有改变 gate/K 数值。

可辩护的表述是：提出一种 representation-preserving bypass，在 list 与 bitmap 的表示边界上消除 expand-then-rebuild，并把完整性校验折叠进已有序列化拍，降低高扇出长尾而不增加额外扫描阶段。

## 7. C1 更新与下一步

采用 BPB 后，双 workspace、共享单 Serializer、stage 边界清空的 C1 模型从 9992 拍完成 45 个 head，相对 C0 的 14078 拍减少 29.02%，加速 1.409x `[模型]`。该收益尚不能作为 RTL 结果。

下一步实现 C1 RTL，必须满足：

1. 只复制两个 canonical workspace，不复制 Serializer 和 slot 写口；
2. capture 可以占用空闲 workspace，emit 严格按 sequence tag 提交；
3. backpressure 下 bitmap、descriptor、destination 和 RAW payload 保持稳定；
4. 逐 word 结果与 C0 相同，并在同一 45-head 流上实测 makespan；
5. C1 若达不到模型收益，按 RTL 实测重写论文结论。

## 8. 证据边界

- 45-head latency、bit-exact 回放和负向 abort 是 `[rtl]`；
- C1 结果是 `[模型]`；
- generic cell 是开放综合结构代理，不是 PPA；
- 当前 trace 仅覆盖单个真实 window；
- 尚缺目标库 DC/STA/SAIF、SRAM macro、映射后等价、全 encoder FPS 与 valid825 部署审计。

