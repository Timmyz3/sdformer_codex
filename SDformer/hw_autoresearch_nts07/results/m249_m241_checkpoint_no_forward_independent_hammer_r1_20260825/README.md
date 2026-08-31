# M249：M241 checkpoint/no-forward accumulator 独立打铁评审

## 结论

评分 **86/100**，`P0=0 / P1=5 / P2=5`。

判定为：**固定 1-cycle 行为宏、8-lane、单个真实 context 的 standalone island 有条件 GO；宏集成、物理性能和任何加速倍率 NOGO。**

M241 的实现不是空壳。两个原封存 seal 均精确通过，独立从 M40/M158 原始事件重建了 126 条 ordered descriptor，从 M41 原始 checkpoint 重新计算了全部权重地址与 signed integer oracle：504 次 destination update、4,032 个 lane check、8 个真实负事件均为 0 mismatch。56 次 weight macro read + 448 次 cache hit 的 **9.0× read-work reduction** 也成立。高 half 的最大 local row 303 正确映射到 dense address 687，而不是旧拼接形式的 815。

但这些证据还不能晋级为速度结果。M241 只覆盖 sample 5 / operator 0 / partition 251 / window 7，真实 126 条全是 full4，只实现 8/96 lanes；M152 完整 producer、window identity、四层 finite trace、final scan/requantization 都未连接。

## 关键打铁发现

1. **宏延迟被硬编码为 1 cycle。** Weight/accumulator 接口没有 response-valid、tag 或 backpressure。独立 VCS 在 1-cycle weight SRAM 下得到正确 `+10`，把响应改成 2 cycles 后得到 stale `0` 而不是 `+11`。因此 no-forward II=1 仍是固定行为宏条件，不是已绑定真实 SRAM 的结论。

2. **overflow 写入 fail-closed，但 completion 不是。** 独立 VCS 观测到同一 overflow transaction 中 `overflow_error=1`、所有 `acc_wr_en=0`，同时 `commit_accept=1`。下游可能把没有落入 accumulator 的 token 当成已完成；原测试又刻意不放 younger token，因此 accepted-younger quarantine 也没有外部可见的 abort 合同。

3. **checkpoint tag 没有绑定到宏数据。** `{partition, epoch, source, half}` 只是内部 cache key；物理 weight address 只有 `{half, source}`，宏响应不带 operator/partition/epoch/window。现有所谓 cache-epoch attack 实际只是错 descriptor epoch，不是 stale loader/macro response。

4. **7,296-bit forwarding payload 是“按名字删除”，不是已测净收益。** M241 同时有 production-width `cache_data`、`s0_hit_data`、`s1_delta`、`s2_delta`。必须用同一 macro cuts 做 M155 forwarding 与 M241 no-forward matched DC/STA，才能报告净 FF/面积/时序收益。

5. **M238 的 1.687017659× 仍被严格封住。** 独立重算 `126,581,635 / 75,032,786` 正确，但 M241 没有在完整 trace 上测 parent/candidate cycles，也没有 SRAM、matched DC/STA、SAIF/PTPX，因此不能写成 achieved speedup。

## 允许口径

- 在严格 1-cycle 行为 SRAM 合同下，M241 是一个通过真实 ordered subset exact VCS 的 8-lane、四 bank、INT8-to-Acc19 standalone island。
- 9.0× 只能称 weight-read work reduction，不能称 cycle 或 energy speedup。
- 可以称 accumulator same-address forwarding payload 按构造不存在，并由同地址 interlock 替代；不能称净存储/面积已降低。
- 1.687017659× 只能称 M238 same-PWP1024 cycle-model target，M241 未 admission。

## 下一关

优先修 overflow commit/abort；再把 loader payload identity、window tag 和宏 response tag/elasticity 接进来。之后跑 full96、四个 bottleneck operator 的完整 M152 finite trace，最后用相同 SRAM cuts 做 M155/M241 matched DC/STA 和 cycle/SAIF/PTPX。完成前 `physical_speedup=false`、`system_speedup=false`、`paper_ppa_ready=false`、`headline=false`。

本评审没有修改 M241 RTL/SVA/TB/contract，也没有启动 DC。`docs/359` SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
