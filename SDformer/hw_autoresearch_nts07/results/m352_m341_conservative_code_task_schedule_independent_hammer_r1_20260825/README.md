# M352：M341 conservative code-task CPU DSE 独立打铁

结论：**90/100，P0=0、P1=1、P2=5**。

M341 的核心数值和当前 NO-GO 都成立。9 个合同输入及 110 个 selected M51 payload（195,840,000 bytes）重新哈希为 0 mismatch，M341 双层 seal 通过；独立做了 351 项 JSON/CSV/histogram/公式检查，0 mismatch。M341、父结果和 `docs/359` 均未修改。

## B0 与预算守恒

B0 的 11 个 module 均逐项等于 M328 group4：FC1 为 4,802,956,800 cycles，selected Conv 为 1,878,719,472 cycles，合计 6,681,676,272 cycles。drop、capture、bucket drain 和 metadata 都为零。

所有 98 个非零 module/aggregate row 均满足：

- `active = dropped + kept`；
- `stage2×8 - lane_idle = kept`；
- `drain = capture + fragmentation`；
- `stage2 = drain + reservoir_stall + tail`；
- `max_abs_raw_error <= max_conservative_bound <= B`，reported bound/capacity violation 均为 0。

B1024 保留 20,020,585,748 个 source-group task，K8 bank lower bound 为 3,942,929,105 cycles，但完整 frozen scheduler 是 10,685,291,808 cycles，只得 0.625315x。

## packing、drain-only 与周期

M341 的稳定 16-bucket 公式执行正确：capture `C=ceil(A/8)`；每个 code 独立打 8-ID word，`R=sum_c ceil(n_c/8)`；fragmentation `F=R-C`。全 population 的 `C=4,808,840,856`、`R=9,763,838,328`、`F=4,954,997,472`，平均 fragmentation 为 4.887742 cycle/task。

registered reservoir 对每个 word 至少花一拍；先按八个 bank 各 issue 一个，再在 post-issue occupancy 加 kept suffix 不超过 16 时接收。最终 `E=R+capacity_stall+tail`。B1024 中为 `9,763,838,328 + 58,771,482 + 813,336,106 = 10,635,945,916`。

完全落在 dropped prefix 内的 word 令 `word_kept=0`，仍然花一拍。因此 drain 在所有非零 B 下固定不变。这个行为与 frozen contract 一致，但 sealed artifact 没有单列 full-drop/partial/kept word 数，无法直接估算 pointer-skip 或 cutoff 后跨 code repacking。

two-context flow-shop 递推也正确。B1024 的 additive total 是：

`stage2 10,635,945,916 + stage2-wait 49,330,196 + metadata 15,696 = 10,685,291,808`。

capture-context wait、startup、final drain、stream-order penalty 和 overlap floor 是重叠诊断项，不能再次相加。

## NO-GO 是否充分

对 **M341 frozen stable-16-bucket、所有 packed word 必须 drain、two-context route**，NO-GO 非常充分：七个非零点全部低于 1.0，最佳只有 0.625315x；甚至只保留不可省的 bucket-word drain，其 9.7638B cycles 已高于 6.6817B baseline，理论上限也只有 0.684329x。11 个 module 的所有非零点都慢于各自 baseline，Amdahl 稀释不能把它变成系统加速。

但这不能证明所有 conservative cumulative-budget 结构都失败。跳过 fully-dropped bucket words、cutoff 后跨 code 重新打包、只发 cutoff 加 bank queue 等结构不在 M341 中。建议把决策名称限定为 `NO_GO_M341_FROZEN_STABLE16_BUCKET_ALL_WORD_DRAIN_ROUTE_B`。

## 指标命名问题

- `total_speedup` 是 10 个 FC1 加 1 个 selected Conv 相对 M328 K8 的 selected-scope schedule speedup，不是全网 speedup；M341 正确保持 `system_speedup=false`。
- B0 虽然严格 bypass selector，`registered_reservoir_stage2_cycles` 和 `two_context_pipeline_cycles` 却填入 baseline K8 cycles，属于字段复用，不是 selector overhead。
- `b0_exact` 在非零预算 row 也被置为 true，应移到 module/top-level 身份账本。
- 4.0x 只表示 498,816-byte persistent beta table；加入 M341 声明的 scratchpad、ping-pong 和 reservoir logical state 后为 518,512 bytes，即 one-bit reference 的 4.157942x。

最终口径：**GO M341 frozen CPU ledger、B0、整数 bound 和当前架构 NO-GO；NO-GO GPU/RTL/新思/系统倍速/论文贡献，也不得把当前结果扩写成整个 conservative-code 家族的死亡证明。**
