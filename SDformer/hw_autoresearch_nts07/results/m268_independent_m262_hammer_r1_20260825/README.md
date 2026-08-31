# M268：M262 FC1 独立打铁评审

结论：`87/100`，`P0=0 / P1=2 / P2=5`。允许 M262 作为 **8-lane 小宽 FC1 descriptor 生命周期模块证据**继续使用；不允许把 trace 比率晋级为 full96、完整 FC1/FFN、系统或 headline 指标。

## 独立复验结果

- 对冻结 RTL/SVA/TB/filelist/contract 做了 exact-SHA 预检；用 Synopsys VCS V-2023.12-SP1、新 seed `2680825` 从源码重新编译并仿真。
- VCS：`5 tiles / 1 empty / 18 descriptors / 14 clean cycle checks / 32 commits / 4 attacks`；六类 stall 均非零，15 个 coverpoint 均命中，assertion failure 为 0。
- wrong-SHA：独立污染 contract 后 analyzer 非零退出，且未创建输出目录；clean replay 的 JSON/CSV 与作者封存 payload 逐字节同 SHA。
- 作者、trace、作者 VCS、独立 VCS、独立 replay 五组 manifest 全部校验通过。

## 周期与协议复核

单 descriptor 的状态代价严格为：factor request/response `1+2`，weight request/response `1+2`，每个有效 context 的 Acc read/response/write `1+1+1`，所以为 `6+3×popcount(mask)`。独立枚举 popcount 1..8 得到 `9,12,15,18,21,24,27,30`。

非空 tile 固定开销为 `header 1 + init 8 + commit(8×3) + done 1 = 34`；没有漏计 init、response、commit 或 done。空 tile 的合法 header 只有在 `done_ready` 时接受，并与 done 同拍完成，不产生 factor/weight/Acc 访问。

stale response 出现的第一拍，`protocol_error/fault_event` 同拍拉高并隔离存储副作用；`abort_valid` 是状态寄存到 `ST_ABORT` 后的下一拍，而不是第一拍组合输出。abort payload 在 backpressure 下保持。signed19 overflow 在 `acc_write_valid` 之前组合隔离，随后 sticky fault；commit 不可见。

## 冻结 M230 独立复算

独立从 100 条 raw record 重建 10 samples × 10 binary FC1 modules，以及每个冻结 96-lane block 的 12 个 8-lane 串行 slice。100 条 JSON、CSV 和 aggregate 均为 0 mismatch。

| 模式 | 同端口 8-lane serialized lifecycle cycles |
|---|---:|
| dense | 798,024,960,000 |
| bit-sparse | 110,840,148,144 |
| context-factorized | 66,282,442,128 |

因此 factorized/bit 的 lifecycle cycle 改善为 `1.6722399565×`；weight request reduction 为 `2.5800602657×`。bit/factor 使用相同 factor、weight、Acc read/write 和 commit 端口；Acc update 与 commit 数相同。比率仍是假定固定响应延迟、无 backpressure、无地址/Bank 冲突的 aggregate mapping。

## 打铁问题

P1：

1. IDLE 中非法 mode/count header 会被永久 backpressure，不会进入 sticky abort。冻结合法输入不受影响，但不能宣称 malformed header 全域 fail-closed。
2. `1.672240× / 2.580060×` 尚未包含地址级 SRAM 容量、bank conflict、queueing 或物理时序；在这些补齐前不能作为完整 FC1 或系统 headline。

P2：

1. SVA 只断言 retire cycle `>=9`；精确公式由 directed TB 检查，clean suite 未逐一覆盖 popcount 1..8。
2. empty 用例只在 reset 后执行；随后出现 empty header 时，transaction-local debug request/commit counters 不会清零。
3. 没有 directed done-backpressure、malformed header/factor-shape sweep。
4. producer analyzer 不拒绝重复 JSON key，且 analyzer 单独调用时只检查 VCS receipt 内容；exact runner 才额外固定 receipt SHA。
5. `factor_base + descriptor_index` 没有显式地址溢出检查。

## 评分

- 身份与可复现性：19/20
- RTL 生命周期与数值语义：24/27
- VCS/SVA/负向测试：20/25
- trace 数学与公平性：19/23
- claim discipline：5/5

总分 `87/100`。决策是 `GO_FOR_SMALL_WIDTH_MODULE_EVIDENCE_ONLY`。

本评审未启动 DC，未修改 `docs/359`，未扩大到 full96、full-trace RTL、完整 FC1/FFN、系统或 headline。

