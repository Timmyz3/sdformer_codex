# M507 APEC-G2 same-resource cycle fast-kill 独立 preflight 评审 r1

日期：2026-08-27  
范围：只读审查 M507 contract/analyzer、M501 封存身份和本地 ExSpike 官方 artifact；未 import/运行生产 analyzer，未启动 VCS/DC/PT，未修改生产文件或 `docs/359`。

## 裁决

**`NO_GO_REVISE_BEFORE_ONE_SHOT_EXECUTION`，58/100。**

不得用当前 r1 执行“唯一一次”的 production fast-kill。身份、输入和 prior-art 边界是干净的，但周期模型没有实现 M501 独立评审授权的 same-resource 计费合同：它对 candidate 加了 overlap scratch 的两次 destination read/commit，baseline 却没有对称的两个正常 destination-vector output/commit；同时用布尔字段代替真实 SRAM/端口守恒，queue/bank conflict 也没有进入模型。这个 r1 可以作为一个极保守的特定 schedule 上界，不能作为不可逆的 `KILL_M501_M507_HARDWARE_LINE` 依据。

## 1. 通过项：身份与 claim boundary

| 对象 | SHA256 / 结果 |
|---|---|
| M507 contract | `a2646134822d2074bc810004576dc0ffc6be04a5f4417b08c477d9c2a8a90410` |
| M507 analyzer | `213976d42c83b7f3512b62e35c2c9e6a7763e1953d67e618517dc5897291db92` |
| `docs/359` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |
| M501 result seal | 完整验证通过 |
| M501 independent-review seal | 完整验证通过 |

contract 正确锁定 validation horizontal-G2，train-only 18 sequence 只作 calibration，且明确 `standalone_rtl_novelty=false` / `system_speedup=false`。当地官方 ExSpike 仓库是 commit `51accc76936588705255487d101fcc80092b98ce`。`weight_acc.v:132-157` 明确有 `overlap_cal_res` 并用它 seed 后续 position accumulator；`weight_top.v:439-467` 将各 WPE 结果送入 elastic FIFO。因此“APEC 是 ExSpike 直接 prior art，M507 只能做 workload/cycle audit”的边界正确。

## 2. P0 blocker：candidate/baseline destination 路径不对称

生产脚本 `record_cycles()` 中：

- baseline 为 `bitmap_read_cycles + bexec`（第 224 行），没有任何 destination-vector output/commit 或 output-psum bank 更新项；
- candidate 为 `bitmap + compare + cexec + scratch_cycles`（第 252-253 行），其中 `scratch_cycles=3*scratch_pass`，两次 read 被同时命名为 destination commit（第 245-251 行）。

无论 overlap 是在 local WPE register 中 seed，还是通过 1R1W scratch 取回，两臂最终都要产生左/右两个 shifted destination vector 并进入相同 output sink。当前模型没有定义 baseline 的这两次正常传输是否被 `bexec` 包含，也没有定义 candidate 的 scratch read 是“seed residual accumulator”还是“最终 commit”。因此无法排除 candidate-only 重复计费或 baseline 漏计。

独立边界算例显示这个不对称会主导裁决：

| 左/右事件 | taps | baseline | candidate | r1 ratio |
|---|---:|---:|---:|---:|
| 各 1 个，完全 overlap，interior | 9 | 146 | 462 | 0.3160x |
| 各 1 个，无 overlap，interior | 9 | 146 | 147 | 0.9932x |
| 各 1 个，完全 overlap，top edge | 6 | 98 | 309 | 0.3172x |

interior overlap 组的 `462` 周期中有 `387=3×129` 周期来自 candidate-only scratch path。在用这个数做永久 KILL 前，必须先画出两臂完整的 `compute -> accumulator -> output sink` 周期时序，共有路径对称计费，只有 overlap-specific 增量计入 candidate。

## 3. P0 blocker：same-resource gate 是自由布尔量

`no_free_sram_or_ports` 只检查三个 contract 声明：baseline/candidate scratch 都等于 16,416 B，以及 `same_top_ports_frequency_lanes_and_sram is True`（第 481-484 行）。它没有从交易 ledger 推导端口守恒，也没有将 scratch 从某个已准入的 240 KiB 布局中挖出后重算 reload/tiling。

当前尚未计入/验证：

1. pair bitmap buffer + exact comparator 的容量和端口；
2. overlap 生成期间 16.03 KiB 向量由哪个已收费 accumulator 承载；
3. output psum bank 的 read/modify/write 映射和 shifted-destination conflict；
4. 8-bank weight 路径的实际 bank mapping（目前只是 aggregate 128 B/cycle）；
5. 240 KiB 剩余容量对已选 baseline 的 tile/reload 影响。

因而 `same_top_cycle_model=true` 只是声明，不是可审计结果。

## 4. P0/P1 周期与 ledger 缺口

### 4.1 scratch 同步返回少收尾拍

contract 声明 1-cycle synchronous response，但脚本对每个 read pass 只计 `scratch_pass`，没有最后一个 response/commit tail。在“write/read 串行，两次 read 也串行”的 contract 下，interior 至少应从 `3×129=387` 改为 `387+2=389`，top edge 从 `258` 改为 `260`。这是 candidate 少收，和第 2 节的对称性问题独立。

### 4.2 要求的 stall 类别没有实现

`scratch_port_conflict_cycles` 被初始化为 0，之后从未增加；`queue_occupancy` 和 `weight_bank_conflict` 在源码中不存在。“用全局串行避免 conflict”可以是一个合法 schedule，但应把因串行而产生的 stall 显式记入，不能输出恒为 0 的 conflict 计数来满足“显式报告”。

### 4.3 M501 ledger 只复现 validation aggregate

脚本只与 M501 的 validation horizontal-G2 三个 aggregate 整数对账（第 437-451 行），train 18 sequence 只从 raw manifest 重算，没有与 M501 train overall/per-sequence ledger 对账。另外，M507 用 `values != 0` 得到 overlap，但没有像 M501 一样验证每个非零 payload bit pattern 确实等于锁定 codeword。虽然当前 manifest SHA 是冻结的，一次性 fail-closed 脚本仍应将这些条件直接编码。

### 4.4 geometry/completeness 检查不完整

border tap 计数对冻结 `3x3/stride1/pad1/dilation1`, `15x20` 几何是合理的，width 20 也没有 G2 tail；但 M507 没有自行验证 manifest 中的 module geometry/output shape，也没有验证 sample×operator 笛卡尔积无重复/无缺失。精确 SHA 降低了当前风险，但不足以支撑 schema 变更后的 fail-closed 声明。

## 5. r2 最小修改门

不要改 axis/G2/阈值，也不要运行 r1 后再补测。在消耗唯一一次运行配额前，r2 必须：

1. 冻结一个具名 baseline storage/sink 组织，列出两臂相同的 bitmap/weight/accumulator/output-psum/SRAM 容量和端口 ledger；`no_free_sram_or_ports` 由 ledger 计算，不得读自由布尔字段。
2. 对 baseline/candidate 都写出两个 destination vector 的生成、传输和 sink commit；共有项对称收费，只把 compare/overlap save/seed 的增量收给 candidate。
3. 显式模拟 1-cycle scratch response tail，以及串行导致的 scratch stall；删除恒 0 的伪 conflict 指标或实现它。
4. 将 8-bank weight mapping、output bank conflict/backpressure 纳入 ledger；如果明确选择全局串行，记录相应 stall，不得当作零冲突。
5. 对 validation + train overall/per-sequence 全部复现 M501，并增加 payload exact-codeword、geometry、sample×operator completeness 检查。
6. 重新锁 analyzer/contract SHA，再做一次独立 preflight；只有 preflight GO 后才执行那唯一一次 production fast-kill。

## 6. 结论与可用口径

本评审不否定 M501 的 `1.379628678x` exact event-work opportunity，也不改变 APEC 是 ExSpike prior art 的裁决。它只阻止用一个不对称、端口守恒未实现的 r1 cycle model 作永久 GO/KILL 裁决。

可复现检查器：`audit_m507_preflight_independent.py`；它不 import 生产 analyzer，只做锁 SHA、静态账本检查和三个手推边界组。
