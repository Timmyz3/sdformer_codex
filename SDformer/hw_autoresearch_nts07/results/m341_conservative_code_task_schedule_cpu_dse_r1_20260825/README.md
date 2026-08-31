# M341：4-bit conservative-code task-schedule CPU DSE

结论：**NO-GO**。完整 frozen group4 population 上，最佳非零点是 `B=1024`，总周期 `10,685,291,808`，相对相同 M328 K8 baseline 的 `6,681,676,272` 只有 `0.625315x`。没有点达到 `1.15x`，所以不准入 GPU modified-forward，更不准入 RTL、VCS、新思或论文贡献。

本轮新增独立合同和 CPU 分析器，未修改 M328/M336、现有合同、RTL 或 `docs/359`。第一次和 exact-SHA 重跑的全部非零 aggregate row 完全一致；第二次仅修正 B0 汇总 histogram，使被 bypass 的 capture/bucket 分布明确为零。

## 冻结执行模型

范围与 M328 group4 完全相同：10 个 FC1、1 个 selected patch Conv、110 个 record、`1,013,760,000` 个 token/destination-group task。

每个非零预算 task 都逐项执行：

1. 输入升序 active ID，每拍 8 个，从当前 group 的 432 B 8R scratchpad 查 4-bit code；
2. 按固定上界 `U={0,9,17,26,34,43,51,60,68,77,85,94,102,111,119,127}` 稳定分成 16 桶；
3. capture=`ceil(A/8)`，drain=`sum ceil(n_c/8)`；
4. 累计 U 丢弃最大 stable prefix；保留第一项超预算 source 和后续全部 source；
5. 一个 16-ID registered reservoir 每周期先对 8 个 modulo bank 各 issue 至多一个 ID，再在容量允许时接收一个 bucket word；
6. 两个 448-active context 采用 blocking flow-shop 递推。record 间不 flush；每个 module/group 启动空 context，完成最后 task 后切 group；
7. 498,816 B metadata 主表按 group-major 256 bit/cycle 预取，完整 sweep 精确为 `15,696 cycles`。

`B=0` 完全 bypass metadata、capture、bucket 和 reservoir，直接执行 frozen K8 baseline。

## B0 对账

- M328 baseline：`6,681,676,272 cycles`；
- M341 B0：`6,681,676,272 cycles`；
- active source/group task：`35,106,857,184`；
- task：`1,013,760,000`；
- drop、bound、raw error、capture、drain、metadata：全零；
- 11/11 module baseline 均逐层等于 M328。

因此零预算无损和 frozen work/cycles 复现通过。

## 全 budget 结果

| B | Drop | kept K8 lower bound | Bucket drain | Registered stage2 | Total | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000% | 6,681,676,272 | 0 | 6,681,676,272 | 6,681,676,272 | 1.0000x |
| 16 | 0.0952% | 6,676,853,157 | 9,763,838,328 | 11,039,417,618 | 11,082,211,330 | 0.6029x |
| 32 | 1.6299% | 6,592,349,402 | 9,763,838,328 | 11,036,826,397 | 11,079,733,546 | 0.6031x |
| 64 | 4.2103% | 6,432,868,103 | 9,763,838,328 | 11,018,326,964 | 11,062,314,521 | 0.6040x |
| 128 | 8.4049% | 6,171,282,634 | 9,763,838,328 | 10,982,419,876 | 11,028,183,801 | 0.6059x |
| 256 | 15.3278% | 5,742,422,133 | 9,763,838,328 | 10,927,887,758 | 10,974,894,171 | 0.6088x |
| 512 | 26.4278% | 5,038,094,024 | 9,763,838,328 | 10,827,612,751 | 10,875,724,565 | 0.6144x |
| 1024 | 42.9724% | 3,942,929,105 | 9,763,838,328 | 10,635,945,916 | 10,685,291,808 | **0.6253x** |

所有非零点都有 dynamic witness，integer bound violation=`0`、capacity violation=`0`，但 accuracy 统一为 false。

## 为什么彻底失败

M336 预设计指出，要过 1.15x，平均 bucket fragmentation 必须低于约 `0.9861 cycle/task`。实际 task 分布是：

| 指标 | Mean | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Active source | 34.6303 | 20 | 98 | 152 | 448 |
| Capture | 4.7436 | 3 | 13 | 19 | 56 |
| Bucket drain | 9.6313 | 9 | 19 | 26 | 64 |
| Fragmentation | **4.8877** | 6 | 8 | 9 | 13 |

实际 fragmentation 共 `4,954,997,472 cycles`，是允许量的约 `4.96x`。根因是 16 个 beta class 各自对 8-ID word 向上取整；active 较稀时，桶内 partial-word padding 比 capture 本身还贵。

这不是 reservoir 小优化能修复的：即使把 bank reservoir、startup/drain 和 metadata 全部免费，单独 bucket drain 已有 `9.7638B cycles`，相对 baseline 也只有 `0.6843x`。

在最佳 B1024：

- 4-bit coding 将 drop 从 M328 exact-beta 的 45.4944% 降到 42.9724%；
- kept K8 lower bound 从 `3.7871B` 增至 `3.9429B`，只恶化约 4.11%；
- bucket drain 已经是 `9.7638B`；
- registered reservoir 再加入 `58,771,482` capacity-stall 和 `813,336,106` tail cycles；
- 相对 `max(drain, kept issue)` 的 stream-order penalty 为 `871,807,724 cycles`；
- 两 context 仍有 `49,330,196` stage2-wait/startup bubble；
- metadata prefetch 只有 `15,696 cycles`，不是瓶颈。

因此主要失败不是 conservative code 精度，也不是 metadata bandwidth，而是 stable 16-bucket word fragmentation。

## Layer/record 分布

B1024 下：

- FC1 aggregate：`0.588355x`；
- selected Conv：`0.744955x`；
- 11 个 module 全部低于 1.0，范围约 `0.4371x–0.7522x`；
- 110 个 record 的 attributed speedup 范围约 `0.4345x–0.7548x`。

完整 880 条 record-policy、88 条 module-policy、24 条 aggregate-policy 和 task histogram 均已输出。record cycle 是连续 group-major timeline 的增量归属；同一 module/group 的 record 边界没有额外 flush。

## 数值与容量

- conservative code 始终满足 `U>=exact beta`；
- 所有非零点满足 `|raw INT8 error|<=sum exact beta<=sum U<=B`；
- B1024 最大 raw signed INT8 error=`982`，最大 conservative bound=`1024`；
- maximum active=`448`；
- maximum bucket occupancy=`100`；
- maximum reservoir occupancy=`16`；
- 448-active/bucket/reservoir capacity violation=`0`；
- 持久 metadata=`498,816 B`，相对 `124,704 B` one-bit mask 正好 `4.000x`。

这些仍只是 CPU frozen-trace schedule，不是 RTL cycle、PPA、energy 或 system speedup。

## 最终 admission

决定固定为：`NO_GO_CONSERVATIVE_CODE_AFTER_CPU_TASK_SCHEDULE`。

- GPU modified-forward：不准入；
- valid825：不准入；
- RTL/VCS：不准入；
- DC/Formality/PT/PTPX：不准入；
- DATE contribution/headline：不准入。

若以后重启 cumulative-budget，只能另开机制，首先消除 per-beta-class partial-word padding；不能把 M336 的乐观 `1.3895x` 或 M341 的 kept-K8 `1.6946x` 当成实际性能。
