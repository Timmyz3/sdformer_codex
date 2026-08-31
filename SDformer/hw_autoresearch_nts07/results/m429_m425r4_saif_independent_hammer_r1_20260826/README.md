# M429：M425R4 real-H67 direct RTL SAIF 独立打铁评审

结论：**91/100，P0/P1/P2 = 0/1/3**。M425R4 的功能回放和 direct RTL SAIF 作为“探索性 subset activity”通过；允许进入 fail-closed mapped annotation 诊断，**不允许直接进入功耗/能量结论**。实际 mapped coverage 未达到并独立证明 `>=95%` 前，PTPX paper-power 仍是 NO-GO。

## 独立复算

- 双封：M425R4、M425 subset、M410R2 stimulus/VCS、M408 stimulus/VCS、M411 hammer、M416 DC 的 manifest 与 seal 全部独立通过。
- 选择：sample 0；operator 0/1/2/3；每个 operator 的 partition 为 `0,27,...,405`，共 64 phase。逐行与冻结 M410R2/M408 上游做 byte-exact 比较，无漂移。
- 账本：192,000 rows；pass1 61,285；early 11,923；zero 93,037；pop1 25,755；PWP rows 63,067；narrow/wide 87,906/416,630；contribution 921,166；reconstructed lane 48,435,456。全部独立复现。
- VCS：compile/sim rc 均为 0；功能 mismatch、accepted-transaction unknown、final protocol error 均为 0；SVA cover 为 PWP 504,536、release 64、global-fault 0，无 assertion failure。
- SAIF：`DURATION=6,288,008.5 ns`；14,537 entries；13,124 entries 有非零 TC。4 个 `protocol_error` 对象均为 `T0=6,288,009, T1=TX=TC=0`。

## 窗口与失败恢复

UCLI 在 21.5 ns 开启，首个 workload drive 在 22.0 ns；最终 PASS 之后 `$stop`，随后同一仿真时刻执行 `power -disable` 和 `power -report`。stop 发生在 6,288,030.0 ns，因此 `stop-21.5 = 6,288,008.5 ns`，与 SAIF duration 精确一致；2,096,003 个 3 ns measurement cycle 也与 testbench marker span 精确一致。没有漏计 workload transaction，也没有多计 post-workload cycle，只含 0.5 ns 的首拍前静默段。

R1、R2、R3 均在独立目录且有 do-not-cite marker；R4 新编译、新目录，R4 SAIF SHA 与 R3 不同。R4 的两个 symlink 都是 VCS 目录内相对链接，不指向旧 run，未发现恢复污染。

## 必须修的映射门槛

M416 的 top 名称仍是 `m405_q32_elastic_selected_slice`，但内部已完全 flatten；RTL SAIF 则保留 `u_matcher/u_balanced` 与 `u_adapter`。现有 basename-only mapping helper 从 4,100 个 mapped register instance 中只能构造 1 个 explicit pair。这不是 PrimeTime coverage 数字，但足以证明不能只凭 strip path 或当前 helper 宣称映射完成。

下一步只允许做 mapped annotation 诊断：使用 hierarchy-aware DC/SVF name map 或 gate-level activity，独立解析真实 annotation coverage；只有 coverage `>=95%` 且相关 unannotated object 通过审计，才可运行并引用 PTPX。M416 删除的 320 bit 是 10×32 个 unread debug counter，它们本来就不存在于 mapped design，不能误算成 mapped-netlist annotation failure。

## 其余 P2

1. `fifo_tile_q[1]`、`fifo_narrow_q[1]` 各有 TX=13,486 ns。它们在 slot 1 为 live head 时可条件性驱动输出，不能写成“结构上不可观察”。本 workload 为 always-ready，任何 live output 都会当拍 accept，accepted payload/metadata unknown audit 为 0；准确表述应为“两个内部 slot-1 SAIF TX entry，但本 workload 的 accepted transaction unknown 为 0”。不得把 TX 擅自填成已知翻转。
2. testbench 的 `time_ns` 标签实际配合 `%0t` 输出的是 ps；这是日志标签问题，不是窗口问题。封存 run 不改，后续 TB 改名为 `time_ps` 或显式打印 ns。
3. 64 phase 是单 sample 的系统抽样，不是全 population、全网络或统计代表性保证。任何 activity/power 列都必须保留 subset scope。

边界：本评审没有运行 PTPX，没有给 power/energy 数字，没有把 subset SAIF 写成 full-population/full-network/paper power，也没有修改 `docs/359`。
