# M360 G12 ATLIF remaining-budget early-stop predesign hammer

结论先行：数学机制成立，但当前性能证据偏弱。`ATLIFTernaryPSN` 是 `T×T` PSN temporal affine `addmm` 加阈值，不是逐时递推 LIF。对量化整数域可用未处理项的严格区间证明 binary/ternary 输出类别已不可改变；但冻结 H67 ep35 的实际部署是 93 个 `official_atlif` 二值运行点、输入仍是宽幅有符号值，不能因为输出二值就宣称乘法消失。

## 独立定向重算

在现有 sample-0 DP-TME 定向向量上，以合法 `INT8 [-128,127]` 输入范围和每 site 静态 `sum_t |W[t,s]|` 降序发射：

- 45 个 T10 site、14,400 个输出 lane，整数 hidden/event 均 0 mismatch；重排后的 158,400 个前缀无 Acc24 overflow。
- 12,747 / 144,000 个 source terms 可证明跳过，即 8.8521%；这是 term-work/energy 机会，不是周期收益。
- 45 个 32-lane×10-output tile 全部仍要走到 k=10，保守 tile-wide cycle skip 为 0；即使按 output-time row group 计也只有 17 / 4,500 = 0.3778%。
- 该向量集故意混合 ordinary、near-threshold、max-amplitude，且只有 sample 0，所以不能替代 S10 机会率；它只是一条偏弱的 directed predesign 反证。

## 严格判定式

固定 issue order `pi`，令 `P[t,k]` 为已处理前缀，未来输入满足 `x[s] in [ell[s],u[s]]`。逐项取乘积最小/最大并求和得到 `Rmin[t,k]`、`Rmax[t,k]`，则 `L=P+Rmin <= h <= P+Rmax=U`。

- official binary：`L >= Theta` 可提前发正事件；`U < Theta` 可提前发 0。
- asymmetric ternary：`L >= Theta_pos` 发正；`U <= Theta_neg` 发负；`L > Theta_neg && U < Theta_pos` 发 0。
- symmetric binary abs：`L >= Theta || U <= -Theta` 发 active；`L > -Theta && U < Theta` 发 0。

等号方向不能放宽，因为模型在正、负阈值相等时都会触发。

二值源 `{0,a}` 或三值源 `{-a,0,+a}` 且 `a` 静态时，可把乘法变成预缩放权重的 add/sub/skip；一般 signed INT8 输入仍需完成已发射前缀乘法，只省尚未发射的后缀 MAC。

## 硬件代价与共存边界

T10 在每次至少发一个 source 后检查 9 个 stop point，完整双边 suffix table 为 180 个 bound word：24 bit 时 540 B，约为 M265 dense T10 133 B 配置的 4.06 倍。若强制 `[-A,+A]` 对称输入合同，可只存一个 remaining magnitude，降到 270 B；更合理的微架构是每个输出时间保留 remaining-bound 寄存器并逐列扣减，但仍要付十路更新和比较逻辑。

M360 的 dense bound 不能直接叠加到 M265 rank-3 两级候选。rank-3 先算 `right*x`，再算 `left*latent`；只在第二级早停不会退回已经支付的第一级 MAC。最终必须在 dense G12、两级保守 latent bound、或 rank-3 stage-2-only 三者中做一个可执行选择。

固定计算账本中 dense T10 ATLIF 仅占约 11.79%，即使全部消失，上限也只有 1.1336×。替入既有 rank-3 候选后，所有剩余 ATLIF 周期再消失的上限约 1.0668×。因此 G12 最多是次级 exact energy/cycle 优化，不是多倍系统头条。

## 分级决策

- CPU mechanism：`GO_COMPLETE`。当前 directed 重算已证明公式和边界，但 8.85% term skip/0 tile-cycle skip 不支持直接写 RTL。
- A800 S10 streaming capture：`GO`，这是 48 小时内信息价值最高的动作。只需在 GPU 上归约 `resolved_at_k`、tile/row/lane mask、范围/溢出和 bounded witness，不必转储全尺寸 tensor。
- RTL：`NO_GO_NOW`。只有 S10 达到零失配/零范围违规、至少 35% suffix term skip、至少 25% 可执行 issue-cycle 降低、投影固定计算收益至少 1.03×，并且 bound/config 开销低于节省值时再晋级。

Predesign readiness：`65/100`，`P0=0 / P1=4 / P2=5`。完整结构化证据、A800 最小 payload 和门槛见 `m360_g12_atlif_remaining_budget_early_stop_predesign_hammer_r1.json`。所有速度数字均 `headline=false`、`system_speedup=false`。
