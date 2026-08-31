# M386 G12/ATLIF S10 gate recompute independent hammer

M386 从 M366 的 810 条原始 sample×site row 独立重算全部统计，结论是：**停止 G12 dense remaining-budget RTL**。严格整数界本身正确，但代表性 S10 的 term skip、可执行 cycle 和 fixed-context uplift 均远低于晋级门槛。

## 独立一致性

- 10 个冻结样本，每样本 81 个 live site，共 810 row；45 个 T10、36 个 T2。
- 逐行检查 11,340 个 work/resolution/term/cycle/tile/exactness 恒等式。
- 重新聚合 81 个 site，再组合 T10/T2；site aggregate、combined counter、fixed projection 和 promotion gate 均为 0 mismatch。
- checkpoint overlay 210/210，missing=0、unexpected=0；论文身份为 H67 ep35/no_running，未混 PAFT ep4。

## T10 主结果

| 指标 | 观测值 | 门槛 | 判定 |
|---|---:|---:|---|
| suffix term skip | 6.5771% | ≥35% | FAIL |
| 32-lane executable issue-cycle reduction | 0.06761% | ≥25% | FAIL |
| conditional fixed-context | 1.0000797× | ≥1.03× | FAIL |
| integer early mismatch / bound violation | 0 / 0 | 0 / 0 | PASS |
| signed-Q8 range violation | 35 | 0 | FAIL |
| metadata/config/compare net energy | 未证明 | 必须为正 | FAIL-CLOSED |

per-lane term skip 基本不能在 32-lane issue group 内对齐。最好的单个 T10 site 也只有 18.99% term skip；最好的单 site issue-cycle reduction 只有 1.117%。45 个 T10 site 中，没有一个达到 35% term gate，也没有一个达到 25% cycle gate，因此结果不是被某个大层的权重稀释。

M360 curated sample0 的 8.8521% term skip 到代表性 S10 下降为 6.5771%，相对下降 25.70%。S10 没有揭示隐藏的强机会。

## 数值桥反馈

sample0 冻结 scale 在 S10 上出现 35 个 T10、19 个 T2 signed-Q8 越界。越界率极低，但 exact contract 要求 0，因此未来任何 integer ATLIF deployment 都需要代表性 per-site calibration 或 QAT。

T10 integer reference 与 float event 有 87,775,830 个 flip，占 11.709B event 的 0.7496%。这不是 G12 early-decision 错误，但说明 integer bridge 必须经过 paired valid825 accuracy 才可晋级。重新校准 scale/QAT 也无法救回 G12 性能，因为 cycle reduction 距 25% 门槛仍差 24.93 个百分点。

T2 attention diagnostic 为 18.20% term skip、7.56% issue-cycle reduction，同样不过门；它不否定 RQTB/attention 的其他独立优化，只否定把 dense remaining-budget 当作强 cycle 轴。

## 收口

- G12 只保留严格 proof 和负面 design-space evidence；不写 RTL，不跑 VCS/Synopsys。
- 不与 rank-3 ATLIF 或任何系统倍速做加法。
- 不作为 DATE 贡献头条；正文需要篇幅时可作为设计筛选/消融，否则省略。
- M386 不承认 accuracy、energy、PPA、system speedup 或 headline。

候选硬件价值评分：**18/100（KILL）**；证据与审计完整度评分：**96/100（PASS）**。低分来自机会本身，不是验证缺失。
