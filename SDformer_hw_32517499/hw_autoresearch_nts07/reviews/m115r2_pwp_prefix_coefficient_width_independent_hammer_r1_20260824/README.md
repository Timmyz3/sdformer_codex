# M115-r2 PWP prefix-coefficient width：独立打铁复审

日期：2026-08-24

结论：**94/100，P0=0 / P1=3 / P2=5。** M115-r2 已正确撤销 r1 的 signed20/signed22 minimum-width 错误，并把 signed19/checkpoint 与 signed21/dense 限定为 exact-once 条件下的 mathematical candidates；没有冒充 RTL、VCS、PPA 或 speedup。

## Prefix coefficient 独立复算

四个 eligible PWP center/target 组合及所有操作排列：

| center | target | 操作 | 全部 prefix | max abs |
|---:|---:|---|---|---:|
| 0 | 0 | 无 | `0` | 0 |
| 0 | 1 | `+positive` | `0→1` | 1 |
| 1 | 0 | `+anchor,-negative` | `0→1→0`; `0→-1→0` | 1 |
| 1 | 1 | `+anchor` | `0→1` | 1 |

另独立枚举四个 escape center/target 组合：escape 不应用 anchor，raw target 为 0 次或一次 `+1`，最大前缀同样为 1。任意跨 term 的全局交错下，每项系数仍在 `{-1,0,+1}`，由三角不等式得到 `abs(accumulator) <= sum(abs(weight))`。

exact-once 是必要条件。对 anchor、正/负 correction 或 escape raw event 任意复制一次，攻击序列即可把某项系数推到绝对值 2；若 retry 无界，则没有有限位宽证明。M115-r2 正确把 integrated accepted-transaction exact-once miter 标为 false。

## signed19 / signed21 与原始 payload

四个冻结权重 payload 的全部 3,072 个输出通道重新从 signed INT8 字节按 `I_KY_KX_O` 布局求和：

| op | min sumabs | max sumabs | max channel |
|---:|---:|---:|---:|
| 0 | 113,538 | **218,338** | 360 |
| 1 | 79,336 | 204,866 | 185 |
| 2 | 87,029 | 207,239 | 513 |
| 3 | 82,093 | 190,753 | 126 |

- signed18 正上限 131,071，不覆盖 218,338；signed19 正上限 262,143，因此 checkpoint bound 需要 19 bits。
- 冻结对称量化范围为 `[-127,127]`，dense bound 为 `6912×127=877,824`；signed20 正上限 524,287，signed21 正上限 1,048,575，因此 dense bound 需要 21 bits。
- 四个 accumulator-init payload 均为 3,072 个零字节，四层均 bias-free；M115-r2 当前只通过 M41 SHA 间接固定这项假设。

## W384 storage 与 384 B valid 修正

独立公式：

`ceil((2×128×384×2 + 314 + 384×8 valid + 384×8×96×bits)/8)`

`384×8=3,072 valid bits=384 B`，五种位宽的 corrected bytes 都比无 valid 版本恰好多 384 B。

| signed bits | 无 valid | corrected | vs signed24 saving |
|---:|---:|---:|---:|
| 19 | 725,032 B | **725,416 B** | **184,320 B / 20.26%** |
| 20 | 761,896 B | 762,280 B | 147,456 B / 16.21% |
| 21 | 798,760 B | **799,144 B** | **110,592 B / 12.16%** |
| 22 | 835,624 B | 836,008 B | 73,728 B / 8.10% |
| 24 | 909,352 B | **909,736 B** | 0 |

这些仍是 descriptor+minimum metadata+valid+accumulator 的 logical lower bound，不是 SRAM macro capacity/area/energy。

## Revocation、manifest 与边界

- r1 revocation 的 analyzer/result/contract SHA、触发 review SHA 与触发 manifest SHA 全部匹配当前字节；撤销项完整覆盖 signed20/22 minimum 以及 signed19 rejection。
- M115-r2 producer manifest 13/13 通过，覆盖所有六个 direct analyzer inputs 与四个 weight payload，修复了 r1 的 manifest closure 问题。
- result/contract 明确保持 integrated exact-once miter、signed19 RTL/full-lane commercial VCS、foundry macro、macro-inclusive PPA、cycle、physical/system/headline 全为 false。
- 小缺口：机器枚举只有四个 eligible PWP case，escape 仅以文字合同出现，因此 `all_legal_term_cases...enumerated=true` 略宽；独立 escape 枚举不改变数学结论。

下一步应先做 accepted-operation 唯一 ID scoreboard/miter，覆盖 stall/retry/reset/flush/escape，再做 signed19 overflow SVA 与 full-lane commercial-VCS 宽参考比对。机器审计见 `m115r2_independent_audit.json`，详细评分见 `m115r2_pwp_prefix_coefficient_width_independent_hammer_review.json`。

本评审只写本目录，未修改生产 analyzer/result/contract、r1 revocation 或 `docs/359`；`docs/359` SHA 仍为 `dedde7ce...`。
