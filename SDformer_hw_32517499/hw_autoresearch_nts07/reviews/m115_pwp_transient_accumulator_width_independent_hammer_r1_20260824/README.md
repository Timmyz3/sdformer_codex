# M115 PWP transient accumulator-width：独立打铁复审

日期：2026-08-24

结论：**80/100，P0=1 / P1=3 / P2=5，要求修订。** 四个冻结 INT8 payload 的 3,072 个输出通道已全部从原始字节重算，M115 的 `2×sumabs`、signed20/signed22、W384 762,280/836,008 B 及节省算术都正确且安全；但“每项最多两次操作”不能推出“单项瞬时系数可到 ±2”。合法 PWP/correction/escape 与任意服务顺序下，单项前缀系数只在 `{-1,0,+1}`，所以当前证据不能宣称 signed20/22 是 required，也不能拒绝 signed19/21。

## 原始 payload 独立复算

四个 payload 均为 `I_KY_KX_O_C_ORDER`、`768×3×3×768` signed INT8，每个 5,308,416 B，且没有 `-128`。按 `index % 768` 聚合每输出通道的 6,912 个权重绝对值：

| op | min sumabs | max sumabs | max channel | once bits | loose 2× bits |
|---:|---:|---:|---:|---:|---:|
| 0 | 113,538 | **218,338** | 360 | 19 | 20 |
| 1 | 79,336 | 204,866 | 185 | 19 | 20 |
| 2 | 87,029 | 207,239 | 513 | 19 | 20 |
| 3 | 82,093 | 190,753 | 126 | 19 | 20 |

四份 accumulator-init payload 也逐字节验证为全零。完整 3,072 通道 ledger 和四个 u32le digest 在 `m115_independent_audit.json`。

## P0：两次操作不等于两倍瞬时系数

M108 的冻结系数关系是：

`target = center + positive_correction - negative_correction`

逐个穷举 eligible/escape、center/target，并枚举每项所有合法操作顺序：

- `center=1,target=0` 是唯一两操作情形：`+1 anchor` 和 `-1 correction`。两种顺序的前缀分别为 `0→1→0` 或 `0→-1→0`。
- `center=0,target=1` 只有一次 `+1` correction。
- `center=1,target=1` 只有一次 `+1` anchor。
- escape 没有 PWP anchor，raw target event 至多一次。

所以最大绝对操作次数确为 2，但任意顺序下最大绝对前缀系数为 **1**。`2×sumabs` 是安全的 total-variation 上界，却不是 tight/required 上界。同一 exact-once 前提下，checkpoint signed19、dense signed21 才是当前抽象模型的紧候选。

反过来，若 stall/retry/duplicate 可以让已接受的 anchor 或 correction 重放，那么 exact-once 前提失效，无界重试也会同时击穿 `2×` 上界；仅增加到 signed20 并不能解决协议重复执行。

## W384 存储复算

固定 descriptor+metadata+valid 为 199,994 bits，所有 byte ceiling 独立复算：

| 口径 | signed bits | combined bytes | vs signed24 saving |
|---|---:|---:|---:|
| 当前实现 | 24 | 909,736 | 0 |
| M115 checkpoint 保守上界 | 20 | **762,280** | **147,456 B / 16.21%** |
| exact-once checkpoint 候选 | 19 | 725,416 | 184,320 B / 20.26% |
| M115 dense 保守上界 | 22 | **836,008** | **73,728 B / 8.10%** |
| exact-once dense 候选 | 21 | 799,144 | 110,592 B / 12.16% |

19/21 位目前只允许称软件抽象候选；尚无相应 integrated RTL、commercial VCS、foundry SRAM 或 PPA 准入。

## Strict/manifest 与 claim boundary

- 当前输入/输出 strict JSON 均可解析；duplicate key、NaN、Infinity、duplicate manifest path、malformed hash、`..` traversal 和 payload byte mutation 攻击均被拒绝或检出。
- M115 producer manifest 七项全部验证，但遗漏 analyzer 实际读取的 M41 result、M108 analyzer/result、M114 result 与 M112 receipt，因此不是 self-contained input closure。
- M115 正确保持 RTL/VCS、foundry macro、macro-inclusive PPA、cycle、physical/system/headline 全为 false。没有把 logical byte saving 冒充硬件性能或 PPA。
- 这些 byte 数仍是 logical lower bound，不含额外 controller/grace、ECC、macro rounding/periphery、interconnect 或 weight SRAM。

建议发布 M115-r2：把 `2×` 降级为保守 fallback；用 exact-once accepted-transaction miter 封死 stall/retry/reset/escape 后，再做 signed19 checkpoint 与 signed21 dense 的 RTL/VCS。生产 analyzer/result/contract 与 `docs/359` 本评审均未修改，后者 SHA 仍为 `dedde7ce...`。

机器审计见 `m115_independent_audit.json`，评分与 findings 见 `m115_pwp_transient_accumulator_width_independent_hammer_review.json`。
