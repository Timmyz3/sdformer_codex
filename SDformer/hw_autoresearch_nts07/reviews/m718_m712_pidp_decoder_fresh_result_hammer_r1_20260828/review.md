# M718｜M712 PIDP decoder fresh-result receipt-blind hammer r1

## 裁决

**93/100，P0/P1/P2 = 0/2/3。** M712 canonical result 可以准入为 deterministic optimistic CPU fast-kill：**全 decoder PIDP 维持 `KILL_NO_RTL`**。允许另立一个 weight-fit selective PIDP 的 fail-closed CPU 合同，但本评审不授权该合同直接执行，不授权 RTL/VCS/EDA，也不准入 `1.474346×` 为论文或可执行加速。

本评审没有 import 或执行作者 analyzer。独立程序解包全部 120 个 M699 bitpack，重建 1200 个 record-timestep 行，并用独立 C LRU oracle 与不同采样位置的小型 Acc24 oracle 检查作者输出。没有使用 GPU 或 EDA。

## 独立结果

- M712 result、最终 author handoff、M699、M705、M686 的 recursive member set、member SHA、inner manifest 与 outer seal 全部通过。
- 120 records = 3 sequences × 10 samples × 4 modules；独立生成 1200 行，其中 headline 900、D1 diagnostic 300。
- 1200 行的 active source、contributor、K8 group、三种 A1 ledger、PIDP ledger、traffic、cache entry/refill/LRU miss 逐项与 author rows 比较，**0 mismatch**。
- 四层 K3/S2/P1/OP1 的 source-forward 与 destination-inverse topology 共 36 个 tap map，0 mismatch；四层 aggregate SHA 全部与 report 一致。
- 独立 Acc24 oracle 在每个 sequence 的 sample 0/9、time 0/4/9、四层、四个 edge/center/hash destination 上做 288 probes，0 mismatch；四层本地 INT8/scale SHA 也与 author 一致。它只证明 schedule equivalence，不是 checkpoint 数值准入。

## 全 PIDP：必须 KILL

固定 strongest baseline 的 ratio-of-sums 独立复算为：

| Fixed A1 | headline cycles |
|---|---:|
| A1-OSG | **21,583,106,050** |
| A1-SC8 | 21,658,089,144 |
| A1-ISO8 | 38,695,606,824 |

全 PIDP 为 43,401,838,140 cycles，故 `A1-OSG/PIDP = 0.497285529253×`。三个序列分别为 `0.498295606359× / 0.497862964913× / 0.495713994265×`。虽然 materialized descriptor+psum logical bytes 降为 0，cycle 回退远超 5%，traffic gate 也失败。

因此全 PIDP 的唯一合法结论是 CPU fast-kill 下的 **`KILL_NO_RTL`**。`0.497×` 不是物理测得的 slowdown，更不是论文性能数字。

## selective 1.474346×：组合算术真实，选择没有 runtime oracle

D3 的选择由冻结层尺寸决定：

| Module | static weight tile identities | logical cache entries | frozen choice |
|---|---:|---:|---|
| D0 | 384 | 16 | A1-OSG |
| D1 | 98 | 16 | A1-OSG |
| D2 | 25 | 16 | A1-OSG |
| D3 | **13** | **16** | **PIDP** |

它没有读取 sample、sequence、density、实际 miss 或 runtime 状态。headline 组合严格来自同一 1200-row 账本：

`4,305,988,872 (D0 A1) + 4,439,367,778 (D2 A1) + 5,893,744,290 (D3 PIDP) = 14,639,100,940 cycles`

因此 `21,583,106,050 / 14,639,100,940 = 1.474346419118×`。三个序列独立 ratio-of-sums 为 `1.473789914009× / 1.474046488751× / 1.475199126208×`，没有 average-of-ratios。

所以 selective diagnosis 不是 selective cherry-pick，也不是把不同 denominator 拼接；其**数学组合成立**。但其每个 PIDP row 仍是候选偏乐观下界，因此不能从“组合正确”跳到“可执行加速成立”。

## D3 cache-fit 攻击

D3 每个 weight tile 是 `16×96×3×3×INT8 = 13,824 B`。十三个 tile 为 179,712 B；另加两行 bitpacked source buffer 8,064 B、Acc24 288 B、control 8,192 B，总计 196,256 B，在 240 KiB logical budget 下尚余 49,504 B。若按 report 的 16-entry cache，则总计 237,728 B，余 8,032 B。

所有 300 个 D3 row 的 active tile identity 都恰为 13。capacity 16 时，285,113,756 次 weight reference 只有 `13×300=3,900` 次 cold miss。因此 D3 `2.178199×` 的 cache-fit 根因真实。

但这是硬 cliff：独立把 capacity 降到 12 后，D3 miss 升为 164,155,303，selective cycles 升到 37,620,297,360，`A1/selective = 0.573709076339×`。新合同必须要求所有 metadata、macro rounding 与端口组织扣除后仍至少有 13 个可用 entry；当前 fully-associative logical cache 不是物理 SRAM closure。

## 公平性敏感性：正号还在，但余量很薄

| 非准入 sensitivity | A1/selective |
|---|---:|
| canonical：PIDP 10 cycles/group | 1.474346× |
| PIDP 改为与 A1 相同 15 cycles/group | 1.265320× |
| A1 source ingress 也按 128-bit word | 1.431103× |
| **同时使用 128-bit A1 ingress + PIDP 15 cycles/group** | **1.214176×** |

联合公平点只比 1.20 gate 高 1.18%，且仍未收费真实 bitmap SRAM read ports/latency、priority/group formation、bank conflict、candidate control 与物理 timing。因此允许下一份 CPU 合同的目的应是**快速判死或闭合**，不能预写 selective 加速结论。

## P1

1. **selective 正点仍是 optimistic composition。** 新合同必须把“128-bit A1 ingress + PIDP 15 cycles/group”作为 primary，不得以 10-cycle canonical 点作为 GO；必须收费 PIDP-only 端口/控制，aggregate ≥1.20 且每序列 ≥1.05 才保留。
2. **13-entry capacity 是不可绕过的硬门。** 新合同必须显式跑 capacity `{12,13,16}`，并要求扣除全部逻辑 metadata 后 `>=13`；physical macro/port 未闭合时继续禁止 RTL/PPA。

## P2

1. Acc24 probe 使用本地 per-output-channel symmetric INT8，D1 使用 folded-theta diagnostic weight；不得升级为 checkpoint numeric/accuracy。
2. 三个选定 S3×10 cohort 不是 population generalization，也不是 full-decoder/system speedup。
3. author handoff alias 在本评审过程中由 `f753a113.../0ed506f8.../60a553b4...` 重封为包含 15-cycle sensitivity 的最终 `b0876d61.../c0e98847.../6167a159...`；canonical M712 result 始终未变。后续必须按最终 SHA tuple 引用 handoff，不能只引用可变目录名。

## 允许的新 CPU 合同

允许**起草并另行静态审查** weight-fit selective PIDP CPU contract，至少包含：

1. 四个 layer config bits 在读任何 row 前冻结：D0/D1/D2=A1-OSG、D3=PIDP；禁止 runtime/sample/sequence/density/miss selector。
2. 同一 M699 S3×10 population、同一 fixed A1-OSG、ratio-of-sums 和每序列分列。
3. primary 使用 128-bit A1 ingress、PIDP 15 cycles/group；10-cycle 只作 optimistic ablation。
4. capacity `{12,13,16}`；扣除全部 metadata 后 D3 `>=13` 为硬门。
5. 收费 bitmap SRAM port/latency、priority/K8 formation、control、refill、dense commit，禁止免费 overlap。
6. positive 仍只为 CPU diagnostic；fresh result hammer 后才讨论 minimal RTL。

## Claim boundary

准入：canonical M712 result 身份与 1200-row 算术；全 PIDP 在冻结 optimistic CPU coordinate 下 KILL；`1.474346×` 仅为 verified-but-non-admitted static composition；允许起草新的 selective CPU contract。

不准入：物理 slowdown、selective executable speedup、checkpoint 数值/accuracy、RTL/VCS/DC/Formality/STA/power/energy/PPA、full-decoder/full-network/system speedup或论文 headline。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
