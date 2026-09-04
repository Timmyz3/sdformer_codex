# M715｜M710 Decoder temporal-delta fresh independent hammer

日期：2026-08-28  
角色：fresh independent reviewer  
方式：receipt-blind；核心比值由 `independent_hammer.py` 从 M699 canonical 的 120 个 bitpack 直接重算，**未 import、未执行作者 M710 analyzer**。作者 CSV/summary 只在独立重算完成后用于逐记录对照。  
最终裁决：**M710 负结果 KEEP；N2 temporal-delta 机制 KILL_NO_RTL。**

## 1. 独立重算结论

| 口径 | Full active | `t0 + XOR(t,t-1)` | Delta / Full |
|---|---:|---:|---:|
| Source events | 487,063,997 | 647,158,492 | 1.3289× |
| K3/S2/P1/OP1 legal-tap events | 4,347,688,168 | 5,773,004,554 | 1.3279× |
| Scalar product work | 522,485,404,224 | 706,579,826,784 | **1.3523436656252985×** |

其中 delta source 严格守恒为：

```text
45,314,522 t0-active + 601,843,970 adjacent-timestep XOR
= 647,158,492 delta sources
```

冻结快杀门要求 `delta/full < 0.70`。独立结果不仅未过门，而且在 memory、state、address、signed-delta service 全部尚未收费前，product work 已增加 **35.2343666%**。

## 2. 逐层、逐序列和分布

### 2.1 Per module ratio-of-sums

| Module | Full product work | Delta product work | Delta / Full |
|---|---:|---:|---:|
| D0 | 82,593,173,376 | 115,319,210,496 | 1.396231744× |
| D1 | 86,324,920,128 | 131,763,963,456 | 1.526372260× |
| D2 | 83,243,665,248 | 130,660,224,768 | 1.569611626× |
| D3 | 270,323,645,472 | 328,836,428,064 | 1.216454548× |

四层全部回归；不是 D1 scaled-binary 单层导致。

### 2.2 Per sequence ratio-of-sums

| Sequence | Delta / Full |
|---|---:|
| interlaken_01_a | 1.344800975× |
| thun_01_b | 1.349038456× |
| zurich_city_12_a | 1.363122770× |

跨序列方向一致。30 个 sample 聚合范围为 `1.335897187–1.370803383×`；120 个单记录范围为 `1.185024094–1.608278581×`。因此 **120/120 records、30/30 samples、3/3 sequences、4/4 modules 均大于 1.0**。

## 3. 独立几何与守恒

审阅器没有采用作者输出的 tap count。对每个输入 `(y,x)` 和 `ky,kx in [0,2]` 独立枚举：

```text
oy = 2*y - 1 + ky
ox = 2*x - 1 + kx
0 <= oy < 2H, 0 <= ox < 2W
```

四层均只出现 `4/6/9` 三种合法 multiplicity，且逐坐标与“左上角 4、单 top/left 边界 6、其余 9”完全一致：

| Module | Input H×W | Output H×W | #4 | #6 | #9 |
|---|---:|---:|---:|---:|---:|
| D0 | 15×20 | 30×40 | 1 | 33 | 266 |
| D1 | 30×40 | 60×80 | 1 | 68 | 1,131 |
| D2 | 60×80 | 120×160 | 1 | 138 | 4,661 |
| D3 | 120×160 | 240×320 | 1 | 278 | 18,921 |

全部 120 条记录通过以下独立守恒：

- `delta_sources = initial_sources + transition_sources`；
- full/delta source 分别等于 `n4+n6+n9`；
- legal events 分别等于 `4*n4 + 6*n6 + 9*n9`；
- product work 分别等于 `legal_events * Cout`；
- 总体只用 ratio-of-sums，没有 mean-of-ratios 替换。

## 4. D1 theta mask 身份

D1 的 30 条记录全部满足：

- route 为 `EXACT_SCALED_BINARY_BITPACK`；
- `coerced=false`、`rounded=false`、`thresholded=false`；
- `theta_gate_pass=true`，`other_finite_count=0`、`nonfinite_count=0`；
- bitpack active count 与 `theta_count`、raw `nonbinary_finite_count` 完全相等；
- raw SHA 在 record/raw/scaled-binary audit 三处一致。

独立将 runtime theta 按 IEEE754 little-endian float32 编码，得到：

```text
value   = 0.9999954104423523
uint32  = 1065353139
hex     = b3ff7f3f
SHA256  = 5df16d346190fdd928ee71a5c3e1dbeaf4d9b71985167bd7eccbdf1d87cc3721
```

四项与 M699/M705 canonical 一致。此处只准入 **theta mask product-work count**；仍不准入 folded-weight deployment 或 decoder numeric equivalence。

## 5. Canonical、SHA 与 seal

| 包 | Members | Manifest SHA | Outer-seal file SHA | 结果 |
|---|---:|---|---|---|
| M699 payload | 122 | `27b35748...46053` | `eaf975a9...e18c` | 全部通过 |
| M705 fresh review | 5 | `2c53369d...f50a9` | `26781f5d...24ff7` | 全部通过 |
| M710 result | 6 | `e4841621...309d6` | `ea0b5081...c128` | 全部通过 |
| M710 author handoff | 3 | `23ba07ea...d18370` | `a8565625...95930` | 全部通过 |

另外：

- M699 contract SHA `43d3b024...8fc7` 与 payload manifest 身份一致；
- M699 payload `manifest.json` SHA `e2d7c92a...3dc0` 与 member manifest 一致；
- M705 `review.json` SHA `6af48fb2...bd3` 与 M710 contract 绑定一致；
- M710 contract SHA `9234a517...5132` 与作者身份一致；
- `docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

独立逐行对照发生在重算之后：作者 `per_record.csv` 的 120 条记录、8 个整数字段及 ratio 全部 0 mismatch；作者 summary 的 overall legal events、product work 和 ratio 也全部一致。

## 6. P0 / P1 / P2

| Severity | 数量 | 结果 |
|---|---:|---|
| P0 | 0 | 无身份、payload、几何或核心算术错误。 |
| P1 | 0 | 无逐记录、聚合、守恒、gate 或 seal 不一致。 |
| P2 | 0 | 作者已正确限制 D1 与 product-work claim boundary。 |

评分：**99/100，ADMIT_NEGATIVE_RESULT**。扣 1 分不是证据缺陷，而是该审计按合同不覆盖 state identity、signed delta numerical bridge、address-timed memory 或 cycles；冻结门已失败，因此不应继续为这些项目投入资源。

## 7. KEEP / KILL

### KEEP

- KEEP M710 的 negative-result receipt，作为 decoder 时域复用消融；
- KEEP `delta/full=1.3523436656252985`，但标签必须是 **legal-tap scalar product-work regression**；
- KEEP M699/M705 的三序列 decoder payload 作为其他 exact decoder 候选的 canonical 输入。

### KILL

- **KILL N2 temporal delta before RTL**；
- 不进入 state-identity、address-timed cycle、memory、VCS 或 DC；
- 不把这个结果改写成 cycle slowdown、energy increase 或 system performance；这些从未测量；
- 不把 D1 mask count 改写成 theta-fold deployment 或 decoder numeric equivalence。

## 8. Claim boundary

唯一 fresh-admitted 结论是：

> 在 M699/M705 canonical 的 120-record H67 ep35 decoder payload 上，按真实 K3/S2/P1/OP1 legal tap 计数，`t0 + adjacent-timestep XOR` 的 scalar product work 为 full-active 的 1.3523436656252985×，增加 35.2344%；因此冻结 `<0.70` pre-memory gate 失败，N2 不进入 RTL。

cycles、speedup、system speedup、accuracy、numeric bridge、state identity、RTL、VCS、EDA、energy、PPA 和 DATE headline 仍全部为 false。

