# M710｜Decoder exact temporal-delta legal-tap product-work 作者交接

日期：2026-08-28  
角色：作者证据生产者，不是最终评审  
状态：**READY_FOR_FRESH_INDEPENDENT_REVIEW；N2 `KILL_NO_RTL`**

## 结论

在 M699/M705 已准入的三序列、30 样本、D0--D3 共 120 条 decoder payload 上，逐
真实 K3/S2/P1/OP1 合法 tap 重算：

| 口径 | Full active | `t0 + XOR(t,t-1)` delta |
|---|---:|---:|
| legal-tap events | 4,347,688,168 | 5,773,004,554 |
| scalar product work | 522,485,404,224 | 706,579,826,784 |

`delta/full = 1.3523436656252985`，即 temporal delta 在进入任何 state/memory 收费前，
已经令 product work **增加 35.2344%**。冻结 fast-kill 要求严格 `<0.70`，因此失败。
四层均回归：

- D0：`1.396232`；
- D1：`1.526372`；
- D2：`1.569612`；
- D3：`1.216455`。

三个序列分别为 `1.344801/1.349038/1.363123`；30 个 sample 聚合比范围为
`1.335897--1.370803`。这不是异常样本或单层造成的。因此 N2 不进入 state-identity、
地址化 cycle、memory 或 RTL：**KILL_N2_NO_RTL**。

## 方法与严格边界

每个 payload 按 `[T=10,B=1,C,H,W]`、little-bit-first、C-order 解包。对输入坐标
`(y,x)` 枚举真实 ConvTranspose 输出边界，合法 tap 数只可能为 `4/6/9`：左上角 4，
仅位于 top/left 一条边界为 6，其余为 9。两个 work 口径为：

```text
full  = Σ_t active(t,y,x,c) × legal_taps(y,x) × Cout
delta = active(t0,y,x,c) × legal_taps × Cout
      + Σ_t=1..9 XOR(active(t),active(t-1)) × legal_taps × Cout
```

D1 没有被二值化或近似。只读取 M699/M705 准入的 exact `{0, runtime-theta}` mask，
固定 `theta=0.9999954104423523`、IEEE754 uint32 `1065353139` 和内容 SHA
`5df16d34...3721`。本结果只数 mask product work，明确保持
`folded_weight_deployment=false`、`decoder_numeric_equivalence=false`。

每条记录满足：

- `delta_sources = t0_active + transition_sources`；
- `legal_tap_events = 4*n4 + 6*n6 + 9*n9`；
- `product_work = legal_tap_events × Cout`。

最终总体采用 ratio-of-sums，不使用 mean-of-ratios。

## 可复验包

冻结源：

- analyzer SHA：`526a36c367af915fdba4daaa8754cb33922fbe7dde327ee307d32464ddcb8296`；
- contract SHA：`9234a517c4fab185a4ae2d0a2b5bc76f41181125510ca35da03fbe0dda4e5132`；
- tests SHA：`897182961ea18486e79258746e03f13c5f10d1bbca514366bec7fc36f6ac8171`；
- runner SHA：`1e642aa796837d4fec77d68d57f7fbf71d4190a10ada5aaf9a36570e7c00c0a4`。

作者冻结结果：

`results/m710_h67_decoder_temporal_delta_legal_tap_product_work_r1_20260828/`

- summary SHA：`9c797a6932723a9e5fedc4b78c060fce0620aa84191ca1f052e56ce5203e0757`；
- member manifest SHA：`e48416210c0140eb19c6fb97e7804887e16c5d6a3b6f247e5b9d61f7358309d6`；
- outer-seal file SHA：`ea0b508127361001e74f5ff94a5ace08420371c27eec4617fec82502297ac128`。

复跑命令只适用于不存在 canonical result 的全新拷贝/审阅沙箱；当前 canonical 是
immutable，runner 会拒绝覆盖：

```bash
hw_autoresearch_nts07/system_simulator/scripts/run_m710_decoder_temporal_delta_legal_tap_product_work_r1.sh
```

作者测试 6/6 通过，最终冻结身份执行一次且 exit 0。开发阶段曾有一次 superseded、
未封存身份在识别 D1 nested statistics schema 前失败；它没有创建 canonical/staging
结果，已在 `handoff.json` 透明记录，不作为证据。

## Claim boundary

唯一可送 fresh review 的 claim 是：**在冻结 120-record payload 上，exact temporal
delta 的 legal-tap scalar product work 相对 full-active 增加 35.2344%，所以 N2 未过
pre-memory fast-kill。**

cycles、speedup、system speedup、accuracy、numeric bridge、RTL、VCS、EDA、DC、
Formality、PTPX、energy、PPA、DATE headline 全为 false。最终准入必须由另一个 fresh
reviewer receipt-blind 重算；本作者交接本身不是 admission。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
