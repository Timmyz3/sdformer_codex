# M510 one-shot 结果独立打铁

日期：2026-08-27  
结论：`GO__COVERAGE_GAP_AND_AGGREGATE_BOUNDS_ADMITTED__EXACT_TRACE_REQUIRED_BEFORE_RTL`  
评分：**99/100**  
P0：**0**  
P1：**0**

## 准入结论

M510 已足以正式确认：旧 operator ledger 遗漏 H67 四层
`ConvTranspose2d`，`620,302,905 cycle/frame` 只能作为
`included-scope 96-lane activity-weighted` 分母。S100 aggregate-count 分析界
也可准入。

但这不是 exact trace，更不是 EPD RTL 或系统倍速。下一步只授权 dedicated
S10 decoder capture 和后续 A0/A1/EPD 同资源 cycle fast-kill。

## 封存与身份

| 项 | 结果 |
|---|---|
| output `SHA256SUMS` SHA | `5d485890766de3d0aa99f0f95273d94b13f50d02cc38d86b8a55023fe2657d04` |
| output seal-file SHA | `97899f462e881c70c53fe58f9c23b2245cd42bcf245d33f7cfb16ab4adf377b5` |
| attempt final `SHA256SUMS` SHA | `97996d3b8627d262e537b22d3b44ba0973e0c5c43625bd819026eee4bf61affe` |
| attempt final seal-file SHA | `e02f9a6925a2fde50b61d101b1cba5cc12c50dcc9dc25b2f81717367f7806fd2` |
| output 成员/outer seal | PASS |
| attempt initial/final 两层 seal | PASS |
| output 额外/隐藏文件 | 0 |
| attempt 额外/隐藏文件 | 0 |
| analyzer start/end SHA | `117384e...` PASS |
| contract SHA | `4bda9f...` PASS |
| analyzer 数值输入当前复核 | 10/10 PASS |
| attempt `identity.sha256` 当前复核 | 6/6 PASS |
| `docs/359` | `dedde7ce...` PASS |

`POSTAUDIT_PASS` 中的 output seal-file SHA 与当前 output 一致，而且它被
attempt final seal 绑定。

## 独立数值复核

结果与执行前已封存的 `independent_bounds_r2.json` 逐层和总量对上：

| 量 | 独立复核值 |
|---|---:|
| active products/S100 lower | 1,637,926,293,504 |
| active products/S100 upper | 1,761,318,549,504 |
| dense products/frame | 78,848,509,440 |
| ideal decoder cycles/frame @96 lanes | 170,617,322.24--183,470,682.24 |
| corrected included-scope envelope | 790,920,227.24--803,773,587.24 |
| decoder share | 21.5720%--22.8262% |
| decoder-free ceiling | 1.2751--1.2958x |
| dense/bit-sparse opportunity | 4.4767--4.8139x |

四层输入为 `(1536,15,20)`、`(770,30,40)`、`(386,60,80)`、
`(194,120,160)`，Cout 为 `384/192/96/96`。D3=96 已再次确认。

## Claim boundary

可以写：

- 四层反卷积覆盖缺口已确认；
- S100 aggregate-count tight bounds；
- decoder 约占修正 included-scope envelope 的 21.6%--22.8%；
- dense/bit-sparse 有 4.48--4.81x 的算法机会。

不可以写：

- `4.48--4.81x EPD speedup`；
- exact coordinate/product trace 或 per-sample cycles；
- RTL/VCS/DC/Formality/PTPX/energy/PPA；
- `1.28x system speedup` 或 DATE headline。

## 已解决的审计过程观察

结果复核过程中曾捕获 docs510 的一次 post-run 临时编辑，当时 SHA 为
`4a6e7315...`。在本评审封存前，文件已恢复为 attempt 绑定的
`9406211a...`；现在从 hardware root 复核 `identity.sha256` 为 6/6 PASS。
因此该项只作为审计过程记录，不是 live P1。

## 裁决

`GO_TO_EXACT_S10_CAPTURE_AND_A0_A1_EPD_FASTKILL_ONLY`。RTL 仍为 **NO-GO**，直到
exact coordinates 和 EPD/A1 同资源门同时通过。
