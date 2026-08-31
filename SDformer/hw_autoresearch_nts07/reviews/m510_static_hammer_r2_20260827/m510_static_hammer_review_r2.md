# M510 r2 独立静态打铁评审

日期：2026-08-27  
结论：`STATIC_GO__ONE_SHOT_AUDIT_ONLY_UNDER_EXACT_SHA_RUNNER`  
评分：**95/100**  
P0：**0**  
P1：**2**  
生产 audit 执行：**否**

## 结论

r1 的 MS decoder 类链 P0 已修复。r2 可以在 exact-SHA one-shot runner 下执行
一次 aggregate coverage-gap audit。它只能准入“四层反卷积被旧 profiler
遗漏 + S100 aggregate-count 界”；不准入 exact cycle、RTL 或系统倍速。

## 关键通过项

1. **实际类链已固定。** r2 新增 `Spiking_STSwinNet.py`
   SHA `b8d969f9...`，并断言 config 为 `MS_SpikingformerFlowNet_en4`、
   unet 为 `MS_Spikingformer_MultiResUNet`、transpose 类为
   `MS_SpikingTransposeDecoderLayer`。限定类区间内检查
   `x = self.sn(x)` 早于 `x = self.deconv(x)`。由于相关源码都受 exact SHA
   约束，冻结 ATLIF nonzero 数可作为遗漏反卷积的 source population。
2. **4/6/9 容量正确。** 对 K3/S2/P1/output-padding1，
   `o=2i-1+k`。`i=0` 裁掉 1 个一维 tap，最后一个 input 坐标不被裁；
   所以二维只有 top/left 裁剪，空间 source 容量为 `4/6/9`。
3. **四层 topology 正确。** Cin/H/W 依次为
   `(1536,15,20)`、`(770,30,40)`、`(386,60,80)`、
   `(194,120,160)`；Cout 为 `384/192/96/96`。D3 确实是 96。
4. **独立整数重算全部对上。**

| 量 | 独立结果 |
|---|---:|
| lower products/S100 | 1,637,926,293,504 |
| upper products/S100 | 1,761,318,549,504 |
| dense products/frame | 78,848,509,440 |
| ideal 96-lane decoder cycles/frame | 170,617,322.24--183,470,682.24 |
| corrected included-scope envelope | 790,920,227.24--803,773,587.24 |
| decoder share | 21.572%--22.826% |
| decoder-free ceiling | 1.2751--1.2958x |
| dense/bit-sparse opportunity | 4.4767--4.8139x |

`4.48--4.81x` 只是 A0 dense 对 activation-zero skip 的 opportunity，不是 EPD/A1
创新倍速。若要对当前修正分母产生约 `1.10x` sensitivity，独立求解得
EPD/A1 需约 `1.662--1.728x`；文档使用 `1.66--1.80x` 作保守门没有夸大。
5. **S100/per-sample 口径安全。** 产品数先以 S100 总量封存，除以
   100 后才称 per-frame mean；所有 exact-coordinate、per-sample cycle、measured
   speedup、energy、PPA 和 headline 字段均为 false。
6. **旧分母已正确降级。** `620,302,905` 只保留
   `included-scope 96-lane activity-weighted` 身份，不再充当 strict
   full-network 分母。

## P1（不阻塞受控 one-shot）

1. analyzer 会校验 contract 中的 analyzer SHA，但 expected contract r2 SHA、
   docs510 SHA 和本评审 seal 并不由 analyzer 外部固定。因此禁止直接手工
   invocation；必须用 runner 固定 analyzer `117384e...`、contract `4bda9f...`、
   docs510 `940621...`、docs359 `dedde7...` 和本评审外层 seal。
2. `not output.exists()` 到 `os.replace()` 之间仍有极小 TOCTOU 窗口。
   本次用单一 owner/one-shot 发布约束即可；通用封存 helper 后续应加独占锁。

## 授权边界

- exact-SHA runner 下一次 aggregate audit：**GO**
- audit sealed PASS 后的 dedicated S10 decoder input capture：**GO**
- sealed PASS 后的 A0/A1/EPD 同资源 cycle fast-kill：**GO**
- RTL/VCS/DC/Formality/PTPX：**NO-GO**，必须等 exact S10 和 EPD/A1 过门
