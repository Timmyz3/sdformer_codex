# M1551：M1547 S2 CCBS retained decoder fast-kill 独立打铁

裁决：**FAIL_P0_DESTINATION_DEBT_ACCUMULATION_DOMAIN**，61/100，P0=1，P1=1。

M1547 的文件身份、封存树和本地算术可以复现，但它的 PASS **不准入**。必须先做 destination-owned successor，重新计算后才能决定是否申请 FC/patch capture。

## 已独立复核

- current Python 与 CPython 3.6 的 test、preflight 全部 PASS；
- M1521 122 个 sealed member 与 M1547 3 个 sealed member 全量 SHA/population 一致；
- 从 M1521 重新选出三序列、每序列位置 `0/4/9`、四 decoder 层，共 `36 call × 64 site = 2304 site`；
- 从 ep34 checkpoint 独立载入四个 FP32 ConvTranspose 权重，tensor SHA 全部匹配；
- 三个 block axis 的 metadata、局部 bound/exact ratio、epsilon drop、dynamic witness 数字全部复现。

严格 metadata 门没有被四舍五入：

| source group × output tile | M bytes | M / hypothetical INT8 | vs old G11 | 严格 `>=8x` |
|---|---:|---:|---:|---:|
| 8×16 | 12,432 | 0.1741% | **7.9768339768×** | FAIL |
| 16×16 | 6,240 | 0.0874% | 15.8923× | PASS-local |
| 32×16 | 3,144 | 0.0440% | 31.5420× | PASS-local |

这里的 INT8 仅是假设的容量分母；ep34 仍没有正式 INT8 数值权威。`PASS-local` 只表示作者的局部 metadata/ratio 算术过门。

## P0：debt 所有者错误

M1547 在每个 `source-site × output-tile` 开始时把 debt 重置为零。可是 K3/S2 ConvTranspose 的一个内部 destination 在二维空间最多同时收到四个 source-site 的贡献。举一个最小反例：四个 source-site 各自在 `epsilon=0.1` 下接受 `0.09` debt；M1547 四次都会通过，但最终同一 destination 累积 `0.36`，已经超过 `0.1`。

所以当前 `epsilon=0.1` 的 98.91%–99.64% drop 只能描述错误 accumulation domain 上的 source-local 决策，不能称为有界机会，更不能触发 FC/patch production capture。

successor 必须二选一：

1. 按 `destination × output-tile` 保存 debt，跨全部合法 source-site 与 kernel tap 累计；或
2. 用严格证明的最大 contributor multiplicity 对预算做保守拆分。

两种实现都必须保留边界、tail 和 padding/stride 映射，并收费 debt SRAM/端口。`epsilon=0` 的 zero-bound 事实不受该反例影响，但仍没有周期、流量或能量结果。

## P1：fetch-before-compute 尚是声明

`decision_before_weight_fetch=true` 是结果里的结构常量，不是 address-timed schedule。当前没有收费 metadata 端口、bank conflict、queue、tail 或真实 weight fetch。因此只能作为 successor 的设计要求，不能写成 cycles、traffic、energy 或 RTL 证据。

## 最终边界

- M1547 本地数值复现：是；
- M1547 PASS 准入：否；
- FC/patch capture 请求：否；
- destination-owned CPU successor：允许；
- cycles / traffic / energy / AEE / RTL / paper headline：全部 false。
