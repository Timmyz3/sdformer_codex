# M1604｜M1575 S2 下一步最小实验第一性原理独立审阅

日期：2026-09-01

状态：`PASS_FIRST_PRINCIPLES_REVIEW__RETAIN_DECODER_S2_FOR_ONE_PAIRED_TEST__NO_GO_EXECUTION__NO_GO_RTL`

## 结论

S2 暂时保留为唯一一条新的有损候选，但现在不能跑 RTL，也不能把 M1575 的 30.168% 写成周期缩短或加速。M1575 真正证明的是：在 ep34 的三条 DSEC 序列、30 个样本、四层 decoder ConvTranspose 上，CCBS16 的 activity-relative 规则能标出一批可丢 block；它没有 paired AEE、地址计时、真实流量或周期。

当前本机不满足一次合法 production experiment 的输入条件，因此本审阅的执行裁决是 `NO_GO_EXECUTION`。应先补齐精确数据，再只跑一次预注册的 paired AEE + same-resource address-timed replay；同一个 epsilon 档同时过精度和周期门，才准许 RTL。

## eligibility 不能换算成加速

| 预算 | block eligibility | hypothetical INT8 weight-byte eligibility | 相对 ε=0 新增 weight eligibility |
|---|---:|---:|---:|
| ε=0 | 22.1723% | 21.9993% | — |
| ε=0.02 | 25.1652% | 24.8886% | 2.8893 pp |
| ε=0.1 | 30.1682% | 29.9303% | 7.9310 pp |

`30.1682%` 只是 decoder block eligibility。它不能按 `1/(1−0.301682)` 包装成 `1.432×`：固定 issue、metadata/debt、burst 对齐、cache reuse、bank 冲突、psum RMW、terminal/close/commit 都还收费。尤其无损 ε=0 已经拿到约 22% eligibility；有损 ε=0.1 的真实边际机会只有约 7.93 个百分点，是否值得精度预算必须由地址计时结果回答。

## 对象边界

- M1575 的实际对象是 decoder D0–D3 ConvTranspose，不是 FC、patch embed，也不是 C1 的四层 bottleneck Conv3x3。
- 因而当前不能把 S2 写成 FC/patch 机制；若要扩展，需要新的 retained payload、metadata、paired AEE 和同资源基线。
- S2 与 C1 没有直接算子重叠，但会共享 C2 typed-K8、Acc24、weight/psum SRAM、端口和完成协议。最终只能在一个统一 replay 中相加测得的算子周期，禁止相乘局部倍率。

## 数据质量审计

独立脚本复核了 M1575 的 5/5 成员和外层 seal、final ep34 capture 的 manifest/ordered trace/外层 seal/checkpoint，以及 M1542/M1562/M1572 的 exact SHA。

- 40/40 event tensor 均在本机，合计 491,525,120 B，SHA 全部与 final capture manifest 一致。
- GT 与 valid mask 各只有 10/40；缺少 `interlaken_01_a`、`thun_01_b`、`zurich_city_12_a` 各 10 个，共 30 GT + 30 mask。精确文件清单见 `audit.json`。
- 没有 unpruned/ε=0/ε=0.02/ε=0.1 的 40-sample paired predictions。
- M1575 仅含三条 decoder cohort 的 30-sample retained payload；若周期也严格用 same-40，还缺 `zurich_city_09_a` 十个样本的 D0–D3 payload/decision trace。
- 没有冻结的 S2 candidate 与 strongest exact baseline 的同资源地址计时结果，也没有 O16 到独立 bank/burst 的物理映射。
- M1575 用 FP32 ep34 权重生成 M(G,O)，byte proxy 却假设 packed INT8；部署定点权重、scale、向上取整的整数 bound、metadata 位宽/饱和规则尚未闭合。

这些都是阻断项，不是可忽略的表格空白。

## 唯一准许的下一次实验

预注册四条 arm：unpruned、ε=0、ε=0.02、ε=0.1。不得看完 AEE 后另挑 epsilon。

paired AEE 必须使用同一 ep34 checkpoint、同一源码/配置/预处理、eval mode、确定性 seed、相同 recurrent-state reset 和 final capture 的同序 40 样本。主指标为 40 样本所有 valid pixel 聚合后的 paired ΔAEE；同时报告 40 个 sample 和四条 sequence 的 paired 结果。

准入门：

1. ε=0 对 unpruned prediction bit-exact，`max_abs_prediction_error=0`、`ΔAEE=0`；
2. 同一个预注册 lossy arm 的 overall valid-pixel-weighted `ΔAEE≤0.02`；
3. 每条 sequence 的 valid-pixel-weighted `ΔAEE≤0.03`；
4. 同一个 arm 在同一 96-lane、Acc24、3 ns、相同 SRAM/bank/burst/metadata/debt 条件下，相对 strongest exact decoder `BIT_TYPED_K8` 的 local decoder ratio-of-sums cycle speedup `≥1.15×`；
5. O16 必须对应可独立抑制的 bank row 或 external burst。若仍读 O96/1152-bit 全行后再 mask，fetch saving 记 0；
6. 40/40 AEE 与 decoder replay、部署定点 quantization bridge 全部闭合。

只要任一项失败，就保持 `NO_GO_RTL`。即使只测到 byte reduction，也只能作为附录/能量候选，不能在 M1604 下启动 RTL。

完整可执行合同见 `next_experiment_contract.json`，复核入口见 `independent_readonly_audit.py`。

## Claim boundary

本审阅只读；没有 GPU、EDA、RTL、AEE、周期、流量、能量或新性能数字；没有修改 M1575、final capture、docs/359 或任何作者证据。
