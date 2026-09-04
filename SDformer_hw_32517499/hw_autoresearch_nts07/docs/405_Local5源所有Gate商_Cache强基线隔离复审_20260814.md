# Local5 source-owned gate quotient 与 W4 cache 隔离复审

## 1. 裁决

`ADMIT_AS_ISOLATION_RTL_ONLY_STRONG_MFEP_CACHE_UNRESOLVED`

本轮只回答一个窄问题：在相同 source-ordered 流、相同 W4 product cache、
相同 TCFM5 和合法 1RW Acc 下，是否仍需要把同一
`{source,lane,gate}` 的 destination 合并成 bitmap。

答案为 **是**。但本轮 `C` 是 source-one-hot cache，不是 destination-local
MFEP+W4，故不得把周期比写成打赢全局强基线，更不得修改 `docs/359`。

## 2. 实验边界

- `[rtl]` Verilator `--assert`；
- 真实权重、100-group production cohort；
- `OUT_DIM=2` tile，不是 encoder；
- `ACC_BACKEND_KIND=1`，合法单端口 1RW Acc；
- `C`：source-ordered one-hot destination issue + W4；
- `QC`：source-owned equal-gate destination-mask issue + W4；
- 两边逐组检查相同 expected Acc32，0 mismatch；
- `docs/359` SHA 保持
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

证据：

- `results/local5_qc_c_population100_20260814/verilator_assert.log`
- `results/local5_qc_q_population100_20260814/verilator_assert.log`
- `results/local5_qc_cache_isolation_ablation_20260814/report.json`

## 3. 结果

| 项 | C: source-one-hot+W4 | QC: source-mask+W4 |
|---|---:|---:|
| busy cycles | 296,310 | 170,486 |
| term issue | 222,649 | 74,131 |
| destination update | 222,649 | 222,649 |
| product start | 5,556 | 5,561 |
| tag compare | 890,596 | 296,524 |
| memory wait | 198,745 | 73,496 |

- 隔离周期比：`296310 / 170486 = 1.7380x`；
- QC 在 61 组更快，39 组持平，0 组更慢；
- product start 几乎相同，周期收益来自少发 ready-valid term 和少做 tag lookup，
  不是少做乘法；
- QC+W4 为 170,486 周期，现行 QC 无 cache 为 170,269 周期，W4 自身慢
  217 周期，故 cache 不作为性能贡献。

## 4. 为什么不能称为完整强基线

profile 中 destination-local MFEP+W4 的对象是 174,289 条命令和 5,551 次
product start。现行 production inverse-stencil 接口在 score 后只保留
`K_self`，依靠关系转置在 source 侧恢复 consumer；destination MFEP 则需要
同一 destination 的五个 candidate K。直接把旧 MFEP 接入当前生产 tile 会新增
另一套 K metadata/执行路径，不是本轮的窄参数消融。

所以：

1. `1.7380x` 只能证明 source-owned bitmap 相对 source-one-hot 的发行收益；
2. 它不能替代 MFEP+W4 的全局强基线；
3. 现有 `[prof]` 已证明 MFEP+W4 和 source-owned+W4 的 product start 基本相同，
   但还没有同生产边界的周期/PPA；
4. 不为此再建第三套生产后端，避免用工作量抬创新分。

## 5. DATE 影响

- Local5 主叙事仍是一条：QS early-out -> inverse-stencil/FCSR ->
  source-owned issue -> TCFM5；
- W4 是强竞争实现，不是贡献；
- 本轮提高了 source-owned 发行边界的可辩护性，不改变创新分；
- Local5 当前仍按创新 `3.1/5`，不因 `1.7380x` 上调；
- 下一步停止 Local5 新机制扩张，转 Motion 多窗达到性与双线完整度。

