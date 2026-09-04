# M571｜M548 waterfall r2 mechanical repair author handoff

日期：2026-08-28  
状态：`R2_SOURCE_ONLY_REPAIR_COMPLETE__FRESH_INDEPENDENT_HAMMER_REQUIRED__NOT_SHAREABLE`

## 唯一数值修复

M569 独立审阅判定 M548 r1 只有一个 P1：

```text
213376 / 245760 = 0.868229166666...
```

因此新的不可变 r2 只把
`paper_waterfall_rows[2].ratio` 从 `0.868131510417` 修正为
`0.868229166667`。r1 保持原文件、原 SHA，不原位改写，并明确按 M569 裁决为
`DO_NOT_SHARE`。

## 冻结边界

- 所有整数输入和 waterfall 整数守恒式不变。
- `2.038776477138` 仍只表示 arithmetic-work reduction。
- `1.741232213066` 仍只表示四个 bottleneck Conv、单序列、十样本的 exact CPU-model cycle speedup。
- H67 ep35、51,840,000 rows、八个 output blocks、240 KiB budget 与 213,376 B modeled storage 不变。
- RTL、VCS、Synopsys PPA、energy、system speedup、DATE headline 仍全部为 false。
- Prosperity/Phi 仅作为已引用的 evaluation-structure 方法来源；无 first/novel claim。

## 新身份与双封

- r2 contract：`contracts/m571_m548_m528_prosperity_phi_style_waterfall_contract_r2_20260828.json`
- contract SHA256：`eb67b5a6c84121b4f650bf7f60178bd7e14c9d07f5a52e615454070862070901`
- contract member sidecar SHA256：`71dddfd590f1e237775df0cca97c38bfc313385dc127e825942008e2fed6a370`
- contract outer-seal file SHA256：`996c58937da74e3ac58bbcc9b7df9492b441e520df9349ea01dc9817c4e59b00`

r2 合同已经绑定 M569 `review.json`、`review.md`、成员 manifest 和外层 seal 的
精确 SHA。fresh reviewer 必须独立验证差分、算术和 claim 边界；author 没有自我
准入。

## 零执行回执

本修复没有运行 EDA、VCS、runner、训练、远端命令或大型 CPU 任务；没有创建
result、attempt 或 launch admission。`docs/359` 未修改，SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

