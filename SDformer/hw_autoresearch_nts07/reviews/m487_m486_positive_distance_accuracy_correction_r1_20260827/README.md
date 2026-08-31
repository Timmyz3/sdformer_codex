# M487：M486 正距离 near-match 精度门纠错

日期：2026-08-27

## 裁决

M486 将 `tau=3` 列为“先做 accuracy fast-kill、通过后再写 RTL”的候选，这一项遗漏了
M312r2/M313r2 已完成的后续 paired valid825 证据，现予以撤销。Conv 的所有
positive-distance near-match residual-elision 路径维持 **NO-GO**，不得继续搜索更大的
`tau`，也不得开发相应 RTL。

## 决定性证据

- 相同 PAFT checkpoint SHA256：`cf4833b2...e9cca`。
- `tau=0`、running-BN、valid825 baseline AEE：`1.4691506710196987`。
- 冻结的选择性 `tau=[1,0,1,1]` candidate AEE：`1.498478640643033`。
- 绝对 AEE 增量：`0.029327969623334393`，超过预注册门 `0.02`。
- candidate 的 positive-distance snapped partition 数为 `245,630,707`，因此这不是
  未激活路径或计数器空跑。
- M313r2 合同红线明确规定：valid825 失败后关闭 positive-distance near-match，且不得
  继续在 validation set 上搜索。

M307 的 S10 结果（`Delta AEE=0.014362`）只是筛选证据；它不能覆盖后续 825 样本的
失败。其 SHARED96 机会也只有 `1.292379x` vs bit-sparse、相对 exact `1.048245x`，
不足以支持绕过 accuracy 门继续迭代。

## 对当前路线的影响

1. 撤销 M486 candidate 3 的 `MEASURE-ONLY` 建议，改为 `CLOSED_NO_RTL`。
2. 不占用远端 A800 运行 `tau=3`；当前用户训练保持不受干扰。
3. 有损近匹配只作为负结果/消融，不进入 DATE 主贡献或性能表。
4. 可立即开发和收口的 RTL 仍只有：M479 Conv capture path，以及 M485 FC2
   shared-context bank-coissue 公平 Pareto。

## Claim boundary

本纠错只关闭 positive-distance near-match 分支，不改变 `tau=0` exact PWP、M479、
M485 或任何封存证据。没有新增 cycle、accuracy、PPA、energy 或 system-speedup claim。
`docs/359_DATE终局冻结_20260813.md` 未修改。
