# M487 positive-distance / tau3 独立纠错打铁评审

## 裁决

**97/100，纠错 GO；当前冻结研究身份下的 `tau>0` positive-distance
near-match residual-elision RTL 永久 NO-GO。**

M486 把 `tau=3` 留作一次 accuracy fast-kill，这与已经执行完毕的 M313r2
预注册 stop rule 冲突。M307 只证明 `zurich_city_09_a` 十帧上
`tau=[1,0,1,1]` 的筛选点 `Delta AEE=0.014362 <= 0.02`，其合同同时明确
`valid825=false`、`rtl=false`、`system_speedup=false`。后续 M312r2/M313r2
在同一 checkpoint、running-BN、相同有序 825 帧上完成 paired evaluation：

- baseline AEE：`1.4691506710196987`；
- candidate AEE：`1.498478640643033`；
- `Delta AEE=0.029327969623334393`；
- 预注册门：`0.02`；超门 `0.009327969623334393`，即预算的
  `1.466398481x`；
- candidate 实际触发 `245,630,707` 个 positive-distance replacement，
  占三个启用算子 `3,207,600,000` partition vectors 的 `7.657772384%`，
  因而不是“只走 exact hit”的空测试。

M313r2 合同红线逐字要求：valid825 失败后关闭 positive-distance
near-match elision，且不得继续 validation-set search。`tau=3` 是同一
checkpoint/catalog/mechanism 上更晚提出的阈值搜索，不得用来救回该分支，
也不得占用 A800、VCS 或 DC 资源。

## 独立核验

两份内层 manifest 均 `sha256sum -c` 通过；两份外层 seal 均正确绑定各自
manifest。baseline/candidate 的 `per_frame.csv` 各有 825 行数据，
`(file, sequence, valid_pixels)` 顺序完全相同。

从逐帧 CSV（十位小数打印值）独立重算得到：

| 聚合 | baseline | candidate | Delta AEE | 对 0.02 门 |
|---|---:|---:|---:|---:|
| frame-equal（CSV 重算） | 1.469150668813 | 1.498478641498 | 0.029327972685 | FAIL |
| pixel-global | 1.421619254603 | 1.449192685686 | 0.027573431083 | FAIL |
| sequence-balanced（CSV 重算） | 1.410775545560 | 1.435103814678 | 0.024328269118 | FAIL |

主门应使用 receipt 中未截断的 primary metric，即
`0.029327969623334393`。三种辅助聚合都失败，仅说明结论对聚合口径稳健，
不得事后替换预注册主指标。18/18 个序列的 frame-equal mean AEE delta
均为正；825 帧中 636 帧变差、189 帧改善，最差单帧 delta 为
`+2.0333518979`。

## 评分

| 维度 | 分数 | 说明 |
|---|---:|---|
| 身份与双 seal | 20/20 | 合同、receipt、profile、CSV、内外层 manifest 均核对 |
| paired AEE 算术 | 25/25 | 原始 primary delta 与 CSV/三种聚合独立复算一致指向 FAIL |
| stop rule 解释 | 24/25 | 当前身份永久关闭明确；不虚构 tau 单调误差定理 |
| claim boundary | 20/20 | exact tau0、当前 lossy 路径、未来新身份严格分开 |
| 可复查性 | 8/10 | 原始逐帧 CSV 有十位小数截断，但不影响门判断 |
| **总分** | **97/100** | **GO correction / NO-GO tau>0 RTL** |

## P0 / P1 / P2

### P0

1. 撤销 M486 的 `tau=3` accuracy fast-kill；不得启动 GPU、不得新写
   positive-distance comparator/suppress RTL。
2. 下游计划、表格和摘要不得把 `1.5948x` frozen-trace opportunity 当成
   accuracy-admitted、hardware-admitted 或 system speedup。
3. 保持 M313r2 的 `NO_GO_MODIFIED_FORWARD_ACCURACY_GATE` 为当前分支终态，
   不得尝试 `tau=2/3/...` 或重新挑 layer subset 来搜索 valid825。

### P1

1. 保留 `tau=0` exact PWP/parent-delta 路径；M313r2 不否定 exact hit、
   M479、C2 或其他无损机制。
2. 将 M307 的十帧乐观点和 M313r2 的 825 帧失败成对放入消融/负结果，
   用来说明小样本筛选不能替代完整精度门。
3. 只读审计所有引用 M486 的后续计划，删除“tau3 尚待验证/通过后写 RTL”
   的队列项。

### P2

1. 若未来重开，必须作为**全新算法研究身份**：新 checkpoint/catalog、
   train-only 选择、预注册固定 tau/layer policy、未被本轮查看过的独立
   holdout，并重跑 baseline、accuracy、cycle、PPA 全链；不能继承本轮数字。
2. 唯一不属于有损重开的语义例外，是对某个 `distance>0` case 完整证明
   omitted correction 恒为零，并以 bit-exact miter 封口；该机制应改名为
   exact residual-zero gating，而不是 near-match approximation。

## Allowed claims

- M307 是四个隔离 Conv 的 frozen-trace/S10 screen；selected tau1 policy
  在十帧上 `Delta AEE=0.014362`，但没有 valid825、RTL、PPA 或系统准入。
- M313r2 的 paired valid825 primary `Delta AEE=0.029327969623334393`，
  失败于预注册 `0.02` 门；当前 M87 PAFT ep4 / M77 catalog / enable023
  positive-distance path 已关闭。
- `245,630,707` 个 positive-distance replacements 被真实执行；失败不是
  exact-only 空跑造成。
- `tau=0` exact 子集不受该 accuracy failure 影响。
- M486 的 tau3 数字至多是未准入 frozen-trace opportunity，可作为负结果
  或 motivation，不是论文性能点。

## Forbidden claims

- “tau3 尚未测，所以还是开放候选”或“可以继续试更大 tau/换 layer subset”。
- 用 M486 的 `1.5948x` 作为 accuracy-admitted、RTL-admitted、全网或 headline
  speedup。
- 把十帧 M307 的 `0.014362` 当作 valid825 精度证明。
- 声称误差对 tau 数学单调、因此 tau3 必然更差；现有证据是预注册的
  procedural stop，不是单调性定理。
- 声称 M313r2 永久否定所有未来 checkpoint、所有 positive-distance 算法，
  或否定 tau0 exact/M479/C2。
- 把 total snapped `476,535,897` 全部称作有损 population；其中
  `230,905,190` 是 distance-zero exact hits，真正 positive-distance 数是
  `245,630,707`。

## 证据边界

本评审只读核对 M307、M312r2、M313r2 的合同、receipt、profile、逐帧 CSV
和 seal；没有运行 GPU、VCS 或 DC，没有修改生产 RTL，也没有修改
`docs/359`。冻结文件 SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
