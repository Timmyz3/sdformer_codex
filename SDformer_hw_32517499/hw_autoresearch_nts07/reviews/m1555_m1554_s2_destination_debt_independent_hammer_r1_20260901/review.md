# M1555｜M1554 S2 destination-owned debt 独立打铁

评审对象：提交 `16272e3ce4a0f20ec83b5e44ca62fe26b4938907` 的 M1554 retained-decoder CPU screen
裁决：**M1554 已修复 source-site debt reset；条件授权一次紧凑 FC/patch capture，但当前 reference 必须在任何 S2 AEE、性能或 RTL 准入前修复。**

## 1. 可确认的正结果

- 独立 hammer 没有 import 作者分析器，从冻结 checkpoint 与 M1521 bitpack 重建了 `3 sequence × 3 sample × 4 decoder layer = 36 calls`，每 call 64 个 destination，共 2304 个 destination observation。
- 对四层完整 production geometry 穷举验证了 `K3/S2/P1/OP1`：`o = 2i - 1 + k`，每个 destination 恰有 1、2 或 4 个合法 contributor。另用非对称 tap/channel 张量逐点对上 PyTorch `conv_transpose2d`，kernel-flip 攻击能被区分。
- 权重 layout 确认为 `[Cin,Cout,Ky,Kx]`；tail group、tail output tile、little-endian bitpack、tap 索引、边界 destination 均通过。
- M1554 的 debt 现在确实由 `destination × output_tile` 持有；每次决策前，`A(G)` 已累计该 destination 的全部合法 source site 与 tap。抽样 exact contribution 全部满足 `|exact| <= M(G,O)A(G)`，0 次 bound violation，positive-bound/exact-zero collision 为 0。
- 作者生产结果可逐 byte 重跑，SHA256 同为 `3763bbe9...c88d85`。作者 synthetic test 在当前 Python 与 CPython 3.6 都 PASS；独立 hammer 也通过 CPython 3.6 语法编译。

## 2. 三个 block 点的独立复算

| G×O | metadata / 假设 INT8 权重 | 相对旧 G11 压缩 | bound/exact median / p90 | 裁决 |
|---|---:|---:|---:|---|
| 8×16 | 0.1741% | **7.9768×** | 2.899 / 6.433 | 严格低于 8×，FAIL |
| 16×16 | **0.0874%** | **15.8923×** | **3.880 / 8.317** | 作者门通过，但只承认 metadata + dynamic witness 存活 |
| 32×16 | 0.0440% | 31.5420× | **5.073** / 11.205 | median 超过 4，FAIL |

16×16 的 1232/3120 个静态 block 在不同 observation 中同时出现 keep/drop，说明它不是静态 mask；这足以保留 compact capture 资格，但不是性能证据。

## 3. reference 的第一性原理审查

作者在 `analyze_m1554...py:169-176` 使用：

`reference = layer_max_abs_weight × Cin × spatial_contributor_count`。

它在数学上是合法的**绝对最坏容量上界**：每个 source/tap/channel 都 active 且都命中全层最大权重时，输出误差不会超过它；独立复算也确认所有 active block-bound mass 都不超过该 reference。

但它不适合作为相对输出、状态或 AEE 预算。16×16 下，真实 destination 的 total active bound mass 只占该容量 reference：

- median：5.3165%；
- p90：9.3621%；
- max：19.9567%。

因此 `epsilon=0.1` 时得到 99.2146% drop，主要是在说“多数 observation 的全部 active-bound mass 小于假想 dense 最大容量的 10%”。把同一批 block 改用 `sum_G M(G,O)A(G)` 作为每个 destination 的 activity-relative safe reference，drop 立即从 **99.21% 降到 29.68%**。

所以 99.2% 不能写成可执行 pruning、周期、流量、能量或精度机会，也不能据此开 RTL。它只保留为 global-capacity diagnostic。

## 4. 最终裁决与下一步

**条件允许 incremental FC/patch capture**，因为：

1. destination debt-domain bug 已关闭；
2. 16×16 metadata 为假设 INT8 权重容量的 0.0874%，相对旧 G11 小 15.89×；
3. 动态 keep/drop witness 真实存在；
4. capture 是紧凑数据获取，不是 AEE 或性能实验。

条件是 capture 必须保留 group magnitude / fixed-point code 与 weight-block identity，使后续能同时计算 global-capacity 和 activity-relative safe reference。**M1554 的 99.2% 不作为 capture 成功门，更不作为 S2 准入数字。**

共享 capture 到位后的顺序：先闭合无损 TSBG 的 exact same-resource fast-kill；随后仅对 S2 16×16 用修复 reference 重筛。重筛前不授权 paired AEE、周期/流量/能量、RTL/VCS/EDA 或论文 headline。

## 5. Claim boundary

- 本评审没有生成 AEE、accuracy、cycle、traffic、energy、speedup 或 system speedup；
- 没有开发或授权 RTL、VCS、DC/PTPX；
- 所有 metadata 容量仍基于“FP32 权重求 bound、假设 INT8 权重作容量分母”，不是量化硬件 authority；
- `decision_before_weight_fetch` 仍是结构假设，尚未由地址计时重放测得。

机器可读复算见 `independent_recompute.json`；独立实现见 `hammer_m1554_destination_debt.py`。
