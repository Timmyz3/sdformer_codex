# M569｜M548 Prosperity/Phi-style waterfall independent hammer

日期：2026-08-28  
模式：fresh independent、严格只读、zero EDA/VCS/runner/remote  
结论：**FAIL_ONE_P1__R1_DO_NOT_SHARE__MINIMAL_R2_REQUIRED**  
评分：**94/100；P0/P1/P2 = 0/1/0**

## 裁决

M548 r1 的两个核心性能台阶、全部整数 savings/overhead、作用域和 claim
边界均通过独立复算；仓库中也没有发现把 `2.038776x` 冒充 cycle 或 system
speedup 的 paper-facing 表述。但是第三个 waterfall 行的容量利用率小数写错：

`213376 / 245760 = 0.868229166666...`，不是合同中的
`0.868131510417`。

这个错误不改变 `213,376 B < 240 KiB`、容量余量 `32,384 B` 和本地机制的
性能判断，但违反了合同自己的 exact-waterfall 要求。因此 r1 不可分享；只能由
新的 r2 身份机械修正该小数并再次接受 independent hammer，不能原位改写 r1。

## 冻结来源与严格 JSON

- 被审合同 SHA256：
  `ccaf1a4bd02b8ab416535133accfda427280f36f1e2583590b233e259635c402`。
- M528 result SHA256：
  `778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1`，
  与 M548 冻结值一致。
- M528 `SHA256SUMS` SHA256：
  `4556a3383507e81ad9883f59bb55bb3d4fd08e7ec03977b215108b5ce4565073`，
  内层 manifest 和外层 seal 均逐项通过。
- M535 prior hammer `review.json` SHA256：
  `cd169aef03e6420287e3dda4ef8c7f833155ba9aa45b6902490980b29edbd8a0`，
  其 review manifest 和外层 seal 均通过。
- M548、M528 result 和 M535 review 均用 duplicate-key-rejecting parser 独立解析，
  三者无重复 key。

## 独立整数复算

| 项 | 独立复算 | 裁决 |
|---|---:|---|
| 八个 output block 的 bit issues | `92,640,472 * 8 = 741,123,776` | PASS |
| product arithmetic issues | `363,513,992` | 与冻结 M528 PASS |
| arithmetic-work reduction | `741,123,776 / 363,513,992 = 2.038776477137639...x` | 合同 12 位舍入 PASS；只能叫 arithmetic-work reduction |
| executable local cycle speedup | `757,946,784 / 435,293,339 = 1.741232213066325...x` | 合同 12 位舍入 PASS；只能叫 four-Conv exact CPU-model cycle speedup |
| bit non-arithmetic cycles | `757,946,784 - 741,123,776 = 16,823,008` | PASS |
| candidate non-arithmetic cycles | `435,293,339 - 363,513,992 = 71,779,347` | PASS |
| extra non-arithmetic tax | `71,779,347 - 16,823,008 = 54,956,339` | PASS |
| arithmetic issues eliminated | `741,123,776 - 363,513,992 = 377,609,784` | PASS |
| net cycles eliminated | `757,946,784 - 435,293,339 = 322,653,445` | PASS |
| waterfall conservation | `377,609,784 - 54,956,339 = 322,653,445` | PASS |
| macro-rounded capacity | `213,376 / 245,760 = 0.868229166666...` | **FAIL：r1 写为 0.868131510417** |
| capacity margin | `245,760 - 213,376 = 32,384 B`（预算的 `13.177083%`） | PASS |

为帮助后续论文 waterfall 解释，独立派生但不作为新 headline 的量为：算术 issue
减少 `50.950974%`，最终 cycle 减少 `42.569406%`；单口/调度/commit 的额外税
消耗了算术 savings 的 `14.553738%`，因此捕获了 `85.446262%` 的算术机会。

## Scope 与 claim-policy 审计

M548 与冻结 M528 对齐：H67 ep35、一个 sequence、十个冻结样本、四个 bottleneck
Conv3x3、`51,840,000` trace rows、八个 output blocks、240 KiB budget。M528 原始
claim boundary 的 `rtl/vcs/synopsys_ppa/energy/system_speedup/date_headline` 均为
false；M548 继续冻结这些 false，并明确禁止把 `2.0388x` 称 cycle、把
`1.7412x` 称 full-network、把局部或外部倍率相乘、把 M472 归为 ours。

paper-facing 全库扫描只发现：

- M548 自身将 `2.038776x` 正确标为 arithmetic-work reduction；
- `docs/524` 两处都将其正确标为 arithmetic-work reduction，并明确禁止称为
  cycle/system speedup；
- 没有发现反向误标。

因此性能分母没有混用，唯一阻断项是容量 ratio 的算术错误。

## 文献方法与 novelty 边界

Prosperity 原论文把 product-density/计算机会、bit-sparsity 消融和最终架构周期结果
分层，并在 evaluation 中另报面积/功耗；Phi 也把理论 pattern opportunity、exact
architecture 结果与 PAFT 有损增量分开，并显式计入 buffer/DRAM 建模。M548 借用的
只是这种 evaluation structure，合同已经给出论文/官方 artifact 链接，且明确写明
`no first/novel claim`。这不是把 Prosperity/Phi 机制改名为本项目 novelty。

本项目允许的对象边界仍仅是 signed H67 source rows、dead-write-only single-port
parent capture 和 240 KiB 资源约束；在 RTL/VCS/Synopsys PPA、memory-inclusive
energy 与 decoder-complete system 结果闭合前，`1.7412x` 不得进入系统 headline。

## P1 与最小修复

**M569-P1-01｜Physical-capacity ratio arithmetic error**

- 位置：M548 r1 `paper_waterfall_rows[2].ratio`。
- 当前值：`0.868131510417`。
- 正确值：`0.868229166667`（12 位小数），由冻结整数
  `213376 / 245760` 得到。
- 影响：容量结论不变，但 exact waterfall 机器行不可审计一致，r1 不能
  shareable。
- 最小修复：创建不可变 r2，仅改 ratio、更新 contract id/date/status/自身 seal，
  其余整数、scope、claim flags 和文献边界不得变化；然后由 fresh reviewer 复核。

本审阅没有运行 EDA、VCS、runner、训练、大型 CPU 任务或远端命令，也没有修改
M548 r1、M528 result、M535 prior review、`docs/524` 或 `docs/359`。
`docs/359_DATE终局冻结_20260813.md` SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
