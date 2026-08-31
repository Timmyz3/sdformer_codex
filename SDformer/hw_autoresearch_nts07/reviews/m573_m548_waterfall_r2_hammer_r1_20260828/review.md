# M573｜M548 waterfall r2 fresh independent hammer

日期：2026-08-28  
模式：fresh independent、严格只读、zero EDA/VCS/runner/training/remote/large CPU  
结论：**PASS__R2_SHAREABLE_WITH_FROZEN_SCOPE_AND_CLAIM_BOUNDARIES**  
评分：**100/100；P0/P1/P2 = 0/0/0**

## 裁决

M571/M548 r2 已把 M569 指出的唯一 P1 机械修正为：

```text
213376 / 245760 = 0.868229166666...
12 位小数 = 0.868229166667
```

r1 保持原文件和 SHA，继续按 M569 裁决为 `DO_NOT_SHARE`。结构化归一化比较确认，
r2 相对 r1 只有合同 identity、date、status、新增 M569 repair provenance，以及上述
容量 ratio 修复；其余整数、两个性能倍率、scope、claim policy 和 literature boundary
完全一致。

因此 r2 可作为论文局部 waterfall 分享，但只允许三层分母各自独立表述：

- `2.038776477138x`：same-coordinate bit 相对 product 的 **arithmetic-work reduction only**；
- `1.741232213066x`：四个 bottleneck Conv、H67 ep35、单序列、十样本的
  **exact CPU-model local cycle speedup only**；
- `0.868229166667`：`213,376 B / 245,760 B` 的 modeled macro-rounded capacity ratio。

本审阅不准入 RTL、VCS、Synopsys PPA、energy、decoder-complete/full-network/system
speedup 或 DATE headline，也不允许将三个 ratio 相乘或把外部 M472 `2.459487x`
改写为本项目结果。

## 身份、严格 JSON 与双封

- r2 contract SHA256：
  `eb67b5a6c84121b4f650bf7f60178bd7e14c9d07f5a52e615454070862070901`；成员
  sidecar SHA256 `71dddfd590f1e237775df0cca97c38bfc313385dc127e825942008e2fed6a370`；
  外层 seal 文件 SHA256 `996c58937da74e3ac58bbcc9b7df9492b441e520df9349ea01dc9817c4e59b00`。
- immutable r1 SHA256：
  `ccaf1a4bd02b8ab416535133accfda427280f36f1e2583590b233e259635c402`。
- M569 `review.json/review.md/SHA256SUMS/outer-seal-file` SHA256 分别为
  `916ca63d78125f2379fbf67a246f45c0a833e2f9c8a99b1c113bc5c2cf826fcd`、
  `c696c0722c2e407b95f8f09ac9b43d8ffdb58ddc1d961bb5badf1d641b837c75`、
  `e7b163406f0f62ec9b7c01c1b9417fa088ede92b44cd76c691ce9fde63cf692e`、
  `4b01b8cfc3c79c5e81a0672d91f010b2fee832e313351a024038838deb464f95`；
  成员和外层 seal 逐项通过。
- author handoff `json/md/manifest/outer-seal-file` SHA256 分别为
  `d13e6dd7d492f375ee296d58fe2e04e5fae1eda8bce6d80c89a2a38268bd821b`、
  `2453680072b87f8e16ec53d9bcc424261b6df470a724bb3c7f66ae4360b9c7f3`、
  `b32831054e141787009bd7f59ac75800f151e6905dd4eeb8124dc7b5aacc561d`、
  `6c74888b068c6da60f2afa130dd248d2dd493b41f63f8ace04e152985f2651be`；
  成员和外层 seal 逐项通过。
- M528 result SHA256：
  `778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1`；
  M535 prior review SHA256：
  `cd169aef03e6420287e3dda4ef8c7f833155ba9aa45b6902490980b29edbd8a0`。
- r1、r2、M569、M528、M535、request 和 handoff 均使用 duplicate-key-rejecting
  parser 解析，重复 key 数均为 0。

## 独立算术复核

| 项 | 独立复算 | 裁决 |
|---|---:|---|
| 八个 output block 的 bit issues | `92,640,472 * 8 = 741,123,776` | PASS |
| arithmetic-work reduction | `741,123,776 / 363,513,992 = 2.038776477137639...x` | 12 位舍入 PASS；work only |
| local cycle speedup | `757,946,784 / 435,293,339 = 1.741232213066325...x` | 12 位舍入 PASS；four-Conv CPU cycle only |
| bit non-arithmetic cycles | `757,946,784 - 741,123,776 = 16,823,008` | PASS |
| candidate non-arithmetic cycles | `435,293,339 - 363,513,992 = 71,779,347` | PASS |
| extra non-arithmetic tax | `71,779,347 - 16,823,008 = 54,956,339` | PASS |
| arithmetic issues eliminated | `741,123,776 - 363,513,992 = 377,609,784` | PASS |
| net cycles eliminated | `757,946,784 - 435,293,339 = 322,653,445` | PASS |
| waterfall conservation | `377,609,784 - 54,956,339 = 322,653,445` | PASS |
| capacity ratio | `213,376 / 245,760 = 0.868229166666...` | r2 12 位舍入 PASS |
| capacity margin | `245,760 - 213,376 = 32,384 B` | PASS |

所有冻结整数均与 M528 result 一致；M528 的
`rtl/vcs/synopsys_ppa/energy/system_speedup/date_headline` 仍全为 false。

## 差分、仓库误标与文献边界

对 r1/r2 去除允许变化的 identity/date/status/repair provenance，并把唯一容量 ratio
归一化后，结构化对象完全相等。r2 已精确绑定 M569 的 JSON、Markdown、成员 manifest
和外层 seal；r1 精确 SHA 未变化。

仓库 paper-facing 扫描没有发现把 `2.038776x` 写成 cycle/system speedup，或把
`1.741232x` 写成 full-network/system speedup。`docs/524` 保持正确的 work/local-cycle
分列；r1 的错误 ratio 只保留在 immutable r1、M569 诊断与 repair provenance 中，均明确
标记不可分享或为修复历史。

Prosperity/Phi 映射与 r1 完全冻结：只借用并引用 opportunity/work、architecture-cycle
和 memory-cost 分层的 evaluation structure。H67 的 claim 仍只限 signed source rows、
dead-write-only single-port parent capture 与 240 KiB resource boundary；没有 first/novel
或外部结果归属升级。

## 冻结与零执行

本审阅没有运行 EDA、VCS、runner、训练、远端或大型 CPU 任务，没有创建 result、
attempt 或 launch admission，也没有修改 r1、r2、M569、M528、M535、`docs/524` 或
`docs/359`。`docs/359_DATE终局冻结_20260813.md` SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 缺陷计数

- P0：0
- P1：0
- P2：0

最终裁决：**M571/M548 r2 在既有 scope 与 claim 边界内 shareable；不授权任何新的
执行、物理、能量或系统 headline。**
