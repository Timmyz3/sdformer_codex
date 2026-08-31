# Motion/H67 全网覆盖与 RQTB 共存独立审计 r1

## 结论先行

当前证据足以定位热点，但不足以声称全网实测周期、带宽或系统加速。按现有 96-lane activity-weighted RQTB 包络，H67 的周期热点是 Conv3x3（42.01%）、其他算子（32.12%，其中 FFN expand+contract 为 25.76%）和 ATLIF（20.64%）。attention（含 Q/K）只有 5.19%，预测头只有 0.044%。因此只继续压 RQTB 核心或预测头都无法成为系统头条。

RQTB 与 M147/M148/M149 在功能域上可以共存：RQTB 服务 attention core，M147 链针对四层 bottleneck Conv3x3，网络算子范围不重叠。但物理共存尚未成立，SRAM 端口、PWP1024 带宽、M149 向 accumulator 的多组写回、跨模块元数据桥和有限缓冲调度都没有闭合。

独立评分为 **6.4/10**：热点与身份可信，系统周期和流量证据仍弱，不能作为 DATE 性能表或摘要数字。

## 五类覆盖

十样本 ordered trace 共 1,840 条 execution record，分类如下：

| 类别 | 动态 record | 当前 RQTB 周期模型/帧 | 周期占比 | 假设该类完全免费时的 Amdahl 上限 |
|---|---:|---:|---:|---:|
| Conv3x3 | 110 | 260,619,577 | 42.0149% | 1.7246× |
| ATLIF | 930 | 128,020,500 | 20.6384% | 1.2601× |
| attention/RQTB | 480（360 个 Q/K/proj + 120 个 attention core） | 32,162,811 | 5.1850% | 1.0547× |
| prediction head | 40 | 271,156 | 0.0437% | 1.00044× |
| 其他算子 | 280 | 199,228,861 | 32.1180% | 1.4731× |
| 合计 | 1,840 | 620,302,905 | 100% | — |

这里的“精确”只表示五类之和与当前 `rqtb_total` 包络逐周期项相等。operator 周期来自 activity-weighted lane model；attention core 是十样本中每 block 一个 T450 window 的 VCS 均值按 stage window 数扩展；它们不是全网 RTL wall-cycle 测量。`system_summary.claim_boundary` 仍写着 sample0，而同文件的详细 attention 对象和 VCS receipt 都是十样本；本审计采用后两者，并将这处身份文本不一致记为 P2。

RQTB attention core 本身从 3,656,069 降至 3,090,731 cycles/frame，局部 1.1829×，放入同一包络后只得到 **1.00091×**。不得把这个数字与 M147 或其他局部倍率相乘。

## Source-work 口径

profile100 的 79 个 Conv/Linear 活动 product term 可以精确分组，但不能与 ATLIF temporal MAC、attention pair evaluation 混成一个五类百分比：三者服务单位不同。

| Conv/Linear 类别 | 活动 product term/帧 | 该 operator-only 账本占比 |
|---|---:|---:|
| Conv3x3 | 25,019,478,953 | 52.5844% |
| attention Q/K/proj | 3,408,196,400 | 7.1631% |
| prediction head | 26,030,743 | 0.0547% |
| 其他算子 | 19,125,969,583 | 40.1978% |
| 合计 | 47,579,675,679 | 100% |

补充但不可直接相加的单位：ATLIF dense arithmetic 是 12,289,968,000 MAC/frame；ordered s10 的显式 attention 是 15,120,000 pairs，即 1,512,000 pairs/frame。十样本 selector 只覆盖二值合格算子，selected work 为 310,640,413,576，其中 Conv3x3 54.2870%、其他算子 45.6793%、预测头 0.0337%；attention 因 temporal axis 不合格而未进入该分母。

## 流量证据

当前只有“每个 operator 输入输出均落地”的 INT8 materialize-all 上界代理：

| Conv/Linear 类别 | 激活字节/帧代理 | operator-only 占比 | 唯一 INT8 权重字节 |
|---|---:|---:|---:|
| Conv3x3 | 1,075,200,000 | 38.6777% | 21,690,720 |
| attention Q/K/proj | 580,608,000 | 20.8860% | 6,469,632 |
| prediction head | 108,024,000 | 3.8859% | 1,536 |
| 其他算子 | 1,016,064,000 | 36.5504% | 18,809,856 |
| 合计 | 2,779,896,000 | 100% | 46,971,744 |

这个分母包含后来由 attention RTL anchor 取代的 `attn.proj`，所以只能用作 profile materialization 上界，不能当当前执行流量。ATLIF 另有 2,922,480,000 bytes/frame 的完整时序输出 payload，但它与网络中间张量重叠，且不代表 SRAM read/write。RQTB 内部 K/slot/SCS、M147 descriptor/PWP 和 DRAM 都没有 address-timed transaction。因此本审计明确拒绝给出五类真实 SRAM/DRAM 流量占比。

## RQTB 与 M147/M148/M149 共存

功能关系是“图上分层、物理资源可能复用”：

- RQTB 改 attention pair service；M147 的 held-out 账本只覆盖四个 bottleneck Conv3x3。算法结果域不重叠，顺序执行时可以时间复用硬件。
- M147 的 1.8054× 只是相对 M143r2 的 held-out ideal opportunity。若没有 same-destination combine，回放是 0.9877×，即反而略慢。
- M148 已证明 presence tuple 守恒，logic-only DC 为 2,183.96 µm²、3 ns setup slack +0.6266 ns；但没有 sign/negate、算术、macro 或 commit。
- M149 已有 exact-SHA VCS seal，72 descriptor、246 tuple、156 output group、90 combined tuple 和 2 次协议攻击通过；它能够 signed-negate 并合并同 destination。但审计封存时 DC 仍未完成/封存、M148 还未与它集成，并明确假设四个 96-lane vector 已经可用。

五个 P1 共存缺口：

1. M149 满 descriptor 输入是四个 96×INT8 vector，即 3,072 payload bits；单个 PWP1024 每拍只有 1,024 bits。M149 单岛 II=1 不能推出端到端 II=1。
2. M149 最多输出四组 96×11-bit contribution，accumulator bank/writeback/冲突代价未实现。
3. M148 不带 negate，而 M149 需要 negate 和 vector；两者之间缺少带 sequence/context 的 typed bridge。
4. RQTB 的 slot/K-store 与 M147 的 descriptor/PWP/destination state 若共享 SRAM，没有端口仲裁；若独立部署，则面积与功耗未计价。
5. RQTB window retirement 与 M147 sequence/row/partition barrier 没有统一有限 FIFO 调度，未证明 deadlock、starvation、reset/context isolation。

因此结论是：**功能独立，物理共存未 admission**。最短闭环不是大而全的系统 scheduler，而是一个有界 adapter replay：`M148 tuple + sign bridge -> PWP1024 finite service -> M149 combine -> banked accumulator commit`，同时把 RQTB 作为另一种 typed transaction 放进同一 SRAM 端口仲裁器，测真实 stall recurrence。

## 最多两个非 Conv 优化建议

1. **ATLIF 可训练低秩 temporal kernel + resident late-scale/requant**。当前周期占比 20.64%，完全移除的上限是 1.2601×。已有 M26 rank2 arithmetic lower bound 可把 ATLIF issue 从 128,020,500 降到 64,594,800；只在该候选最终被训练、数值和端口闭合时，单独替换当前 ATLIF 项的系统包络上限是 1.1139×。创新性 8.0/10，实现风险高。硬件应反哺算法侧约束 rank2、q24-to-q8 舍入/饱和和 tile-resident intermediate，而不是先写无训练锚点的更多 RTL。
2. **FFN expand-contract event-resident fusion + structured intermediate suppression**。两层合计 159,784,111 cycles/frame，占 25.76%，完全免费上限 1.3470×；materialize-all 中间张量 write+read 代理为 700,416,000 bytes/frame。创新性 6.5/10，实现风险中高。单纯 fusion 不能宣称省掉计算周期；需要 address-timed memory model 证明流量收益，若要同时减算术则必须让算法侧训练结构化 intermediate sparsity/低秩约束。

不建议再投资源给预测头：即使完全免费也只有 1.00044×。不建议把 RQTB core 当下一性能主线：core 即使完全免费也约 1.0050×，attention 含 Q/K 全部免费也仅 1.0547×。

## 评分与缺口

| 维度 | 分数/10 |
|---|---:|
| 数据身份 | 9.0 |
| 算子覆盖 | 9.0 |
| 周期证据 | 5.0 |
| 流量证据 | 3.0 |
| 共存集成 | 4.0 |
| 声明卫生 | 10.0 |
| 综合 | **6.4** |

- P0：0。
- P1：5，即上面的 SRAM/带宽、vector delivery、writeback、scheduler、metadata/interface 缺口。
- P2：3，分别是 summary 内 sample0/十样本身份文本不一致、s10 ordered trace 与 profile100 ledger 不是同一总体，以及 source-work 三种单位不可直接相加。

结论等级：`PASS_COVERAGE_QUANTIFIED_COEXISTENCE_NOT_ADMITTED`。没有修改 production，也没有修改 `docs/359`；其审计时 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复跑

```bash
python3 results/motion_fullnetwork_rqtb_coexistence_review_r1_20260824/build_review.py
sha256sum -c results/motion_fullnetwork_rqtb_coexistence_review_r1_20260824/source_manifest.sha256
sha256sum -c results/motion_fullnetwork_rqtb_coexistence_review_r1_20260824/manifest.sha256
```

机器可读详情见 `motion_fullnetwork_rqtb_coexistence_review.json`。
