# DATE 第六轮独立 Architecture 复审

## 1. Recommendation

独立 architecture agent 在读取最新 `report.json/report.md`、模型脚本和
`docs/173-176` 后给出：

```text
研究继续：GO
架构冻结：有条件 GO
performance/PPA sign-off：NO-GO
DATE 投稿：Weak Reject
```

该结论接受。数值冲突时，以
`results/phi_prosperity_dual_line_sim_20260729/report.json` 的 v2 schema
和 `docs/176` 为准。

---

## 2. 三档候选

| 候选 | 结构 | 结论 |
|---|---|---|
| C0 保守 | Direct32 + exact fallback | 永久公平基线 |
| C1 均衡 | 共享 TARE-4 W4；Motion fixed64/M4；Local5 守恒 destination；stage bypass | 唯一推荐候选 |
| C2 激进 | C1 + ISHD/PPDI/Phi-like | Motion ISHD 淘汰；Local5 ISHD 仅保留探针 |

禁止把 W8、ISHD、PPDI 和 Phi 假设命中率相乘后形成“组合加速”。

---

## 3. 双线唯一 RTL 机制

### Motion

```text
temporal-pair static anchor
-> exact residual
-> dense direct fallback
-> SCS/term
-> fixed64/M4 exact multicast
```

stage-conditioned bypass 是静态配置，不做有损裁剪。ISHD、PPDI 和 TTB4
目前均不冻结为 Motion 主机制。

### Local5

```text
self anchor
-> four-direction exact residual
-> boundary-aware stencil schedule
-> dense direct fallback
-> Shiftmax5/MFEP
-> multiplicity-preserving destination
```

最重要的实现缺口是：

> `local5_row_context_tare_engine` 已通过叶/行级验证，但 Local5 默认
> `score_gate_term/window` 主链仍使用 direct row engine。

因此本轮之后第一个 RTL 整改不是再造新编码，而是把 TARE 作为参数化默认路径
接进相同顶层，同时保留 direct 模式做公平消融。

---

## 4. 新颖性判断

唯一可能达到 DATE 架构层、但尚未完成的抽象是：

> **把算法给定的时间/空间关系提前 lower 成 anchor-target descriptor，
> 将在线关系发现改写为 ZERO/LIST4/DIRECT 三路径 exact residual dataflow，
> 并由同一物理执行核处理 temporal-pair 和 spatial-stencil。**

它的价值来自执行边界变化：

1. relation 在软件/descriptor 端已知，不运行 matcher；
2. anchor 驻留，target 只传 exact residual；
3. ZERO、bounded sparse、dense fallback 共享提交次序；
4. 双线复用同一执行 substrate，而不是复制异构核。

当前仍只能称为共享算子微架构，因为：

- Motion 完整 `PAIR_SCORE` ordered row 未闭环；
- Local5 TARE 未进入默认窗口链；
- 没有统一 lowering/descriptor ISA；
- 没有 full encoder 与同约束 PPA。

ISHD、PPDI、TTB/STT、双 context、3-bank 均不能单独列为 DATE 主贡献。

---

## 5. 下一轮淘汰门槛

| 项 | 门槛 |
|---|---|
| 数值正确性 | T450 hardware-order、mask、score/term/destination/final 全部 0 mismatch |
| 模型校准 | 每 stage 与总周期模型/RTL误差 `<10%`，目标 `<5%` |
| 子系统性能 | raw `>=1.15×`；lane 归一 `>=1.15×` |
| 尾延迟 | ordered p99/mean `<=1.25` |
| 系统收益 | full encoder Amdahl `>=1.10×` |
| Motion fixed M4 | RTL lane 归一低于 `1.15×` 则淘汰 TARE 主线 |
| Local5 ISHD | delta6 escape `<=1%`，含 fallback 后 fabric bits `>=50%` 降低 |
| PPDI | exact command `>=15%` 降低且 EDP/面积归一吞吐 `>=10%` 改善 |
| Phi-like | exact、train/test 分离，EDP `>=15%` 改善且面积增量 `<=5%` |
| PPA | 同 SRAM/SDC/PVT，Fmax 留 `>=10%` 裕量，主机制 EDP `>=15%` |

---

## 6. Sign-off 缺口

### Performance

- Local5 无 post-G0 ordered FIFO；
- Motion projection 尚无 profile100 cycle-accurate RTL；
- T450 仍是外推；
- 模型未校准真实 pipeline overlap；
- 无 full encoder Amdahl/FPS。

### PPA

- 绝对面积/功耗预算未冻结；
- node/PVT/SRAM macro/SDC 未冻结；
- 无 STA/SAIF/leakage/clock-tree；
- `500 MHz` 仅是换算频率；
- lane 归一不是面积归一。

### 验证与交付

- T450 `DEST_W=9`、halo、line buffer 和 accumulator 深度未回归；
- 无随机 SRAM latency 与 full final backpressure 分布；
- CDC 为单时钟 N/A，但 reset/DFT/memory map 未签核；
- 无 target-library LEC；
- High 风险尚未绑定 owner。

---

## 7. 评审驱动的立即动作

1. 参数化接入 Local5 TARE 默认主链；
2. 同 stimulus 跑 direct 与 TARE，逐 score/gate/term/destination 比较；
3. 增加 ZERO/SPARSE/DENSE、invalid mask、随机反压和 active reset；
4. 增加 `DEST_W=9/T450` 编译与边界用例；
5. 只有该同顶层消融通过，才继续 Local5 ISHD；
6. Motion 继续 fixed M4，等待修正后 profile 决定 PPDI。

下一次 DATE 复审必须基于这组整改后的实际 RTL 证据，而不是新命名。
