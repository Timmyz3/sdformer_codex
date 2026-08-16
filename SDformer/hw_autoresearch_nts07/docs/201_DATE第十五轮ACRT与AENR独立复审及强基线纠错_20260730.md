# DATE 第十五轮 ACRT/AENR 独立复审与强基线纠错

## 1. 复审评分

| 审稿角色 | Recommendation | 综合分 | 新颖性 |
|---|---|---:|---:|
| DATE审稿人 | Weak Reject | 2.6/5 | 3.0/5 |
| 架构/电路审稿人 | Weak Reject / Borderline | 3.1/5 | 3.7/5 |

共同结论：

> ACRT 比 NC-FIP 更像真实跨阶段架构，但仍只是值得证伪的候选。其核心只能
> 合并写成一个端到端机制，不能把 histogram、relation transduction 和
> banking 拆成三项独立贡献。

---

## 2. 审稿人认可

1. all-class histogram 在整数加法、相同 exp LUT、相同 row max 和相同 gate
   RNE 合同下，可精确替代 active-token sum；
2. T450 最坏 `450×65535` 不溢出32-bit row sum；
3. class-to-gate relation transduction 不恢复 token stream，确实改变
   normalization/projection 边界；
4. segment-local C→G fold 与 segment-major 1G+4K 有可实现的物理组织；
5. ACQN 40,020 checks 和 FCIP Acc reference 为算法分解提供了基础证据。

---

## 3. P0

### 3.1 缺少强中间基线

旧 `1.37x` 同时包含：

- all-class normalization；
- 去 active replay；
- W4/segment banking；
- relation transduction。

必须比较：

```text
B1: all-class denominator + W1 ordered member replay + G1
B2: all-class denominator + W4 ordered member replay + G1
```

本轮已补。

### 3.2 LOAD端口未计

W4 LOAD 可能同拍：

- 更新四个不同 class count；
- 更新同segment内四个不同 C class word；
- 更新32个 K-lane bank的多bit mask；
- 更新occupancy、active count和prefix。

普通1R1W SRAM不能直接满足。首个 RTL 必须 W1，W4 另做冲突合并和端口 PPA。

### 3.3 AENR阈值

评审读取的是修改前快照，指出 gated nonzero event 属于未来信息。当前脚本已
修正为 LOAD 期可知的原始 `active_lane_events`，但仍有样本过拟合：

- 只有45行；
- E20/E32同结果；
- 同一数据选择阈值并报告收益；
- fullres可能有gate-zero和不同event分布。

因此 AENR 不进入首版 RTL。

### 3.4 端到端整数链

ACQN 和 FCIP reference 是两段证明，尚不是单一逐拍链：

```text
hist -> denominator -> gate -> C→G -> G∩K -> term -> Acc
```

还需 burst backpressure、overflow、epoch 和 row-last。

---

## 4. 强基线结果

| ready | all-class replay W1 | all-class replay W4 | ACRT | ACRT/W1 | ACRT/W4 |
|---:|---:|---:|---:|---:|---:|
| 100% | 49.24 | 31.89 | 51.82 | 0.950x | 0.615x |
| 90% | 51.18 | 33.78 | 53.64 | 0.954x | 0.630x |
| 75% | 54.82 | 37.38 | 56.73 | 0.966x | 0.659x |

结论：

> ACRT 当前不是吞吐候选。它相对 W1 强基线慢约3.5–5%，相对理想 W4 明显
> 更慢；唯一可能的晋级理由是关系状态、宽总线活动和 PPA 显著更低。

关系 payload：

- B1 G4×L×T目录：20,736 bit；
- ACRT C16+G4+K32+context4：9,072 bit；
- AENR E20合计：10,052 bit。

该容量优势尚未转化为面积或能耗。

---

## 5. 架构冻结

ACRT 保留为：

> **低状态、局部端口的 normalization-to-projection PPA 候选。**

不再宣称：

- 相对最佳架构加速1.37x；
- AENR加速1.42x；
- zero-cost class fold；
- W4同物理成本；
- 三项独立DATE贡献。

只有同宏综合后同时满足以下条件才晋级：

1. 面积至少降低20%，或总面积不增且功耗明显下降；
2. EDP改善至少15%；
3. Fmax下降不超过5%；
4. aggregate throughput相对W1强基线不低于0.95x；
5. 多样本p99 slowdown低于1.10x；
6. relation traffic与wide fanout能耗有SAIF证据。

否则 ACRT 作为负结果归档。

---

## 6. 下一方向

Motion ACRT 已到“必须靠PPA而非继续解析模型”的阶段。架构创新下一轮优先
转向 Local5 固定五点拓扑：

- 固定3×3邻域和五方向关系；
- producer/consumer距离有界；
- 可用三行甚至方向分离line buffer；
- 关系生命周期可在wavefront前沿退休；
- 有机会真正减少T450全窗口状态和片上搬运。

Local5 仍需等待fullres exact/profile完成；在此之前先审计现有Local5 RTL与
Grok补充结果，建立同样的强基线和物理端口合同。

