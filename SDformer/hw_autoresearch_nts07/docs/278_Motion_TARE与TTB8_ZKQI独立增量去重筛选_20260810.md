# Motion TARE 与 TTB8-ZKQI 独立增量去重筛选

> 日期：2026-08-10  
> 本轮唯一问题：TARE temporal residual 是否能在现有 TTB8-ZKQI 强基线上继续提供独立硬件收益。  
> 证据标签：`[prof]`、`[模型]`、`[待验证]`；本轮没有新增或修改 Motion RTL。

## 1. 先给结论

1. `[prof]` 新脚本读取 fullres T450、sample0、一个窗口、全部 12 个 H67
   attention block 的真实 Q/K 位流，覆盖 `138` 个 head-row 和 `31,050`
   个 temporal pair；
2. `[prof]` 重新得到 `14,554` 个 non-K-zero active pair，与现有
   TTB8-ZKQI 的 aggregate count 一致；当前没有逐 row/pair 比较 active
   bitmap，不能写成工作集身份完全一致；active pair 的时间 score 相等率为
   `82.0256%`；
3. `[prof]` 对全部 `31,050` 个 pair 检查
   `anchor raw16 + updated-lane delta = target raw16`，raw16/Q7 均为零失配；
4. `[模型]` 旧 TARE-4 在 fullres T450 active path 上有 `51.2780%`
   dense fallback，不能从旧 W9/T162 结果直接迁移；
5. `[模型]` 当前唯一值得进入下一轮的是 **TARE-16**：score-lane work
   减少 `40.8631%`，dense fallback 为 `1.7246%`，lane-only 面积归一
   score 吞吐代理为 `1.3107x`；
6. `[模型]` TARE-16 的 score 吞吐仅为双 Direct32 的 `0.9830x`；只计
   每个 dense fallback 追加一拍的理想服务模型从 `101,707` 增至 `101,958`，
   即 `0.9975x`；该数没有计 detector、compactor、双 score packet、流水气泡、
   反压和频率，不能称 RTL 保守上界；
7. `[待验证]` TARE-W8/W16 只被**条件准入为 active-score 前端面积/能耗候选**，
   W16 是首选 RTL 点而非冻结最优。
   必须把 detector、compactor、RNE、dense replay 和随机反压纳入同接口 RTL，
   证明确有全前端面积归一吞吐或总执行 EDP 收益，才能进入论文贡献。

## 2. 为什么必须重做这轮

旧 TARE DSE 的主要证据来自 W9/T162 或 profile100 count trace。当时 W4/T4
fallback 较低，且相对单 Direct32 可显示接近 `1.89x` 的周期收益。但当前
Motion 强基线已经变化：

```text
旧比较：Direct32（T0/T1 串行） vs TARE-4
新比较：TTB8-ZKQI + 双 Direct32（T0/T1 并行） vs TTB8-ZKQI + TARE-W
```

继续使用单 Direct32 会制造弱基线。并且 profile100 只有 Q/K count、overlap、
motion 和 ordered metadata，没有逐 lane 的 Q0/Q1/K0/K1 身份，无法精确恢复
temporal update mask。因此本轮只能使用 sample0/all-12 的 raw bit trace，
并明确禁止跨 sample 外推。

## 3. 工作集身份校准

### 3.1 冻结输入

- manifest：
  `results/h67_fullres_ep30_t450_all12_bit_trace_20260805/manifest.json`；
- 分辨率：`480x640`；
- window：`2x15x15`，即 T450；
- sample：`zurich_city_09_a_0001`；
- 每个 block 捕获一个窗口；
- 全 12 block、四 stage、共 138 个 head-row；
- 每个 head-row 有 225 个 temporal pair。

脚本逐文件验证 manifest 记录的 SHA256。12 个 block 的名字和顺序也作为
合同检查，任一缺失或漂移都会停止执行。

### 3.2 ZKQI 身份复现

对每个 temporal pair 定义：

```text
active = any(K0) or any(K1)
update_mask = (Q0 xor Q1) or (K0 xor K1)
```

结果：

| 指标 | 数值 |
|---|---:|
| total pair | 31,050 |
| active pair | 14,554 |
| active ratio | 46.8728% |
| active score equal | 11,938 |
| active score equal ratio | 82.0256% |

`14,554` 与现有 ZKQI 行级 RTL/报告的 aggregate count 一致，因此本轮没有
明显的筛选总量漂移。但现有检查没有逐 row/pair 对照 active bitmap；在补齐该
对照前，只允许写 count calibration，不能写 pairwise identity。

## 4. TARE 精确语义

单 lane 的 alpha-XNOR raw16 贡献为：

```text
Q=1,K=1 -> 64
Q=0,K=0 -> 1
otherwise -> 0
```

Motion bias `16*popcount(K0 xor K1)` 对两个时间 score 相同。故：

```text
raw_anchor = alpha_xnor(Q0,K0) + motion_bias
delta      = sum(raw_lane(Q1,K1) - raw_lane(Q0,K0)), only updated lanes
raw_target = raw_anchor + delta
score_q7   = RNE(raw_target / 16), exactly once at the end
```

`[prof]` 全部 `31,050` 个 pair 的 raw16 mismatch 为 `0`，Q7 mismatch 为
`0`。该结果证明代数等价，但不证明当前 TARE-16 已完成集成 RTL。

## 5. 与 TTB8-ZKQI 强基线的 DSE

强基线 active score 前端每拍用两个 Direct32 同时算 T0/T1，lane proxy 为 64。
TARE-W 每项先用 Direct32 算 anchor：

- `update=0`：32 lane work；
- `1<=update<=W`：`32+update` lane work；
- `update>W`：32-lane anchor 加 32-lane exact replay，共 64 lane work，且
  共享 anchor 核需额外一拍。

| W=T | dense fallback | score-lane work减少 | score吞吐/双Direct32 | lane-only面积归一吞吐 | TTB8理想串行fallback模型 |
|---:|---:|---:|---:|---:|---:|
| 2 | 10,409 / 71.5199% | 13.5948% | 0.5830x | 1.0975x | 112,116 / 0.9072x |
| 4 | 7,463 / 51.2780% | 22.6263% | 0.6610x | 1.1752x | 109,170 / 0.9316x |
| 8 | 3,321 / 22.8185% | 34.0281% | 0.8142x | 1.3027x | 105,028 / 0.9684x |
| **16** | **251 / 1.7246%** | **40.8631%** | **0.9830x** | **1.3107x** | **101,958 / 0.9975x** |
| 32 | 0 | 41.2067% | 1.0000x | 1.0000x | 101,707 / 1.0000x |

面积归一吞吐仅为：

```text
(active / (active + dense_fallback)) / (32 + W)
-------------------------------------------------  normalized to 1/64
```

它没有计入 update detector、32-to-W priority compactor、原子 T0/T1 packet、
控制、RNE、流水气泡、反压、SRAM 和频率，
只能用于淘汰，不得进入 DATE PPA 主表。

W8 与 W16 的 lane-only 面积归一吞吐只差约 `0.61%`。W16 的吞吐风险更低，
但 32-to-16 compactor 也可能更大、更慢；因此下一轮必须同时保留 W8/W16，
不能按当前代理直接冻结 W16。

## 6. 分层异质性

| Stage | pair | active | active score equal | active update均值 |
|---|---:|---:|---:|---:|
| S0 | 1,350 | 346 | 99.7110% | 2.081 |
| S1 | 2,700 | 235 | 94.8936% | 3.153 |
| S2 | 16,200 | 5,658 | 83.5984% | 5.885 |
| S3 | 10,800 | 8,315 | 79.8557% | 5.670 |

主要 dense fallback 集中于 `S2.B0` 的 148 项和 `S3.B0` 的 98 项；若干
block 的 active pair 为零。这说明：

1. 固定 TARE-4 是错误的 fullres 配置；
2. W=16 是对单 sample/all-12 分布的最低风险定标；
3. 是否按 stage 切换 W 需要额外物理 lane gating 或多模式控制，当前没有跨
   sample lane trace 支持，不准提前扩展为“自适应宽度架构”；
4. 这组异质性更适合支持 clock/operand gating，而不是支持近似 pruning。

## 7. 与现有数据流的组合边界

```text
TTB8 metadata scan
      |
      +-- K-zero --> three-class exact seed --------+
      |                                             |
      `-- active pair --> temporal update detector  |
                            |                       |
                            +-- <=16: anchor+R16    |
                            `-- >16 : anchor+replay |
                                      |             |
                                      v             v
                                  RQTB class commit
                                           |
                                           v
                                shared SCS / gated-K
```

TARE-16 不替换 TTB8、ZKQI、RQTB 或 SCS，也不改变这些模块的语义。它只替换
active pair 的双 Direct32 score 前端。因此：

- Q0/Q1/K0/K1 仍必须被 detector 读取，Q/K SRAM traffic reduction 为 0；
- K-zero pair 继续走 ZKQI 三类精确常量注入，不进入 TARE；
- RQTB 仍按两个 Q7 score 是否相等决定一次或两次 class commit；
- SCS 和 gated-K backend 周期不因 TARE 自动减少；
- dense fallback 若不能和 TTB8/后端重叠，只会增加周期。

## 8. DATE 创新性判断

本轮**没有新增一个可独立列出的 DATE 贡献**。TARE 的“锚点加稀疏 residual”
可映射到 Prosperity/Phi 一类 exact reuse 思路；新价值在于把它放到 Motion
all-binary temporal score 与 ZKQI 强基线之后，并用真实 update mask 定标。

若下一轮 RTL/开放物理代理成立，可将其作为 Motion 主架构下的第二级 exact
执行机制：

> ZKQI 先从 token-time 层面消除 zero-K score work，TARE-16 再从 active
> pair 的 lane 层面消除时间不变贡献；两级筛选作用于正交稀疏维度并保持
> raw16/Q7 bit-exact。

但若全前端面积归一吞吐不大于 1，或总执行 EDP 无净收益，这个机制必须作为
负结果保留，不能靠“层次化稀疏”命名进入贡献列表。

## 9. 下一轮唯一准入任务

1. 将旧 hard-coded TARE-4 叶核参数化，建立 W8/W16 两个候选；
2. 只接入 TTB8 active-pair 出口，保持 ZKQI/SCS/backend 不变；
3. 候选必须原子输出 T0/T1 两个 score；residual16 最坏 delta 为
   `[-1024,+1024]`，使用至少 12-bit signed，不能沿用 delta4 的 10 bit；
4. 建立与双 Direct32 同接口、同输入、同输出、同随机反压的 A/B/C miter；
5. 检查 raw16、Q7、RQTB class commit 和 gated-K 最终输出，并定向覆盖
   update-count 0..32、16/17 边界和 `delta=+-1024`；
6. 对完整前端而非 lane 数做同约束 Yosys/OpenROAD 开放代理；
7. 过门槛后才讨论 SAIF/DC/PTPX；不过门槛立即否决。

## 10. 产物与复现

- 脚本：`scripts/profile_h67_tare_zkqi_overlap.py`；
- 测试：`tests/test_profile_h67_tare_zkqi_overlap.py`；
- JSON：`results/h67_tare_zkqi_overlap_t450_20260810/report.json`；
- 中文摘要：`results/h67_tare_zkqi_overlap_t450_20260810/report.md`。

```bash
python3 -m unittest tests.test_profile_h67_tare_zkqi_overlap
python3 scripts/profile_h67_tare_zkqi_overlap.py
```

当前结果：`3/3 PASS`，分析状态为 `CONDITIONAL_ADMIT`。

## 11. 独立 DATE 评审

独立审稿结论：**ADMIT，仅准入下一轮最小 RTL 筛选**；不准入架构冻结、
DATE 贡献列表或 ASIC PPA 主张。

评分：

| 维度 | 分数 |
|---|---:|
| 包级质量 | 3.4/5 |
| Motion 线整体 | 3.2/5 |
| 创新性 | 2.4/5 |
| DATE 就绪度 | 2.3/5 |

评审没有发现推翻 raw16/Q7 恒等式或报告数值的 P0，但指出：

1. 只能称 aggregate count calibration，不能称 pairwise active identity；
2. 旧 TARE-4 只输出 target score，下一轮必须补原子 T0/T1 packet；
3. residual16 必须扩展 signed delta 位宽；
4. W8/W16 代理差距不足以冻结宽度；
5. `101,958` 只是理想服务模型，不是 RTL 保守上界；
6. 单 sample/all-12 足以准入 RTL，不足以进入论文主结果表。

评审冻结的下一轮硬门槛：

- 138 行、四种反压、Icarus/Verilator+SVA 下 raw T0/T1、Q7、RQTB class、
  gated-K、Acc32 零失配且无丢失、重复、死锁；
- 定向覆盖 update-count 0..32、16/17、`delta=+-1024` 和真实 251 次 W16
  fallback；
- W16 无反压 preload-inclusive 周期不超过 `101,958`，反压模式相对强基线
  回退不超过 1%；
- 完整 active-score 前端的开放代理面积归一吞吐至少 `1.10x`，同 trace
  row-top EDP 活动代理至少改善 10%；
- 任何硬门槛失败即否决 TARE。OpenROAD/活动结果仍只能标为开放代理。
