# Motion全量Ordered周期分布与自适应旁路否决

## 1. 本轮目的

上一轮Motion ZKQI已经完成138条sample0/window0真实head-row的三方RTL、活动事件分账和共同宏开放物理闭环，但DATE审阅指出最主要的本机证据缺口仍是单样本外推。本轮不改RTL，只完成以下工作：

1. 解码H67 ep30 fullres T450的100 sample、1200个attention ordered trace；
2. 从Q/K-count、overlap和motion count精确重建H67 Q7 score；
3. 从RTL状态机推导RQTB2S、PairBitmap-ZKQI和TTB8-ZKQI无反压周期；
4. 先在sample0/window0的raw Q/K vector和138行RTL日志上做零残差校准，再统计全部window/head；
5. 冻结后续门级SAIF的非挑样本抽样合同；
6. 量化逐行自适应旁路的理想上界，避免为极小尾部收益继续堆控制。

本轮证据等级为`[prof]+[rtl校准模型]`。它不是全量逐bit RTL replay，也不是门级SAIF、功耗、能量或ASIC PPA。

## 2. 第一次失败与口径修正

第一次全量执行在校准阶段fail closed：row1的模型active descriptor为131，而RTL `zkqi_slots`为132。根因不是score或周期错误，而是两个计数器语义不同：

- `zkqi_slots`统计score quotient descriptor；当双时间片score不等时，即使一个时间片K为零，也会形成两个score quotient；
- active descriptor只保存需要gated-K发射的条目，同一pair只有一个非零K时间片时只形成一个active entry。

修正后分别记录`candidate_descriptors`和`active_descriptors`，不把两者合并。重新校准得到138行、1794个RTL字段零失配。这个失败说明term/descriptor统计必须绑定消费阶段，不能用一个“descriptor数”贯穿所有数据流。

## 3. 周期模型与零残差校准

### 3.1 H67 score重建

对每个时间片、window、head和spatial pair，ordered trace给出：

- `q_count`；
- `k_count`；
- `overlap`；
- 两时间片共享的`motion = popcount(K0 XOR K1)`。

RTL score可重建为：

```text
same_zero = 32 - q_count - k_count + overlap
raw = 65*overlap + 32 - q_count - k_count + 16*motion
score_q7 = round_to_nearest_even(raw / 16)
```

sample0/window0的全部Q/K-count和score逐项与raw 32-bit Q/K vector一致，失配为0。

### 3.2 单槽TTB前端

TTB8不是`29 + active_pairs`的串行模型。真实RTL只有一个bundle descriptor槽：

- 空bundle可在旧descriptor消费期间继续扫描；
- 非空bundle必须等待槽空；
- 旧descriptor最后一个pair可与新descriptor接收同拍发生；
- 新descriptor从下一拍开始消费。

按该协议逐拍推导的前端周期在138行上与RTL完全一致。

### 3.3 后端周期

无反压后端精确公式为：

```text
backend = 3 + occupied_classes + 2*active_descriptors
          + emitted_K_tokens + I(active_descriptors > 0)
RQTB2S = 225 + backend
PairBitmap-ZKQI = 225 + backend
TTB8-ZKQI = depth1_ordered_bundle_cycles + backend
```

校准结果：

| 项目 | 结果 |
|---|---:|
| raw vector/profile count与score | 0 mismatch |
| RTL校准行 | 138 |
| RTL校准字段 | 1794 |
| 三方逐行周期残差 | 0..0 cycle |
| 12 block active/K/motion bundle trace | 36/36组交叉检查通过 |

## 4. 全量覆盖

| 覆盖项 | 数量 |
|---|---:|
| sample | 100 |
| attention record | 1200 |
| block-window | 132000 |
| head-row | 672000 |
| temporal pair | 151200000 |

这是全部100 sample、全部12个attention block、全部window和全部head，不是抽样统计。

## 5. 全量周期与工作事件

| 指标 | RQTB2S | PairBitmap-ZKQI | TTB8-ZKQI | TTB8相对基线 |
|---|---:|---:|---:|---:|
| 执行周期 | 282254101 | 282254101 | 178614211 | 1.5802x |
| 含225拍preload | 433454101 | 433454101 | 329814211 | 1.3142x |
| score次数 | 151200000 | 36306009 | 36306009 | -75.99% |
| Q/K读取bit | 20906132672 | 6199701824 | 6199701824 | -70.35% |

结论保持因果分账：

- PairBitmap与TTB8都利用exact zero-K work gating，因此score和读bit相同；
- PairBitmap仍逐pair扫描，周期与RQTB2S完全相同；
- TTB8额外利用层次issue skipping，才得到周期收益；
- 读取bit和score次数是`[prof]`工作事件，不能乘任意常数后称为功耗或能量。

## 6. 多样本与尾部分布

### 6.1 Sample级

含preload的逐sample加速：

| min | mean | p50 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|
| 1.3062x | 1.3143x | 1.3141x | 1.3225x | 1.3260x | 1.3269x |

100个sample之间波动较小，说明sample0/window0的方向正确，但该窗口含preload的1.147x并不代表总体均值；全量均值更高。因此主表应使用全量`[rtl校准模型]`分布，sample0/window0 RTL只承担校准和bit-exact证据。

### 6.2 Stage级

| Stage | active pair | score减少 | Q/K读bit减少 | 执行加速 | 含preload加速 |
|---:|---:|---:|---:|---:|---:|
| S0 | 22.77% | 77.23% | 72.00% | 1.6547x | 1.3401x |
| S1 | 7.48% | 92.52% | 90.58% | 2.8396x | 1.5669x |
| S2 | 29.71% | 70.29% | 63.54% | 1.4258x | 1.2541x |
| S3 | 54.81% | 45.19% | 38.25% | 1.1576x | 1.1138x |

S1最稀疏、收益最大；S3最稠密、收益最小。后续架构优化应优先减少稀疏行header成本或S3稠密行的无效控制，而不是假设四个stage共享同一最优元数据粒度。

## 7. 真实负结果：逐行自适应旁路不值得实现

逐head-row结果：

| faster | equal | slower | slower比例 |
|---:|---:|---:|---:|
| 653650 | 4381 | 13969 | 2.08% |

全部slower行都有100% active pair，且只慢1拍。即使假设一个不占面积、不耗周期的理想选择器能逐行取`min(RQTB2S, TTB8)`：

- 也只额外节省13969拍；
- 相对TTB8总含preload周期仅减少0.0042%；
- 总加速仅从1.314237x变为1.314293x。

因此本轮明确否决按row动态选择RQTB2S/TTB8的控制器。该机制会增加mode判定、mux、验证和物理控制扇出，而理想上界已经不足万分之一。慢1拍的稠密行保留为论文负结果。

## 8. 下一架构候选

stage和密度分布表明，更值得评估的是更粗TTB粒度：

1. bitmap总信息量仍是225 bit，不复制活动元数据；
2. 将8-bit mask读出改成16/32-bit分段，可减少极稀疏行header数；
3. 代价是更宽priority-select、descriptor mask和可能更长组合路径；
4. 必须先做B4/B8/B16/B32全量周期、mask选择器面积和固定5 ns物理对照，再决定是否进入RTL主线；
5. 未完成同约束RTL与物理代理前只能标为`[模型候选]`，不能列为DATE贡献。

这条候选比逐行双模式更符合本网络特点：S1有大量近空行，S3有高密度行，而bundle宽度改变的是header摊销与mask选择器成本，不改变任何score或Shiftmax语义。

## 9. SAIF预提交合同

当前567 MiB profile没有原始Q/K bit身份，只保留count trace，因此不能生成真实门级切换。已冻结：

- 10个等间隔sample：0、11、22、33、44、55、66、77、88、99；
- 每个block固定首、中、末window；
- 覆盖全部head，共4140条head-row；
- 三方使用相同输入、相同5 ns约束和相同目标SRAM宏；
- 主功耗模式无反压，固定反压只作敏感性；
- 后续门级SDF+SAIF与PTPX必须按metadata、score、SRAM、SCS、emit分层报告。

合同保存在`results/h67_zkqi_multisample_ordered_20260809/saif_capture_manifest.json`。GPU空闲后只能按该合同抓取，不能根据收益事后挑样本。

## 10. 复现与证据入口

CPU-only复现：

```bash
sim_h67/run_h67_zkqi_multisample_profile.sh
```

产物：

- 中文报告：`results/h67_zkqi_multisample_ordered_20260809/report.md`
- 结构化结果：`results/h67_zkqi_multisample_ordered_20260809/report.json`
- SAIF合同：`results/h67_zkqi_multisample_ordered_20260809/saif_capture_manifest.json`
- 统计脚本：`scripts/profile_h67_zkqi_multisample_ordered.py`
- 单元测试：`tests/test_profile_h67_zkqi_multisample_ordered.py`

## 11. 下一轮

1. 以本轮全量trace评估B4/B8/B16/B32的周期、尾部和元数据读结构；
2. 只选择一个具有明确周期收益且priority-select代价可控的粒度进入最小RTL和同约束开放物理对照；
3. 每轮完成后继续独立DATE审阅；
4. Local5 watcher继续等待同窗全head真实profile，结果到达后立即转入checkpoint绑定、12-block调度和真实trace闭环；
5. Motion的现有SCS/NMF/DCTF回归保持，不因Local5优先级而停止维护。
