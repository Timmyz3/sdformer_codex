# Joint admission：统一 tile-score 目标的有限规则对照

2026-10-14。**只实现纯CPU选择器、合成验证及已有proposal重放；没有运行模型、GPU、恢复或新的质量评价。** 这是既有四状态诊断提示的候选规则，不声明新颖、必要性或已成立的光流增量。原图/源模块/已封存结果均未修改。

输入复用既有完整pair+singleton proposal、相同tile幅值分数与真实有效点cost。三种主要规则统一最大化 **sum(selected TILE scores)**，避免旧组最大分数每组只计一次造成目标函数变化；旧目标仅作独立来源控制。motion、rotated90、horizontal_local均允许传入，不按方向是否有效筛选拓扑。

| `rule` | pair允许状态（1=slow） | singleton允许状态 | 优化目标 |
|---|---|---|---|
| `free` | 00/10/01/11 | 0/1 | selected tile score之和 |
| `at_least_one` | 10/01/11 | 0/1 | 同上 |
| `all_or_none` | 00/11 | 0/1 | 同上 |
| `legacy_group_max` | 00/11 | 0/1 | 旧selected group max-member score之和，每组一次 |

分数按原Python int/float的精确Fraction比较；同分时选择“已选tile IDs升序元组”的字典序最小者。目标预算为**精确900真实点**，不可达返回`INFEASIBLE_EXACT_COST`和null mask，不自动改预算、不丢样本。一般接口允许0–2048用于合成检查，当前真实重放固定900；与另一个R0/R1恢复合同中“cap900、逐target共同可达”的规则分开，不能混用标签。

完整pair互斥且覆盖全部active tile，因此free组内四状态的笛卡尔积恰好覆盖所有自由tile mask。统一tile求和与原independent的singleton目标相同，同样的精确比较和tile-ID tie使free与原independent严格等价，既不依赖pair方向，也不引入额外信息。`at_least_one`/`all_or_none`只有约束变化；它们的proxy分数不能超过free最优值，但质量是否更好必须用真实任务结果判断，不能把proxy下降当作方法失败。

每个mask仍须由调用方应用于完整T10×真实9×9尾块×全通道。cost是该层真实空间点；只有固定C/T和相同分支时才对应相同slow/fast算术量。预测、fast、控制、BN、存储及跨层成本均不在本DP的点数中，不能称周期/能耗收益。当前DP使用Fraction和Python状态表，不是已综合控制器。

## API

```python
from select_joint import select_exact
result = select_exact(saved_proposal, 900, rule='at_least_one')
if result['feasible']:
    mask = result['slow_mask']  # 二维bool；True保留完整slow块
else:
    # 保留目标/拓扑/不可达结果；不能降预算、重采样或把它算作质量失败。
    pass
```

返回selected tile IDs、精确objective分数、真实点数、逐group状态、pair 00/10/01/11计数、全部≤目标的可达cost及DP操作元数据。`selected_group_ids`表示至少选一成员的组，另有`fully_selected_group_ids`，以免部分pair被误记成整组服务。对legacy目标另外保存selected TILE score，两个目标不混在同一排名里。

## 实际CPU验证

[7项合成检查](cpu_checks.json) PASS（0.324秒）：独立穷举小图的四种规则最优mask、正/负/零/浮点同分、交错pair和tail异成本、free与旧independent严格等价、legacy与旧DP一致、至少一块但singleton可不选、精确预算不可达、空tile/输入畸形和不修改输入。合成小图穷举枚举所有mask，不复写DP作为reference。

[四帧重放](replay_four_frames.json)于0.257秒完成，只读 [已完成的控制结果](../selection_controls/run_v1/results.json) 中的proposal和部署mask。源SHA：`2611a15d76a5bc2d550412c40f3ad0efba02524795e0ce254f3aaabaceb9fba2`。不读取GT/events/预测数组，不调用torch或模型。

- **16/16 free等价检查通过**：四帧×四拓扑，mask、selected IDs、Fraction分数和真实cost完全等于旧independent。
- **16/16 legacy等价检查通过**：完全复现对应旧DP及保存的实际部署mask。
- 三个非独立拓扑×四帧均能精确900，无不可达项；`at_least_one`有6/12处改变free mask，全部来自rotated/local，各3帧。
- **motion的4/4帧均不改变free mask**：自由选择本已给每个motion pair至少一块slow。当前四帧/900点/同前缀和同fast下，两者输出必相同，不能把普通independent的质量重新命名为motion新收益。
- tile-sum同开关相对旧group-max只改变2/12处，均为rotated；当前motion/local掩码不变。修正目标不自动挽救已观察到的motion排名。

| Target | motion至少一块 vs free | rotated至少一块 vs free | local至少一块 vs free | tile-sum同开关 vs legacy发生改变 |
|---|---:|---:|---:|---|
| 01a0083 | 0 | 2 | 2 | 无 |
| 05a0165 | 0 | 0 | 0 | 无 |
| 01a0030 | 0 | 2 | 2 | rotated：4 |
| 05a0352 | 0 | 2 | 2 | rotated：4 |

数字是selected tile IDs对称差，**不是删减点数、AEE或PE服务收益**。这四帧仍是backbone-seen、反复用于探索的训练侧开发样本，不是新确认集。

可重复执行（输出路径必须不存在）：

```bash
python3 test_joint_admission.py
python3 replay_proposals.py \
  --results ../selection_controls/run_v1/results.json \
  --results-sha256 2611a15d76a5bc2d550412c40f3ad0efba02524795e0ce254f3aaabaceb9fba2 \
  --output replay_four_frames_NEW.json
```

## 下一有限判断，尚未执行

先完成R0/R1普通底座恢复，再由主会话固定执行范围。本模块不加GPU队列，不因当前motion约束不活跃而自动扫更低预算或更多方向。

1. 在预先固定的下一开发target列表和同一预算上，先做不读GT的proposal重放，完整记录三种拓扑的约束活跃率/不可达率；所有方向均保留，不挑能产生差异的样本。若motion至少一块仍处处等于free，此范围内该候选没有行为增量，可以停在此机制结果，无需重复前向。
2. 若有非重复mask，统一使用同fast权重、同恢复、同数值图和实际总成本，比较共同free以及三拓扑的至少一块、同开关；legacy单列检验目标变化。先对完整候选集合做mask去重，别名保留，完全同mask/同fast只执行一次并明确不算独立支持。
3. 只有真实新增任务损伤在全部GT/B/L/unknown及跨序列上支持方向条件，且超过同输入的普通局部/旋转控制，才能推进光流关联增量。单纯禁止00、较低proxy分数、普通恢复改善或一个方向偶然获胜，都不够。

最强反解释仍是普通邻域保护、幅值排序或预算可达性已解释结果；目前尤其存在“motion新约束根本未改变free决策”的直接反例。相同预算下约束有害、不可达或无行为差异都应原样保留，不能回头调分组再称预先假设得到验证。
