# Kill A：lifting 系数舍入裕度证书静态上限

数据源：`algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/constant_compilation_result.json`（fast_temporal_recovery_lifting40 常量编译结果）。
系数格式：signed16/f12 RNE, no clipping on actual stage320 values；状态格式：signed24/f14; RNE then saturation after every lifting half-step；Q=4096。

证书条件：|q|·d < m₀ = Q/2 − |r₀|。静态上限 = |q|<Q/2 的系数比例
（只有这些系数在某个状态下才可能被精确跳过）。

| 模型 | n | 系数范围 | \|q\|<Q/2 | \|q\|<Q/4 | \|q\|<Q/8 | 零系数 |
|---|---:|---|---:|---:|---:|---:|
| fast_raw_diagonal | 40 | [-19046, 18426] | 0.4500 | 0.2250 | 0.1000 | 0.0000 |
| fast_shared | 40 | [-19048, 18428] | 0.4250 | 0.2250 | 0.1000 | 0.0000 |

## 判读（C3 卡杀门1：|q|<Q/2 比例 < 30% 即停）

见下结论节。注意：该比例是**静态资格上限**，动态可达率还受 |r₀| 分布限制，
只能比它低；报告中不把静态上限当作实测跳过率。

**过静态门**：全部模型 |q|<Q/2 比例 ≥ 30%，C3 卡保留，进入动态测量。
