# 跨18序列固定36tile：组件服务与独立网络质量

周期覆盖固定36tile的原生source到最后I24，模型加载一次、逐tile加载source/origin/start；AEE覆盖完整825帧网络，两个覆盖范围不同。不同function行是有损方案取舍，不是同函数加速比。

| 函数 | 执行 | 组件周期 | 825逐帧AEE |
|---|---|---:|---:|
| spatial_q11 | expanded_OS | 1,182,084 | 1.254431 |
| spatial_q11 | ordinary_factor | 980,690 | 1.254431 |
| spatial_q11 | general_Winograd | 978,062 | 1.254431 |
| moment | ordinary_factor | 980,690 | 1.296504 |
| moment | general_Winograd | 908,354 | 1.296504 |
| native_tap | ordinary_factor | 849,931 | 1.285953 |
| native_tap | general_Winograd | 937,429 | 1.285953 |
| moment | three_product | 865,250 | 1.296504 |
| unconstrained | three_product | 865,250 | 1.258343 |
| moment | source_box_two_tap | 883,130 | 1.296504 |

同环境NB0逐帧AEE为1.447936665574317；候选使用既有matched-dense学生/粗头，整段优势不能归因于本次剪枝或分解。

全部周期来自Verilator组件仿真；共同算术与服务合同不等于独立裁剪后同面积或同Fmax。展开OS权重容量、通用Winograd额外状态、各路径所有配置/恢复费用见各报告。无VCS/DC/PT/Formality/PPA、无整网RTL周期或FPS。

unconstrained为两相位函数，已与其独立展开gold核对；尚未实现同函数直接两相位RTL。moment与unconstrained的三项硬件相同，不把控制少一项说成moment独占。

[完整ready/BP及两套同帧64tile表](component_quality.csv)、[825质量明细](quality/QUALITY_REPORT.md)、[独立评审](review_stage_final.md)。
