# 固定 m2 置零的 phase3 强控制

授权范围：只导出数值模型、静态界与 gold，不训练、不运行 GPU/RTL、不改变已冻结的 moment/native_tap 模型。母模型为 `../../spatial_winograd_inputs` 的 Q11。

- A：Winograd 域直接整 N8 组置零。固定全部 192 个 `(rank,N8)` 的 m2 为零，保留母 U0/U1/U3；没有拟合、比例扫描或活动选择。
- B：合法整数三tap投影可能比自由 U 置零损失更多；奇数恢复不能成为拒绝这个有损控制的理由。
- 本控制不主张 X。它不是平移不变 1×3 卷积，不能提供伪造的普通 g。两个水平位置分别采用 `[U0,U1,U1−U0,0]` 与 `[0,U1−U3,U1,U3]`，再与竖直 q1 展开成独立两相位权重。
- 物理系数 `physical_coeff3[o,r,k]` 的 k 顺序为原 m0,m1,m3，signed13；不提供歧义字段 `q2`。D 仍是 `[z0−z2,z1+z2,z2−z1,z1−z3]`。仅计算 M0/M1/M3。
- raw `p2_even=M0+M1`、`p2_odd=M1−M3`，不除2。`p_int` 即 raw p2，`output_scale=母/2`，按该 scale 重新构造 a_q40；b、identity→J、单次 RNE26/I24 保持。须证明 D/各 M/各恢复及跨 stripe 累加 signed32，wide signed64。
- 输入使用母模型同135 tile及36 live记录（18序列×tile128/9664）。phase3 由 M 路径和两相位 expanded-W 独立一致验证；不得把母 P/I24 当新函数 gold。另独立重算 moment/native_tap 的36记录，旧135 gold不重写。
- 局部误差统一对母 Q11：raw 比较 p2 与2×母p，BN尺度输出与I24另报，两水平相位单列。局部误差不等于网络 AEE。实际 RTL 周期、网络质量均由其它负责人测量。
