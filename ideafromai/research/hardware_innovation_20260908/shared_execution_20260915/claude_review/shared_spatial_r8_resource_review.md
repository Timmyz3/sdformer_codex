共同空间链/R8 资源初审（2026-09-15）：只读两个 PLAN、top/context 的数据阵列、读表达式、执行归属和消费生命周期。两臂确有相同的容量上限、服务口及核心算术数量；没有发现每 context 偷加一套 producer 算术或提前覆盖未消费输出。**当前不是相同物理网表或等面积/Fmax 的证明。** 本页不引用仍在收口中的性能排名，不执行 RTL/EDA，不改变实现。

| 项目 | spatial_rr | r8_reference | 判断 |
|---|---|---|---|
| context | 两个状态/存储 leaf | 两个状态/存储 leaf | 都由顶层原子 RR 授权。 |
| source | 每 context 1536×10 = 1920 B | 相同 | 1536 字与 origin32 分别实际握手；双 context 都加载后才 launch，没有一臂免费边加载边执行。 |
| Z | 每 context 8×40×52 = 2080 B | 相同 | 每次全局最多一个 Z owner。空间只使用低32位、其余清零；R8 使用52位但目前仅前10行。是相同声明容量和服务权，不代表两算法同等利用容量。 |
| psum | 每 context 8×480×32 = 15360 B | 相同 | 单次每bank读或写；消费结束后才回收整个 batch。 |
| W 读服务 | 单 owner、256bit mux，Q1/Q2互斥 | 单 owner、256bit mux，q/v/plane/class/rep互斥 | 原子资源位拒绝时不推进相应读/写。系数实际有效位宽不同，不能只按256bit口推等存储面积。 |
| 静态数据 | q1 4608 B + q2 7488 B；live144 B | q2592 B + v1536 B + plane2592 B + class2592 B + rep96 B，另live/排列 | 阵列和布局本来不同，应分列容量，不能声称 W 存储相同。 |
| producer | 唯一8×signed19×13乘法及8×32 carry链 | 相同数量/宽度 | 两套leaf都接外部算术结果，没有各自数据乘法器。空间cut15；R8另有cut13/10/8/5及pop16等逻辑。 |
| Q2/cache | 每context 8×24×13 =312 B | 24×8×16 =384 B | R8声明容量更大，属于给予强控的预算上限；不是两个完全相同的物理数组。 |
| I24消费 | 唯一 i24_consumer 实例 | 源文件逐字相同、唯一实例 | 同8×32×32乘法、FP32→J20及wide/I24控制；a/b与输入gold按各自函数独立。 |
| wide借用 | 唯一8×64链，15/30/45/60断进位 | 唯一8×64链，13/26/39断进位 | 消费者与Q1借用互斥、同wide RR。模块并非同源配置；只能声称同链数量与权限，若要字面同物理多格式链应另行统一。 |

证据入口：[空间 top](../spatial_rr/interleave_stream.sv)、[空间 leaf](../spatial_rr/rr_context.sv)、[R8 top](../r8_reference/interleave_stream.sv)、[R8 leaf](../r8_reference/rr_context.sv)、两树的 `wide_phase_alu.sv` 与 `i24_consumer.sv`。Z/psum 各 context 分开声明，访问的全局互斥由资源位与 grant 实现；没有据此声称已映射到某一份单端口 SRAM 宏。两臂的 metadata/holding、bitmap/pop、变换尾字和 M 状态不是相同逻辑，PLAN 已按各自资源列项，此边界应保留在性能结论旁。

生命周期检查：两树的 shared grant 对 source/W/Z/psum/ALU/wide 冲突作原子决定；不冲突时可双 context 前进。borrow 与 consumer add 不会同时占链，DRAIN 的 p 读取也经过 psum 授权。消费按 tile 顺序，每 tile 480 个 I24 输出接受完成后才释放相应 owner；第二 tile 消费完成后才进入下一 batch 的 SOURCE。空间第二 stripe 的 p 读取有第一 stripe ownership 检查；未见来源装载跨越未消费输出。拒绝保持与逐状态计数的动态验证仍以两位负责人最终收据为准，本页不是独立重跑证明。

配置类型更正：此前把 kind6 的864个配置字写成 plane 配置是本审阅的误判，现已撤回。实际 [R8 leaf 第338行](../r8_reference/rr_context.sv:338) 为 `6:k_live[cfg_addr[9:0]]<=cfg_data[0]`；该支持表既用于普通路径跳过无效K，也用于bitmap构造。实际 `bp_q` 与 `plane_live` 在 [top 的 kind4 权重接收分支](../r8_reference/interleave_stream.sv:488) 随同一批 q 权重写入，没有另一个 kind6 plane 装载阶段。因此不能声称“不跑bitmap即可省864拍”，也不再保留此前的 bitmap-resident 优化建议。主表最强 R8 mode7 本就支付同一 k-live 冷配置，更正不改变已测服务排名。

本轮资源审阅支持使用“共同容量和服务权限、相同核心算术数量、相同完整消费者”的比较口径；不支持“所有硬件完全相同”或用当前不同有损函数的 raw 差值作误差正确性检验。算法质量由各自已冻结全网评估给出，数值正确性由各自独立 gold 给出，完整服务由各自同一 source/identity 输入与实际加载/退休费用给出。

收尾核对：[COMPARISON.md](../COMPARISON.md) 的六行、36个服务值均与各自冷配置首遍记录相同，全部满足 `service=total+go` 且 `go=1`；六个变慢百分比也重算一致。共同预算、不同进位/附属逻辑及不同学生函数的边界与本审阅一致。收据：[comparison_service_check.json](comparison_service_check.json)。本次只读核数，没有重跑实验。
