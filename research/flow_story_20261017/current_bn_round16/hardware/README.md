# R16：有条件的 current-BN 服务与工作量筛查

本目录仅有 CPU 几何、事务和依赖检查，没有新网络前向、GPU、RTL 或 DC。`service_model.py` 的加权 BN 数学路径对应已被数值检查否决的 V1；其服务数字只能解释资源代价，不能作为可部署执行器的收益。V2 恢复原生全域 BN，单独看 `V2_DELTA_AND_SINGLE_LAYER_PILOT.json`，不能套用 V1 的 BN 节省。

复算命令（输出目录必须不存在）：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B research/flow_story_20261017/current_bn_round16/hardware/run_service.py --output-dir /tmp/flow_r16_service_new
```

`actual_v1/CPU_CHECKS.json` 中 8 项检查通过；`TINY_TRANSACTION_TRACE.json` 保存逐笔 DMA beat。生产几何来自 SHA 钉住的独立 R13 审计和实际 external54 BN 形状记录，未读取 AEE 排名。`INDEPENDENT_IMPL_FC_COUNT_REVIEW.json` 独立读取并哈希 1,288 个保存的 FC 输入数组：276 case、644 路径，每 FC 合计 473,300 行，与钩子记录一致；该计数 PASS 不改变数学 `NOT_ADMITTED`。

## 明示资源和依赖

- 2 MiB activation 存储，其中 128 KiB 预留统计/系数；512 KiB weight 存储；4 KiB 累加器寄存器。块内保存完整 T=10，块大小按 C 和真实尾格计算。
- activation 为 16 个每 tick 单读或单写 word 的 bank；weight 计算期 32 个 bank 各读 1 word，DMA 装载期分时写，计算与装载串行。统计 RF 假定 16 读+16 写通道，容量计入预留区；其面积与端口实现未验证。
- 共享 DMA 64 B/tick。每个 FC 权重 output stripe 在有限 SRAM 内装载后即计算，再换下一条 stripe，不把完整 FC 权重矩阵假定为常驻。各空间块重新读取权重，PSN 的 T×T 权重和 bias 也计流量。
- FC 为 row8/output32 阻塞映射，FP32 FMA 的 K 递归延迟假定 4 tick；统计 SIMD16、算术延迟 4 tick、rsqrt 延迟 16 tick。**没有绑定 RTL/DC 算术单元。**输出累加器落地、BN 读写和残差读/加/写分时计账；物理 bank 地址、路由和调度最优性未知。
- 每个 MLP 两道全域归一化屏障。显式 centered two-pass variance 又引入均值完成→方差完成依赖：选中八层为 16 道归一化边界/32 道矩完成边界。先完成全域统计，再对本层任意代表点归一化；不允许逐 tile 自行完成两个 BN。

## 两种寿命策略

`spill` 保存 compact FC1/FC2 输出各一次，在方差和归一化阶段各读一次。compact U、原始残差和完整输出在外存，不能把完整窗口特征免费留在 2 MiB 内。`recompute` 留存 compact U、不保存 FC 输出，按均值、中心方差、归一化的依赖重新生成：选中层 FC1/PSN1 **5 遍**、FC2/PSN2 **3 遍**。每遍均重新计行数、权重流量、PSN 与 BN1 的必要归一化。未选四个 MLP 在所有比较中均采用相同 dense spill 服务。

| 动作/寿命策略 | 全 12 层名义每 FC 行 | 实际 FC1 行 | 实际 FC2 行 | 相对已知共同工作分母的乐观取消份额 |
|---|---:|---:|---:|---:|
| native / spill | 558,000 | 558,000 | 558,000 | 0 |
| 八层 CC / spill | 153,600 | 153,600 | 153,600 | 56.6830% |
| 八层 CC / recompute | 153,600 | 360,000 | 256,800 | 34.9272% |
| P / spill | 164,200 | 164,200 | 164,200 | 49.7663% |
| P / recompute | 164,200 | 413,000 | 288,600 | 7.5150% |
| checker / recompute | 356,520 | 1,374,600 | 865,560 | −78.2228% |

份额是另一个明确假定“每个 scalar MAC/add/multiply/threshold 占共同 PE 一个 issue”的必要条件筛查，**不是上面的服务日程 tick 比值，也不是周期或能耗收益**。分母仅包括完整十二 MLP 和共同 external54 BN 的已知算术；缺失 attention、conv、上下采样、预测器等会进一步压低正份额。rsqrt 单独计数，没有用 FLOPs 除 PE 数当周期。在该假定下 checker 重算新增工作已经超过取消工作，P 重算的乐观份额也低于 10%；这两条可便宜否决该寿命策略。CC spill 尚有 producer 工作空间，但 V1 已被数学否决，不能据此启动部署。

共同部分没有遗漏成“免费”：未选八个 BN 每前向 115,200,000 元素，已包含在四个完整 MLP 内；external54 为 976,584,960 元素，逐层保存算术/三读一写库存，**未引用旧串行 BN 周期作整网 baseline**。其 producer 融合、实际生命周期及周边层仍未闭合。普通评分与光流信息评分使用同一 compact 执行器、同一 mask，执行工作差为零；评分器本身的成本待补。

## V2 与单层 pilot 的界限

V2 selected16 BN 重新处理完整 322,560,000 元素。相对八层 CC 的 V1 weighted 36,864,000 元素，增加 285,696,000 元素及 2,211,840,000 个假定 scalar issue。若两者采用**另行声明的共同完全物化策略**（Z 写一次、BN 输入三读、输出一写），BN 路径差额为 5,713,920,000 B；BN1 gather、compact 回存/重读另列。这个策略不同于 V1 producer 融合 mean 的日程，不能混用相减。V2 的 BN2 输出已经是完整 branch，不再重复计最终 expand。诊断 moments/cell equality/BN2 gather 不是生产执行器，但实际 CPU/GPU验证开销仍应单列。

单层 `s2b2` pilot 的目标 FC 行为 12,000→1,400，全十二层为 558,000→547,400。十二 MLP 每块 FC MAC 均为 14,155,776,000，单层 CC 最多取消十二层 FC 总 MAC 的 **7.361111%**；按上述已知共同工作假定，即便加 PSN 节省减 pool 新增，乐观份额也仅 **6.832274%**。因此单层 pilot 只适合验证动作是否有光流信息价值，不能承担整网 10% PE 收益主张；后续必须扩大有效动作范围并闭合真实 trace。

文件中的 tick 仅为声明机器的一条阻塞服务序列。未闭合部分还包括预测器/索引控制、全域 gather/布局转换的真实地址服务、FP32 数值顺序、RTL 综合面积频率，以及完整 attention/conv 分母。数学修正若再次失败，应停止此执行路径；通过也仅允许回到固定小规模动作信息实验，不能把归一化实现当光流贡献。
