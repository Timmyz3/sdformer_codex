# arXiv:2410.23082 · FlexSpIM

- uid/来源: `MAIN-R347`｜arxiv_2410.23082+本地excerpt（`p0_excerpt_batches/batch_04_residual.json`）
- 题名: An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise Weight/Output Stationarity
- 精读深度: 方法级（仅方法摘录窗口）+依据：贡献三点——层间 WS/OS（膜）可切换驻留、NR×NC 任意纵横比 operand shaping、非固定比例的权/膜位宽；摘录多为 Fig.1 CIM 相位/进位选择示意图，§II 数据流文字细部偏图。**更深全文卡已有** `flexspim_iscas2025.md`（local_author_fulltext）；本卡只记 arXiv 摘录窗，不覆盖该全文卡

## 可继承 A
按层选择驻留对象（权 vs 膜/OS）+ 任意 operand shaping——作有限 RF 下「谁该驻留」系统对照（借入≠X；与已有 FlexSpIM 全文卡一致）。

## 强对照 B
固定纵横比/仅 WS 的 SNN CIM；权膜位宽锁死比例；无事件驱动的层优先执行流。

## 可差分 X线索
CIM 宏≠ lifting 数字残差链 X；可借驻留选择对照 F5 生存期，宣称 CIM-X 则越界。

## 与 F1–F7 / Stage B 关系
F5 旁证；不进主岛。不抢 Stage B。与 `flexspim_iscas2025.md` 联读，避免双计为两套独立标题。

## 不可搬用边界
40nm 数字 CIM 实测/外推≠本地 AEE；摘录图重、相位时序以全文卡为准；勿搬 79–90% 能量外推。

## 可复用 idea 点
- HS-min：每层选内存更小操作数驻留
- carry-select 链式多比特 ↔ 邻接通信限制
- shaping 未用列 standby 的能量波动上界叙事
- 负结果只停「CIM 驻留名替换数字链」类比

## 杀门建议
无同端口重读/带宽分项改善，或 CIM 相位不可映射 → 停类比，不杀有限 RF 家族。
