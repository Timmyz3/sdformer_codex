# Fusion ROUND1 短杀门卡（B / A / X / 对照）

日期：2026-09-13  
用途：box 上 **RTL→测→杀→下一个** 的一页杀门措辞；**≠** Codex Stage B / 整链净服务 / TCAS 标题 X。  
纪律：借入底座 ≠ X；负结果只停**本布局**；禁动生产树 / `main.tex` / `nts07`。  
姐妹页：`f1_f2_paper_design.md`、`f3_f7_paper_design.md`、`ab_fusion_candidates.md`。

通用一票否决（任一候选）：
1. 为过门改 same-port / 同状态 / 同背压合同  
2. 把通用 CSE/指令融合写成 lifting/家族标题 X  
3. 未评网络函数就报 AEE；或用节点比冒充周期  

---

## F1　lifting 源活动 × r1 结构剪枝

- **B**：固定广播域内共同删物理源字（含同步系数/阈值），适合共享供数。  
- **A**：lifting T10 图；双 PED 消费者；Gustav 供数分母；da4ml CSE；2:4/Wanda 同等权限。  
- **X（待证）**：按 lifting 改变后的源活动 + 非因果 T10×双 PED 联合扰动选可删字（≠只换损失名）。  
- **对照**：ordinary 窄稠密/hidden50；HiNM/VENOM-CRISP；同 lifting 无剪枝；禁止只打 s2b3。  
- **杀**：不胜 HiNM/窄稠密；只降 W² 不降物理字/门/流误差；AEE 破 1.259 或相对 +0.005；并集仍满读且无服务余量 → **停该剪枝布局**。

## F2　半步检查点 × 组有损共同完成

- **B**：半步/RNE 检查点预测整组门字；接受则发预测 θg 并退休，否则继续原算。  
- **A**：lifting 半步+RNE；Gustav 共享请求；SparseInfer/BitFair 预测链（借入）。  
- **X（待证）**：共同完成挂在 lifting 半步边界与 r1 多消费者并集费用。  
- **对照**：无预测共享同步；独立每神经元预测+同组关闭；静态窄层；lifting 无预测。  
- **杀**：不胜窄层与「独立预测+同组关闭」；预测开销≈再做 PSN；接受后 AEE 爆；组尾被最慢消费者钉死 → **停该检查点布局**。

## F3　Prosperity 联合图挂 lifting 源 DAG

- **B**：在 lifting 源 DAG 上选公共虚节点/部分 lane/有限供权包（非整 W 产品 mask）。  
- **A**：Prosperity 森林语义；Gustav 有限计划；lifting vs ordinary 260/159+35 图同分母。  
- **X（待证）**：产品式共享挂到 lifting **源图节点**，而非再扫 mask。  
- **对照**：ordinary+同等联合图；纯 da4ml CSE；−0.09045% mask 控制（负/薄）；lifting 无图。  
- **杀**：不优于 CSE+部分 lane；净省&lt;5% 且精度不守；环/多父背压变差；抬旧 0.09% 当 X → **停该挂 DAG 布局**（家族保留）。

## F4　末5 部分合并 + 证书

- **B**：仅末5 跨 RNE 近似合并 + 轻量证书；不确定回退原半步图。  
- **A**：lifting 末5 门前推权限；gate_collapse 全合并负结果边界。  
- **X（待证）**：范围缩到末5 + 显式证书/回退（≠已停全合并）。  
- **对照**：原 159+35；全合并 241（负）；ordinary cutoff；无证书盲目合并。  
- **杀**：含证书后成本≥原图；未评函数报 AEE；回退率高净服务≤原图；瓶颈非 RNE 却强开 → **停该部分合并布局**。

## F5　依赖生存期 × 中间量释放

- **B**：约束同时未完成广播组数，T10+PED 消费完才释放 lifting 中间量/S。  
- **A**：Avalanche 排列/完成写出（借入）；Gustav F_live/NR4；lifting 中间写回语义。  
- **X（待证）**：把非因果延长占用的生存期**写入训练目标**（≠原样 SpMM last-use）。  
- **对照**：固定双组配对；滚动择优；hidden50；Avalanche 无训改图。  
- **杀**：乐观上界已无 5–10% 余量却仍训；训后不胜固定配对/窄稠密；只减局部字数 → **停该生存期布局**。

## F6　有界 clip × lifting 可取消列

- **B**：严格区间证门提前完成；组内消费者全完成后取消对应 lifting/源时间列生产。  
- **A**：残差链 BN/门/消费者；finite 请求组；SpikeX/PACT 作对照底座。  
- **X（待证）**：取消是否在 lifting **列结构**上足够集中以偿还比较/mux。  
- **对照**：同裁剪逐门界；整字退休；PACT；SpikeX；ordinary 列取消率。  
- **杀**：不胜整字界+PACT；收紧破 AEE 门；比较开销吃掉节省 → **停该 clip×取消布局**。

## F7　Gustav 打包对齐 lifting 因子组

- **B**：CPTB/NRV/源∩W 物理打包边界对齐 lifting 因子组/半步写回。  
- **A**：Gustav 物理链（底座）；lifting matchings/半步图。  
- **X（待证）**：仅打包/行导航与 lifting **共设计**（Gustav 本身≠X）。  
- **对照**：同 Gustav + ordinary 源序打包；逻辑 NRV；无对齐双指针。  
- **杀**：物理字/周期不优于 ordinary 序+同 Gustav；只改善逻辑代理；破坏同端口公平 → **停该对齐布局**（不杀 Gustav）。

---

## ROUND1 建议筛序（box，不抢 Codex）

1. **已有表达力**：F2(MP2)、F4(MP4)、F1 剪源字控制(MP3)、通用融合 iface(MP5) — 只当通路沙盒。  
2. **下一刀优先测杀**：F1 索引/对齐税是否吃掉删字收益；F4 证书路径 vs 原图有效成本；F7 对齐 vs ordinary 序同 Gustav（若有最小物理包）。  
3. **后置**：F3（Prosperity 旧负多）、F5/F6（偏训练，RTL 轮次难一次杀清）。  
4. **每刀产出一行**：候选 | 测了什么 | PASS/KILL | 停哪层布局 | 是否误抬 X。

Codex 主线（Stage B / open_fusion_execution / ready·completion）**禁止**本轮 divert。
