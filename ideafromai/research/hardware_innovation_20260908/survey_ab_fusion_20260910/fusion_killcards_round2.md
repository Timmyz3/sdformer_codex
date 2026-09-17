# Fusion ROUND2 短杀门卡（接 ROUND1 记分牌）

日期：2026-09-13  
前提：ROUND1 仅 **cand1 KEEP_PROBE**（16→8）；cand2–5 均 **KILL_LAYOUT**（见 `rtl_fusion_SCOREBOARD.md`）。  
ROUND2 目标：**深化 cand1** + 新试 **F3 / F7 / BN 类**。仍 ≠ Stage B / ≠ 标题 X；不碰 Codex。

通用否决（同 ROUND1）：改同端口合同；通用 CSE 冒充 X；未评函数报 AEE。

---

## R2-A　深化 cand1：`source_commit_preview_bypass`

ROUND1 已证：同 1-port 下「commit 旁路」相对「存后读」周期减半（探针级）。  
本轮要证/杀的是：**旁路在更真的前后依赖下是否还删读写**，而不是再扫玩具 TB。

- **B**：source→preview 路径上，commit/旁路直供，减少 store-then-reload。  
- **A**：同端口/同状态；既有 dual-consumer 到达语义；ROUND1 cand1 模块为底座。  
- **X（待证，极窄）**：不是家族标题；最多是「NEXT_INTERFACE 保序直供少一次读写」的接口差分。  
- **对照**：  
  1. 存后读（ROUND1 基线）  
  2. 旁路 + **假** ready（应不优于真依赖）  
  3. 旁路但 **打乱** source/preview 序（应功能 FAIL 或回退）  
  4.（若有）双消费者同时拉 preview / gate 的并集读  
- **杀（KILL_LAYOUT / 降回 KEEP 边界）**：  
  1. 接上真实/模拟 ready·completion 后周期 ≥ 存后读  
  2. 只在 always-ready 赢、一加背压或第二消费者就打平/变差  
  3. 为赢而放宽端口或打乱保序  
  4. 写成 lifting/Gustav/CSE 标题 X  
- **KEEP 升级条件（仍非 X）**：真依赖下稳定少 ≥1 次有效读写，且功能保序；记作 **KEEP_INTERFACE**，继续探针，不进生产树。

---

## R2-B　F3：Prosperity 联合图挂源 DAG（新试）

- **B**：在（玩具）源 DAG 上选公共虚节点/部分 lane，同端口计划。  
- **A**：Prosperity 森林语义（借入）；ordinary 源图同等联合图权限。  
- **X（待证）**：共享挂在结构化源节点，而非整 W mask。  
- **对照**：无虚节点的逐节点算；ordinary+同等联合图；纯 CSE 折叠。  
- **杀**：同端口服务不优于 CSE/无图；引入多父等待使周期变差；净省薄（<5% 代理）却抬标题；复读旧 −0.09% mask → **KILL_LAYOUT**（家族保留）。  
- **注意**：ROUND1 未测 F3；本轮若 TB 过简，最多 **KEEP_PROBE**，禁止晋级 X。

---

## R2-C　F7：Gustav 式打包对齐（新试）

- **B**：源字/行打包边界对齐「因子组/半步」标签，减跨组拆包。  
- **A**：同 Gustav 风格双指针/NRV 玩具分母；对齐 vs 不对齐。  
- **X（待证）**：仅打包导航共设计；Gustav 本身 ≠ X。  
- **对照**：同链 + ordinary 源序打包；逻辑计数代理（应弱于物理字/周期）。  
- **杀**：物理字/周期不优于 ordinary 序；只改善逻辑行代理；给一侧多 bank 破公平 → **KILL_LAYOUT**。

---

## R2-D　BN / 动态范围类（新试，窄刀）

候选方向（择一刀，勿并行摊开）：
1. **BN 后 clip 区间 → 门提前完成**（F6 玩具子集）  
2. **固定 BN vs 动态 BN 在同端口下的额外读/写税**（对照，非 X）

- **B**：用范围/BN 状态减少门或列上的无效工作。  
- **A**：残差链门比较；clip/PACT 对照底座。  
- **X（待证）**：取消/提前完成是否集中到可偿还的列/组（极难在玩具 TB 成立）。  
- **对照**：无 clip 全算；整字退休；同精度静态窄层。  
- **杀**：比较/mux/BN 状态税 ≥ 节省；精度代理破门；只减旗标不减 job 周期（同 ROUND1 cand2 模式）→ **KILL_LAYOUT**。

---

## ROUND2 建议筛序

1. **R2-A 深化 cand1**（最高优先：用户/Codex 也盯「保序直供是否真删读写」）  
2. **R2-C F7 对齐**（纯数据路径，易杀清）  
3. **R2-B F3**（易虚报，杀门从严）  
4. **R2-D BN**（最后；易与训练混淆，RTL 轮次只杀「税≥省」）

每刀一行记分：候选 | 测什么 | PASS/KILL | 停哪层 | 是否误抬 X。  
已杀布局（cand2–5）**不复活**，除非对照轴与杀门理由被显式推翻。

