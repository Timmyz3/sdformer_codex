# sources_read.md — 实际读过的路径与关键摘录出处

> 说明：本执行端为 box-scoped，未能经 zmd `machineId` SSH 直连 ismd；证据来自与远程 HEAD 对齐的本地跟踪树  
> `sdformer_codex@22fc4aa1`（`ideafromai/research/hardware_innovation_20260908/`），并交叉先前会话落在 box 的 Codex 复核摘录。  
> 远程权威路径仍以 `/home/zhumd/work/ideafromai/...` 为准；落盘目标即该树下本目录。

## 必读（已读）

1. `…/hardware_innovation_20260908/README.md`  
   - 顶栏工单：Prosperity/Gustav 收口→第二队列；下一主实验锁定 lifting40 Stage B（ordinary dense/raw vs `fast_raw_diagonal`）；shared 仅消融；禁止训练/量化/RTL/主稿修改。  
   - 相对 AEE +0.013178 未过 0.005；绝对 1.259 过。

2. `…/hardware_innovation_20260908/CURRENT_LINES_AND_PLAN_20260910.md`  
   - §1 晋级补正；§2 主候选/支线/第二队列表；§5 四步：先共同执行比较→≥15% 潜力才成对恢复→再写 X→隔离 RTL。  
   - 明确负结果只停布局；TCAS-II 语境；不改生产树/主稿。

3. `…/fast_temporal_recovery_lifting40/net_cost_one_page.md`  
   - AEE / CSE / 十帧代理 / shared 相对 raw 分项；下一步同资源排程；gate_collapse 停全合并。

4. `…/fast_temporal_recovery_lifting40/implementation_inputs.md`  
   - Stage B 输入表：两轴 npz/常量/DAG/数值实现路径；r1 服务边界；可复用 finite_service / Gustav resident 边界；明确 `schedule_compare_same_port` 应先排静态 source 图；三源 gate 捕获缺口。

5. `…/prosperity_gustav_reopen_20260910/README.md`  
   - 三探针结果与「未证明标题级 X」；收口第二队列；不挤 Stage B；不杀家族。

## 目录扫描

- 主战场顶层、`lifting40/`、`psn/`、`motion/` 存在。  
- `schedule_compare_same_port/`：**不存在**（顶层与 lifting40 下 find 无同名交付目录）。

## 对齐核实

- git HEAD：`22fc4aa195518d569ca3f88d39b9ddef57649551`（短 `22fc4aa1`）— 与「已知对齐」一致。  
- Codex 会话 jsonl：本端未直读远程 `~/.codex/sessions/...01a07507....jsonl`；交叉了 box 上既有复核 `/workspace/codex_review_01a07507.md`（结论：Stage B 未开工、近期曾偏航 Prosperity 叙事，后由 README 工单拉回 Stage B）。

## 关键数字摘录出处

| 陈述 | 出处 |
|---|---|
| raw AEE 1.232979… / shared 1.247809… | `lifting40/fixed_lifting_valid825/result.json`（`net_cost_one_page.md` 表） |
| dense/raw AEE 1.219801…，Δ=+0.013178 | `CURRENT_LINES…` §1；`net_cost_one_page.md` |
| 260 vs 159+35 RNE；十帧~15% 代理 | `net_cost_one_page.md` |
| shared−raw：−1.130% / −4.442% / +0.166% NRV | `net_cost_one_page.md`；`source_cost.json` |
| Prosperity mask +0.09045% 级薄增量 | `prosperity_gustav_reopen_20260910/README.md` |
| Stage B 未开工、输入已定位 | README 顶栏；`implementation_inputs.md` |
