# RTL 微探针规格（≤2｜仅 box，不写 ismd 生产树）

给 **管理** 在 bot 本地电脑（box）用 iverilog/yosys 粗验。与本工作融合相关，但 **最小接口**。

---

## MP1 — `same_port_credit_probe`（优先｜服务 Stage B 思想）

### 目的
粗判「同端口 / 同背压」计分母是否可在 RTL 级用极小信用计数器表达；**不是** 完整 schedule_compare。

### 模块边界
- 输入：`req_a`, `req_b`（1-bit 请求，模拟 ordinary vs lifting 两源争用同一服务口）
- 输入：`grant`（1-bit，端口每拍最多服务 1）
- 状态：`credit`（小数位宽，如 4-bit），`stall_a/stall_b` 计数
- 输出：`serviced_a`, `serviced_b`, `stall_count`

### 对照
| 变体 | 行为 |
|---|---|
| C0 基线 | 固定优先 A 再 B |
| C1 公平 | 交替/轮询 |
| C2 背压 | credit 用尽则两边 stall，恢复后按 C1 |

### 成功粗判
- 波形/计数显示 **同一时刻只有一侧 serviced**（same-port）
- 两变体在相同 `req` 轨迹下 stall 计数可复现
- yosys 能综合出无锁存推理（组合环）的净表

### 明确不做
- 不接 nts07 / 生产 PSN
- 不宣称净服务%
- 不写 ismd 路径

### 建议目录（box）
`/workspace/rtl_microprobes/mp1_same_port_credit/`（`probe.v`, `tb.v`, `Makefile`）

---

## MP2 — `group_accept_continue_probe`（次优｜服务 F2）

### 目的
粗判「一组门字：预测接受 vs 继续重算」控制通路的时序与互斥；对应 F2 半步检查点接口可行性。

### 模块边界
- 输入：`pred_valid`, `pred_accept`（组级 1-bit）, `true_gate_bus`（宽度参数化，默认 8）
- 输入：`recompute_done`
- 输出：`use_pred`, `issue_recompute`, `retire`
- 互斥：`use_pred` 与 `issue_recompute` 同拍不得同时为 1

### 对照
| 变体 | 行为 |
|---|---|
| B0 | 永不接受预测（总是 recompute） |
| B1 | 一律接受预测（安全开关关掉） |
| B2 | `pred_accept` 控制；拒绝则 recompute 后 retire |

### 成功粗判
- 断言互斥成立（tb 检查）
- 接受路径 retire 延迟 < 重算路径（可用拍数计数）
- 与「独立 per-bit 预测」相比，组级控制线数量更少（参数扫描 8/16）

### 明确不做
- 不接真实 θg/PED
- 不训练
- 不在 ismd 落 RTL

### 建议目录（box）
`/workspace/rtl_microprobes/mp2_group_accept/`（`probe.v`, `tb.v`, `Makefile`）

---

## 与融合候选映射
- MP1 ↔ Stage B 同分母思想 / 间接支撑 F1 排队理由  
- MP2 ↔ F2  
- **不选** F3/F7 做 RTL：接口过大或旁路主岛
