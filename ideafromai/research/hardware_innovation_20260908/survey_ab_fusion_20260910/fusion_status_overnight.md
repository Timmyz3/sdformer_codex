# 融合过夜状态（fusion_status_overnight）

日期：2026-09-10（过夜交付稿）
约束：不抢 Stage B；不改 main.tex / 生产 RTL / nts07；负结果只停布局；RTL 粗验在 **box**，不在 ismd 生产树。

## 1. 现在唯一主实验（锁定）

lifting40 **Stage B**：ordinary dense-source/raw vs `fast_raw_diagonal` 的 same-port / same-state / same-backpressure **schedule compare**。  
`schedule_compare_same_port/` 仍未开工。本目录一切融合建议不得抢档。

## 2. 可试 / 不可试（相对 Stage B）

| 状态 | 对象 | 说明 |
|---|---|---|
| **现在可准备、不可开训** | F1 设计页（对照轴含 HiNM/窄稠密/开源2:4·Wanda） | Stage B **后**可挂；依赖净服务结论 |
| **第二队列可设计** | F2 半步检查点×组共同完成 | 不抢主实验；Stage B 若显示非并集瓶颈则降级 |
| **第二队列谨慎** | F3 Prosperity 联合图 | 家族保留；旧 G4 负结果只停布局 |
| **底座维护** | F5 Gustav 对齐打包 | 借入≠X；可并行准备接口 |
| **旁路** | F7 稀疏注意力 | 不进主岛 |
| **低优先** | F4/F6 | Stage B 后可挂 |
| **禁止** | 改生产 RTL/main.tex；空开融合实验；把节点比写成 PPA | — |

## 3. 建议 RTL 粗验（≤2，在 box 跑）

详见 `rtl_microprobes_for_box.md`。摘要：

1. **MP1 — same-port 背压计数微探针**（服务 Stage B 同分母思想，不做生产树改动）  
2. **MP2 — 组接受/继续 1-bit 控制微探针**（服务 F2 接口可行性粗判）

管理负责在 box 上搭 iverilog/yosys 并跑；本页只给模块边界与对照。

## 4. 文献精读进度（诚实）

见 `p0_deepread_progress.md`（随批更新）。目标 P0=246；未完成不得虚报全文。

## 5. 明早仍缺时的底线

- 至少：进度表诚实 + 已完成 idea 卡全集落盘 + 本页 + 融合修订 + 2 个 RTL 微探针规格  
- 理想：P0 卡接近全集 + `idea_synthesis.md` 完整簇汇总

## 6. RTL 粗验回写（管理｜box）

来源：管理报告；路径 box `/workspace/rtl_microprobes/`；摘要 `OVERNIGHT_RTL_SUMMARY.md`。

| 探针 | 结果 | 含义 |
|---|---|---|
| MP1 same-port credit | C0/C1/C2 仿真 PASS；同拍两侧不同时 serviced；yosys ~179 cells，无锁存 | same-port/背压计分母在 RTL 微探针级可表达 |
| MP2 group accept/continue | B0/B1/B2 互斥断言 PASS；接受约1拍，重算均值约6–7拍；W8~921 / W16~985 cells | F2 组级接受/继续控制通路可行 |

**诚实边界**：不是 Stage B 净服务%，不是 TCAS PPA；不因此提前开训或改生产 RTL。

