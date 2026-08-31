# M320：M311 matcher 时序架构预设计打铁

结论：92/100，P0=0、P1=1、P2=2。建议下一版 RTL 采用 **两级 elastic 4×4 balanced tournament**。这只是 `GO RTL`，不是 timing、frequency、area 或物理实现 admission。

## 135-level 根因

M314 最坏 setup path 从 `in_pattern[4]` 到 `out_selected_pattern_reg[15]`：data arrival 2.8836 ns、required 2.8841 ns、slack 0.0005 ns。QoR 报告 135 levels / 2.68 ns critical path。

独立解析首条路径得到 135 个组合 cell arc：4 个 adder、15 个 mux、28 个 buffer/inverter、84 个其它比较/选择逻辑、4 个 XOR。资源报告另有 16 个 popcount/distance datapath block 和 15 个 16-bit tie comparator。RTL 中 `best_distance/best_center` 在 16-center `for` 循环内递推，形成 15 级 source-ordered compare/mux winner chain；这才是主要深度来源，不能只在模块前后加寄存器。

## 三方案比较

| 方案 | 新增 pipeline FF | 估计总 FF | winner 层数/级 | 两周期 latency | II=1 | 面积口径 | 决策 |
|---|---:|---:|---:|---:|---:|---|---|
| input register slice + 原串行 matcher | 275 | 323 | 15 | 是 | 是 | baseline + 796.95 um² direct FF | NO-GO |
| 8×2 tournament | 66 | 114 | 3 / 1 | 是 | 是 | 2160–2280 um²，非 DC | backup only |
| 4×4 balanced tournament | 108 | 156 | 2 / 2 | 是 | 是 | 2280–2400 um²，非 DC | **GO RTL** |

单纯 output slice 可少加寄存器，但也完全不切断 matcher 组合路径。input slice 虽去掉 0.2 ns external input delay，却把 256-bit center bundle 全部寄存，并保留 15 级 winner recurrence，不能解决 M314 P1。

8×2 比 4×4 少 42 个 FF，按当前 enabled-FF 单元直接估算少约 121.716 um²；但第一级是 popcount 加三层 winner，第二级只有一层 winner，时序明显不均衡。为解决 0.5 ps setup margin，4×4 多付这部分寄存器更合理。

## 4×4 推荐数据通路

候选统一表示为 `{distance[4:0], center[15:0]}`，比较顺序为 `(distance, packed unsigned center)` 的升序最小值。该规则是全序，pairwise winner 具有结合性；因此 serial、4×4 和 8×2 在任何分组下保持相同 minimum-distance 与 lower-packed-center tie 语义。

Stage 0：

- 对 16 个 center 并行执行 XOR 和显式 balanced popcount16。
- 另算原 pattern population。
- 每四个候选通过两层 balanced pairwise winner，得到四个 local winner。
- 注册四个 `{center16,distance5}` 共 84 bit，加 original16、population5、tau2，共 107 data bit；另有 valid，共 108 FF。

Stage 1：

- 四个 local winner 再经过两层 balanced winner。
- 使用随 transaction 携带的 population/tau 执行 `population>=2 && best_distance<=tau`。
- 生成 selected/original、distance、population、tau、snapped、exact、positive，写入现有 47-bit output payload + valid。

只复位两个 valid；invalid 时 payload 是 don't-care，避免给 107-bit payload 增加无意义 reset fanout。accepted tau 的 admission 域仍严格为 0/1。

## Elastic ready/valid

采用两格无旁路 elastic pipeline：

```text
s1_ready = !s1_valid || out_ready
s0_ready = !s0_valid || s1_ready
in_ready = s0_ready
```

每一级只在本级 ready 时更新 valid/payload，否则必须保持稳定。若 output stall 且 s1 满，s1 冻结；s0 空时仍可吸收一个新 transaction，随后两级都满并反压输入。释放 stall 时可同拍 retire 旧 s1、将 s0 推入 s1、并接受新输入，不丢失、不重复、不乱序。

模型验证：4096 个 continuous-ready transaction 在 4098 cycle 内完成，连续接受 4096 个、填充后 II=1；另有 20,000 transaction 随机 offer/backpressure，6704 stall cycle、最大 occupancy=2、0 mismatch。该结果只证明架构协议模型，必须由新 RTL 的 VCS/SVA 重做。

## 时序和验证门槛

仍使用冻结 3 ns pre-macro SDC，但把以下作为 RTL admission 门槛而非结果预测：

- 每一级 worst setup slack 必须至少 0.50 ns。
- 每一级 mapped logic levels 不超过 60。
- exact-SHA VCS 对 M311 serial oracle 全字段 0 mismatch。
- pop<2 guard、tau0 exact-only、tau1 distance0/1、local/global lower-center tie 全部有 assertion 和 cover。
- 连续 ready 至少 4096 次连续 accept；两级满 stall、空 s0 吸收、同时 retire/shift/accept 必须覆盖。
- `accepted-retired == valid0+valid1`，occupancy 始终在 `[0,2]`。

若 4×4 未达到 0.50 ns margin，优先把 popcount 单独切成第三级，而不是退回 8×2；但这会显著增加 16 个 candidate tuple 的寄存器成本，必须重新 DSE。

## Claim boundary

本里程碑仅 admission：M314 135-level 根因、balanced tournament 数学等价、两级 elastic 协议模型，以及 **4×4 为下一版 RTL 的唯一 GO 方案**。

不 admission：M320 RTL、VCS/SVA/Formality、DC 面积/时序/频率、0.50 ns 实测裕量、中心 SRAM、功耗能效、完整 Conv、accuracy、系统加速或论文 headline。

复算：`python3 results/m320_m311_timing_architecture_predesign_r1_20260825/analyze_m320_m311_timing_architecture.py`。
