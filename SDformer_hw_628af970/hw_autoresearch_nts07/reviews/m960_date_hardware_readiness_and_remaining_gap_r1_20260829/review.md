# M960 | DATE 硬件就绪度与剩余缺口独立审阅

## 总裁决

截至 M959，硬件证据完成度约 **58%**，DATE Accept readiness 为
**3.4/5**：Borderline Reject / Weak-Accept 边缘。若今天冻结投稿，独立估计
accept 概率约 35%–45%。强项是证据纪律、C2 等带宽物理面积效率、C1
功能闭合和 C3 setup/area；决定性短板是 C1 新结构尚无 DC/RTL-cycle、
decoder 不完整、最终 checkpoint 未绑定、全系统 Table-A 仍为零行。

P0/P1/P2 = **3/6/4**。本审阅只读；未改源码，未运行 EDA/GPU/remote。

## 分线状态

| 线 | 完成度 | 已完成且可保留 | 不等 checkpoint 继续 | 必须等 checkpoint | 预计剩余 |
|---|---:|---|---|---|---|
| C1 | 60% | M959 foundry-UNIT_DELAY 功能 negative-attack PASS；M623 九宏 parent-scratch 模型能量 -38.23% | 先解决 M934“任何 assertion fail 禁止 DC”与 M959 单条预期 fail 的门槛冲突；RTL 同账本 replay；M935 macro-aware DC；若过门再 Formality/PT，并补全 213,376 B 存储义务 | 最终 parent-hit/activity、SAIF/PTPX、能量与可能的参数重综合 | ckpt 前 1.5–3 天；失败一次再加 1–2 天；ckpt 后 0.5–1 天 |
| C2 | 75% | 1.0167× 等带宽周期、4.541× throughput/mm²、77.61% logic area saving；3 ns setup/area | SRAM 固定延迟/宏账、Formality/PT hold、component-only SAIF/PTPX | trace 加权周期、最终 value class、energy/frame 和系统贡献 | ckpt 前 0.5–1.5 天；后 0.5 天 |
| C3 | 60% | Fixed-T10 logic-only 62,433.5 µm²、3 ns setup PASS | PT hold/STA、系数/状态存储、Formality、吞吐分母 | 最终 T/rank/系数、利用率和动态功耗；参数变化时重综合 | ckpt 前 0.5–1.5 天；后 0.5–1 天 |
| Decoder | 35% | D0 一行 exact diagnostic；D0/D2/D3 exact-support；D1 输入 exact 0/theta；M950 bounded-prefix source 已封 | D1/D2/D3 10K/100K 门、streaming 资源验证、四层聚合器和 D1 numeric bridge 预备 | 最终 D0–D3 payload、D1 数值类、四层同资源/multi-sequence replay | ckpt 前 0.5–1 天；后 1.5–3 天 |
| System | 15% | C2 component annex 1 行 | 十算子/17 宏 schema、基线、地址时序内存模型、聚合 validator | ordered trace、decoder-complete cycles、SRAM/DRAM、SAIF/PTPX、energy/FPS、Table-A | ckpt 前 0.5–1 天；后 2–3 天 |
| Checkpoint rebind | 10% | M940 失效/复用草案；最后封存监测仍是 epoch5、无 checkpoint | 准备 identity/load/export/range/trace/replay 工具；并行做物理闭合 | 选最终 epoch、valid825、missing=0、整数导出、多序列 trace | 外部等待未知；checkpoint 出现后 1–1.5 天才进入 replay |

## 当前能写、不能写的数字

- **C1：** M935 模型 `438,541,979 cycle`、`1.7338× vs zero / 1.7283× vs bit`
  仍是 projection，不是 RTL measured 或 timing-admitted。M959 只能写成带限定语的
  functional negative-attack PASS。M623 的 `38.2283%` 仅是九个生成宏的组件模型。
- **C2：** 可以写 `1.0167×` equal-bandwidth directed cycle、`4.541×`
  throughput/mm² 和 `77.61%` logic area saving，三者必须同句写明 K8 vs K1×8、
  logic-only、pre-macro、非系统。
- **C3：** 可以写 28 nm logic-only 3 ns setup/area；不能写 throughput、speedup、
  power 或 paper PPA。
- **Decoder：** D0 `20,548,766` cycles 是单行 diagnostic；`4.0359×` 是请求压缩比，
  不是加速。Prosperity `3.0876×` 是外部 exact-support subset，不是 ours。
- **System：** Full-system Table-A production rows = 0。旧 `620.3M` 分母漏了四层
  decoder，不能复活；decoder sensitivity share 为 21.57%–22.83%。

## 真正关键路径

checkpoint 前不应停工：立即并行做 C1 M935 DC/replay、C2/C3 PT/宏、decoder
bounded-prefix 和 system schema。checkpoint 出现后才串行做：最终选择与 valid825 →
整数导出/多序列 trace → D1 miter/四层 decoder → 全系统 cycle/traffic/energy →
Table-A/B/C hammer。

如果 checkpoint 到位且 C1/D1 都不返工，仍需约 **4–6 个集中日**；若 C1 timing
或 D1 numeric bridge 返工一次，则 **7–10 日**。排除训练等待，剩余约 **8–12
person-days**。从 M959 的封存截点看，8 月 31 日前做到完整 paper-ready 不可信；
可信目标是把 checkpoint-independent 的组件物理门和系统基础设施尽量关掉。

## P0

1. C1 新 M935 没有 timing-closed DC 或 RTL-cycle replay；且 M934 clean-assertion
   DC 门与 M959 单条预期 negative-test assertion 尚未显式衔接。
2. 最终 Motion checkpoint、valid825 和硬件 rebind 均未封存，ep35 的 activity/cycle/
   energy 不能转名。
3. Decoder-complete 与 full-system 同资源证据缺失，Table-A 仍为零生产行。

结论不是停止，而是“双关键路径并行”：物理闭合不等 checkpoint；最终 workload、
decoder、power 与系统表必须等 checkpoint 后统一重放。
