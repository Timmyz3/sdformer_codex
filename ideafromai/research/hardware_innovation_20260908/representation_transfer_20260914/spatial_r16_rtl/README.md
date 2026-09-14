# Spatial R16：完整普通空间因子 RTL

已经完成原生 96×4×4×T10 source → Q1 竖 3×1 → 精确 Z15 → Q2 横 1×3 → raw p32 → FP32 identity/J20/wide/I24。A 是普通空间低秩分解的完整迁移；双 P、最终 Z 支持、三 tap 权重复用与普通消费者为实现底座，不据此声称新颖 X。网络质量由根代理的同网实测报告承担，本目录不借旧浮点/R8 AEE。

| 64 tile 集 | 直接 OS raw core | 因子 raw core | 直接 OS raw 冷服务 | 因子 raw 冷服务 | 因子完整 I24 冷服务 |
|---|---:|---:|---:|---:|---:|
| held | 2,337,920 | 2,079,514 | 2,446,720 | 2,179,098 | 2,334,322 |
| disjoint | 2,742,188 | 2,256,006 | 2,850,988 | 2,355,590 | 2,510,814 |

冷服务 = 每组首次模型配置 + 每 tile 的 1536 source 配置、1 原点配置、1 启动拍 + 全部计算/输出；权重配置不在 64 tile 内重复收费。直接展开每模型 10,368 拍、因子 1,152 拍；完整消费者另 24 拍。ready raw 的同函数冷服务分别下降 **10.9380% / 17.3764%**（core 11.0528% / 17.7297%）。主分母已转为 `../spatial_r16_direct/OS_SUMMARY.json` 的 output-stationary 同预算 bitmap 强臂，已独立 572 命令与完整逐状态验证；其输出逐值等于同一整数因子函数。旧 `SUMMARY.json` 的 Gustav event/psum RMW 臂仍保留，29.5488% / 38.4415% 只属于该旧臂，不能作为主增益。此处没有推测直接展开的完整 I24 周期。

资源并非等面积：双方执行资源合同为 8×32 ALU、单 256 bit 权重服务和同 p_mem；因子另实际使用 8 个 19×13 乘法，直接展开可用但闲置。直接展开 W32 为 331,776 B，因子静态 Q1/Q2 为 12,096 B；双方均有 1,280 B 八 bank 中间存储，直接 OS 实存 1,080 B 位图，因子实存 Z。因子另有 **Q2 cache 312 B、静态支持 144 B、最终位置支持 80 B**。完整资源和端口/寄存器边界见 `resource_contract.json`，无综合面积、时序或能耗结论。

Q1 将相邻 x 两个 Z15 放在同一 32 bit bank 字，通过 bit15 carry cut 复用八条 ALU；每个合法输入先读取 16 字局部窗口，再由真实门字生成 2×4 个 T10 掩码。两个 R8 条带重读 source 的费用全部计入。Q2 在完整 Z 扫描之后使用最终支持，每个 N8 输出组实际预取 R8×3tap×N8 到 312 B cache；Z15 符号扩展到 19，再乘 signed13。stripe0 写 p_mem，stripe1 实读并累加；最后才向消费者按原 og/P/T/lane 顺序输出，任何阶段没有中间 RNE。

位宽仅对固定模型 admission：Z ∈ [−8562, 7427]，任意 Q2 前缀绝对界 504,162,009；因此 signed15 与 signed32 足够。`prepare.py` 对每个 fixture 重算原生门、Z、p，并与独立展开 W 卷积及导出 real gold 比对。全零/全一、随机、尾项、合法 q1 正负方向和 padding 污染均保留固定因子，未用截断伪造宽度支持。

验证：raw 与完整消费者各 **572 命令**，均含 15 小集、held128–191、disjoint4000–4063，ready/BP 和不 reset 连续两遍；总计 142 个独立 fixture（135 真实、7 合成）。每一路 raw/J/wide/I24 分别检查 2,196,480 值、Z 732,160 值；raw-only 再独立检查同量 raw/Z。所有数值、逐状态与资源计数、消费者成本恒等式、跨遍同值/同周期通过。BP 在实际 source/weight/raw/identity/output 服务上发生。报告见 `raw_verification.json`、`stream_verification.json`，逐命令 `results_*.jsonl`、`stream_*.jsonl`。

当前 ready Q2 乘加占 core 的 held 56.49%、disjoint 55.24%，每 tile 平均 18,353.8125 / 19,471.3125 个实际 scalar-Z×N8 服务；Q1双P分别 1,604.65625 / 2,110.28125。固定 source 装载阶段 3,264 拍，Q2 cache 装载 576 拍、最终支持扫描 80 拍、两条带位置装载/存储 1,920 拍，均未藏在 oracle 中。完整普通消费者 ready 每 tile 增加 2,425 拍。最值得另开有界接口的是连续 Q2 的三 tap 空间重用/变换，并同时收费输入变换、重建和新增位宽；本目录保持普通 A，新的函数/位宽必须使用自己的 gold。

复现入口：`run.py --stage all --prepare`；也可 `--stage raw` 或 `stream`。`spatial_core.sv` 为生产者，`spatial_stream.sv` 接复用的单上下文 `i24_consumer.sv` / `wide_phase_alu.sv`；后者仅追加 wide debug 端点，算术与原消费者一致。配置接口冻结于 PLAN，source origin.hex 是物理 source 窗口原点（output origin−1），不能再减一次。只在本目录写入，无生产、训练、量化调整、EDA 或 Git 提交。
