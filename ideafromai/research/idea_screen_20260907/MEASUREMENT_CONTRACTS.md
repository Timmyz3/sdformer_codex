# 测量合同（给 Codex，本 Agent 不跑）

每条都是「先有收据再 RTL」。禁止用 ep35 数字填 ep34 表。

冻结对照：Motion C12 ep34 SHA 前缀 `4bbaf7fc`，valid825 AEE 1.199514。  
新精度门：AEE ≤ 1.259 且 < 1.5848。

---

## M0 — ep34 注意力份额（原 T0，决定 O1 能不能当电路面）

**问：** 注意力分数叶、Q/K 投影、瓶颈 Conv、FC1/FC2、解码、ATLIF T=10 各占信封多少？  
**输入：** ep34 与 C1/C2 同一账本的周期或 MAC-proxy。标 [model] 或 [VCS]。  
**过门：** 叶 < ~2% → O1 只能当叶论文，摘要禁止系统倍率。  
**现状：** 未做。历史 ~0.59% 是旧信封。

## M1 — ep34 QK 普查（把本目录 ep35 脚本换身份重跑）

脚本可参考 `screen_all.py` / `analyze_t1_t4.py`，但必须换 **ep34** packed Q/K。  
输出与 `T1_T4_ep35.json` 同字段：token dirty、K 零、leaf_needed、**15×15 窗 OR dirty**、run length。  
**过门：** 窗干净比例若仍 ~10% 且 S3=0，禁止卖整窗 Shiftmax memo。  
**现状：** 只有 ep35 路径 100 样本。磁盘上 **没有** 同格式 ep34 QK 包（有过 failed valid825 QDQ 尝试，`m2044_..._FAILED_DO_NOT_CITE`）。

## M2 — Shiftmax 行与 token 对齐（原 T1 生命门）

**问：** 分母属于「一个 query 对 450 key」还是「一个 head 的 15×15」还是 RTL 另一行？  
在该行粒度上重算 dirty。  
**过门：** 行 dirty ≈ 100% → dirty 只当能量门。  
**现状：** 用 15×15 OR 做了 ep35 预演，dirty 90.1%。

## M3 — TSBG 同资源 RTL（O5）

对照：同等 bank/端口/行缓存容量的 ordinary LRU。  
候选：B8（B4 诊断）。  
**过门：** 局部周期 ≥1.15×，或周期退化 ≤5% 且权重字节 ≥30%。CPU 3.89× **不算**过门。  
身份：已有 ep34 CPU `tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902`。

## M4 — C2 共享和 vs 直接 FTP（O4）

同一 FC2 捕获、同一 Y 端口/位宽/反压。  
候选：A6 有界槽；对照：直接完整 T FTP；可选 RSR++。  
**过门：** 完整时间线（含排空）不差于 FTP。加法 −42% 不算过门。

## M5 — stage2 单块换核 10 帧 AEE（O3）

从 ep34 微调或冻结权直接换（先冻结，炸了再微调）。  
三路：Motion-XOR / 公开 SDSA / QKFormer 线性 Q-K。  
**过门：** 子集 AEE 方向清楚；全量才引用 1.259 门。  
**禁止：** 12 块一起换。

## M6 — PAFT 仅在 ep34（O6）

旧 PAFT-ep4 running AEE≈1.47 **禁止**当精度证据。  
若做：从 ep34 训，running-BN 推理，valid825。

## M7 — 晚知 θ 接真实 BN（O7）

A8 RTL 已功能过。输入改成网上真实区间/残差，输出恢复率、带宽、错误恢复次数。  
合成 TB 区间不算网上证据。

## M8 — 解码器完整表（O8）

现有 `m1681_ep34_decoder_d0_shard_*` 不是 Table-A。需要 D0/D2/D3 闭合行。

---

## 明确不是本 Agent 的下一项

- 训练、overlay 改核、新 RTL、DC/PT、ZCU102 上板  
那些全部是 Codex（或用户点名的实现 Agent）。
