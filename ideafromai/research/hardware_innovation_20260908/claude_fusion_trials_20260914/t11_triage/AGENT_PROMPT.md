# T11a 全库分桶任务指令（每个 agent 处理一个 batch）

## 任务

读取 `t11_triage/input/batch_<N>.json`（100 条论文记录，字段 id/name/title/venue/cat/A/B/X/untried/code），
对**每一条**给出五选一处置，把结果**增量写入** `t11_triage/output/batch_<N>.jsonl`
（每判完 ~25 条就用 Bash `cat >>` 追加一次，防中途失败丢全部进度）。

每行 JSON 格式：
```json
{"id":"W0123","name":"...","disposition":"...","family":"...","reason":"一句话理由"}
```
APPLICABLE_NEW 额外必填 `"how"`（套用方式，具体到本项目哪个数据/通路）和
`"trial_type"`（"numeric" 或 "rtl"）。

**必须覆盖输入文件里的全部 100 个 id，一个不漏。**

## 项目语境（判断基准）

SDformerFlow：事件相机光流，Swin 风格脉冲 Transformer（SNN），DSEC 数据集，
软硬件协同设计，目标 TCAS-II 5 页短文（只容一个机制）。当前在找硬件创新点。

- 推理输出为二值 {0,θ}（AT-LIF；θ 推理时折入下层 W）；内部有连续通路
  （I24 残差、PED）供门判决使用。
- 现有硬件资产：C2 权重广播 TSBG 调度核（Verilator/UVM 已闭环）。
- **当前最强候选 C1（已 RTL 零差）**：组级块浮点（10 判决共享指数）+
  MSB-first 位平面串行供数 + 逐判决精确区间证书锁定即停；供数拍比 17.3–17.7%
  （诚实 FX 基线 24 拍/组）。
- 质量门：AEE 优于 1.445353（valid825）；准入门：同端口/同状态/同背压净服务 ≥15%。

## 已否决方向（命中即 FAMILY_COVERED，不必立项）

1. 块级聚合/块级剪枝/块级条件完成（粒度定律：384 通道求并后 78.5% 块必有失败通道）；
2. 静态 per-lane 位深（57.9%，弱）；静态锁深表（误 fire 37–38%，锁深是样本性质）；
3. 事件前端运动对齐 delta（事件已是时间对比编码，warp 仅降 4.8%）；
4. 连续稠密状态上的舍入跳过证书（命中率 0.04%）；
5. 逐词优先级打包（11–12% 但同端口元数据 +5 拍 → 32%，元数据税）；
6. 通道序整词供数（59–61%）、仅符号终止（2.3–7.1%）；
7. I24 消费者直接位平面化（冷启动 -14~16%，偏工程适配）；
8. 加速器平台级宏指标（TOPS/W、突触事件/s）当净服务——不接受平台旁路计费；
9. 通用 LLM/大模型专用机制（域失配：本网小尺寸、无长序列、无 MoE）。

## 五类处置定义

| 处置 | 判据 |
|---|---|
| INCORPORATED | 机制已在 Claude 侧 T1–T10 试验中实际照抄/融合（BitFair式动态终止、R-Sparse分流、MotionDeltaCNN对齐、lifting40舍入证书、ConvReflex静态锁深、AO-BFP打包、EITCE通道序、ECHO符号、BitL LUT、MINT查表等）或已被 Codex 侧 fusion_ten_trials 实测 |
| FAMILY_COVERED | 机制家族已被上述试验/否决方向覆盖，无需单独立项（注明被哪个家族吸收） |
| APPLICABLE_NEW | 机制未被覆盖、与本项目语境（事件相机/SNN/位平面供数/证书终止/门路消费者/背压计费）真实适配 → 给出 how + trial_type |
| APPLICABLE_BLOCKED | 适配但被硬条件挡（付费墙/代码不公开且机制不可从摘要重建/域失配依赖），注明挡点 |
| NOT_APPLICABLE | 域完全失配（非 NN 加速/非相关算法；纯理论/数据库/图计算/机器人控制等） |

## 判决纪律

- **宁严勿滥**：绝大多数 arXiv 关键词噪声（PDE 谱方法、最优传输、GPU 故障、
  3DGS、视频生成、联邦学习等）直接 NOT_APPLICABLE。
- APPLICABLE_NEW 只给**每个机制家族最强的代表**，同族其余标 FAMILY_COVERED。
- reason 用一句中文说清关键差分或失配点。
- 只依据输入记录内的信息判断，不做网络检索。
- 用 `python3`（无需 numpy）做 json 读写与计数。

## 完成自检

写完后运行一次核对：输出行数 == 100、id 集合与输入完全一致、disposition 枚举合法。
在最终回复中报告：五桶计数 + APPLICABLE_NEW 的 id 列表。
