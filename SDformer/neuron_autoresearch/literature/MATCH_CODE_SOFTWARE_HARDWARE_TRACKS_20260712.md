# Match-Code 软件候选线与硬件实现线（2026-07-12）

## 1. 共同边界

- 软件候选必须是 all12 同构 attention、one-sided binary ATLIF105、无 SC/TX stage mix、无
  old native carrier。
- short run 只检查实现；算法判定必须完成 DSEC full30 + standard valid825。
- 硬件线分为 bit-exact 实现优化和改变数学语义的候选。后者必须回到软件线完整训练。
- `total_spikes`/`energy_uj` 是 spike-activity proxy，新增 matching、projection、SRAM/NoC 和
  control 必须单列。

## 2. 软件线

| ID | 公式核心 | 论文迁移 | 当前状态 | 主要风险 |
|---|---|---|---|---|
| H73 DE9 | 3x3 跨时 offset 的 n11/n00 双证据 Shiftmax，18维静态 codebook | event-flow correlation + dual evidence | 已实现，H66 后 full30 排队 | silence 证据可能过强；18xD 投影增加参数/算术 |
| H74 MC49 | 固定49-offset alpha-XNOR Shiftmax，49维静态 codebook | EEMFlow 固定位移描述子 | 已实现，H73 后 full30 排队 | halo/带宽与49xD投影成本高 |
| H75 AX17 | 横纵半径4、中心共享的17-offset alpha-XNOR Shiftmax | Flow1D 二维搜索正交分解 | 已实现，H74 后 full30 排队 | 无完整二维联合 offset；属于启发式迁移而非原论文复现 |
| XD13 | 13-offset 匹配后动态聚合 K | sparse correlation | 暂不启动 | 保留 weighted-K，硬件叙事弱于 H73/H74 |

H73/H74 的创新假设是：事件光流需要显式保留跨时位移通道，而 H60 的逐 token 标量 gate 会丢失
位移索引。静态 codebook 把固定描述子映射到 D 通道，不读取动态 K value carrier。两项都采用
同一 TTX checkpoint 和 full30 协议，结果可直接比较。

主线晋级顺序：先满足 NB0 AEE +5% 与 spikes -20% 硬门槛，再比较 H60 AEE1.5016/AAE9.8431，
最后加入 matching/projection/SRAM 操作成本。只对最终前二名补随机种子。

## 3. 硬件线

### 3.1 DE9 数据通路

每个 query token/head 读取另一时刻的 3x3 K halo。九个 offset 对 32-bit Q/K 做 bitwise event
match 与 silence match、popcount，再分别进入 Shiftmax9。18个定点描述子通过静态
`18x32` codebook 投影。codebook 可按 head 常驻片上 SRAM/ROM；不能把静态权重误写成“无乘法”，
除非 RTL 实际采用 shift-add/低比特阵列并给出综合结果。

### 3.2 MC49 数据通路

49-offset 表固定，因此 address generator 不需要学习型 router，但需要更大 halo/line buffer。
每路对32 lanes做 alpha-XNOR score，Shiftmax49 后进入静态 `49x32` codebook。它是精度上界，
不是默认低成本设计；论文必须报告有效 offset 读取、边界 mask、codebook traffic 和投影周期。

### 3.3 可叠加的 bit-exact 优化

- T=2/head_dim32 的 64-bit temporal-pair co-residency，减少是否成立取决于 baseline address trace。
- 固定 offset 顺序与 K halo reuse，只改变调度和缓存，不改变 score。
- codebook weight-stationary：每 head 权重常驻，descriptor 流式输入。
- zero descriptor/invalid boundary clock gating，只跳过数学上严格为零的操作。

Progressive pruning、offset early exit、近似 codebook 压缩会改变输出，必须作为新的软件候选重新
full30，不能直接计入 H73/H74 的精度结果。

### 3.4 AX17 数据通路

AX17 读取另一时刻同一9x9 window 的整条横轴和纵轴位置，中心只读一次。17路32-bit匹配后
进入 Shiftmax17，再经静态`17x32` codebook 投影。它借用 Flow1D“二维搜索可按正交方向分解”
的动机，但没有照搬原论文先做动态1D softmax attention、再做另一轴 correlation 的公式；因此
论文只能称 Flow1D-inspired axial Match-Code。相对 MC49，它减少 offset、halo读和投影权重；
相对 DE9，它覆盖更大轴向位移但丢失近邻对角联合位移。

## 4. 审计资产

- config/queue：`entrypoints/make_h73_h74_match_code_configs.py`、`entrypoints/run_match_code_after_h66.py`
- load chain：`neuron_autoresearch/experiments/h73_h74_match_code/load_chain_audit.json`
- operation model：`entrypoints/audit_attention_candidate_ops.py`
- deploy quantization：`entrypoints/run_h60_family_deploy_eval.py`
- literature basis：`literature/idea_mining_20260711/notes/DEEP_IDEA_MINING_UPDATE_20260712.md`
