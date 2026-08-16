# Local5 真实 Descriptor 证据链与 DS-FLM 晋级门槛

> **2026-08-01 纠错：本文件中的“`k_orig`严格0/1”合同已被真实 fullres
> post-G0 运行证伪。** Local5 score 使用二值 `k_event`，但 value/projection
> 使用 ATLIF `{0, theta_block}` 阈值幅度。旧 receipt/run identity 和基于
> `k_event` 直接替代 `k_orig` 的 projection exact 声明全部失效。修正合同为：
> `k_orig` 有限、非负、正值支持集与 `k_event` 完全一致，且每个 callback
> 只有一个非零标量幅度；硬件必须将该 block 的 `theta_block` 预折叠到
> projection weight，并重新完成定点与 RTL 等价。详见
> `docs/227_VL_GSTTB_DVCO_ABIC与Local5阈值幅度纠错_20260801.md`。

日期：2026-07-31

## 1. 本轮目的

DS-FLM 的 lane-major/gate-major 双顺序只能由真实 Local5 descriptor 决定。
此前 1494-term 数据来自 `3×6×2` 定向 RTL slice，证据等级是
`[rtl-directed]`，不能代表 fullres 部署 workload。本轮不修改训练和网络
算法，只补齐 fullres post-G0 的 source-major descriptor 导出、分析与自动
执行链。

## 2. 导出合同

`scripts/profile_local5_hardware_features.py` 新增并固化
`qfit_relation_transpose_source_descriptor_v3`。每个 source descriptor
包含：

| 字段 | 含义 |
|---|---|
| `source_id/plane/y/x` | source 的逻辑位置 |
| `source_k_bitmap` | source 的 self-K lane bitmap |
| `incoming_gates[5]` | 五个候选角色对应消费 destination 的 gate |
| `valid_mask[5]` | 五个角色是否存在合法消费关系 |
| `descriptor_group_offsets` | sampled window-head 到 descriptor 区间的边界 |

五槽保持候选角色顺序：

1. self；
2. up；
3. down；
4. left；
5. right。

必须注意 source 与 destination 的方向相反。例如某 source 出现在
`up` 槽，消费它的是下侧 destination。v3 同时冻结 candidate role 和
source-consumer relation；旧 v2 中残留的 east/west 命名被强制拒绝。

本轮复核软件实现
`_binary_alpha_xnor_stencil_attention()` 后确认，软件候选的严格顺序是
`self/up/down/left/right`。此前 T450 向量生成器和 RTL 水平方向把
role3/role4 写成了 `right/left`，因此旧 miter 只证明了两个同错实现互相
一致，不能作为软件等价证据。现已统一为：

| role | destination-major 候选 | source-major consumer |
|---:|---|---|
| 0 | self | self |
| 1 | up | down |
| 2 | down | up |
| 3 | left | right |
| 4 | right | left |

修复覆盖 relation-transpose、dynamic retirement、TCFM-5、Affine-4、
Linear-5、Role-Sharded 及其测试参考；旧方向相关签核全部作废后重跑。

合同不再只绑定顶层文件，而是绑定 relation-transpose 顶层、retirement
scheduler、同步 SRAM bank、两组 SVA、Python 向量生成器、miter TB 和执行
脚本的规范路径与 SHA256。post-G0 仍要求：

- Local5 attention；
- Q7 score；
- Q1.7 gate；
- integer LUT Shiftmax hardware order；
- 真实 invalid-candidate mask；
- `crop=null`；
- `480×640`；
- `window=2×15×15`；
- `scale_factor=1`。

## 3. 分析器

新增 `scripts/analyze_ds_flm_descriptor_manifest.py`。它默认运行
fail-closed 正式模式，只接受 `evidence_level=post_g0`、v3 descriptor
合同以及全部通过的 qualification。它执行以下检查：

- offset 从 0 开始、单调、数量为 `groups+1`；
- 所有 descriptor payload 长度一致；
- gate 数组严格为 `[N,5]`；
- lane-major 与 gate-major 的 term 集合完全相同；
- 每个 group 分别展开 source-major descriptor 与 destination-major item，
  严格比较 `(destination,lane,gate)` 更新多重集及 multiplicity；
- 两顺序首项、末项以及 descriptor 结束状态相同；
- 空 K、零 gate、无效角色不生成 term。

输出包括：

- descriptor 非空比例；
- active lane、unique gate、term 的 mean/p50/p95/p99/max；
- 分 stage 分布；
- 两顺序的 lane/gate/mask 实际序列 Hamming；
- descriptor 内 lane run；
- 每个 sampled group 的结构统计。

Hamming 只表示控制和数据位活动，不是功耗。sampled group 使用互素步长在
展平的 window-head 空间中轮转抽样，不是固定首尾点，也不是完整 workload
总量；它不能用于推断跨不连续 window 的 LRU 命中率。

## 4. 自动执行

`scripts/run_local5_qfsa_profile_after_fullres.py` 已接入以下串行流程：

1. 等待 fullres deploy follower 的
   `ALL COMPLETE fullres deploy followup`；
2. 从 valid825 ranking 选择 rank-1 checkpoint；
3. 导出 100-sample、每 block-sample 4 个互素轮转 window-head 的 post-G0
   trace；
4. 运行原 QFSA/FCSR replay；
5. 运行 DS-FLM source descriptor 分析；
6. 重新校验 manifest、replay、descriptor report、run identity 和当前
   relation RTL 的 SHA256；
7. 只有 `acceptance.json` 为 `accepted=true` 才写最终完成标记。

run identity v3 与 release receipt v2 同时绑定：

- 本次 watcher UUID；
- deploy status 的启动前字节数与前缀 SHA256；
- 只能出现在该前缀之后、且同一行包含 H67/H66d 的完成标记；
- ranking、config、best epoch、rank-1 checkpoint 与 relation RTL 传递依赖；
- profiler、attention、checkpoint loader、模型、dataset、trace loader、
  replay、descriptor analyzer 和 acceptance 的规范绝对路径及 SHA256；
- 数据集分层抽样算法与 window-head 抽样算法。

旧 manifest 没有 v3 descriptor 合同、qualification 或当前 run identity
时不会被复用。固定 append log 中的历史完成行不能释放新 watcher。当前
watcher 已在整改期间主动停止；只有本节全部回归通过后才重新启动。

## 5. 验证结果

执行：

```bash
/opt/conda/envs/sdformerflow/bin/python -m unittest \
  tests.test_local5_ordered_trace_sink \
  tests.test_ds_flm_descriptor_analysis \
  tests.test_local5_postg0_acceptance \
  tests.test_et3_ordered_trace_replay
```

结果：`24/24 PASS`。

覆盖：

- destination-major 原有 term 顺序保持；
- 1×3 与两时间面 3×3 合成拓扑的 N/S/E/W relation transpose 精确方向；
- 互素轮转抽样覆盖全部 head；
- 正式 100 sample × 12 block × 4 group qualification；
- 正式模式拒绝 coverage、T450×32、run identity 或 RTL 绑定不合格的输入；
- manifest/NPZ 往返和 SHA256 防篡改；
- post-G0 fullres/W15/T450×32 合同；
- DS-FLM 两顺序 term 集合与状态不变量；
- v1/v2 descriptor 合同拒绝；
- 左右 role 互换即拒绝；
- 等覆盖基数但非预注册 flat index 即拒绝；
- deploy status 历史前缀改写即拒绝；
- replay 或 descriptor 落盘报告任一字段篡改即拒绝；
- cohort 必须为 100 个非空且唯一 sample key，并绑定全部 group sample ID。
- 非二值 `k_orig` 或 `k_orig != k_event` 即拒绝；
- source/destination 任一 gate 被篡改，即使同步重算 payload SHA256，仍因
  更新多重集不等价而拒绝；
- 正式 dataset cohort 必须使用跨 sequence 比例分层时间中点索引。

另外新增 Python→RTL T450 relation-transpose miter：

```bash
bash sim_qfit/run_qfit_relation_transpose_python_miter.sh
```

测试向量由同一 Python source-descriptor 参考模型生成，覆盖
`15×15×T2`、32-bit K、9-bit gate、N/S/E/W 边界、运行时 invalid mask
以及随机输出反压。结果：

| 工具 | 描述符 | 随机 stall | 结果 |
|---|---:|---:|---|
| Icarus | 450 | 98 | PASS |
| Verilator + SVA | 450 | 126 | PASS |

上述表是 role3/role4 修复后的重新签核结果，旧的同错 miter 结果已废止。
此外重新运行所有受影响 RTL：

| 回归 | 结果 |
|---|---|
| relation-transpose mode 0/1/2 + three-row stripe | PASS |
| TCFM-5 source multicast/projection | PASS |
| Affine-4 exact replay | PASS |
| Role-Sharded projection | PASS |
| Local5 score→relation tile | PASS |
| Local5 attention-to-projection，四后端 | PASS |
| Verilator SVA 与 Yosys check | PASS |

四后端同一输入的端到端整数金参考均为零失配：

| 后端 | cycles | descriptors | terms | updates |
|---|---:|---:|---:|---:|
| TCFM-5 | 1607 | 36 | 1498 | 2332 |
| Affine-4 | 1692 | 36 | 1498 | 2332 |
| Linear-5 | 1706 | 36 | 1498 | 2332 |
| Role-Sharded | 1607 | 36 | 1498 | 2332 |

这些是 `3×6×T2` 定向 RTL slice 结果，只用于正确性和局部周期回归，不是
fullres workload 性能结果。

该 miter 证明的是 relation-transpose 位序、方向、边界和握手的 RTL
一致性，不证明 DS-FLM 的周期、面积或能耗收益。

### 5.1 首轮独立 DATE 复评与整改

首轮独立审稿分数为 `2.0/5`，结论是 profiling/export G0 不可冻结。主要
问题与整改如下：

| 首轮问题 | 整改 |
|---|---|
| watcher 可能信任旧 report | 删除仅凭 evidence level 的提前返回，最终统一重放、重分析、重验收 |
| post-G0 自声明、coverage 不足 | 强制 100 sample、12 block、每 block-sample 至少 4 group、全部 head coverage |
| relation RTL 仅记录未校验 | run identity 与正式分析器同时重算当前 RTL SHA256 |
| 固定四点抽样偏置 | 改成跨 sample 的互素轮转 flat window-head 抽样 |
| 单 active-lane 的 gate-major run 公式错误 | 改为从实际调度序列计算 run |
| 方向测试只覆盖 E/W | 增加两时间面 3×3 N/S/E/W 与 T450 Python→RTL miter |
| G3 缺测量协议 | 在 6.3 节冻结同约束 PPA/EDP 与统计协议 |

整改后仍不把 G0 声称为冻结；必须通过下一轮独立 DATE 复评。

### 5.2 第二轮独立 DATE 复评与整改

第二轮复评分数为 `2.7/5`，G0 仍为 `NO-GO`。审稿人指出正式验收仍可能
只信任报告元数据、历史完成标记、任意 12 个模块名或等基数采样集合。
本轮逐项整改：

| 第二轮问题 | 本轮整改 |
|---|---|
| 验收只检查报告 metadata | 验收时从 manifest/NPZ 重新执行 replay 和 descriptor analysis，并与落盘 JSON 逐字段相等 |
| 历史完成行可释放新 run | watcher 启动时冻结 status 前缀 SHA256；只接受其后新行并写 UUID receipt |
| callback 几何未精确校验 | 正式采集逐元素核验 T2×15×15 self/up/down/left/right index 和 valid mask |
| 只查采样覆盖基数 | 逐 module/sample 重算互素轮转索引，并核验 `flat=window×heads+head` |
| 12 block 可由任意名称凑数 | 强制精确 block 集合 `S0:2、S1:2、S2:6、S3:2`，拒绝重复或额外模块 |
| run 未绑定生产源码 | v2 identity 绑定 11 个规范生产文件路径与 SHA256 |
| cohort 可重复 | 强制 100 个唯一非空 sample key、100 个非空 sequence key 和 group sample `0..99` |

整改过程中新增反例还发现两项真实缺陷：

1. acceptance 中局部变量遮蔽 replay 函数，导致正式重算路径会抛
   `UnboundLocalError`；已修复并加入篡改反例。
2. 软件 `left/right` 与旧 RTL/TB 的 role3/role4 相反；已按第 2 节合同
   修复并将受影响 RTL 全部重新签核。

因此本轮证据强度比第二轮复评时有实质提升，但在第三轮独立复评给出结果
之前，仍不宣称 G0 已冻结。

### 5.3 第三轮独立 DATE 复评与后续整改

第三轮只读复评分数为 `2.9/5`。审稿人确认旧左右合同和缺少双表示逐组
等价这两个 P0 已实质关闭；架构创新性仍为 `2.5/5`，不会因为证据工程自动
上升。该复评读取的是 receipt v2、二值 K 断言和分层抽样落盘之前的快照，
其后又完成以下整改：

| 剩余问题 | 后续整改 |
|---|---|
| 软件输出使用 `gate×k_orig`，RTL 使用 `k_event` | callback 同时传入两者；正式模式逐元素要求 `k_orig∈{0,1}` 且完全等于 `k_event` |
| receipt 可跨固定路径运行复用 | receipt v2 绑定 ranking、best epoch、checkpoint、config 的路径与 SHA256 |
| relation RTL 只绑定顶层 | 扩为 RTL、SVA、向量、TB、执行脚本的 8 项传递依赖闭包 |
| valid 前 100 项有 sequence 偏置 | 改为 sequence 比例分层、序列内时间中点抽样，并预注册索引 |
| 文档仍称 v2 | 本文和代码统一升级到 descriptor v3、run identity v3 |

CPU 预注册结果：valid 集 825 项，100 个索引覆盖全部 18 个 sequence，索引
SHA256 为
`438ca5a4a7dfafdecce389b67ad14162e4b7392873b8c51721a9f846572040e2`。
完整索引和 sequence 配额见
`results/local5_postg0_stratified_cohort_prereg_20260731/`。

当前仍不能冻结 G0，因为真实 fullres post-G0 artifact 尚未产生；这属于外部
训练/部署状态门槛，不再是已知的证据链绕过。待 follower 完成后必须由新
watcher 产生 receipt v2、run identity v3、v3 descriptor 和最终
`accepted=true`。

## 6. 预注册晋级门槛

### G1：真实 workload 门槛

必须先满足：

- Local5 fullres valid825 与 hardware-order exact 完成；
- trace 是 fullres、post-G0、至少 100 个 sample；
- 所有 12 个 attention block 均有 coverage；
- source descriptor invariant 零失败。

否则 DS-FLM 保持 G0 叶模块，不进入论文主贡献。

### G2：双顺序存在价值

先用真实 trace 回放 lane-major 与 gate-major。在尚未标定物理系数时，只
允许报告字段活动和周期，不允许报告能耗。若两模式只是把 lane Hamming
换成 gate Hamming，且任何合理系数范围内都没有稳定优势，则删除 selector，
保留更简单的单模式。

### G3：物理晋级

与现有 W6 late-materialization 强基线在同一 SDC、同一 SRAM macro 和同一
反压条件下比较。测量协议冻结为：

1. **统计单位**：每个 sample 内先汇总全部 12 block，再对 sample 等权；
   同时报告 window 加权结果作为敏感性分析，禁止用 term 数量给稀疏样本
   降权。
2. **数据划分**：selector 阈值只在 profile/train split 标定；valid825
   held-out trace 只做一次最终报告，禁止按测试集调阈值。
3. **性能**：在相同输入/输出反压、真实 SRAM latency 下报告 cycle 的
   mean/p50/p95/p99，并以 sample 为重采样单位给出 95% bootstrap CI。
4. **面积/时序**：W6、lane-major、gate-major、selector 四者使用同一
   工艺库、PVT、SDC、层次边界和 SRAM macro；面积必须包含 selector、
   descriptor SRAM、控制和 glue logic。
5. **功耗**：使用 held-out trace 产生门级 SAIF；同时报告逻辑动态功耗、
   SRAM 动态功耗、漏电和每类访问次数。RTL Hamming 只能解释原因，不能
   替代功耗。
6. **EDP 边界**：以完整 Local5 attention-to-projection subsystem 为边界，
   包含 descriptor 生成、relation transpose、目录、权重读取、投影累加和
   输出反压；独立叶模块 EDP 不进入主表。

双顺序/selector 要成为 DATE 贡献，至少应同时满足：

- bit-exact；
- 平均 EDP 改善不低于 10%；
- p99 周期不回退超过 2%；
- 逻辑面积增加不超过 5%；
- held-out sample 的 95% bootstrap CI 不跨越零收益；
- sample 等权与 window 加权下结论方向一致。

达不到时，应将 DS-FLM 记录为负结果，不用“代数闭包”或“双 stationary”
包装掩盖收益不足。

### 6.1 当前物理环境边界

2026-07-31 实机检查：

- `dc_shell`：未安装或不在 `PATH`；
- `pt_shell`：未安装或不在 `PATH`；
- `vcs`：未安装或不在 `PATH`；
- `/root/private_data` 下未发现可用 `.db/.lib` 标准单元库；
- 未发现 SRAM `.db/.lib` macro。

因此 G3 当前没有可执行的目标工艺配置，不能把 Yosys 结果写成 DC 面积，
也不能生成可信的门级 SAIF 功耗或 PVT 时序。现阶段只能继续：

1. 完成 bit-exact、反压和真实 trace 周期验证；
2. 用同一 Yosys/open-cell flow 做结构趋势筛选；
3. 将逻辑与 SRAM 访问次数分账；
4. 等用户提供目标库、SRAM macro 与许可后，再冻结具体 PVT、时钟、输入
   驱动、输出负载和功耗活动窗口。

该环境缺口不阻止等待真实 fullres trace，但阻止任何“可直接做 DC”或
“达到 DATE PPA 主表”的结论。

## 7. 当前结论

本轮完成的是可审计的真实 descriptor 证据链，不是 DS-FLM 的性能结论。
Local5 fullres 训练和 exact follower 尚未完成，因此现在不能声称：

- gate-major 优于 lane-major；
- selector 节能；
- DS-FLM 优于 W6；
- Local5 已成为硬件主线。

Motion 与 Local5 继续双线保留。Motion 已有 fullres 低 spike 但精度显著
退化的风险；Local5 是否能替代它，首先由 fullres accuracy/exact 决定，
其次才由本报告定义的 descriptor 和 PPA 门槛决定。
