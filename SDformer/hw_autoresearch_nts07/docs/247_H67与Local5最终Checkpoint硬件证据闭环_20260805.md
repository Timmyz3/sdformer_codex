# H67 与 Local-5 最终 Checkpoint 硬件证据闭环

日期：2026-08-05

## 1. 当前算法赢家与候选

- H67 Motion-XOR fullres ep30：480x640、window `2x15x15`、valid825
  AEE `1.3387`、AAE-Benchmark `5.7558`、spikes `81.3086G`。
- NB0 fullres ep29：AEE `1.4454`、AAE-Benchmark `6.1803`、spikes
  `126.1156G`。H67 相对 NB0 的 AEE/AAE-Benchmark/spikes 分别改善约
  `7.38%/6.87%/35.53%`。
- Local-5 旧 fullres 使用过低 LR，不作为最终公平比较。算法侧已排队同 bb1e4、
  同 30-epoch、同 fullres/window15 协议重跑；milestones `13/20` 对齐 H67
  实际执行的三段 LR 轨迹。

## 2. 证据等级不得混写

1. `float`：标准 PyTorch/CuPy valid825。
2. `dyadic Q7/Q1.7`：score/gate 量化，Shiftmax 仍为浮点 `2^x`。
3. `attention-core hardware-order numeric`：Q7 score、Q8 LUT、integer rowsum、
   ceil-pow2 normalization、Q1.7 gate 的 Python 顺序模拟。
4. `post-G0 ordered profile/replay`：真实最终 checkpoint 产生 T450 ordered trace，
   并完成 descriptor/frontier replay 与 acceptance。
5. `RTL-exact`：相同 trace 驱动 T450 SystemVerilog，逐事务 zero-mismatch。
6. `full-network RTL/PPA`：包括外围、SRAM、控制与完整 fixed-point 数据通路；不能由
   attention-core 数值评估替代。

在等级 5 完成前，任何表格和正文只能使用
`attention-core hardware-order numeric`，不得写 fullres RTL-exact。

## 3. 最终 checkpoint 重新绑定要求

每个进入硬件表的算法赢家必须记录：

- config path 与 SHA-256；
- checkpoint path、size、mtime 或 SHA-256；
- `ATLIFTernaryPSN=105`、`Shiftmax=12`；
- `checkpoint_overlay_keys=210/210`；
- `missing/unexpected=0/0`；
- resolution `480x640`、window `2x15x15`、T450、BN `no_running`；
- ordered trace 的样本数、顺序、mask 与 descriptor manifest；
- SV regression 的总事务数与 mismatch 数。

旧 H67 crop/T162 或旧低 LR Local-5 的 profile 可以保留作开发证据，但不能绑定到
新的 fullres 最终 checkpoint。

## 4. 当前自动队列

- H67 ep30 已完成 dyadic 与 hardware-order valid825；float/dyadic/hardware-order
  AEE 分别为 `1.3387/1.3424/1.3417`，AAE-Benchmark 分别为
  `5.7558/5.7625/5.7536`。量化结果稳定，加载审计为 ATLIF `105`、Shiftmax
  `12`、overlay `210/210`、missing/unexpected `0/0`。输出目录：
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/deploy_valid825/`。
- Local-5 公平重跑已于 2026-08-05 启动，首批约 `1.08 s/it`、8 workers；
  训练完成后自动进入硬件 profile 队列：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline.py`。
- 长训练在 ep9/19/29 保存 model+optimizer/scheduler/scaler 成对状态；流水线重启
  自动从最新成对 checkpoint 严格恢复。
- 为启用该恢复合同，训练在首个 ep0 约 6% 时受控重启一次；无 checkpoint/状态
  被复用，重启后从相同 crop ep29 初始化，加载审计和算子计数保持不变。
- 独立单实例 supervisor 在流水线异常退出时最多自动重启 5 次；checkpoint 选择仍由
  model/state 成对审计控制，避免仅加载模型却丢失 optimizer/scheduler 状态。
- supervisor 初版的相对路径识别测试曾误启动第二流水线约 14 秒；该进程在首个
  forward、`0/3672` 时因显存竞争 OOM，未执行 optimizer step 或写 checkpoint，
  原训练未停止。修复后使用进程 cwd 与 argv 合成绝对路径进行精确识别。
- Local-5 的 100-sample post-G0 wrapper：
  `hw_autoresearch_nts07/scripts/run_local5_bb1e4_postg0_profile.py`。
- checkpoint-bound projection RTL watcher：
  `hw_autoresearch_nts07/scripts/run_local5_bb1e4_checkpoint_bound_rtl.py`；它在
  acceptance 后生成 100 组 T450 real-trace projection vectors，并执行
  direct/QGASR SystemVerilog、random-stall SVA、lint 与 Yosys。该证据只覆盖
  post-G0 projection，不等于 full-attention 或 full-network RTL-exact。
- 最终 source-binding 逻辑更新后，启动时已加载旧模块的 post-G0 watcher 被受控终止，
  训练进程保持运行。主流水线将在 train/valid825/deploy 完成后因旧 watcher 返回非零
  触发一次 supervisor 重启；成对 checkpoint、ranking 与 deploy 均按存在性和完整性
  复用，重启只负责以当前代码重新生成 profile/replay/acceptance。独立 checkpoint-bound
  RTL watcher 同步重启并已加载当前代码，因此最终报告不会混入旧 source SHA。
- 计划产物：
  `hw_autoresearch_nts07/results/local5_fullres_bb1e4_postg0_profile100_20260805/`、
  `local5_fullres_bb1e4_postg0_replay_20260805/`、
  `local5_fullres_bb1e4_descriptor_analysis_20260805/`、
  `local5_fullres_bb1e4_postg0_acceptance_20260805/`。
- 训练健康检查：ep0 已在 `3929.79 s` 内完成并进入 ep1，train/validation loss 为
  `2.14747/1.97951`，LR `1e-4`，峰值训练显存 `38.354 GiB`；首个可用于算法裁决和
  checkpoint-bound 硬件证据的预注册检查点仍是 ep9。
- 后续 ep1 也已完整结束并进入 ep2：train/validation loss 降至
  `1.84878/1.70673`，相对 ep0 同时下降约 `13.9%`；无 NaN/Inf。日志中的
  `threshold_updates_frozen=1` 仅表示 homeostatic 更新停止，optimizer 阈值梯度仍开启。
  ep1 不保存 checkpoint，硬件 watcher 继续只等待 ep9 以后最终 rank-1，不能
  从当前 train/validation loss 生成提前的 profile 或 RTL 权重证据。
- profiler 协议已补齐 standard evaluator 的 BN `no_running` 语义，并在产物中记录
  config/checkpoint identity、T450 几何、ATLIF/Shiftmax 数量与 load audit；H9
  missing/unexpected 非零时直接失败。此前未带该元数据和 BN 口径的 profile 只作开发
  证据，不进入最终表。
- H67 ep30 fullres profile watcher：
  `hw_autoresearch_nts07/scripts/run_h67_ep30_fullres_t450_profile.py`。它等待 Local-5
  整条流水线和 post-G0 acceptance 完成、GPU 显存低于 8 GiB 后，运行 H67
  profile100，并导出 1 个真实样本的 all12 block Q/K/Q1.7 gate trace；审计要求四
  stage 与 12 records 完整覆盖。
- H67 计划产物：
  `hw_autoresearch_nts07/results/h67_fullres_ep30_t450_profile100_20260805/`、
  `h67_fullres_ep30_t450_all12_bit_trace_20260805/`、
  `h67_fullres_ep30_t450_all12_bit_trace_audit_20260805/`。

## 5. Local-5 真实 Q/K 到 Shiftmax 的新增闭环

- 旧 T450 full-chain regression 使用 synthetic Q/K/weights；旧 checkpoint watcher
  只覆盖真实 descriptor 驱动的 projection。因此此前不能声称最终 checkpoint 的
  Q/K、score 与 Shiftmax 已被 RTL-exact 覆盖。
- ordered post-G0 trace 现保存 `descriptor_q_bitmap`，与每组 450 个 K descriptor
  严格对齐；独立 trace contract 为 `local5_qk_score_shiftmax_trace_v1`，不修改已有
  descriptor contract v3。
- 最终 Local-5 rank-1 profile 通过 acceptance 后，自动从四个 stage 各取 25 组，
  生成 100 组 T450、共 45,000 个真实 Q/K score/Shiftmax vectors。软件参考独立复算
  alpha-XNOR Q7、valid mask 与 integer Shiftmax Q1.7，并先验证复算 gate 等于 trace
  gate，避免由同一被测实现自证。
- Icarus 与 Verilator 均须 45,000/45,000 zero-mismatch；Yosys 对 score leaf 和
  Shiftmax leaf 分别执行结构检查。对应自动入口：
  `hw_autoresearch_nts07/sim_local5/run_local5_checkpoint_score_trace_checks.sh`。
- 该工具链已通过 synthetic T450 边界 fixture（100 组/45,000 事务）验证；fixture
  只说明生成器、双仿真与 Yosys 流程可运行，不进入论文结果表。真实 checkpoint
  结果须等待本轮 Local-5 fullres 训练、valid825、profile 与 acceptance 完成。
- 最终组件级准确表述为：
  `checkpoint-bound post-G0 Q/K-score-Shiftmax RTL-exact` 与
  `checkpoint-bound post-G0 descriptor projection RTL-exact`。projection 目前仍使用
  确定性 synthetic weights，旧 full-chain 也仍是 synthetic workload；因此不能把
  两项拼接后写成 full-attention 或 full-network RTL-exact。
- Local-5 profiler 已直接补齐 BN `no_running`，并在 identity 中记录
  ATLIF/Shiftmax、overlay/missing/unexpected、config/checkpoint 与全部 source SHA；
  新增 score generator/reporter/TB/runner 也进入 source binding。
- 复审发现旧 projection replay 只对真实 K/gate 做了 checkpoint binding，权重仍是
  手工 synthetic 函数；且 QGASR runner 的汇总器未显式传当前 manifest，存在报告读取
  旧默认 manifest 的 provenance 风险。现已新增 all12
  `checkpoint_projection_contract.{json,npz}`，记录真实 raw `proj.weight/bias`、dyadic
  INT8 code/scale、checkpoint SHA 和 BN 边界；ordered trace 与 acceptance 强绑定该
  contract，runner 也强制传当前 vector manifest 并复核 SHA。
- 新 testbench 模式按每个 selected group 的 stage/block/head 加载对应真实 32-lane
  weight slice；不同组之间 reset/reload，避免跨 block/head 权重污染。带正负非平凡
  INT8 fixture 的四 stage T450 回放和旧 synthetic 兼容回归均为 Acc32 zero-mismatch；
  相关 `18` 个 unittest 通过。最终 watcher 已重启加载当前代码。
- 该增强只证明 `real checkpoint INT8 per-head partial projection accumulator`。完整 C 维
  输出还需跨 head 求和、bias、BN、requant 与 residual；尤其当前 `no_running` BN 使用
  输入相关 batch statistics，不能静态折叠，因此仍不得写 full projection/full
  attention/full network RTL-exact。

## 6. RTL-exact 剩余闭环

- 将 H67 与最终 Local-5 算子的 token/address/controller 参数统一到 T450；
- 从最终 rank-1 checkpoint 生成真实 ordered trace，而非 synthetic window；
- 对 score、mask、row max、LUT、rowsum、normalization、gate、projection 逐级比对；
- SystemVerilog 全事务 zero-mismatch，失败必须保留首个 mismatch 的输入与中间值；
- 仅对最终算法赢家做完整 PPA，另一候选保留 attention-core 对照，避免重复扩展两套
  full-network RTL。

## 7. H67 ep30 T450 checkpoint-bound row RTL 追加队列

- 现有 H67 row engine 的 token/address/count 参数可合法实例化 T450；Yosys 已在
  `MAX_TOKENS=450` 下完成 hierarchy/proc/opt/check/stat，报告 0 problems。
- bit-trace manifest 现在强绑定 config/checkpoint SHA-256、fullres/window15/BN 协议、
  ATLIF105、Shiftmax12、overlay210/210 与 missing/unexpected0/0，防止旧 T162 或其他
  checkpoint trace 被误用。
- 新向量生成器要求 all12 block，逐 block/window0/all-head 独立复算 Motion-XOR Q7、
  exp2 LUT、row sum、ceil-pow2 normalization 与 Q1.7 gate，再生成真实 row vectors。
- 新 testbench 检查 active K/gate 输出、zero-K denominator fold、loaded/folded/emitted
  守恒，并加入输入空拍与输出 backpressure；正式结果要求 Icarus、Verilator 双仿真
  zero-mismatch。
- 旧真实 T162 trace 的工具 dry-run 已通过：45 rows、7,290 token vectors、1,054 active
  outputs，双仿真零失配。它只证明工具链和既有 row RTL 的连接正确，不是最终 H67
  ep30/fullres/all12/T450 checkpoint 证据。
- 自动 watcher 已升级并重启；Local-5 整条流水线完成后，将依次产出：
  `h67_fullres_ep30_t450_profile100_20260805`、
  `h67_fullres_ep30_t450_all12_bit_trace_20260805`、
  `h67_fullres_ep30_t450_all12_bit_trace_audit_20260805`、
  `h67_fullres_ep30_t450_score_shiftmax_rtl_20260805`。
- 新 RTL 报告的准确范围是 checkpoint-bound Q/K score、SCS zero-K denominator fold、
  Shiftmax 与 active gate output component RTL-exact；projection、完整 attention 控制、
  SRAM macro、encoder 和 full-network 仍不在该报告范围内。

## 8. 2026-08-05 最终 fail-closed 补强

- Local-5 projection regression 的 Yosys 结构检查已升级为 `check -assert`；运行器把
  当前实际参与仿真的全部 RTL、SVA、testbench 以及 vector manifest 写入
  `source_sha256.txt`。汇总报告逐文件复算哈希，并绑定 source-manifest 本身的 SHA，
  避免只绑定 Python runner、却没有绑定被仿真的 RTL 源码。
- checkpoint-bound 总入口必须同时看到以下四项为 PASS：真实 checkpoint dyadic INT8
  weight binding、random-stall SVA、Verilator lint、Yosys check；同时复核当前 vector
  manifest 与 source manifest SHA。任一项缺失或读取了旧目录都会 fail-fast。
- Q/K score 与 Shiftmax 两个 leaf 的 Yosys 检查也改为 `check -assert`，报告要求两个
  session 都有正常结束标记；Icarus、Verilator 与独立软件 reference 仍须全量零失配。
- 等待 Local-5 最终 rank-1 的独立 watcher 已按当前代码重启（PID `2161370`）。正式
  结果仍严格分为：score/Shiftmax component RTL-exact、真实权重 per-head projection
  partial-accumulator RTL-exact；跨 head 求和、bias、动态 BN、requant、residual、SRAM
  macro PPA 与 full network 不在证明范围内。
- Local-5 训练到标准推理的入口现有独立 fail-fast 门禁：最终 train log 必须证明
  ATLIF105、Shiftmax12、checkpoint overlay210、missing0/unexpected0，之后才允许检查
  五个 model checkpoint 与三个 paired optimizer state 并进入 valid825。该门禁防止
  “文件名正确但新增 overlay 权重未加载”的 checkpoint 被硬件 profiler 接受。
- standard/deploy evaluator 现将 checkpoint SHA-256 和实际 ATLIF/Shiftmax 模块计数写入
  profile；Local-5 的五个 float valid825 与两个 rank-1 deploy profile 均须逐项通过
  fullres/window15/batch1/BN、overlay210、missing/unexpected0/0、ATLIF105、Shiftmax12
  和 checkpoint SHA 门禁。缺 SHA、缺 counts 或同名权重被替换均拒绝复用。
- 当前配置中的 `threshold_freeze_after_step=1224` 只停止独立的 homeostatic
  `threshold_update` 路径，并不冻结 optimizer 对 ATLIF threshold 参数的梯度；配置未启用
  `freeze_threshold_grad_after_step`，threshold optimizer LR 为 `5e-6`。此外
  `official_atlif` 模式不会执行配置中的 `min_threshold/max_threshold` clamp。硬件部署使用
  最终 checkpoint 内的静态 threshold 参数，不需要在线 homeostatic controller。上述语义
  将作为 `threshold_training_semantics` 同时写入 profile JSON 和 bit-trace manifest。
- Local-5 专用 `local5_hardware_features.json`、Markdown 与 ordered-trace manifest 也绑定
  同一语义；post-G0 acceptance 要求 `official_atlif`、boundary1224、optimizer gradient
  未冻结、threshold LR `5e-6`、official clamp inactive 和 checkpoint-static inference
  全部精确匹配。字段缺失或漂移会 fail-closed，后续 score/Shiftmax/projection RTL watcher
  不会释放。包含 standard/deploy/convergence provenance 的相关回归共 `25/25` PASS。
- 算法总流水线在 watcher 退出后还会二次读取 acceptance 与 ordered manifest，复核
  `accepted=true`、100 samples、12 blocks、rank-1 checkpoint 路径和 threshold semantics；
  不再把“watcher 因其他进程持锁而 exit0”误当作正式证据完成。

## 9. 收敛续训后的 checkpoint 版本规则

- H67 ep30 是当前算法最优和已注册 T450 证据锚点，但 ep25->ep30 AEE 仍改善
  `2.47%`，不能把 ep30 写成“已收敛最终权重”。算法侧已新增 H67/NB0 对称 +10
  fullres 收敛审计，严格排在 Local-5 与 H67 ep30 证据任务之后运行。
- ep30 的 profile/bit-trace/RTL 报告永久绑定其 checkpoint SHA，可作为预算30锚点保留；
  若 H67 ep35/40 在 standard valid825 成为新 rank-1，最终论文硬件表必须为新 rank-1
  重新生成 profile100、all12 T450 trace、score/Shiftmax RTL 和 source/checkpoint SHA。
  禁止仅修改报告中的 epoch/path，或把 ep30 zero-mismatch 外推到新权重。
- H67 +10 不改变 attention 结构、T450 几何、ATLIF105、Shiftmax12 和 RTL 数据流；预期
  只改变 checkpoint 参数与真实活动分布。因此 RTL 模块可复用，但 vector、activity、
  spike/energy profile、projection weight contract 和最终证据 manifest 必须重跑。
- +10 的三个预算点 standard profile 均强绑定 config/checkpoint SHA，并再次验证
  fullres/window15/batch1/BN、overlay210、ATLIF105、Shiftmax12；不能只依据训练日志或
  checkpoint 文件名选择后收敛 rank-1。
- Local-5 的最终硬件证据仍只绑定其 own fullres rank-1；H67/NB0 的 +10 队列不会替换
  Local-5 watcher 输入，也不会把 Local-5 的 profile 或 projection contract 复用于 H67。
- 自动 follower：
  `hw_autoresearch_nts07/scripts/run_h67_postconvergence_rank1_profile.py`。它等待 +10
  valid825 完成后解析 H67 rank-1；ep30 路径必须复核既有 RTL report 的 checkpoint
  SHA，ep35/40 路径则生成全新 hardware-order config/profile/trace/vector/report。最终
  选择写入
  `hw_autoresearch_nts07/results/h67_postconvergence_rank1_hardware_evidence_20260805.json`。

## 10. ATLIF/DP-TME 证据边界复核

- 当前 post-G0 profile 对 81 个 functionally-live ATLIF 记录首调用输入统计、事件率、
  margin 以及 Q4/Q6/Q8 参数量化事件翻转；这属于 sampled numeric profile，不是 RTL-exact。
- 现有 `hitflow_dptme_array.sv` 与随机 TB 已证明 T10/T2 packed temporal matrix 的整数
  累加映射、协议、SVA、lint 与 Yosys 可运行，但向量尚未绑定最终 Local-5/H67
  checkpoint，且输入定点 scale、bias/threshold 共同比例和 `event x threshold` 输出
  scale folding 尚未闭合。
- 因此当前准确表述仍是 score/Shiftmax 与 per-head projection partial accumulator 的
  checkpoint-bound component RTL-exact；ATLIF 不能计入该 scope，更不能据此声称完整
  attention/full encoder/full network RTL-exact。
- 最小补全协议固定为：从最终 rank-1 的 100-sample profile 导出 functionally-live
  ATLIF 的真实 T10/T2 输入、weight/bias/threshold；用每 site 静态 power-of-two INT8
  input/weight scale 和 Acc24 共同比例生成 golden；先报告浮点到定点 event flip，再以
  同一整数向量驱动 DP-TME，要求 Icarus/Verilator hidden/event zero-mismatch，并绑定
  checkpoint/config/vector/RTL/TB/SVA/source SHA。输出只称
  `checkpoint-bound ATLIF temporal-matrix component RTL-exact`。
- 输出的 1-bit event 必须同时携带 checkpoint-static threshold/output scale；在完成下一
  Linear/Conv weight folding、BN/requant/residual 与 valid825 精度复验前，不得把
  component zero-mismatch 外推为网络部署精度无损。

## 11. ATLIF checkpoint-bound 工具与队列落地

- 已新增 vector generator、file-driven SystemVerilog TB、双仿真 runner 和 fail-closed
  reporter；最终命令集合严格覆盖 `81=45 x T10 + 36 x T2` 个 live site，每个 site
  在同一 32-lane command 内混合 ordinary、near-threshold 与 max-amplitude lanes。
- machine contract 同时拒绝 input/weight/bias/threshold clip 和 Acc24 overflow；记录
  float-to-fixed event flip、per-site scale、threshold integer、hidden range与 output
  `event + static threshold scale` 合同。RTL hidden/event zero-mismatch 与网络数值误差是
  两个独立字段，前者通过不能覆盖后者。
- synthetic 工具测试的 Icarus 结果为 `25,920/25,920` hidden 和 event zero-mismatch；
  Verilator 编译已通过。由于与 Local-5 训练、OpenROAD 并行时 full32 Verilator fixture
  极慢，正式双仿真不在训练期间占用 CPU，已交给最终 checkpoint watcher 串行执行。
- Local-5 在 post-G0 wrapper 退出前执行 ATLIF vector/RTL；H67 ep30 watcher 与
  post-convergence rank-1 follower 也已接入相同阶段。每次结果绑定自己的
  config/checkpoint SHA，禁止跨 checkpoint 复用。
- fullres 周期口径已另存
  `results/dptme_fullres_w15_port_contract.{json,md}`：15x15 的225 positions下，T10为
  2250拍；T2的G5/G4/G3计算周期为90/114/150，单32-bit出口下G5至少450拍。旧81-position
  34拍结果只作历史兼容，不再进入当前 DATE 表。

## 12. Local-5 当前队列与 ATLIF manifest fail-closed 补强

- Local-5 fullres 训练正在进行，配置固定为 480x640、window `2x15x15`、batch2、
  BN `no_running`、ATLIF105、Shiftmax12、overlay210、missing/unexpected `0/0`；输入权重
  是 Local-5 自身 crop/full30 ep29，不与 NB0/H67 checkpoint 混用。
- 截至 2026-08-05 18:05 UTC 已完成 ep0--2，valid loss 为
  `1.97951 / 1.70673 / 1.80956`，当前进入 ep3。最终硬件向量只绑定 standard valid825
  选出的 rank-1 checkpoint，训练中间 loss 不触发 RTL/profile 释放。
- `report_checkpoint_atlif_dptme_rtl.py` 已增加逐 command 复核：81个唯一 site/tag、
  `45xT10 + 36xT2`、T10/T2 对应 scenario lane 配额、固定25,920个事件、scale 正且有限、
  Acc24 不越界、四类 quantization clip 为0、overflow为0、输出合同为
  `one_bit_event_plus_checkpoint_static_threshold_scale`。manifest summary 与逐命令总计
  不一致时直接失败。
- 新增 `test_checkpoint_atlif_dptme_manifest.py` 的4项正反测试；连同 fullres window15
  两项几何测试共 `6/6` 通过。训练到 standard/deploy/profile 的 provenance 门禁回归
  `25/25` 通过。正式 Icarus/Verilator 双仿真仍等待最终 checkpoint，不将 synthetic
  qualification 写入论文结果。

## 13. Local-5 ATLIF watcher 生命周期闭环

- 当前主流水线早期创建的 embedded post-G0 child 已退出，不能把“主进程仍训练”误当作
  ATLIF watcher 仍在。为消除这一依赖，独立 checkpoint-bound RTL watcher 现也负责最终
  rank-1 的 ATLIF vector 生成与 DP-TME 双仿真。
- 两个可选入口共用
  `results/local5_bb1e4_checkpoint_atlif_dptme_rtl_20260805.lock`；锁内复核 report status 与
  checkpoint SHA，相同则复用，不同或残缺才重新生成。该机制避免 acceptance 发布后两个
  watcher 同时覆盖同一 vector/result 目录。
- 独立 watcher 已重启为 PID `2233542`，依次要求 score/Shiftmax、真实 checkpoint INT8
  per-head projection partial accumulator、ATLIF temporal matrix 均通过，才写总
  `checkpoint_bound_scope.json`。相关回归总计 `31/31` PASS。
- 总证据名称限定为三个 checkpoint-bound component RTL-exact；动态 BN、cross-head、
  requant、residual、完整 attention/encoder/network 仍明确排除。

## 14. DATE 最终算法/硬件证据审计器

- `audit_date_algorithm_closure_20260805.py` 作为最后一级签核，不以 watcher 退出码或文件存在
  代替证据；它重新读取 Local-5 五个 standard profiles、deploy summary、三组件 RTL、
  H67/NB0 +10 summary 与 H67 post-convergence hardware evidence。
- score 和 projection vector manifest、ordered trace、projection contract、ATLIF report、
  H67 row report 均反向校验到各自最终 rank-1 checkpoint SHA。Local-5/H67 epoch 与 ranking
  不一致、协议漂移、scope 缺少 component RTL-exact、或把本地 valid825 冒充 hidden test
  都会失败。
- PASS 输出明确限定为 `checkpoint_bound_component_rtl_exact_not_full_network`；不补齐动态
  BN、cross-head、requant、residual、SRAM macro 和 full-network 的未证明范围。
- 常驻审计 watcher PID `2239223` 已启动；当前正式产物未齐，因此正确状态为 PENDING。
  相关 ranking/profile 正反测试 `3/3` PASS。

## 15. Local-5 post-G0 producer 独立监督

- 运行态审计发现 embedded post-G0 child 已提前退出，原独立 RTL watcher 只等待
  acceptance，不能自行补产 profile/replay/acceptance。该状态若不修复，会在训练结束后先
  触发流水线失败，再依赖 supervisor 重启补做。
- checkpoint-bound RTL watcher 现将 post-G0 wrapper 作为受监督子进程同步执行：acceptance
  缺失时先完成 rank-1 ordered profile、replay、descriptor 和 acceptance，再运行真实 Q/K
  score/Shiftmax、checkpoint INT8 per-head projection partial accumulator、81-site ATLIF
  temporal matrix。
- 新监督器 PID `2243134`、producer PID `2243137` 已存活并等待软件 deploy release；不占用
  训练 GPU。所有 producer 继续共享既有 `flock`，因此不会并发覆盖同一 profile/vector。
- 两项新生命周期测试与既有 pipeline/config 测试合计 `10/10` PASS；该修复只增强证据生成
  的可靠性，不扩张 RTL-exact 声明范围。

## 16. valid825 样本数门禁

- 软件 evaluator、Local-5 pipeline、H67/NB0 `+10` runner 与最终 closure auditor 现均要求
  每个 standard profile 的 `samples == 825`，不再仅凭目录名或 population 标签判断完整性。
- `824` 样本反例测试已通过；新 `+10` runner PID `2245266` 与 closure watcher PID
  `2245267` 已加载该门禁。该检查保证后续 RTL 绑定的 rank-1 来自完整统一评估集合。

## 17. H67 证据复用与 staged resume 预审计

- H67/NB0 staged resume 已在 GPU 队列释放前完成 model/state SHA、hardlink、internal epoch29、
  scheduler epoch29、AMP scaler、固定源 LR 和 RNG 缺失披露审计；正式训练只复用已签名状态。
- H67 ep30 report 和 post-convergence FINAL 不再按文件存在直接复用。复用必须同时绑定当前
  rank-1 epoch、checkpoint path/SHA、score component RTL scope/PASS 以及 ATLIF PASS/SHA。
- checkpoint 内容变更反例测试 `2/2` PASS；加载新逻辑的 watcher PID 为 ep30 `2246496`、
  post-convergence `2246497`。

## 18. Local-5 成对训练状态门禁

- 最终 closure 将读取 Local-5 ep9/19/29 model+state 对，校验 internal epoch、scheduler
  `last_epoch`、13/20 milestones、AMP scaler 及五参数组 optimizer/scheduler LR 全一致。
- model/state SHA 会写入最终机器审计；缺 state、只加载模型、LR 轨迹漂移或 scaler 丢失均
  不得产生 PASS。正常/反例测试通过，closure watcher PID 更新为 `2248718`。

## 19. AAE 可执行公式证据

- 新 receipt 在生产环境重跑 3 项 AAE 数值测试，并 SHA 绑定 metric、evaluator、test 源码；
  同时固定 2-D legacy、Barron 3-D、frame-equal 聚合和 eval batch1 口径。
- 最终 closure 必须重新计算源码 SHA 后才接受 receipt；watcher PID 更新为 `2250001`。

## 20. 软件训练与 OpenROAD 同机争用边界

- Local-5 ep3 与 `local5_out32_allmacro_proxy` detail-route 重叠时，训练由约 `1.05 s/it`
  短时降至 `3.2--4.8 s/it`，GPU utilization 降至 `15--58%`；显存和 worker 均稳定。
- OpenROAD 129线程覆盖64 CPU且系统无 I/O wait，故判定为 CPU/内存带宽争用。软件训练不
  重启，硬件任务也不被干预；该运行时抖动不作为算法或 RTL 结果。
- 后续吞吐逐步恢复到 `1.10 s/it`，训练 PID/optimizer state 未变化，确认无需重启。
- detail-route 后续重负载使 ep3末段再次短时降至 `3--5 s/it`、GPU约11%；到 ep4开始已在
  OpenROAD仍运行时恢复到约 `1.06 s/it`/GPU99%。两段均无 OOM、worker exit 或 I/O wait，
  不作为算法吞吐基准，也不触发训练重启。

## 21. Local-5 relation/profile acceptance 最终绑定

- post-G0 acceptance 现为最终 closure 的直接 REQUIRED，不再只是 score/projection watcher
  的间接前置条件。
- closure 重验 100 samples、12 blocks 和11项 provenance/relation RTL/descriptor/replay/
  projection/threshold 门禁，并要求 acceptance、ordered manifest、run identity 与 AEE
  rank-1 checkpoint SHA 一致。
- 正常/关系 RTL 失效反例测试通过；closure watcher PID 更新为 `2252989`。
- 软件/硬件闭环相关组合回归共 `20/20` PASS，并通过 `py_compile`/`git diff --check`。

## 22. Local-5/H67 当前证据状态与清理边界

- Local-5 fullres30 当前训练 ep3；首个可绑定 model/state 对在 ep9，最终 AEE rank-1 尚未
  产生。因此 score/Shiftmax、projection partial accumulator、ATLIF temporal matrix 三项
  正式 checkpoint-bound RTL report 当前状态均为 `queued`，不能提前计为 PASS。
- 软件流水线会先完成825样本 standard/deploy profile 和 rank-1 SHA 冻结；随后 post-G0
  acceptance、三个 RTL component report 与总 `checkpoint_bound_scope.json` 必须绑定同一
  checkpoint SHA。H67 ep30 和追加训练后的最终 rank-1 使用同样规则独立重跑/复用审计。
- 论文范围继续限定为 `checkpoint-bound component RTL-exact`，不包含 dynamic BN、cross-head
  accumulation、requant/residual、SRAM macro 或 full network。
- 第五轮磁盘清理仅删除6条退休旧微调的非锚点 model/state，保留各自 valid825 rank-1、final
  与配套状态；Local-5/H67/NB0 及硬件向量/profile/report 均受保护。审计见
  `neuron_autoresearch/cleanup_audits/retired_ft_intermediates_20260805.json`。

## 23. Local-5 ep9 早期 checkpoint 签收门禁

- 新独立 watcher 在 ep9 model/state 稳定写完后，立即核验 SHA、internal epoch、scheduler、
  milestones、AMP scaler、五参数组 LR 和 overlay210/missing0/unexpected0；正式输出为训练目录
  下的 `checkpoint_epoch9_early_audit.json`。
- 该门禁提前发现续训锚点损坏，但不替代最终 valid825 rank-1、post-G0 acceptance 或三个
  checkpoint-bound RTL component report。硬件向量仍只能绑定最终 rank-1，不允许绑定 ep9
  早期签收点后提前宣称 RTL-exact。
- watcher PID `2264517` 已独立常驻，相关正常/反例单元测试 `3/3` PASS。
- Local-5 ep3 已完成，train loss `1.6521`，但小验证 loss `2.0182`；当前进入 ep4。该42帧
  验证反弹只标记为 ep9 valid825 前的风险，不改变硬件绑定规则：没有最终 rank-1 SHA 前仍
  不生成或复用正式 component vectors。

## 24. H67/NB0 +10 标签与硬件绑定编号

- H67 的预算30/35/40对应 checkpoint label30/35/40，内部 epoch29/34/39；NB0 同预算对应
  label29/34/39。H67 的 offset1 是历史保存标签合同，不是多训练一轮。
- 配置 generator 已清除继承的旧 `resume_source_epoch: 15`，显式绑定 source budget30 和实际
  checkpoint label30/29。最终硬件 watcher 必须使用 convergence summary 给出的 rank-1
  checkpoint label/SHA，不得按内部 epoch自行拼路径。
- 编号/provenance 与既有 profile、acceptance、closure 回归合计 `14/14` PASS；训练数值参数
  与 staged state 未改变。
- staged-resume 审计现额外绑定 config SHA、source budget 与 checkpoint label；配置 SHA 已存在
  但不一致会 fail-closed，缺失旧字段仅能在模型/状态 SHA、hardlink 和 RNG disclosure 仍通过时
  升级。漂移反例通过后组合回归为 `15/15`；新 runner PID `2268615` 已加载该逻辑。
- H67/NB0 两份现有 `resume_stage_audit.json` 已实际完成升级，config SHA 分别为
  `86db3960...b15d1cbcc` 与 `55aeb36c...20290efe`，并与当前文件复算一致；model/state SHA、
  hardlink、epoch29、scheduler29、scaler/LR 和 RNG disclosure 同时通过。

## 25. 训练恢复 CLI 语义修复

- H9 wrapper 过去会把布尔式 `--resume 1`/`--finetune 1` 误绝对路径化为 `.../1`；上游仅按
  字符串真值判断，故历史功能未失效，但审计命令不规范。现只绝对化真正的模型/保存/MLflow
  路径，两个控制参数保持字面值 `1`。
- 新参数归一化测试通过；Local-5 ep9 后自动恢复和 H67/NB0 +10 均会加载修正入口。该修改
  不改变模型权重、优化器状态、checkpoint 标签或硬件 profile 合同。
- 与 checkpoint/provenance/profile/closure 门禁组合重跑 `16/16` PASS。
- 真实 `resume_model` 集成测试进一步验证 paired-state 自动命名、五组 optimizer LR、scheduler
  epoch/milestones、scaler 和 next-epoch 起点，组合回归更新为 `17/17` PASS。该证据保证后续
  最终 checkpoint/hardware SHA 不会来自只恢复模型的伪续训。
### 26. Local-5 运行时配置身份与硬件证据门控（2026-08-05 19:40 CST）

<!-- LOCAL5_RUNTIME_CONFIG_IDENTITY_HW_GATE_20260805 -->

- Local-5 fullres bb1e4 当前处于 ep4，尚未产生首个 ep9 checkpoint；因此 checkpoint-bound
  score/Shiftmax、真实权重 projection partial-accumulator、81 live-site ATLIF temporal matrix
  三类 RTL-exact 任务均处于正常等待状态，不得复用旧 Local-5 权重冒充本轮证据。
- 本轮训练进程早于配置生成器最终 mtime，故新增
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/enforce_local5_ep9_config_identity_20260805.py`。
  硬件证据只接受 `training_config_identity.json` PASS 后、由 standard valid825 选出的同一
  rank-1 checkpoint SHA；配置身份以 ep9 optimizer/scheduler state 为权威门控。
- 最终可声明范围仍严格限定为
  `checkpoint_bound_component_rtl_exact_not_full_network`。当前链路不覆盖 dynamic BN、跨 head
  汇聚、全网 requant/residual 或 SRAM macro，因此不能写成 full-network RTL-exact。

### 27. 训练身份到硬件向量的直接 SHA 链（2026-08-05 19:47 CST）

<!-- LOCAL5_TRAINING_IDENTITY_TO_RTL_SHA_CHAIN_20260805 -->

- post-G0 profiler 现在先等待 `training_config_identity.json` PASS，并将其 path/SHA 写入
  `post_g0_run_identity.json::source_bindings.training_config_identity`；训练 config 与 ep9 paired
  state 发生任何漂移均拒绝生成向量。
- component RTL watcher 在 acceptance 后再次复算该 source binding，并把训练身份、config、ep9
  state 三个 SHA 写入 `checkpoint_bound_scope.json`。closure 还会独立复算 acceptance 和 RTL 两侧，
  形成 `ep9 runtime state -> rank-1 checkpoint -> T450 profile -> component RTL` 的闭环。
- 门控单测 `3/3 PASS`；新 watcher PID `2286750/2286752` 当前在等待 ep9 身份 PASS，没有占用 GPU。
- ep9 model/state 审计与配置身份签收现由单一 `flock` enforcer PID `2287913` 完成，并在身份
  报告中绑定 early-audit SHA，避免并发写入竞态。

## 28. H67 all12 checkpoint 实权重 projection 补证

- 最终签核复查确认：既有 H67 watcher 对当前 checkpoint 只覆盖 score/Shiftmax 与 ATLIF；旧
  DCTF96 projection 结果来自历史 trace，不能跨 checkpoint 复用。该缺口现已作为 fail-closed
  条件修复，旧结果只保留为架构开发证据。
- vector generator 支持同 stage 多 block，以 `s{stage}_b{block}` 生成 all12 独立向量；每条
  使用当前 bit trace 内的真实 K/gate、dyadic INT8 projection weight 与 acc32 bias，并保存
  source manifest SHA 和 checkpoint run context。
- DCTF96 generator/TB/runner 从旧 T162/8-bit 参数化为 fullres T450/9-bit；超过255个
  destinations 的同类 term 确定性拆分。runner 逐条执行 all12 Icarus final-element bit-exact，
  并按 stage 编译、对 all12 全部执行 Verilator+SVA；report 必须给出 `record_count=12`、`temporal_tokens=450`、
  stage coverage `[0,1,2,3]`、weight mode 和 checkpoint SHA。synthetic token449/450-destination
  S0 Icarus 与 Verilator+SVA 已零失配通过，正式 all12 仍等待最终 checkpoint trace。
- ep30 watcher、post-convergence rank-1 follower 与最终 closure 现在都要求 score/Shiftmax、
  ATLIF、projection 三报告绑定同一 rank-1 checkpoint。总声明仍限定为
  `checkpoint_bound_component_rtl_exact_not_full_network`；动态 BN、requant、residual、decoder
  与 full-network bit exact 仍不在证明范围内。
- trace 复用前必须复算 checkpoint/config/12 NPZ SHA、all12 和 T450；旧 `audit.json` 单独存在
  不再构成复用条件，audit 在每次正式回放前重跑。

## 29. Local-5 活跃启动到硬件证据的进程签收

- 三次启动日志中，唯一有效根训练已由 `/proc` 签收为 PID2097444、父流水线2097439、
  `2026-08-05 14:24:39 CST`；14:28的退出码属于重复实例，不代表有效训练退出。
- `active_launch_provenance.json` 保存完整 argv、进程 start ticks、Python executable、source
  checkpoint SHA及 train/pipeline source SHA。ep9 training identity 直接绑定该文件 SHA，最终
  profile/RTL再绑定 training identity SHA，形成 `active launch -> ep9 state -> rank1 -> RTL` 链。
- 收据不宣称进程读取了后来重写配置的字节；运行时配置仍由 ep9 optimizer/scheduler/scaler
  state 签收。该限制会随最终 component RTL scope 一并披露。

## 30. 最终硬件证据等待器重签收（2026-08-05 20:29 CST）

- H67 ep30 T450、post-convergence rank-1 和 DATE closure 的旧 PID 文件对应进程已退出，日志均
  停在20:08；它们只处于 WAIT，未生成或覆盖任何 RTL PASS 产物。
- 三个任务已用 `nohup + setsid` 重新启动并签收为 PID `2310765/2310766/2310767`；复核均
  `PPID=1`、独立 SID，日志分别重新写入 Local-5 release、equal+10、最终 closure WAIT。
- Local-5 checkpoint-bound RTL producer `2286750/2286752` 与训练 PID2097444仍存活。最终证据
  颗粒度不变：score/Shiftmax、all12 real-weight projection、81-site ATLIF 三项同 checkpoint
  component exact，加 post-G0 profile/acceptance；任一缺失或 SHA 不同均不得签核。
- ep9 config-identity enforcer 的旧 PID 也已退出并重签收为 PID2311400（PPID=1、独立 SID）；
  它继续等待首个 model/state 对，负责把 active launch 与 optimizer/scheduler/scaler 事实绑定到
  后续 profile/RTL，而不提前生成任何硬件 PASS。

## 31. Local-5 ep4 硬件语义健康检查

- ep4 训练完成时 module summary 仍为 `ATLIFTernaryPSN=105`、official one-sided binary activity
  mean `5.2366%`、ternary activity `0`，Shiftmax attention `12`；没有在训练过程中退化为三值
  神经元或混合 attention 图。
- train loss `1.621065`，小验证从 ep3 `2.018181` 回落到 `1.642443`。这只证明当前图继续健康
  训练，不构成 AEE、收敛或硬件签核；正式硬件向量仍等待 ep9/14/19/24/29 valid825 rank-1。
- 软件保存/恢复、身份、profile acceptance、stale RTL复用拒绝等聚焦回归 `25` 项通过；AAE
  公式回归 `3/3` 通过。当前 checkpoint-bound component RTL 状态仍正确标记为 `queued`。

## 32. 跨软件/RTL follower 存活监督

- 新增 DATE closure watchdog，监督 ep9 identity、Local-5 checkpoint-bound RTL、H67/NB0
  equal+10、H67 ep30/post-convergence component RTL 和最终 closure 六项；训练主进程仍由原
  Local-5 pipeline supervisor 管理，职责不混合。
- 每项以 PID cmdline 和独立完成 marker/JSON PASS 双门控：进程死亡且任务未完成才重启，已有
  PASS 不重跑，PID 指向其他脚本不视为存活。三项正反测试通过。
- watchdog PID2319106 已独立常驻，首次 heartbeat 为 `incomplete=6/alive=6`；最多8次重启后
  fail closed。该监督只提高长队列可靠性，不改变任何 RTL scope、checkpoint 或指标。
- completion 还必须看到实际 artifact：Local-5 scope、equal+10 summary、H67 ep30 三组件
  report、post-convergence FINAL 与 closure JSON/MD；日志 marker 单独存在不算完成。更新后的
  watchdog PID2319720 已签收，六个 follower 仍全部存活。
- PID 文件陈旧时 watchdog 会按脚本 cmdline 扫描并收养现有 detached follower，而不是启动一个
  会被 flock 拒绝的副本。四项 watchdog 测试通过，最终加载 PID 为2320213。

## 33. Local-5 ep5 和 checkpoint-bound 硬件队列复核

- Local-5 fullres/window15 已完成 ep0--4，当前 ep5 约13%；训练图仍为105个
  one-sided binary ATLIF 与12个 Shiftmax attention，未转为三值或混合 attention。
- 本轮首个可签收 checkpoint 是 ep9；之前不使用历史 crop Local-5 RTL 结果替代
  fullres 证据。PID2286750 正在等待 training identity PASS 和 valid825 rank-1，之后必须
  产生同 checkpoint SHA 的 score/Shiftmax、real-weight projection 和81-site ATLIF 三类 RTL
  report，再与 T450/all12 post-G0 profile/acceptance 闭环。
- H67 ep30 T450、post-convergence rank-1、equal+10 收敛审计和 DATE closure 等待器均存活；
  watchdog 最近 heartbeat 为 `incomplete=6/alive=6`。当前硬件声明仍严格限制为
  `checkpoint_bound_component_rtl_exact_not_full_network`。
- 本轮磁盘清理只删除已退役 crop 实验的18个 optimizer-state，保留每条线的 ep19
  AEE rank-1 和 ep29 final model。因此不影响历史硬件复核，也不影响 Local-5/H67/NB0
  最终证据链。

## 34. 精度 profile 的 population/聚合绑定

- 新生成的 Local-5 与 H67/NB0 convergence `spike_profile.json` 必须同时保存825帧、
  18个本地 subsequences、valid-pixel 总分母、三种 AAE 聚合以及 validation CSV SHA。
- 最终 closure 会确认 frame-equal 重聚合与原生 AEE/AAE-2D/AE-3D 在 `1e-5` 内一致，
  防止硬件报告引用了不同数据 population 或不同 mask 的精度值。该门禁只增强指标
  provenance，不改变 component RTL-exact 声明范围。

## 35. H67 训练血缘与 Local-5 fullres RTL 状态

- `H67_FULLRES_LINEAGE_RECEIPT_20260805.json` 将 H67 绑定为自身 Motion-XOR crop ep19，经
  fullres ep0/5/10/15 到 ep30；起点 SHA `5ff626a7...3d22ba`，最终 SHA
  `7a484dc1...f37e4a`，没有 NB0 或 Local-5 初始化。最终硬件报告必须继续绑定这个训练血缘
  选出的 rank-1 SHA，不能只按实验名推断权重来源。
- H67 ep30 虽为边界最优，但 ep25到ep30 AEE仍改善2.47%，当前硬件只能先生成 ep30候选证据；
  +10 收敛比较完成后，post-convergence watcher 会为最终 rank-1 生成或复用同 SHA 的三组件
  报告。rank-1 若变化，ep30候选不得冒充最终证据。
- Local-5 fullres 当前进入 ep6，ep9运行时配置身份仍未 PASS。PID2286750 的等待是预期门禁，
  不是漏做 RTL；它将在 ep9/14/19/24/29 valid825 rank-1 确定后，绑定同一 checkpoint 完成
  score/Shiftmax、真实权重投影累加器和 ATLIF 时序矩阵 RTL-exact，并连同 profile/acceptance
  输出。声明颗粒度保持 `checkpoint_bound_component_rtl_exact_not_full_network`。
- 六任务 watchdog 已增加非阻塞 child reaping，并以 `6/6` 单测验证 completion、PID身份、
  detached follower 收养与退出进程回收；重启后已收养当前六条等待链，未触碰训练进程或
  checkpoint。该修改只保证队列可持续运行，不扩大任何 RTL-exact 声明。
- Local-5 初始 post-G0 child 的历史 SIGTERM 不再作为硬件失败依据。pipeline 现在在 child
  非零退出时恢复或加入 canonical producer，并只在 `acceptance.json` 已绑定 rank-1 checkpoint、
  100 samples、12 blocks 和阈值部署语义后放行。这样既不因旧 follower 退出丢掉完整训练，
  也不允许仅凭另一进程仍存活就宣称 RTL/profile PASS。
- equal+10 GPU release 不再以 score/Shiftmax 单报告存在为条件；必须等 score/Shiftmax、ATLIF、
  all12实权重 projection 三报告均绑定 H67 ep30 同一 SHA并完成最终 marker。这样软件续训不会
  与 ATLIF checkpoint-vector 生成争用 GPU，也使“硬件证据完成后释放训练”的含义与最终 closure
  完全一致。
- Local-5 ep5 完成时仍为105个 official one-sided binary ATLIF、12个 Shiftmax，ATLIF activity
  mean `5.4486%`、ternary activity `0`；train/小验证 loss 均继续明显下降。该健康检查只证明
  软件图和活动语义未漂移，不替代 ep9以后的 checkpoint-bound profile 或 RTL-exact。

## 36. ep30 profile/trace 复用门禁与 Local-5 当前状态

- Local-5 fullres 当前在ep6约17%，训练图仍为105个 one-sided binary ATLIF和12个Shiftmax；
  ep5 train/小验证 loss `1.515193/1.411659`，尚未进入首个ep9 checkpoint验收点。硬件 watcher
  仍等待最终 valid825 rank-1，并非漏做；其输出必须覆盖 profile100/T450/all12、relation
  acceptance、score/Shiftmax、真实权重projection partial accumulator和81-site ATLIF temporal
  matrix，全部绑定同一fullres checkpoint SHA。
- 修复 H67 rank-1 恰为ep30时的证据不对称：旧分支只复用三类RTL报告，现额外要求
  hardware-order config、`nts11_hardware_p0_profile.json`、all12 manifest与trace audit。验收会复算
  checkpoint/config SHA、100 samples、480x640/crop null、T2x15x15=450、ATLIF105/Shiftmax12、
  12个NPZ SHA及四stage覆盖；任一缺失或陈旧都不能生成最终PASS。
- ep30 producer、post-convergence binder、DATE closure auditor和watchdog required paths已同步升级。
  H67复用、closure、watchdog测试分别为`4/4`、`8/8`、`6/6 PASS`。新版watchdog PID2364542
  已拉起三个新版等待器，未触碰Local-5训练。
- H67 ep30仍是当前AEE最优，但末五轮改善2.47%，硬件侧只能称“当前候选”，不能称已收敛或
  最终冻结。最终硬件证据必须跟随H67/NB0对称+10完成后的rank-1 SHA；若ep35/40胜出，ep30
  的全部profile/RTL只能作为中间候选证据。
- 新一轮磁盘清理只移除H66a/H66f/H71中被rank-1 ep19在AEE、AAE和spikes三项同时支配的
  ep29模型，回收约2.06GiB；三条rank-1、全部profile/ranking/log/config及当前软硬件主线均保留。
- Local-5 RTL watcher 不再仅凭 acceptance 文件存在或 `accepted=true` 放行。启动三类RTL前必须
  重新解析当前rank-1，并绑定 acceptance/ordered manifest/run identity/hardware config/checkpoint/
  ep9 training identity SHA、100 samples、12 blocks及全部11项acceptance checks；陈旧rank-1证据
  会触发profile producer重建。
- score与projection vector manifest必须沿source manifest回溯到当前rank-1 SHA，ATLIF report也
  必须绑定同一SHA；聚合scope新增checkpoint/rank-1/run-identity/acceptance identity。四项门禁
  测试和closure测试分别`4/4`、`8/8 PASS`，新版WAIT follower PID2369602/2369604已签收。
- ATLIF report 自身包含config/checkpoint完整identity；上层复用现从仅比较checkpoint升级为
  config SHA与checkpoint SHA双门禁，并同步覆盖Local-5 producer/aggregate、H67 ep30/
  post-convergence及最终closure。配置被替换时不得复用旧ATLIF RTL PASS。H67、Local-5、closure
  回归为`5/5`、`4/4`、`8/8 PASS`，最终等待器PID为2372110--2372113及其Local-5子进程2372115。

## 37. Local-5 fullres RTL-exact 队列与 H67 直接 trace 绑定

- Local-5 fullres/window15 训练当前在ep6约44%，首个正式签收点仍是ep9；因此当前没有
  fullres Local-5 RTL-exact PASS 是预期状态，不是遗漏。ep9/14/19/24/29 valid825 rank-1 选出后，
  checkpoint-bound producer 必须用同一SHA完成profile100、T450/all12 trace与acceptance，再完成
  score/Shiftmax、真实权重projection partial accumulator、81-site ATLIF temporal matrix三类
  RTL-exact。历史crop证据不进入最终签核。
- H67 score/Shiftmax report 现在必须直接保存并校验 hardware-order config SHA、source all12 trace
  manifest路径及SHA；projection report也必须直接绑定该config SHA。这样旧trace、旧配置或只替换
  checkpoint中的任一种漂移都会在producer、post-convergence binder或closure层被拒绝。
- 加固后的H67复用回归为`5/5 PASS`，DATE closure为`8/8 PASS`。加载该逻辑的watchdog为
  PID2375263，H67 ep30/post-convergence/closure follower为PID2375265/2375266/2375267；Local-5
  训练PID2097444和Local-5硬件follower未重启。
- H67 ep30仍只是当前候选：其AEE/AAE-2D/AE-3D为`1.3387/6.0147/5.7558`，ep25到ep30 AEE仍改善
  2.47%。最终硬件冻结必须等待H67/NB0对称+10后重新选择rank-1；若ep35/40胜出，所有最终报告
  必须切换到新checkpoint SHA。
- 本轮磁盘清理仅删除三份退役早期短筛ep0模型并回收`1,948,774,400 bytes`，未触碰任何
  Local-5/H67/NB0、TTX/NTS锚点或RTL/profile依赖。审计见
  `neuron_autoresearch/cleanup_audits/retired_early_screen_checkpoints_20260805.json`。
- 新增checkpoint-set fail-closed回归，逐项证明五个eval model `9/14/19/24/29`和三个paired
  state `9/19/29`缺任意一个都不能进入valid825/deploy/profile。Local-5软件流水线测试为
  `8/8 PASS`，checkpoint-bound watcher为`4/4 PASS`，12-block真实权重projection合同为
  `1/1 PASS`；因此当前queued状态不是靠文件名约定，而是可执行门禁。
- equal+10 的GPU释放门同步要求H67 ep30 hardware-order config、profile100、all12 trace/audit及
  三类RTL全部直接绑定同一config/checkpoint SHA；还复核100 samples、480x640/T450、12个trace
  payload SHA与四stage覆盖。score必须绑定source trace SHA，旧config/trace/RTL组合不能提前释放
  续训。新版WAIT runner PID2383761已加载该门禁，当前仍未开训。
- Local-5 acceptance 后的ATLIF与score/projection GPU向量生成现强制串行：checkpoint-bound
  watcher先取得既有ATLIF flock并生成或复用双SHA绑定报告，之后才启动score/projection。
  顺序门禁测试后watcher为`5/5 PASS`，新版父/子PID2385101/2385103已加载；这消除了两个
  producer同时占GPU的OOM风险，不改变component RTL-exact颗粒度。
- score/projection的ordered source manifest现由仅checkpoint SHA提升为checkpoint/config双SHA；
  aggregate scope显式保存hardware-order config identity，最终closure再次沿两条manifest复算。
  缺失或短config SHA会fail closed。watcher/closure回归分别为`6/6`、`9/9 PASS`，加载新版的
  Local-5父/子PID2386788/2386791与closure PID2386789均在WAIT。
- 最终closure现在直接执行论文目标：每个H67/Local-5 rank-1均与同轮NB0比较，要求AEE不劣于
  5%且spikes至少下降20%；至少一个候选必须同时过线。负候选保留为消融，多候选过线时按AEE
  选择最终主线。回归为`10/10 PASS`，加载最新版的closure PID为2389117。
- 标准valid825断点复用不再只看ranking文件存在；五点Local-5及三点H67/NB0都必须先通过完整
  profile合同，否则自动重建评估。软件组合回归`10/10 PASS`，equal+10新版WAIT PID2390244
  已加载该恢复逻辑。
<!-- PROFILE_CHECKPOINT_CONFIG_DOUBLE_IDENTITY_20260805 -->

### 38. Profile 的 checkpoint/config 双身份约束

- Local5 standard、dyadic Q7/Q1.7、hardware-order 三类 profile 不再只绑定 checkpoint；每类必须同时记录并匹配对应 config 的绝对路径与 SHA256。
- 最终 DATE closure 独立复核 480x640、T2x15x15、825 frames、overlay210、missing0/unexpected0、ATLIF105、Shiftmax12，以及 checkpoint/config 双 SHA。H67/NB0 equal+10 profile 采用同一 fail-closed 规则。
- RTL-exact 声明仍限定为 checkpoint-bound component exact：score/Shiftmax、真实权重 per-head projection partial accumulator、ATLIF temporal matrix；不得外推为 full-network RTL-exact。
- 软件合同测试 Local5 `10/10 PASS`、closure `10/10 PASS`；closure watcher 已加载该规则并持续等待最终证据。

<!-- WINDOW15_PRETRAINED_WINDOW_AUDIT_20260805 -->

### 39. Window15/pretrained-window 的硬件解释

- 运行时 token geometry 由 `window_size=[2,15,15]` 决定，即每窗 T450；`pretrained_window_size` 不是硬件 tile 参数。
- 当前 all12 路径安装 `Spiking_QK_WindowAttention3D` 的 Shiftmax forward。静态源码审计确认该 forward 不读取 `pretrained_window_size`，因此 Local5 `[2,9,9]` 与 H67 `[2,15,15]` 的记录差异不会改变 T450 score/gate、projection 或 ATLIF RTL 向量。
- 硬件 profile/RTL 合同继续以 profile 中 `window_size=[2,15,15]`、`tokens_per_window=450` 和四 stage/12 blocks 覆盖为准；不得用 pretrained-window 字段替代实际 trace geometry。

<!-- H67_PROJECTION_PROVENANCE_V2_LOCAL5_EP7_20260805 -->

### 40. H67 projection provenance v2 与 Local5 RTL 队列状态

- Local5 fullres训练当前在ep7约32%，首个可绑定的正式checkpoint为ep9。完整硬件队列仍为：
  ep9/14/19/24/29同协议valid825选rank-1，随后生成float/dyadic/hardware-order profile、
  profile100/T450 all12 trace、score/Shiftmax RTL、81-site ATLIF temporal-matrix RTL及真实权重
  projection RTL。当前无Local5 fullres RTL-exact PASS是输入尚未产生，不是遗漏。
- H67 projection report schema升级到`h67_checkpoint_projection_rtl_exact_v2`。除了数值PASS和
  config/checkpoint identity，现在还逐文件绑定source trace manifest、vector manifest、12个
  record manifest、memh payload、generator与测试、runner、TB、SVA、bind和7个RTL源文件的
  path/SHA256/bytes。闭环复用会实时复算，向量或源码被替换后旧报告立即失效。
- 端到端报告生成及payload篡改测试、H67复用测试、DATE closure测试合计`11/11 PASS`。加载新版
  逻辑的H67 ep30/post-convergence/closure watcher为PID2418138/2418139/2418140；它们等待Local5
  释放GPU后生成H67 source trace并重做v2投影报告。
- 证据颗粒度保持不变：当前可签的是checkpoint-bound score/Shiftmax、ATLIF temporal matrix与
  per-head真实权重projection component exact，不是跨head/BN/requant/decoder的full-network
  INT8 RTL-exact。论文和硬件表必须按此边界表述。
- H67 ep30是当前最优但不是收敛签核点；末五轮AEE仍改善2.47%。最终硬件checkpoint SHA必须跟随
  equal+10后的rank-1，不能因为ep30暂时最好就冻结旧证据。
- 最终closure还要求候选具备收敛资格：Local5若ep29边界最优且末五轮AEE改善大于1%，或H67
  equal+10仍在边界改善，都不能冻结为硬件主线；NB0参考线未平台时最终选择整体失败。回归合计
  该门已进一步收紧为任何边界rank-1均right-censored，不再按小斜率自动放行。

<!-- LOCAL5_PROJECTION_PROVENANCE_V2_20260805 -->

### 41. Local5 component RTL provenance v2

- Local5 `checkpoint_bound_scope.json`升级为v2，逐文件绑定projection report/vector manifest、
  ordered trace manifest/payload、8个memh payload、13个RTL/TB、7个SVA以及runner、generator、
  summarizer源码。最终closure不信任静态PASS字段，而会重新计算全部path/SHA256/bytes。
- 生产形态21行source manifest去重、精确20个RTL/SVA集合及projection/score/ATLIF payload
  篡改反例后，加上site集合与边界收敛反例，全链回归为`32/32 PASS`。新版Local5硬件父/子
  PID2445870/2445874、closure PID2445871、equal+10 PID2440292已处于WAIT；GPU训练
  PID2097444保持运行。
- ATLIF的105表示软件安装wrapper数。93个唯一动态调用site目前由历史同构profile证明，本轮Local5
  仍须profile直接确认；81 replay表示sample0每站点首次调用中，去掉12个结果死亡attn_sn后的
  功能活跃site，每site抽取320个位置、共25,920 events，不是全张量覆盖。score/Shiftmax和真实
  权重projection另链验证；声明仍是component exact，未覆盖跨head、BN/requant、decoder和整网控制。
- score与ATLIF也已纳入递归provenance：score重算RTL/TB、日志、vector/source manifest/payload及
  shell/generator/reporter；ATLIF重算report source集合和manifest内generator、7个mem及contract。
  三类component任一源码或payload漂移都会令aggregate/closure失效。
- ATLIF vector manifest现在直接SHA绑定105 installed、93 called、12 dead-called和81 replayed的
  完整名称集合，reporter验证集合包含关系和dead site类型。最终收敛门也按right-censoring处理：
  Local5 ep29或H67/NB0扩展末点只要仍为rank-1，就不能因斜率小于1%而宣称收敛。
- Local5边界续训配置已预注册：`dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml`
  SHA256为`99f5baef32334762f6e5fb6d00aa61911e5ab91a38583ffa38a21a10b9a537a7`，固定自身ep29
  model/state续接和ep34/39签收点，不改变480x640/T450或硬件结构。当前总回归`35/35 PASS`。

<!-- LOCAL5_EP7_HARDWARE_QUEUE_CLEANUP13_20260806 -->

### 42. Local5 当前硬件队列、收敛边界与第十三轮清理

- Local5 fullres训练已完成ep7并正常进入ep8；ep7 train/小验证loss为`1.4330/1.3504`，相对ep6
  继续下降约`2.61%/5.61%`。首个冻结checkpoint仍是ep9，故当前没有Local5 fullres
  RTL-exact PASS是输入尚未产生，不是硬件流程遗漏。最终rank-1必须完成同一checkpoint/config
  双SHA绑定的profile100、T450/all12 trace、score/Shiftmax RTL、81-site ATLIF temporal-matrix RTL
  及真实权重projection partial-accumulator RTL，聚合声明保持
  `checkpoint_bound_component_rtl_exact_not_full_network`。
- ATLIF capture合同现在有独立直接测试：构造105 installed站点、实际调用93站点、过滤12个
  deployment-dead `attn_sn`后回放81站点，集合和SHA关系均通过。全链回归更新为`36/36 PASS`；
  该合成测试不替代最终Local5 checkpoint trace，实际105/93/12/81仍由最终manifest/report绑定。
- H67 ep30虽为当前rank-1，但末五轮AEE仍改善2.47%；NB0末五轮改善5.56%。最终硬件checkpoint
  不在此刻冻结，必须等待H67/NB0对称+10后跟随收敛合格的rank-1 SHA；若扩展末点继续最优，则
  仍按right-censored处理并继续对称预算，而不是把边界最优写成已收敛。
- 第十三轮清理仅移除42个2026-05-22 H40/probe短筛ep0模型，回收约17.11GiB；所有短筛配置、
  日志、profile和汇总测量保留。该范围不包含任何NB0、TTX/BTTX、H67、Local5、fullres、resume、
  rank-1或RTL/profile依赖，清理后可用空间约278GiB。审计见
  `neuron_autoresearch/cleanup_audits/retired_h40_may_screen_checkpoints_20260806.json`。

<!-- LOCAL5_THETA_FOLDED_CLOSURE_FIX_20260806 -->

### 43. Theta-folded projection 与最终 closure 枚举对齐

- Local5生产链使用`checkpoint_theta_folded_dyadic_int8_head_slice`：阈值已折叠进dyadic权重，随后
  进入真实权重per-head projection partial-accumulator RTL。最终closure此前仍检查未折叠旧枚举
  `checkpoint_dyadic_int8_head_slice`，会错误拒绝新rank-1的正确报告。
- closure已改为只接受theta-folded生产枚举并拒绝旧枚举；改动不改变RTL、向量或数值，只修正
  checkpoint-bound证据的最终签收合同。closure JSON同时绑定auditor自身源码SHA，旧PASS无法在
  审计代码变化后被静默复用。
- 全链定向回归为`38/38 PASS`，加载新合同的closure PID为2458356。Local5训练和profile/RTL
  follower未重启；当前ep8约12%，硬件链继续等待ep9身份及最终rank-1输入。

<!-- EQUAL10_LOCAL5_RTL_RELEASE_GATE_20260806 -->

### 44. Equal+10 训练前的双硬件完成门

- 旧equal+10 release函数只检查H67 ep30三组件证据，未读取Local5 checkpoint-bound aggregate；
  因此H67完成而Local5 projection/score向量仍在生成时存在训练抢GPU风险。
- 释放条件现增加Local5当前rank-1身份：ranking epoch、checkpoint/config path+SHA、aggregate
  component-exact/not-full-network scope、score/ATLIF/projection PASS、theta-folded weight mode和
  watcher completion marker全部一致后才允许训练。最终closure仍独立递归验证payload/RTL源码SHA，
  此处不以轻量调度门替代最终证据签核。
- 正例、旧枚举/checkpoint漂移及criterion文字一致性反例加入后全链`41/41 PASS`。收敛结果
  JSON/Markdown现与代码统一：最大观测预算点为AEE rank-1即right-censored，last5斜率只作描述，
  不再写`>1%`条件。加载新门的equal+10 PID2461925已处于
  `WAIT Local-5 RTL and H67 ep30 T450 evidence release`，Local5训练未重启。

<!-- LOCAL5_H67_RTL_STRICT_SERIALIZATION_20260806 -->

### 45. Local5先于H67的checkpoint-bound RTL严格串行

- H67 ep30 watcher不再以Local5软件pipeline marker加瞬时显存作为释放依据。它现在必须先验证
  Local5最终rank-1的checkpoint/config双SHA、component-exact/not-full-network scope、score/Shiftmax、
  ATLIF temporal-matrix及theta-folded真实权重projection三类PASS和completion marker，然后才可占用
  GPU生成H67 profile100/T450 trace与向量。新PID2464747已加载并处于
  `WAIT Local-5 checkpoint-bound RTL release`。
- 最终GPU顺序冻结为：Local5 fullres训练与ep9/14/19/24/29评估 -> Local5 checkpoint-bound三组件
  RTL -> H67 ep30三组件RTL -> H67/NB0对称+10 -> post-convergence H67 rank-1三组件RTL -> DATE closure。
  该拓扑无环，并避免Local5/H67/equal+10任意两个GPU producer交叉。
- H67复用测试此前为未被unittest发现的自由函数。现在6个H67正反例已显式装入unittest，fixture按
  projection provenance v2构造all12/source/vector/payload/RTL完整绑定；全链实际回归`47/47 PASS`。
  这项修复不改变RTL或模型数值，只确保验收测试确实运行。
- 退役MDR smoke/吞吐checkpoint另回收`9,656,684,544 bytes`；所有当前硬件checkpoint、正式MDR
  ep43、配置、日志和测量均保留，审计见
  `neuron_autoresearch/cleanup_audits/retired_mdr_smoke_bench_checkpoints_20260806.json`。当前Local5 ep8
  约29%，最终fullres RTL-exact尚待rank-1输入产生，旧crop PASS不得替代。

<!-- LOCAL5_SOFTWARE_MARKER_SHA_HARDENING_20260806 -->

### 46. 软件pipeline marker不再替代checkpoint身份验收

- Local5主pipeline的post-G0 acceptance验收已补齐schema、13项acceptance checks、manifest和run
  identity path/SHA、run identity v3、checkpoint/config path/SHA及manifest反向绑定；checkpoint哈希
  缓存同时绑定size/mtime，文件原地漂移后不能复用旧digest。
- 独立supervisor不再看到completion marker就退出。它会重新验证5个冻结模型、3个paired state、
  5份standard valid825 profile、当前rank-1和严格post-G0 acceptance。旧内存pipeline即使产生弱marker，
  supervisor也会拒绝并以新代码恢复；当前训练未重启，只有supervisor更新为PID2471551。
- acceptance等待现在只在严格验证通过后释放，stale文件会触发canonical producer重建而不是反复
  消耗supervisor重试。marker-only、旧acceptance、checkpoint内容漂移和stale恢复反例加入后，全链
  实际执行`49/49 PASS`。软件marker
  仍只表示训练/评估流水线结束；硬件真实性必须继续以checkpoint-bound三组件RTL aggregate和最终
  DATE closure为准。

<!-- AAE_DSEC_FL_CLOSURE_RECOMPUTE_20260806 -->

### 47. 算法指标与硬件checkpoint闭环的共同population门

- 新标准`DSEC_Fl`使用GT flow magnitude的3px/5%准则并以百分数输出；历史prediction-magnitude
  `AEE_outliers`只保留为legacy，不进入论文Fl-all比较。该指标只扩展推理profile，不改变当前Local5
  训练config、模型权重或硬件数据流。
- 最终closure从checkpoint/config SHA绑定的H67/NB0三点profile自行重算rank-1、last5/last10、角度
  平台和边界状态，不再信任summary派生字段。硬件最终checkpoint选择因此不能被陈旧summary误导。
- Local5五个standard点、两个deploy点、H67三点和NB0三点必须共享validation-list path/SHA及18序列
  精确帧数分布，并硬绑定valid825 SHA `7f3dc280...`。只有在相同population下选出的收敛合格rank-1
  才能进入checkpoint-bound score/Shiftmax、ATLIF和projection三组件RTL。
- 主闭环回归`50/50 PASS`，metric/aggregation另`8/8 PASS`；closure等待器已重启加载新合同。

<!-- LOCAL5_DEPLOY_SUMMARY_CONTRACT_20260806 -->

### 48. Local5部署摘要到checkpoint-bound RTL的最终软件门

- 当前Local5训练进程未重启，ep8约76%；ep9 model/state尚未生成，因此fullres Local5 RTL-exact仍是
  排队等待输入，不能用旧crop checkpoint的PASS替代。冻结点仍为ep9/14/19/24/29，最终仅对标准
  valid825 rank-1的checkpoint/config双SHA生成profile100、T450/all12 trace和三组件RTL。
- 由于训练父进程早于标准`DSEC_Fl`代码启动，独立supervisor增加部署摘要重验：float/dyadic/hardware
  必须同时含`AEE/AAE/AAE_Benchmark/DSEC_Fl/total_spikes`，部署两点还必须与其profile逐字段一致。
  若旧内存父进程落下弱marker，supervisor会用当前代码跳过训练并重建推理摘要，随后才允许
  checkpoint-bound score/Shiftmax、81-site ATLIF和theta-folded projection RTL启动。
- 部署摘要缺字段/数值漂移反例加入后，主闭环为实际执行的`51/51 PASS`，指标/聚合另`8/8 PASS`；
  新supervisor PID为`2487791`。清理只移除7个与当前硬件链无关且已有summary的一轮短筛权重，所有
  NB0/TTX/H67/Local5/正式MDR与NTS历史rank-1均受保护，不影响RTL输入可追溯性。实际回收
  `5,163,982,848 bytes`，清理后活动训练和全部保护锚点已复验通过。

<!-- LOCAL5_RTL_WATCHER_DURABILITY_20260806 -->

### 49. Checkpoint-bound RTL等待器的终端生命周期闭环

- 原Local5硬件父/子PID `2457012/2457015`仍绑定`pts/1`，长训练期间存在终端生命周期风险；当时仅
  等待ep9 runtime identity，未开始任何GPU profile、向量或RTL任务，故受控终止旧process group。
- 当前runner以`setsid`、stdin=`/dev/null`、stdout/stderr独立日志方式重启为父/子PID
  `2493350/2493352`；父进程PPID=1且无TTY，子进程继续显示
  `WAIT Local-5 ep9 runtime config identity PASS`。训练PID及其CUDA context未变化，硬件串行拓扑和
  checkpoint/config双SHA合同不变。

<!-- LOCAL5_EP9_RUNTIME_IDENTITY_PASS_20260806 -->

### 50. Ep9运行时配置身份与首个checkpoint硬件输入锚点

- Local5 ep9 model/state已稳定写盘并由独立审计绑定：model大小`591,166,629 bytes`、SHA256
  `695d0541...bfe5f`，state大小`432,588,102 bytes`、SHA256 `3437e4b6...6df9`。state/internal
  scheduler均为epoch9，milestones `{13,20}`、五组LR和AMP scaler全部PASS；没有修改state或停止训练。
- 加载证据仍为ATLIF `105`、Shiftmax `12`、overlay/missing/unexpected `210/0/0`，480x640和
  T2x15x15配置检查PASS。训练已由同一PID连续进入ep10。
- 硬件父/子PID `2493350/2493352`已读取并绑定identity SHA，随后正确进入
  `WAIT fullres deploy follower`。因此ep9只作为恢复与运行身份锚点；最终profile100、T450、score、
  81-site ATLIF和theta-folded projection RTL仍严格绑定五点valid825选出的最终rank-1。
## 51. Local5 ep9直接结构审计与最终RTL边界（2026-08-06）

<!-- LOCAL5_EP9_DIRECT_STRUCTURE_AND_FINAL_RTL_BOUNDARY_20260806 -->

- 软件侧新增
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/checkpoint_epoch9_structure_audit.json`，
  直接读取ep9 full-model对象并与训练身份SHA绑定。结果PASS：ATLIF 105、Shiftmax 12、overlay
  keys 210；ATLIF均为one-sided binary official-ATLIF，attention均为
  `binary_axnor_local5_shiftmax/reuse_k`。
- 该ep9证据只证明首个恢复锚点的结构和权重完整性，不是最终精度或RTL签收。硬件watcher仍必须
  等待full-res五点valid825选出的rank-1，再对同一checkpoint执行T450 profile100、ordered replay、
  acceptance、attention/relation/ATLIF component RTL-exact；任何旧crop或ep9结果都不得冒充最终
  full-res sign-off。
- 当前H67 ep30仍标记为边界最优、未证明收敛；H67/NB0公平+10轮与最终rank-1硬件profile排在
  Local5 checkpoint-bound RTL之后，避免训练与profile抢占同一GPU。

## 52. Local5最终硬件证据队列与AAE签核边界（2026-08-06）

<!-- LOCAL5_FINAL_HW_QUEUE_AND_AAE_MACHINE_RECEIPT_20260806 -->

- Local5 full-res训练保持原PID连续运行，当前ep10约29%。ep9的105 ATLIF/12 Shiftmax/210 overlay
  结构审计只作恢复锚点；最终硬件输入必须是ep9/14/19/24/29共同valid825排名后的rank-1。
- rank-1之后的强制顺序不变：float/dyadic/hardware-order部署评估 -> profile100/T450 all12 trace ->
  relation acceptance -> score/Shiftmax、81-site ATLIF temporal matrix、theta-folded projection三组件
  checkpoint-bound RTL-exact。当前没有full-res Local5 RTL PASS是依赖尚未满足，不是漏项。
- 软件侧新增`NB0_AAE_GAP_DIAGNOSTIC_20260806.json`并纳入最终closure。它确认H67 ep30与NB0 ep29
  均是AEE边界点而非收敛签核；因此最终硬件profile不得在equal+10完成前把H67 ep30永久固化为
  论文rank-1。AAE官方隐藏测试口径不参与本地RTL checkpoint选择。

## 53. Local5 rank-1 ATLIF/attention/projection component exact（2026-08-09）

<!-- LOCAL5_ATLIF_VERILATOR_HANDSHAKE_20260809 -->

- Local5 post-G0 acceptance已以13/13检查通过。最终rank-1为ep29，checkpoint SHA
  `6e0e92a5...c993b`；安装结构为ATLIF 105、Shiftmax 12、overlay `210/210`、
  missing/unexpected `0/0`。
- ATLIF testbench的旧ready/valid驱动存在接收沿后重读ready的调度竞争，表现为Icarus通过而
  Verilator停在command0。负沿驱动payload、稳定采样ready并只跨一个接收上升沿后，两个仿真器均
  完成81命令、25,920 hidden和25,920 event比较，mismatch均为0；lint/Yosys同时PASS。
- aggregate `checkpoint_bound_scope.json`已完成：score/Shiftmax、theta-folded checkpoint真实权重
  projection partial accumulator、ATLIF temporal matrix均为component RTL-exact。该scope明确包含
  `not_full_network`，不得写成整网RTL-exact。
- ATLIF fixed-vs-float局部事件翻转为`1177/25920=4.5409%`，所以
  `deployment_accuracy_signoff=false`保持正确。硬件零误差指RTL相对定点向量，整网精度仍由
  hardware-order valid825签核；两者不能合并成一个PASS。
- H67 ep30 T450 profile100/all12 trace已在Local5 release后启动，后续将生成自身checkpoint绑定的
  score/Shiftmax、ATLIF与12块真实权重projection报告。Local5/H67/NB0收敛队列现补入Local5自身
  ep29->34/39续训，最终若rank-1变化，必须对新rank-1重新绑定硬件证据，旧ep29 scope不能继承。


### DATE 算法/RTL 最终证据闭环

<!-- DATE_ALGORITHM_CLOSURE_AUDIT_PASS_20260805 -->

- fail-closed closure audit PASS；Local-5 rank-1 ep29，H67 rank-1 ep35。
- H67 收敛判定 `operationally_plateaued_or_overfit`，NB0 收敛判定 `operationally_plateaued_or_overfit`；AAE-2D 与 AE-3D 仍分口径报告。
- Local-5 收敛判定 `not_plateaued`，ep24到ep29 AEE改善 `2.301%`；边界仍改善时不得选为最终主线。
- H67 训练血缘由机器收据绑定为自身 Motion-XOR crop ep19 经五段续训到 fullres ep30；没有从 NB0 或 Local-5 初始化。
- Local-5 仅声明 score/Shiftmax、真实权重 per-head projection partial accumulator、ATLIF temporal matrix 三项 component RTL-exact；H67 同样不外推为 full network。

## 55. 交接后源码漂移重签与当前证据边界（2026-08-11 14:43 CST）

- 2026-08-11 03:59 的 closure PASS 后，Local5 projection generator 在当前工作树继续演进，旧
  `checkpoint_bound_scope.json` 因 generator SHA drift 不再能对当前源码复验。该问题是证据
  provenance 漂移，不是 checkpoint 数值或既有 RTL compare 失败。
- 14:38-14:39 对同一个 Local5 full-res ep29 checkpoint 重新执行
  `run_local5_bb1e4_checkpoint_bound_rtl.py`：ATLIF report 在 checkpoint/config/source 绑定均有效时
  复用；score/Shiftmax 与 100 组 T450 real-weight projection vectors/RTL 重新生成并 PASS。
- 最新 scope 已重新绑定当前 generator、runner、vector manifest、payload、RTL/SVA 和日志，
  `validate_local5_projection_provenance` 独立复验 PASS；14:43 DATE closure audit 再次 PASS。
- 当前可写 claim 仍严格限定为 checkpoint-bound component exact：ATLIF temporal matrix、
  score/Shiftmax 和 per-head projection partial accumulator。不得扩写为完整 encoder、decoder、
  BN/requant/residual 或 full-network RTL-exact。
- equal+10 的 Local5 ep39 是软件 AEE 最好点，但本节硬件绑定仍是 ep29；若论文把 ep39 升为部署
  checkpoint，必须对 ep39 重新跑 profile/trace/ATLIF/score/projection 全链，不能继承本次 scope。
- 机器审计：`neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.json`。

## 56. direct-MVSEC 的硬件签核边界（2026-08-11）

- direct-MVSEC 按 CICC/Spike-FlowNet 几何使用 center/random `256x256` 与 window
  `2x8x8`，attention tile 为 `T128`；DSEC full-resolution 使用 `480x640`、window
  `2x15x15`，tile 为 `T450`。两者共享 all12 H67/Local5 算法公式与 ATLIF 结构，但不能共享
  checkpoint 数值证据或 tile 级 cycle/traffic 数字。
- 当前 Local5 ep29、H67 ep35 的 score/Shiftmax、ATLIF、projection component RTL-exact 仅绑定
  DSEC checkpoint/T450 trace。MVSEC-NB0/H67/Local5 seed0 训练完成并冻结 winner 后，若论文报告
  MVSEC 硬件数据，必须从对应 MVSEC checkpoint 重新生成 T128 profile、all12 trace、定点向量与
  component RTL compare；仍不得外推为 full-network RTL-exact。
- CICC 风格模型表先分 float 与 hardware-order/INT8；部署累计表的 fixed800 traffic/cycle/energy
  必须与算法 fixed800 manifest SHA
  `5a1e8312a85d760e708165662e3925141286d3aac10d0a3710ab35aff5dac0bd`、同一 checkpoint SHA 和
  per-sequence sample index 绑定。full-sequence AEE另表报告，不得与 fixed800 的硬件计数混合归一化。
- direct-MVSEC 当前仅是算法训练/推理队列，尚未产生 MVSEC checkpoint-bound RTL PASS；DSEC
  closure PASS 保持有效，但不构成 MVSEC 部署签核。
