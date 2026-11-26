# E0：普通量化诊断与一次固定 RNE4 恢复

2026-10-21。本目录为新实现；旧 R27–R30 源码与结果未改。实现代理只进行了 CPU 微型核验，没有连接服务器或启动 GPU。GPU 由 root 统一调度。源版本见 `SOURCE_FREEZE_V1.json`；实际数据预检、GPU 数值一致性、训练执行和质量仍需各自的实际回执。

## 数值合同

`mixed_integer_core.py` 派生自冻结 R30 core `ceec6924…a13d5`，但低域改为独立执行。完整 T10、全部空间/通道、真实尾部、全部 78 个 current BN 及原 PSN 保持。FC1/FC2 共享原 mask；四 patch conv 各自按输出 9×9 完整块分域。输出按原行顺序重建后才进入 BN。

- Q8：每输出通道 `A=max(abs(W))`，`s8=A/127`，CPU FP32 ties-to-even，`q8=clamp(round(W/s8),-127,127)`。零通道 divisor 设 1，实际 scale 保留 0。
- RNE4：`s4=A/7`，`q4=clamp(round(W/s4),-8,7)`，正常 maxabs 权重只占 `[-7,7]`。执行编码 `enc4=(q4<<4).int8`，低域仅运行 bit4–7，并以 `s4/16` 一次缩放；高域独立运行 q8 的八 plane，再以 s8 缩放。没有借用 q8 的高四 plane。
- 中点诊断：低域先真实执行 q8 的 bit4–7，再加入 `8*popcount(binary_input)`，最后乘 s8。计数、广播、加法记录在 `midpoint_correction`，不能认为免费。dense 参考直接点乘 `drop4(q8)+8`。
- 旧 Q8、旧 M 诊断直接装载 SHA 固定的原 R30 core。新 core 的 `drop4` 仅用于 CPU 一致性对照，不替换旧 M 诊断实现。
- 训练 forward 是实际整数；backward 为声明的 identity STE。dx 使用各域真实有效权重，卷积的两组 masked-output 梯度经完整 halo 汇合；dW 为同原权重的 identity STE，scale/round/clamp 不求导。没有任务损伤监督或新评分器。

RNE4 同时改变舍入和低域尺度，改善不能仅归因于去偏。`enc4*(s4/16)` 是本实现的实际 FP32 合同；不能凭实数等价假称所有极端/次正规浮点值都与 `q4*s4` 逐位相同。CPU 已对测试向量验证；实际整网需三路径核验。

## 三个运行入口与计数

统一入口 `run_joint_recovery.py --task diagnostic|smoke|recovery`。

| task | 更新 | 质量预测 | 完整前向 |
|---|---:|---:|---:|
| diagnostic，启用 root 同意的 `--diagnostic-parity` | 0 | 16 身份×4 状态=64 | 64 正式＋4 工程=68 |
| smoke | 1 个 M 更新 | 0 | teacher＋student＋STE/deploy/dense=5 |
| recovery | 固定 512 | 初始 Q4/M 各64，末端 Q4/Q8/M 各64，共320 | 128＋1024＋9＋192=1353 |

diagnostic 身份固定为 root 的 `../selections_v1.json`，SHA `60b7d6d525d4130f5f0efecaff95511808b2186c3aaba9e9f37a957db32ef31f`。仅使用 `localization_ids` 的原顺序，不读取 GT 数组。它们是已暴露训练域开发身份，不能称独立测试。

四状态为 `original_Q8 / original_M_drop4 / original_M_midpoint / original_M_rne4`，全部从同 W0 开始且不更新参数。两种新 M 在第一个固定身份的正式 deploy 前向上额外捕获全部 PSN、28 个整数输出和四 head；各追加 STE/dense 两次，逐字节比较。4 次额外前向明确是工程，不产生新质量身份。旧两状态不重复工程核验。

recovery 从原 W271482 和空 AdamW 开始，固定 train256 两遍、512 crop/顺序/seed，与 R30 一致。模式 Q4/Q8/M 次数为 256/128/128，M 哈希布局与配额不变。四头监督＋0.1 四头 native 固定 teacher L1，lr 2.5e-5、wd .01、betas .9/.999、eps 1e-8、clip100；477 允许参数、活跃/无梯度数量由实际记录决定。原初始 Q8 只复用已审计引用；新初始 Q4/M 全部重采。末端只用 step512，不读开发 GT 选择终点。

没有断点继续入口。失败输出保留账本和异常，但不得充作成功终点；重试从 W0 和新输出目录开始。smoke 权重绝不用于正式训练。仅 step512 生成终点 checkpoint，先写 `.pt.partial` 再原子改名，包含 optimizer、RNG、源 SHA 和实际状态摘要，仍标 `formal_resume_allowed=false`。

## 源复用与运行模板

仅复用 SHA 固定的 R28 `run_smoke_v2.py` / `run_recovery_v2.py` 作为模型/数据/原 loss/审计 hook helper，及其 R26 原始装载合同。`--r28-root` 需要含 `implementation/` 子目录；`--r26-root` 同旧运行。`--r30-core` **直接指定实际文件**，避免假设远端 R30 是扁平还是有 implementation 目录。该文件只用于 diagnostic。

先将本目录四个 `.py`、冻结文件、root selections 上传到新独立目录。root 核实实际远端文件位置后运行；以下是参数模板，不是本代理执行收据：

```bash
/data/flow_a/envs/v2/bin/python -B E0_ROOT/run_joint_recovery.py \
  --task diagnostic --diagnostic-parity \
  --r28-root /data/flow_a/code/research_20261019/round28 \
  --r26-root /data/flow_a/code/research_20261017/window_precision_round26 \
  --r30-core EXACT_FROZEN_R30_CORE_FILE \
  --members EXACT_ORIGINAL_R27_MEMBERSHIP_JSON \
  --selections EXACT_UPLOADED_SELECTIONS_JSON \
  --mask-contract EXACT_R30_MASK_CONTRACT_JSON \
  --mask-contract-sha256 5a5018e92bfb632b56788dc20f7a3d7a41ce1b732e73f7e78245040fe2236d16 \
  --runner-sha256 EXACT_RUNNER_SHA_FROM_SOURCE_FREEZE \
  --output NEW_DIAGNOSTIC_OUTPUT --wall-minutes 20 \
  --preflight-output NEW_DIAGNOSTIC_CPU_PREFLIGHT_JSON
```

CPU 预检通过后，同样参数加 `--execute`；已存在的预检收据不重写。`--execute` 必须显式给出。全局 GPU lock 在首次 CUDA 初始化前获取，已有输出目录拒绝，超时/失败保存 `results.json`。GPU flag 必须由 root 的独占调度决定。

smoke 使用 `--task smoke --wall-minutes 20`，不需 r30-core/selections/reference-contract。正式使用 `--task recovery --wall-minutes 60`，加 `--reference-contract EXACT_R30_INITIAL_REFERENCES_REMOTE_V1_JSON`；它验证旧双引用合同，但实际只复用其中 initial_Q8，不复用旧 initial_Q4 预测。每个任务使用不同新 output/preflight 路径。

## 输出及审计边界

- `results.json`：实际 forward/update 计数、状态、源绑定、全部预测来源/数组 SHA、真实 BN/仿射调用、工作记录与一致性结果。
- `predictions/<stage>/<target_id>.npz`：head3 原数组；四 head 全部保留摘要。parity head 原数组另存。
- `training_ledger.jsonl`：每一步真实 crop、模式/mask、loss/梯度、teacher/student 调用与实际整数工作；首见模式/最后一步保存完整 kernel 列表，其余只省略重复 kernel 明细，不省略总数/域/shape/表示摘要。
- `weight_error_initial.json`、恢复末端 `weight_error_final.json`：全28层及每输出通道相对真实 W 的均值/MAE/MSE，仅是普通量化诊断；与结果绑定文件 SHA 和对应状态摘要，不作为 AEE 替代物。
- `diagnostic_progress.json` 只报告当前状态内已完成帧数，不是完成收据。runner 不计算任务 AEE，不读取开发 GT 数组；CPU 预检为了来源核对会读取并散列已有 GT 文件字节。

当前未运行真实 GPU。CPU 检查不替代 GPU full-graph 一致性、执行独立审计或原数组质量评估。独立 RNE 权重/scales 的供数、两个 weight slot、真尾使 Q8 始终存在等费用必须重新计算；不能搬用 R30 M 的 11% 模型数字。

## CPU 验证

`test_cpu.py` 的10项包括：真实512日程、root16身份和full/crop配额；tail/完整tile拒绝规则；int64独立黄金值；全部符号plane及−128编码；RNE ties/零通道；中点补偿；原 Q8/M 逐字节一致性；FC STE dx/dW/bias；conv halo梯度；真实原 MLP 和 MS_ResBlock 的 current-BN 三路径及 backward。6次原模块微型前向不计整网前向。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B \
research/flow_story_20261021/e0_ordinary/test_cpu.py NEW_CPU_RECEIPT.json
```

`CPU_CHECKS_V1.json` 是首次通过；之后仅改执行元数据、精确旧 core 文件参数及输出绑定，最终源对应 `CPU_CHECKS_V2.json`。两者保留，不覆盖历史收据。训练器的完整实际模型/数据预检需要 root 在专用服务器执行；本地只核验微型数学与来源合同。
