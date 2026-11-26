# 较后 checkpoint 部署诊断入口

本入口只执行 [PROTOCOL.md](PROTOCOL.md) 冻结的一项诊断：271482 原生 checkpoint，自身176次BN校准，检查8帧 native/FF，首帧FF dual，共193次完整前向、0 optimizer。源 checkpoint 实际路径为 `/data/flow_a/runs/native_crop9_fresh_20261012/checkpoint_step_000271482.pt`。已有135930 BN和恢复权重不作为本入口输入。

## 来源与新增部分

- [run_deployment.py](run_deployment.py) 负责实际checkpoint/sidecar/run-contract/数据/成员SHA校验、校准与检查固定顺序、输出、30分钟上限及共享GPU0锁。原R13 runtime仅提供既有数据路径与原生source合同；checkpoint参数被明确替换为271482。
- [deployment_core.py](deployment_core.py) 新增原生校准FC hook记账、72自身BN导出和54 current-stat切换、FF原始全输入捕获、无损四头/八输入/八分支封存；不定义新的近似计算数学。
- 已审H0 core及其冻结依赖仅复用数学接口 `load_frozen/adapted_probe/forward`；不调用H0或历史collector的prepare/execute。原生loader是 `run_same_shape_reference.load_model`；BN reset及非BN摘要来自 `collect_bn_cma.reset_bn_cma/non_bn_digest`；这些源由 [source_pins.json](source_pins.json) 固定。
- 全部78 BN原生校准时逐次记录真实counter和12 MLP的FC1/FC2行数；仅此次实际得到的24 private BN × 3 buffer落盘。校准后清除54 external BN的running buffers。
- 每FC总行数107745600 = 192 × 558000 + 609600，包含176校准、16正式检查和1次dual。行数不是周期或能效。

## 输出合同

`results.json` schema `flow_later_checkpoint_deployment_v1`，完整状态 `LATER_CHECKPOINT_193_FORWARD_COMPLETE`。`binding`保存实际源、成员、runtime、region、源码和CPU检查SHA，以及完整193步计划。`forward_records`保留193条真实执行记录；校准176条含全部78 counter。

`bn_receipt.json` schema `flow_later_checkpoint_own_bn_v1`，完整状态 `OWN_BN_COMPLETE`；`bn_buffers.npz`为本模型自己计算的72数组。二者在结果中分别由 `own_BN_receipt` 和 `own_BN_buffers` 的 `path/sha256` 绑定。非BN状态须与原checkpoint完全一致，部署前后完整state及BN signature须相同。

`evaluations.endpoint.frames`为固定8帧，`arms.native/FF`均含16区域原始聚合量、最终预测payload SHA和执行记录；`summaries`含总体与两条序列。每帧一个 `endpoint_<id>.npz`，含native/FF最终预测，因此质量输出为8 NPZ、16预测。首帧FF actual/dual另外各有一个20数组无损NPZ，保存四flow、八whole-input、八whole-branch，数组SHA由实际捕获结果计算，且分别与执行head/branch records绑定。raw数组dtype为`float32`，head record的Torch dtype为`torch.float32`，两种字段格式不同。

`calibration_seconds`含176校准及其BN artifact序列化；`evaluation_seconds`含16+1前向、检查预测/raw artifact序列化和闭合审计；总`wall_seconds`还含模型加载与初始化。最终results JSON写盘本身不计入这三个字段。失败保存 `INCOMPLETE_NO_SCIENTIFIC_CONCLUSION`，不自动重跑。

## CPU检查与冻结候选

执行 `/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_deployment_cpu.py`：6项检查PASS，5.304秒，CUDA未初始化。测试复用现有小型12 MLP数学fixture：实际176次前向核所有78 BN counter、导出72数组/切换54 current-stat；真实H0 FF clone捕获8个完整输入、四头与八分支actual/dual逐位相同；失败时清理hook/恢复模块；193生产几何独立重算；9种不合法checkpoint合同拒绝；嵌套global绑定与零训练调用检查。四输出头和外部54 BN在测试中作最小合成适配，测试不冒充完整模型或真实精度验证。

| 文件 | SHA256 |
|---|---|
| run_deployment.py | `93ffc0ed503ea62d67a1d4f25e26c47800b122c0e5d0f5578ff35eb4069fa140` |
| deployment_core.py | `6e6dd6d0cd322f7e230d2a2015dacdfccbcf7b51c6570bd1bbb91cbf72ca82b0` |
| test_deployment_cpu.py | `22d5cc794d7abd9fbbc9b899b24b23dc3fe845b76bdb423669f679c34687b5cc` |
| source_pins.json | `c21a6529c0f22ba29b9087352ed4d3d1ebfd8c172f1a7960062e62d17c513963` |
| PROTOCOL.md | `af3f37f2c59603ca29b63f1609b37632724f90ab153615e58d521c0d37a32fd3` |
| cpu_results.json | `82d08cc2862b156a59eefd82630245a2d78f7eca5280e8b0c9421ef1c84604b5` |

此处只有CPU执行准备；独立源码/CPU复查和实际GPU执行/质量审计另存各自证据。本入口准备未读真实检查质量、未SSH/GPU。原checkpoint真实CPU加载与source合同独立审计位于 `independent_review/`。执行资格及root freeze由root编排，不能用本README代替独立审计。

## root执行接口

[root_launch.py](root_launch.py) 为root维护的纯编排入口，需root生成的 `EXECUTION_FREEZE_V1.json`。remote代码目录固定 `/data/flow_a/code/research_20261016/later_checkpoint_deployment`，新运行目录 `/data/flow_a/runs/later_checkpoint_deployment_20261016/run_v1`。默认不推理；只调用 `run_deployment.py` 以下参数：

```
--runtime-manifest PATH --runtime-manifest-sha256 SHA
--membership PATH --membership-sha256 SHA
--checkpoint PATH --checkpoint-sha256 SHA
--cpu-checks PATH --output-dir NEW_DIRECTORY
--preflight-output FRESH_RECEIPT_OUTSIDE_NEW_DIRECTORY
--wall-minutes 30
```

preflight实际CPU加载checkpoint、核源及数据SHA，创建receipt父目录，但禁止把receipt写入未来执行目录，因此不会提前创建run_v1。仅root在review/freeze/preflight满足后追加 `--execute`。执行仍要求全新输出目录和共享非阻塞GPU锁，无后续训练/H0排队逻辑。
