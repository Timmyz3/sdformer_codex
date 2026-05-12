# SDformer 实验总台账

## 用途

这份文档是整个项目的总实验记录表，用来统一记录每一次训练或验证尝试的：

- 实验目的
- 成功 / 失败状态
- 改动了什么
- 使用了哪个训练脚本和配置文件
- 对应的日志文件
- 对应的 MLflow experiment / run id
- 关联的模型权重、`state_dict`、阶段性报告
- 下一次实验是在什么基础上继续改的

后续每次只要有新的训练、推理、评估或失败排查，都继续往这份文档追加。

## 统一约定

- 当前项目根目录：[SDformer](/root/private_data/work/sdformer_codex/SDformer)
- 兼容软链接根目录：[SDformer](/root/private_data/work/SDformer)
- baseline 根目录：[third_party/SDformerFlow](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow)
- 实验日志目录：[experiments/logs](/root/private_data/work/sdformer_codex/SDformer/experiments/logs)
- 实验报告目录：[experiments/reports](/root/private_data/work/sdformer_codex/SDformer/experiments/reports)
- MLflow 根目录：[experiments/mlruns](/root/private_data/work/sdformer_codex/SDformer/experiments/mlruns)
- 本文件是后续唯一总索引；新实验先追加这里，再写单独 report。

状态说明：

- `成功`：训练或验证完整跑通，并产出了可用结果
- `失败`：中途报错退出，结果不可直接作为最终 baseline
- `中止`：人为停止，通常是为了切换到更合适的配置
- `进行中`：实验仍在运行
- `探测性尝试`：只做链路、路径或环境验证，没有形成完整有效 baseline

## 当前正在运行

- 状态：`无`
- 最近完成实验：第 18 次，原始全量权重 vs bs4 epoch59 推理对比
- 最佳当前权重：[checkpoint_epoch59.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth)
- 最新对比报告：[infer_compare_original_vs_bs4_epoch59_20260425.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/infer_compare_original_vs_bs4_epoch59_20260425.md)

## 数据集前置状态

当前 DSEC baseline 数据准备情况：

- 原始官方 coarse 数据已经下载并整理成 baseline 需要的目录结构
- `saved_flow_data` 已完成全量预处理
- flow 标注训练序列数：`18`
- 训练样本数：`7345`
- 验证样本数：`825`

## 实验总览

| 序号 | 日期 | 类型 | 状态 | 目的 | 主要改动 | MLflow Experiment | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-04-21 | 单序列 smoke | 失败 | 验证最小链路 | 直接使用单序列配置，仍请求 `cupy` backend | `DSEC_single_sequence_smoke` | `0f052c75bc3443d8935609f2c34a36cd` |
| 2 | 2026-04-21 | 单序列 smoke | 成功 | 验证训练、保存、MLflow 全链路 | 改为 `torch` backend，使用 hetero smoke 配置 | `DSEC_single_sequence_smoke_hetero` | `a417c8095b3f42e3a2762b3cf446ccb1` |
| 3 | 2026-04-21 | 全量训练探测 | 探测性尝试 | 尝试直接启动 full baseline | 使用默认 full 配置，未形成有效训练产物 | `Default` | `e7e12389273044869b86f08e50d2f0e5` |
| 4 | 2026-04-21 | 全量训练 | 失败 | 跑完整 baseline | 新建 full + `torch` backend 配置，保留 `AMP` | `DSEC_full_torch` | `5e98b281af454dfd9e17b16099d329dc` |
| 5 | 2026-04-21 | 全量训练 | 中止 | 验证关掉 `AMP` 能否提升稳定性 | 保持 full 配置，关闭 `AMP` | `DSEC_full_torch_noamp` | `76225a9bc6cb4e7099f6f69feceb57d7` |
| 6 | 2026-04-21 | 全量训练 | 手动停止，可恢复 | 尽量兼顾稳定性与速度 | 保留 `AMP`，学习率从 `1e-4` 降到 `5e-5` | `DSEC_full_torch_amp_lr5e5` | `9691153d1b6e495da2411029bdf27a11` |
| 7 | 2026-04-21 | 恢复训练 | 失败，可恢复 | 从第 6 次实验继续训练 | 修复 PyTorch 2.6+ `torch.load` 兼容问题，使用 `--resume True` | `DSEC_full_torch_amp_lr5e5` | `033e8ba0cb71405a8d13a243f837fdb6` |
| 8 | 2026-04-22 | 恢复训练重试 | 失败，可恢复 | 验证 Epoch 19 NaN 是否复现 | 从第 7 次实验保存点继续，未改训练逻辑 | `DSEC_full_torch_amp_lr5e5` | `2a8daca8481243d585100ca99b18c56e` |
| 9 | 2026-04-23 | 恢复训练 | 中断，可恢复 | 从 Epoch 45 继续完成 full baseline | 不改训练逻辑，继续 AMP lr5e-5 | `DSEC_full_torch_amp_lr5e5` | `a78bfc1ee0524671961d687cc7b9bc43` |
| 10 | 2026-04-23 | 恢复训练 | 成功 | 完成第一次全量 baseline | 从 Epoch 54 脱离终端继续训练 | `DSEC_full_torch_amp_lr5e5` | `66d1fc5322004d59a03c8ab132b11830` |
| 11 | 2026-04-23 | 推理评估 | 成功 | 用第一次全量权重跑 DSEC valid 推理 | `cupy` backend，无可视化配置 | 无新增训练 run | `66d1fc5322004d59a03c8ab132b11830` |
| 12 | 2026-04-24 | 后端 benchmark | 成功 | 比较 `cupy` 和 `torch` 后端 | 128 train / 8 valid 小切片 | `DSEC_backend_benchmark` | 多个短 run |
| 13 | 2026-04-24 | 吞吐 benchmark | 成功 | 找 A800 batch/worker 甜点区 | `bs1/4/8/16`，workers=4，TF32 | 多个短 run | 多个短 run |
| 14 | 2026-04-24 | bs4 从头全量 | 中止 | 尝试 `bs=4` full 从头训练 | `torch + bs4 + workers4` | `DSEC_full_torch_bs4_fast` | `455ad1898a8f47669c3f902c57fda2fe` |
| 15 | 2026-04-24 | bs4 继续训练 | 失败 | 从第一次全量权重继续训练 | MLflow 关闭但保存仍调用 MLflow，epoch0 后失败 | 无 MLflow run | 无 |
| 16 | 2026-04-24 | bs4 继续训练 | 成功 | 从第一次全量权重继续优化 | MLflow 记录指标，本地保存 checkpoint | `DSEC_bs4_continue_from_full` | `512913e55f73447db3de2f011af74ec9` |
| 17 | 2026-04-24 | bs4 断点继续 | 成功 | 从第 16 次 Epoch 15 继续到 Epoch 60 | 新增本地 checkpoint 恢复，恢复 optimizer/scheduler/scaler，最后一轮强制保存 | `DSEC_bs4_resume_epoch15_to60` | `98d161a3f7144441a60fa79083e0fffd` |
| 18 | 2026-04-25 | 推理对比 | 成功 | 对比原始全量权重和 bs4 epoch59 | 同一 DSEC valid 推理配置，比较 AEE/AAE/PE | 无新增训练 run | `66d1fc...` vs `checkpoint_epoch59.pth` |

## 详细记录

### 1. 单序列 smoke 失败：CuPy backend 缺失

- 日期：`2026-04-21`
- 状态：`失败`
- 目的：先验证单个 DSEC 序列能否从数据读取走到训练启动
- MLflow experiment：[718671080547918614](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/718671080547918614/meta.yaml)
- run id：`0f052c75bc3443d8935609f2c34a36cd`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/718671080547918614/0f052c75bc3443d8935609f2c34a36cd/meta.yaml)
- 训练日志：[smoke_train_zurich_city_09_a_20260421.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/smoke_train_zurich_city_09_a_20260421.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_single_seq.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml)

本次改动：

- 没有改模型结构
- 只是首次按 baseline 的单序列配置启动训练

结果：

- 启动即失败
- 日志中报错为 `ModuleNotFoundError: No module named 'cupy'`
- 后续又被包装成 `RuntimeError: CuPy backend requested but CuPy is unavailable`

结论：

- 这是环境 / backend 选择问题，不是模型训练本身的 `NaN`
- 也说明 upstream 默认配置会优先走 `cupy` backend，需要显式切到 `torch`

下一步：

- 改用 `torch` backend 再做一次单序列 smoke run

### 2. 单序列 smoke 成功：torch backend 跑通

- 日期：`2026-04-21`
- 状态：`成功`
- 目的：验证从数据读取、训练、保存到 MLflow 记录的完整链路
- MLflow experiment：[640502437107696771](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/640502437107696771/meta.yaml)
- run id：`a417c8095b3f42e3a2762b3cf446ccb1`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/640502437107696771/a417c8095b3f42e3a2762b3cf446ccb1/meta.yaml)
- 训练日志：[smoke_train_zurich_city_09_a_hetero_20260421.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/smoke_train_zurich_city_09_a_hetero_20260421.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_single_seq_hetero.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq_hetero.yml)
- `state_dict`：[state_dict.pth](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/640502437107696771/a417c8095b3f42e3a2762b3cf446ccb1/artifacts/training_state_dict/state_dict.pth)
- 模型记录目录：[m-6ab1a904a91f4da7adb7c80e3c75c71b](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/640502437107696771/models/m-6ab1a904a91f4da7adb7c80e3c75c71b/meta.yaml)

本次改动：

- 明确切换到 `runtime.snn_backend: torch`
- 使用更稳的 hetero smoke 配置
- `use_amp: False`

结果：

- 单序列训练成功跑通
- 确认环境、数据、模型实例化、反向传播和 MLflow 记录链路都可用
- 这是目前第一个真正可复现的 baseline 成功样本

结论：

- `torch` backend 可作为当前稳定 baseline 路线
- 后续 full 训练可以沿用这个方向

### 3. 全量训练探测：默认 full 配置未形成有效产物

- 日期：`2026-04-21`
- 状态：`探测性尝试`
- 目的：在 full 数据上直接探测 baseline 是否可开跑
- MLflow experiment：[745608124752639800](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/745608124752639800/meta.yaml)
- run id：`e7e12389273044869b86f08e50d2f0e5`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/745608124752639800/e7e12389273044869b86f08e50d2f0e5/meta.yaml)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：无独立定制配置，参数显示为默认 `Default` 实验名

本次改动：

- 属于最早的 full 训练启动探测
- 参数里 `data.path` 仍是 `data/Datasets/DSEC/saved_flow_data`

结果：

- run 很快结束
- 没有留下有效 artifact
- 没有形成当前可追溯的独立日志文件

推断：

- 这更像一次路径 / 配置探测性启动，而不是完整训练
- 由于缺少明确日志，这里暂不把它作为正式 baseline 结果，只保留台账索引

### 4. 全量训练失败：AMP 开启时出现 NaN

- 日期：`2026-04-21`
- 状态：`失败`
- 目的：在完整 DSEC 数据上直接跑 baseline
- MLflow experiment：[636237110622814736](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/meta.yaml)
- run id：`5e98b281af454dfd9e17b16099d329dc`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/5e98b281af454dfd9e17b16099d329dc/meta.yaml)
- 训练日志：[full_train_dsec_20260421.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_20260421.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch.yml)
- `state_dict`：[state_dict.pth](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/5e98b281af454dfd9e17b16099d329dc/artifacts/training_state_dict/state_dict.pth)
- 关联模型记录：
  - [m-a0c3dd5779e04b028488828e51f69ad6](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/models/m-a0c3dd5779e04b028488828e51f69ad6/meta.yaml)
  - [m-1dca1fced02c40e99c93781826b97e2c](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/models/m-1dca1fced02c40e99c93781826b97e2c/meta.yaml)
  - [m-763668084a194d1299921b6b3af658a5](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/models/m-763668084a194d1299921b6b3af658a5/meta.yaml)
  - [m-f46c98ded0974a49a7036a51cfb7ec26](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/636237110622814736/models/m-f46c98ded0974a49a7036a51cfb7ec26/meta.yaml)

本次改动：

- 新建 full 训练配置
- 改成 `runtime.snn_backend: torch`
- 使用全量 `saved_flow_data`
- 保留 `optimizer.use_amp: True`

结果：

- 成功进入 full training
- 速度较快，约 `4.3 ~ 4.5 it/s`
- 在 `Epoch 0` 的 `2461 / 7345` step 左右报错退出

关键报错：

- `RuntimeError: Function 'AddmmBackward0' returned nan values in its 1th output.`

判断：

- 当前更像训练数值稳定性问题
- 与早期的 `cupy` 缺失不是同一个根因
- `AMP` 很可能是诱发因素之一

### 5. 全量训练中止：关闭 AMP 后速度过慢

- 日期：`2026-04-21`
- 状态：`中止`
- 目的：验证关闭 `AMP` 后是否能稳定训练
- MLflow experiment：[856037723312197512](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/856037723312197512/meta.yaml)
- run id：`76225a9bc6cb4e7099f6f69feceb57d7`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/856037723312197512/76225a9bc6cb4e7099f6f69feceb57d7/meta.yaml)
- 训练日志：[full_train_dsec_noamp_20260421.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_noamp_20260421.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_noamp.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_noamp.yml)
- 阶段说明：[full_train_dsec_noamp_20260421.md](/home/zhumd/code/sdformer_codex/SDformer/experiments/reports/full_train_dsec_noamp_20260421.md)

本次改动：

- 在 full `torch` 配置基础上，把 `optimizer.use_amp` 从 `True` 改为 `False`

结果：

- 训练可以稳定起跑
- 速度下降明显，约 `1.28 it/s`
- 粗估 `60 epoch` 需要接近 `4` 天
- 因为速度不适合作为当前主线方案，本次人为停止

结论：

- 关闭 `AMP` 有助于稳定性
- 但训练代价太高，不适合当前 baseline 主线

### 6. 全量训练手动停止：AMP 开启 + 学习率下调

- 日期：`2026-04-21`
- 状态：`手动停止，可恢复`
- 目的：尽量同时保留训练速度和数值稳定性
- MLflow experiment：[183153168054988814](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/meta.yaml)
- run id：`9691153d1b6e495da2411029bdf27a11`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/9691153d1b6e495da2411029bdf27a11/meta.yaml)
- 训练日志：[full_train_dsec_amp_lr5e5_20260421.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_amp_lr5e5_20260421.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml)
- 阶段说明：[full_train_dsec_noamp_20260421.md](/home/zhumd/code/sdformer_codex/SDformer/experiments/reports/full_train_dsec_noamp_20260421.md)

本次改动：

- 保留 `AMP`
- 学习率从 `1e-4` 调低到 `5e-5`
- 其他主配置尽量保持不变，方便和失败的 AMP run 直接对照

当前进展：

- 已经明显跨过上次 `NaN` 出现的位置
- 已完整完成 `Epoch 0` 到 `Epoch 11`
- 手动停止发生在 `Epoch 12` 中途
- MLflow run 状态为 `FINISHED`
- GPU 训练进程已停止，显存已释放
- 前几轮速度约 `4.2 ~ 4.4 it/s`，后面部分 epoch 受机器负载影响下降到约 `2 it/s`

当前已记录指标：

- `Epoch 0`: `train_loss=7.2216`, `valid_loss=4.5886`
- `Epoch 1`: `train_loss=6.1457`
- `Epoch 2`: `train_loss=4.4695`
- `Epoch 3`: `train_loss=3.6239`
- `Epoch 4`: `train_loss=2.9671`
- `Epoch 5`: `train_loss=2.5374`, `valid_loss=1.7021`
- `Epoch 6`: `train_loss=2.4001`
- `Epoch 7`: `train_loss=2.1030`
- `Epoch 8`: `train_loss=1.9955`
- `Epoch 9`: `train_loss=2.0225`
- `Epoch 10`: `train_loss=1.8458`, `valid_loss=1.2362`
- `Epoch 11`: `train_loss=1.7333`

当前判断：

- 这次配置相比原始 AMP full run 更稳定
- loss 正在持续下降，说明训练方向目前是正常的
- 下次可以通过 `--prev_runid 9691153d1b6e495da2411029bdf27a11` 从保存状态继续训练
- 恢复训练会从 `state_dict.pth` 记录的 epoch 后继续，而不是严格从 `Epoch 12` 中途的 step 继续

## 后续追加模板

### 8. 恢复训练重试失败：Epoch 50 中途再次 NaN

- 日期：`2026-04-22` 到 `2026-04-23`
- 状态：`失败，可恢复`
- 目的：验证第 7 次实验在 `Epoch 19` 的 NaN 是否稳定复现
- MLflow experiment：[183153168054988814](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/meta.yaml)
- run id：`2a8daca8481243d585100ca99b18c56e`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/2a8daca8481243d585100ca99b18c56e/meta.yaml)
- 训练日志：[full_train_dsec_amp_lr5e5_retry_from_ep18_20260422.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_amp_lr5e5_retry_from_ep18_20260422.log)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml)
- 续训来源 run id：`033e8ba0cb71405a8d13a243f837fdb6`

本次改动：

- 未改训练逻辑
- 从第 7 次实验保存点继续
- 用于判断上一次 `Epoch 19` 的 NaN 是否固定复现

结果：

- 成功越过上一次 `Epoch 19` 的 NaN 位置
- 已完整记录 `Epoch 18` 到 `Epoch 49`
- 在 `Epoch 50` 中途约 `6268 / 7345` step 处中断
- 报错为 `RuntimeError: Function 'AddmmBackward0' returned nan values in its 1th output.`
- 最佳验证记录：`Epoch 45 valid_loss=0.9587`
- `Epoch 49 train_loss=1.3533`
- 当前 `state_dict.pth` 内部 epoch 为 `44`，所以下次从该 run 恢复会从 `Epoch 45` 开始

结论：

- `Epoch 19` 的 NaN 没有固定复现
- 但 AMP/SNN 后期仍存在偶发 NaN
- 原样重试可以推进训练，但不够可靠
- 下一次更建议修 AMP 梯度裁剪顺序，或降低后期学习率 / 关闭 AMP 做最后阶段收敛

### 7. 恢复训练失败：Epoch 19 中途出现 NaN

- 日期：`2026-04-21`
- 状态：`失败，可恢复`
- 目的：从第 6 次实验保存的状态继续训练
- MLflow experiment：[183153168054988814](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/meta.yaml)
- run id：`033e8ba0cb71405a8d13a243f837fdb6`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/033e8ba0cb71405a8d13a243f837fdb6/meta.yaml)
- 训练日志：[full_train_dsec_amp_lr5e5_resume_20260421.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_amp_lr5e5_resume_20260421.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml)
- 续训来源 run id：`9691153d1b6e495da2411029bdf27a11`

本次改动：

- 修复 [utils.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py) 中 PyTorch 2.6+ 的 `torch.load` 兼容问题
- 对 MLflow 保存的完整 PyTorch 模型显式使用 `weights_only=False`
- 对训练状态 `state_dict.pth` 加载也显式使用 `weights_only=False`
- 正确使用 `--prev_runid 9691153d1b6e495da2411029bdf27a11 --resume True`

结果：

- 日志确认 `Model restored from 9691153d1b6e495da2411029bdf27a11`
- 日志确认 `Model resumed from 9691153d1b6e495da2411029bdf27a11`
- 从 `Epoch 12` 开始继续训练
- 已完整记录 `Epoch 12` 到 `Epoch 18`
- 在 `Epoch 19` 中途约 `6119 / 7345` step 处中断
- 报错为 `RuntimeError: Function 'ConvolutionBackward0' returned nan values in its 1th output.`
- 最近一次验证：`Epoch 15 valid_loss=1.4160`
- 最近一次完整训练记录：`Epoch 18 train_loss=1.7098`
- 当前 `state_dict.pth` 内部 epoch 为 `17`，所以下次从该 run 恢复会从 `Epoch 18` 开始

结论：

- 恢复训练链路已经打通
- 后续如果再次暂停，必须同时传 `--prev_runid` 和 `--resume True`
- NaN 仍然存在，位置从前一次的 `AddmmBackward0` 变为这次的 `ConvolutionBackward0`
- 优先怀疑仍是 `AMP` 数值稳定性；另外当前 AMP 梯度裁剪没有先执行 `scaler.unscale_(optimizer)`，需要修正后再继续更稳

### 8. AMP 低学习率从 Epoch 18 重试，训练到 Epoch 50 中途 NaN

- 日期：`2026-04-22`
- 状态：`失败，可恢复`
- 目的：不改代码、不改配置，从第 7 次实验保存点继续，验证 Epoch 19 的 NaN 是否固定复现
- MLflow experiment：[183153168054988814](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/meta.yaml)
- run id：`2a8daca8481243d585100ca99b18c56e`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/2a8daca8481243d585100ca99b18c56e/meta.yaml)
- 训练日志：[full_train_dsec_amp_lr5e5_retry_from_ep18_20260422.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_amp_lr5e5_retry_from_ep18_20260422.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml)
- 续训来源 run id：`033e8ba0cb71405a8d13a243f837fdb6`

本次改动：

- 不改代码
- 不改训练配置
- 继续使用 `AMP=True`
- 继续使用 `lr=0.00005`
- 继续使用 `snn_backend=torch`

结果：

- 成功越过上一次出错的 `Epoch 19` 位置
- 已完整记录 `Epoch 18` 到 `Epoch 49`
- 在 `Epoch 50` 中途约 `6268 / 7345` step 处中断
- 报错为 `RuntimeError: Function 'AddmmBackward0' returned nan values in its 1th output.`
- 最近一次验证：`Epoch 45 valid_loss=0.9587`
- 最近一次完整训练记录：`Epoch 49 train_loss=1.3533`
- 当前 `state_dict.pth` 内部 epoch 为 `44`，所以下次从该 run 恢复会从 `Epoch 45` 开始

结论：

- Epoch 19 的 NaN 不是固定坏样本导致，因为重试后已经通过
- AMP/SNN 数值不稳定仍然存在，但位置具有随机性
- 在不改逻辑的前提下，可以继续通过 checkpoint 恢复推进训练

### 9. AMP 低学习率从 Epoch 45 继续，Epoch 54 中途静默停止

- 日期：`2026-04-23`
- 状态：`中断，可恢复`
- 目的：不改代码、不改配置，从第 8 次实验保存点继续，尽量完成 60 epoch 全量训练
- MLflow experiment：[183153168054988814](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/meta.yaml)
- run id：`a78bfc1ee0524671961d687cc7b9bc43`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/a78bfc1ee0524671961d687cc7b9bc43/meta.yaml)
- 训练日志：[full_train_dsec_amp_lr5e5_retry_from_ep45_20260423.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_amp_lr5e5_retry_from_ep45_20260423.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml)
- 续训来源 run id：`2a8daca8481243d585100ca99b18c56e`

本次改动：

- 不改代码
- 不改训练配置
- 继续使用 `AMP=True`
- 继续使用 `lr=0.00005`
- 继续使用 `snn_backend=torch`

结果：

- 已完整记录 `Epoch 45` 到 `Epoch 53`
- 最后日志停在 `Epoch 54`，约 `2696 / 7345` step，进度约 `37%`
- 日志末尾没有 `Traceback`、`RuntimeError`、`nan` 或 `Killed`
- 系统检查时训练进程已不存在，GPU 也已空闲
- MLflow 文件后端曾残留显示 run 状态为 `RUNNING`，检查确认进程不存在后已手动标记为 `KILLED`
- 最近一次验证：`Epoch 50 valid_loss=0.9863380775219057`
- 最近一次完整训练记录：`Epoch 53 train_loss=1.3865732051386486`
- 当前 `state_dict.pth` 内部 epoch 为 `53`，所以下次从该 run 恢复会从 `Epoch 54` 开始

结论：

- 这次不像前几次 AMP NaN，当前证据更像外部中断、终端断开、会话被杀或进程无堆栈退出
- 训练质量没有明显恶化，`valid_loss` 仍在 1.0 左右波动
- 如果继续“不改变任何东西”，可以从 run id `a78bfc1ee0524671961d687cc7b9bc43` 恢复到 `Epoch 54`

### 10. AMP 低学习率从 Epoch 54 继续，后台脱离终端完成全量训练

- 日期：`2026-04-23`
- 状态：`成功`
- 目的：不改代码、不改配置，从第 9 次实验保存点继续，避免终端断开导致进程被带走
- MLflow experiment：[183153168054988814](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/meta.yaml)
- run id：`66d1fc5322004d59a03c8ab132b11830`
- MLflow run 元数据：[meta.yaml](/home/zhumd/code/sdformer_codex/SDformer/experiments/mlruns/183153168054988814/66d1fc5322004d59a03c8ab132b11830/meta.yaml)
- 训练日志：[full_train_dsec_amp_lr5e5_retry_from_ep54_detached_20260423.log](/home/zhumd/code/sdformer_codex/SDformer/experiments/logs/full_train_dsec_amp_lr5e5_retry_from_ep54_detached_20260423.log)
- 关联脚本：[train_flow_parallel_supervised_SNN.py](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 关联配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml](/home/zhumd/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml)
- 续训来源 run id：`a78bfc1ee0524671961d687cc7b9bc43`

本次改动：

- 不改代码
- 不改训练配置
- 继续使用 `AMP=True`
- 继续使用 `lr=0.00005`
- 继续使用 `snn_backend=torch`
- 启动方式改为 `setsid` 脱离当前终端，并把输出写入独立日志文件

结果：

- 日志确认 `Model restored from a78bfc1ee0524671961d687cc7b9bc43`
- 日志确认 `Model resumed from a78bfc1ee0524671961d687cc7b9bc43`
- 已完成 `Epoch 54` 到 `Epoch 59`
- MLflow 状态为 `FINISHED`
- 当前 `state_dict.pth` 内部 epoch 为 `59`
- 最终完整训练记录：`Epoch 59 train_loss=1.3590569656182503`
- 最近一次验证：`Epoch 55 valid_loss=0.9676900447868719`
- 学习率：`1.5625e-06`
- 最大 GPU 显存记录：约 `4.962 GiB`
- 最后一轮耗时：`3058.12` 秒，约 `50.97` 分钟

结论：

- baseline 全量训练已经完成，可以作为后续神经元优化、体素化改进、注意力改进、剪枝实验的初始化基线
- 本次完成阶段没有出现 NaN 报错
- 后续需要基于该 run 的权重做推理，并生成中文训练/推理报告

备注：

- 启动前做过一次 45 秒前台诊断 run：`46c1f8a52573406b8c678f13f3285f0e`
- 该诊断 run 只用于确认恢复链路，不作为正式实验，已标记为 `KILLED`

后面每次新实验都按下面这个模板续写：

```md
### X. 实验标题

- 日期：
- 状态：
- 目的：
- MLflow experiment：
- run id：
- MLflow run 元数据：
- 训练 / 推理日志：
- 关联脚本：
- 关联配置：
- 关联权重：
- 关联报告：

本次改动：

- 

结果：

- 

结论：

- 

下一步：

- 
```

## 维护原则

- 不覆盖旧实验记录，只追加
- 失败实验也保留，因为它们对定位问题很有价值
- 每次新增配置文件时，在这份台账里明确写出“基于哪个配置改的”
- 每次新增日志文件时，在这份台账里和 `run id` 双向关联
- 训练完成后，把最终推理和指标也回填到对应实验条目里

## 2026-04-23/24 新增实验记录

### 11. DSEC valid 推理评估：第一次全量训练权重

- 日期：`2026-04-23`
- 状态：`成功`
- 目的：验证第一次全量训练得到的权重在本地 DSEC valid split 上的推理指标
- 权重来源 run id：`66d1fc5322004d59a03c8ab132b11830`
- 推理脚本：[eval_DSEC_flow_SNN.py](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/eval_DSEC_flow_SNN.py)
- 推理配置：[valid_DSEC_supervised_no_vis.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_no_vis.yml)
- 推理日志：[infer_dsec_cupy_20260423_130347.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/infer_dsec_cupy_20260423_130347.log)
- 推理报告：[infer_dsec_cupy_20260423_130347.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/infer_dsec_cupy_20260423_130347.md)

结果：

- `AEE = 2.3922855504879763`
- `AAE = 12.012944526824297`
- `AEE_PE1 = 0.5333474421983256`
- `AEE_PE2 = 0.24919899437101742`
- `AEE_PE3 = 0.15811654052745058`

结论：

- 本地 valid split 推理链路跑通
- 该 AEE 高于论文 benchmark 数值，当前不能直接等价比较，因为评估 split / checkpoint / benchmark 口径不同

### 12. 后端 benchmark：cupy vs torch

- 日期：`2026-04-24`
- 状态：`成功`
- 目的：确认当前 A800 上 `cupy` 和 `torch` 哪个 SNN backend 更快
- 配置：[train_DSEC_supervised_SDformerFlow_en4_backend_benchmark.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark.yml)
- 日志：
  - [backend_benchmark_cupy_20260423_0034.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/backend_benchmark_cupy_20260423_0034.log)
  - [backend_benchmark_torch_20260423_0034.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/backend_benchmark_torch_20260423_0034.log)
- 报告：[backend_benchmark_20260423_0034.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/backend_benchmark_20260423_0034.md)

结果：

- `cupy`: `train_samples_per_sec = 0.7318`
- `torch`: `train_samples_per_sec = 0.7486`

结论：

- 在当前服务器和代码路径上，`torch` 略快于 `cupy`

### 13. A800 吞吐调优 benchmark

- 日期：`2026-04-24`
- 状态：`成功`
- 目的：寻找 A800 上更合适的 `batch_size` / dataloader worker / runtime 设置
- 报告：[throughput_tuning_20260424.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/throughput_tuning_20260424.md)
- 相关配置：
  - [backend_benchmark_bs1w4.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark_bs1w4.yml)
  - [backend_benchmark_bs4w4.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark_bs4w4.yml)
  - [backend_benchmark_bs8w4.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark_bs8w4.yml)
  - [backend_benchmark_bs16w4.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark_bs16w4.yml)

结果摘要：

- 原始基线：`1.0631 samples/s`
- `bs1 + workers4 + runtime tuned`: `3.2858 samples/s`
- `bs4 + workers4 + runtime tuned`: `7.5272 samples/s`
- `bs8 + workers4 + runtime tuned`: `7.6922 samples/s`
- `bs16 + workers4 + runtime tuned`: `6.5619 samples/s`

结论：

- `bs4` 和 `bs8` 是当前甜点区
- `bs4` 速度接近 `bs8`，显存压力更小，后续主线优先用 `bs4`

### 14. bs4 full 从头训练

- 日期：`2026-04-24`
- 状态：`中止`
- 目的：用 `bs4` 加速 full training
- MLflow experiment id：`886437313958210942`
- run id：`455ad1898a8f47669c3f902c57fda2fe`
- 配置：[train_DSEC_supervised_SDformerFlow_en4_full_torch_bs4_fast.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_bs4_fast.yml)
- 日志：[train_full_dsec_torch_bs4_20260423_165845.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_full_dsec_torch_bs4_20260423_165845.log)
- 报告：[full_train_dsec_torch_bs4_20260423_165845.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/full_train_dsec_torch_bs4_20260423_165845.md)

结果：

- GPU 利用率和吞吐明显提高
- 但从头跑 `bs4` 的 loss 明显高于已完成的 `bs1` baseline
- 后续判断：更合适的路线是从 `bs1` 已训好权重继续 `bs4` 微调，而不是从头训练

结论：

- 该 run 不再作为当前主线继续

### 15. bs4 从第一次全量权重继续训练，MLflow 关闭版

- 日期：`2026-04-24`
- 状态：`失败`
- 目的：从第一次全量训练权重继续 `bs4` 小学习率优化
- 续训来源 run id：`66d1fc5322004d59a03c8ab132b11830`
- 配置：[train_DSEC_supervised_SDformerFlow_en4_bs4_continue_from_full.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_continue_from_full.yml)
- 日志：[train_bs4_continue_from_full_20260424_102645.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_bs4_continue_from_full_20260424_102645.log)
- 报告：[train_bs4_continue_from_full_20260424_102645.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/train_bs4_continue_from_full_20260424_102645.md)

结果：

- 成功加载旧权重：`Model restored from 66d1fc5322004d59a03c8ab132b11830`
- `Epoch 0 train_loss = 2.1728915300735063`
- 吞吐：`train_samples_per_sec = 9.3107`
- 失败原因：关闭 MLflow 后，保存模型逻辑仍调用 `mlflow.pytorch.log_model`，导致 `Could not find experiment with ID 0`

结论：

- 训练方向正确，loss 比从头 bs4 明显正常
- 需要保留 MLflow 或改为本地 checkpoint 保存
- 已修训练脚本，使关闭模型 artifact logging 时可以本地保存 checkpoint

### 16. bs4 从第一次全量权重继续训练，MLflow 指标开启，本地 checkpoint

- 日期：`2026-04-24`
- 状态：`成功`
- 目的：加载第一次全量训练完成权重，用 `bs4`、小学习率继续优化
- 续训来源 run id：`66d1fc5322004d59a03c8ab132b11830`
- MLflow experiment id：`234701887556936306`
- run id：`512913e55f73447db3de2f011af74ec9`
- 配置：[train_DSEC_supervised_SDformerFlow_en4_bs4_continue_from_full.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_continue_from_full.yml)
- 脚本：[train_flow_parallel_supervised_SNN.py](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 日志：[train_bs4_continue_from_full_mlflow_20260424_025025.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_bs4_continue_from_full_mlflow_20260424_025025.log)
- checkpoint 目录：[bs4_continue_from_full_mlflow_20260424_025025](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_continue_from_full_mlflow_20260424_025025)
- 最近可用 checkpoint：[checkpoint_epoch15.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_continue_from_full_mlflow_20260424_025025/checkpoint_epoch15.pth)
- 最近可用 state_dict：[checkpoint_epoch15_state_dict.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_continue_from_full_mlflow_20260424_025025/checkpoint_epoch15_state_dict.pth)

本次改动：

- `batch_size: 4`
- `lr: 1e-5`
- `n_epochs: 20`
- `runtime.snn_backend: torch`
- `detect_anomaly: False`
- `cudnn_benchmark: True`
- `allow_tf32: True`
- `n_workers: 4`
- `persistent_workers: True`
- `prefetch_factor: 4`
- `non_blocking: True`
- `SDFORMER_USE_MLFLOW=1`
- `SDFORMER_MLFLOW_MODEL_LOGGING=0`

结果：

- 日志确认已加载旧权重
- 完整跑完 `20` 个 epoch：`Epoch 0` 到 `Epoch 19`
- MLflow run 状态为 `FINISHED`
- 最终训练 loss：`1.5016479405277572`（Epoch 19）
- 最终训练吞吐：`9.2827 samples/s`，约 `2.32 it/s`，注意这里 batch size 为 `4`
- 最后一次验证 loss：`1.9024231867356733`（Epoch 15）
- 最高显存：约 `18.552 GiB`
- 大模型 artifact 不写入 MLflow，checkpoint 写到本地目录
- 本地 checkpoint 保存到 Epoch `0,1,2,3,4,5,6,7,9,10,12,15`；最后一个可用完整模型为 `checkpoint_epoch15.pth`

结论：

- 训练本身正常结束，没有发现 `Traceback`、`Killed` 或 `Exception`
- 由于保存策略没有在 Epoch 19 自动保存，当前用于推理评估的首选权重是 `checkpoint_epoch15.pth`
- 下一步应用 `checkpoint_epoch15.pth` 跑 DSEC valid 推理，对比第一次全量 baseline 的 AEE `2.3923`

### 17. bs4 从第 16 次 Epoch 15 本地 checkpoint 断点继续

- 日期：`2026-04-24`
- 状态：`成功`
- 目的：第 16 次只配置了 `20` 个 epoch，本次从其最后可用 checkpoint 继续训练到 `Epoch 60`
- MLflow experiment id：`793360624675302370`
- run id：`98d161a3f7144441a60fa79083e0fffd`
- 配置：[train_DSEC_supervised_SDformerFlow_en4_bs4_resume_epoch15_to60.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_resume_epoch15_to60.yml)
- 脚本：[train_flow_parallel_supervised_SNN.py](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 恢复逻辑：[utils.py](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py)
- 日志：[train_bs4_resume_epoch15_to60_20260424_163657.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_bs4_resume_epoch15_to60_20260424_163657.log)
- checkpoint 目录：[bs4_resume_epoch15_to60_20260424_163657](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657)
- 续训来源模型：[checkpoint_epoch15.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_continue_from_full_mlflow_20260424_025025/checkpoint_epoch15.pth)
- 续训来源训练状态：[checkpoint_epoch15_state_dict.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_continue_from_full_mlflow_20260424_025025/checkpoint_epoch15_state_dict.pth)

本次改动：

- 新增本地 checkpoint 加载支持：`--prev_runid` 可以直接传 `.pth` 文件
- `--resume True` 时自动读取同名 `_state_dict.pth`
- 恢复内容包括 optimizer、scheduler、AMP scaler 和 epoch 号
- `loader.n_epochs` 改为 `60`
- 保存策略补强：train loss 创新低或到最后一轮都会保存 checkpoint，避免最终 epoch 权重不落盘

启动确认：

- 日志确认：`Model restored from local checkpoint ... checkpoint_epoch15.pth`
- 日志确认：`Training state resumed from local checkpoint ... checkpoint_epoch15_state_dict.pth`
- 日志确认从 `Epoch 16` 开始
- 后台 PID：`150229`

结果：

- 完整跑到 `Epoch 59`
- 最终训练 loss：`1.436652611142533`
- 最佳训练 loss：`1.436652611142533`（Epoch 59）
- 最终 checkpoint：[checkpoint_epoch59.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth)
- 最终 state_dict：[checkpoint_epoch59_state_dict.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59_state_dict.pth)
- MLflow 状态为 `FINISHED`

### 18. 原始全量权重 vs bs4 epoch59 推理对比

- 日期：`2026-04-25`
- 状态：`成功`
- 目的：用同一套 DSEC valid 推理配置，对比原始 full baseline 与 bs4 续训最终权重
- 推理配置：[valid_DSEC_supervised_no_vis.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_no_vis.yml)
- 报告：[infer_compare_original_vs_bs4_epoch59_20260425.md](/root/private_data/work/sdformer_codex/SDformer/experiments/reports/infer_compare_original_vs_bs4_epoch59_20260425.md)
- 原始全量日志：[infer_compare_original_full_20260425.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/infer_compare_original_full_20260425.log)
- bs4 epoch59 日志：[infer_compare_bs4_epoch59_20260425.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/infer_compare_bs4_epoch59_20260425.log)

结果：

- 原始全量：`AEE=2.3922855505`，`AAE=12.0129445268`
- bs4 epoch59：`AEE=1.3306697985`，`AAE=7.8131987781`
- AEE 相对降低约 `44.38%`
- AAE 相对降低约 `34.96%`

结论：

- bs4 续训后的最终权重在同一推理口径下明显优于原始 full baseline
- 训练 loss/valid loss 不能单独作为当前优化成败判断，应以后续 DSEC valid AEE/AAE 为主
