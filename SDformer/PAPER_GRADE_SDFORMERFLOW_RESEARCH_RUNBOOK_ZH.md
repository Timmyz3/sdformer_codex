# SDformerFlow 论文研究级 Runbook

## 1. 文档目的

这份文档不是“怎么把脚本跑起来”的速查表，而是一份面向论文研究的执行路线图。

目标是把你后续的工作拆成 6 条清晰主线：

1. 建立可复用的 smoke 流程
2. 建立完整 DSEC baseline
3. 建立正式评估与对比协议
4. 建立 MVSEC 泛化评估路线
5. 建立模块改进 / 注意力改进 / 剪枝研究路线
6. 建立最终论文表格、图和结论的产出路径

---

## 2. 先给结论

你当前已经完成的不是“论文级复现”，而是：

- 单序列 `zurich_city_09_a` 的 smoke 版预处理
- 单序列 1 epoch smoke 训练
- 单序列验证集 smoke 评估

这一步的意义是：

- 说明代码链路是通的
- 说明 DSEC 原始数据 -> `saved_flow_data` -> 训练 -> 评估这一套没有结构性断点
- 说明你之后改模块时，有一条可以快速回归的功能测试链

但这一步还不等于：

- 完整 DSEC baseline
- 官方 DSEC benchmark 结果
- MVSEC 正式实验
- 论文里的消融与对比

因此后续必须把“开发 smoke”与“论文实验”明确区分。

---

## 3. 当前仓库里最重要的脚本与配置

### 3.1 DSEC 训练与评估

- 训练入口：
  - `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
- DSEC 评估入口：
  - `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`

### 3.2 MVSEC / MDR 路线

- MDR 训练入口：
  - `third_party/SDformerFlow/train_mdr_supervised_SNN.py`
- MVSEC 评估入口：
  - `third_party/SDformerFlow/eval_MV_flow_SNN.py`

### 3.3 当前真正可用的关键配置

- DSEC 完整训练：
  - `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`
- DSEC 单序列 smoke：
  - `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml`
- DSEC 完整验证：
  - `third_party/SDformerFlow/configs/valid_DSEC_supervised.yml`
- DSEC 单序列验证：
  - `third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq.yml`
- MDR 训练：
  - `third_party/SDformerFlow/configs/train_MDR_supervised_SDformerFlow.yml`
- MVSEC 评估：
  - `third_party/SDformerFlow/configs/eval_MV_supervised.yml`

### 3.4 你这次新打通的辅助脚本

- 单序列预处理：
  - `tools/prepare_dsec_single_sequence.py`

它现在已经不是最初版本，而是支持：

- `--device auto/cpu/cuda`
- 断点续跑
- 跳过已生成样本
- CUDA 加速体素化
- 进度输出

---

## 4. 研究执行总路线

建议把整个研究拆成 5 个阶段。

### 阶段 A：开发 smoke

目标：

- 确认任何代码改动都不会把基础链路搞坏

完成标准：

- 单序列预处理成功
- 单序列 1 epoch 训练成功
- 单序列评估成功

作用：

- 这不是论文结果
- 这是你的“功能测试”

### 阶段 B：完整 DSEC baseline

目标：

- 用完整 DSEC 训练集和验证协议，跑出你自己机器上的可信 baseline

完成标准：

- 完整 DSEC `saved_flow_data`
- 60 epoch 训练完成
- 本地 validation 指标可复现
- 训练日志、run id、模型权重都可追踪

### 阶段 C：正式 benchmark / 对外对比

目标：

- 把 baseline 放进可与论文、benchmark 和其他方法比较的框架里

完成标准：

- 本地 validation 指标稳定
- 确认 benchmark 口径
- 如需官方提交，生成对应测试预测结果

### 阶段 D：改进研究

目标：

- 改模块、改注意力、改训练策略、做剪枝和加速

完成标准：

- 每种改动有 baseline 对照
- 有消融表
- 有精度、复杂度、速度三类指标

### 阶段 E：论文产出

目标：

- 把实验结果组织成论文表格、图、结论和局限性

完成标准：

- 有主结果表
- 有消融表
- 有可视化案例
- 有效率分析

---

## 5. 阶段 A：开发 Smoke Runbook

这是你后续每次改代码都应该先跑的一条链。

### A1. 数据准备

目标：

- 只准备一个 DSEC 序列，例如 `zurich_city_09_a`

完成标准：

- 原始目录存在：
  - `data/Datasets/DSEC/train_events/zurich_city_09_a`
  - `data/Datasets/DSEC/train_optical_flow/zurich_city_09_a`

### A2. 单序列预处理

命令：

```bash
cd /root/private_data/work/SDformer
python tools/prepare_dsec_single_sequence.py \
  --root data/Datasets/DSEC \
  --sequence zurich_city_09_a \
  --num-bins 10 \
  --valid-stride 10 \
  --device cuda \
  --progress-every 5
```

完成标准：

- `saved_flow_data/gt_tensors` 有 638 个样本
- `saved_flow_data/mask_tensors` 有 638 个样本
- `saved_flow_data/event_tensors/10bins/left/zurich_city_09_a` 有 638 个样本
- `train_split_seq.csv` 有 574 行
- `valid_split_seq.csv` 有 64 行

### A3. 单序列 smoke 训练

命令：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

完成标准：

- 训练进入 `Epoch 0`
- 能跑完整个 epoch
- 有 MLflow run id
- 有 `state_dict.pth`

### A4. 单序列 smoke 评估

命令：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python eval_DSEC_flow_SNN.py \
  --config configs/valid_DSEC_supervised_single_seq.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow \
  --runid <TRAIN_RUN_ID>
```

完成标准：

- 能恢复训练好的模型
- 能跑完整个验证集
- 能输出 `AEE` / `AAE`

你已经完成了这条链，所以后面任何改动都应该先复跑这一条。

---

## 6. 阶段 B：完整 DSEC Baseline

这是后续论文研究最重要的一步。

### B1. 明确数据范围

不要用单序列做论文基线。  
完整 DSEC baseline 应该至少使用 upstream 脚本里列出的训练序列。

当前 upstream 的 `flow_sequences` 包含 18 条训练序列，位于：

- `third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_preprocess.py`

测试序列列表也在同一脚本中定义，但官方 test benchmark 与本地 validation 不是一回事。

### B2. 完整预处理目标

你最终需要的不是 raw 目录，而是完整的：

```text
data/Datasets/DSEC/saved_flow_data/
  event_tensors/
  gt_tensors/
  mask_tensors/
  sequence_lists/
```

完成标准：

- 全部训练序列都进入 `event_tensors/10bins/left`
- `gt_tensors` / `mask_tensors` 数量与样本数一致
- `train_split_seq.csv` / `valid_split_seq.csv` 可被 `DSECDatasetLite` 正常读取

### B3. 为什么这一步要单独做

因为论文级训练最容易死在这里：

- 目录名不一致
- split csv 缺失
- 预处理 bin 目录和 loader 目录不一致
- raw 传输损坏
- 单序列脚本不等于全量预处理脚本

建议做法：

1. 单独写一个“完整 DSEC 预处理 wrapper”
2. 不要把“生成数据”和“训练”耦在一起
3. 预处理完成后先做数量与路径校验，再启动训练

### B4. 完整 DSEC 训练

命令：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

当前配置里的正式训练关键项：

- `n_epochs: 60`
- `batch_size: 1`
- `crop: [288, 384]`
- `n_workers: 4`
- `milestones: [10, 20, 30, 40, 50, 70, 90, 120]`

### B5. 阶段 B 完成标准

你只有在下面 5 个条件都满足时，才算拥有“完整 DSEC baseline”：

1. 完整 `saved_flow_data` 已生成
2. 60 epoch 训练跑完
3. 本地 validation 指标可复现
4. 模型权重和 run id 可追踪
5. 同配置重复一次结果在合理波动范围内

---

## 7. 阶段 C：正式评估与 benchmark

### C1. 本地 validation 和官方 benchmark 的区别

必须明确区分这两件事：

1. 本地 validation
2. 官方 DSEC benchmark

本地 validation：

- 直接使用本地 `saved_flow_data`
- 用 `valid_split_seq.csv`
- 自己本地算 `AEE/AAE`
- 适合调参、做消融、快速比较

官方 DSEC benchmark：

- 官方 test set GT 不公开
- 需要生成预测结果并按 benchmark 规则提交
- 适合对外报告最终成绩

所以研究节奏应该是：

1. 先本地 validation 做内部对比
2. 再决定哪些版本值得上 benchmark

### C2. 本地 DSEC 验证命令

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python eval_DSEC_flow_SNN.py \
  --config configs/valid_DSEC_supervised.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow \
  --runid <TRAIN_RUN_ID>
```

### C3. 当前仓库的评估兼容性说明

你当前仓库里已经修过两处兼容问题：

1. `eval_DSEC_flow_SNN.py`
   - 对缺省 `vis.store_att` 等可选键做了兼容
2. `utils/utils.py`
   - `load_model()` 兼容新版 MLflow 模型目录

因此后续基于 run id 恢复模型评估是可用的。

### C4. 阶段 C 完成标准

1. baseline 的本地 validation 指标稳定
2. 至少一个改进版本在同协议下优于 baseline
3. 如果准备对外展示，再决定是否做官方 benchmark 提交

---

## 8. 阶段 D：MVSEC / MDR 路线

这部分不要和 DSEC 单序列 smoke 混淆。

### D1. 当前仓库的真实 MVSEC 路线

从代码结构看，这个仓库当前是：

- 在 `MDR` 上训练
- 在 `MVSEC` 上验证 / 测试

对应文件：

- 训练：
  - `third_party/SDformerFlow/train_mdr_supervised_SNN.py`
  - `third_party/SDformerFlow/configs/train_MDR_supervised_SDformerFlow.yml`
- 评估：
  - `third_party/SDformerFlow/eval_MV_flow_SNN.py`
  - `third_party/SDformerFlow/configs/eval_MV_supervised.yml`

### D2. 训练命令

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python train_mdr_supervised_SNN.py \
  --config configs/train_MDR_supervised_SDformerFlow.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

### D3. MVSEC 评估命令

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python eval_MV_flow_SNN.py \
  --config configs/eval_MV_supervised.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow \
  --runid <TRAIN_RUN_ID>
```

### D4. 阶段 D 完成标准

1. MDR 训练可以稳定跑通
2. MVSEC 至少一个标准 test sequence 可稳定评估
3. `dt1` / `dt4` 协议明确
4. DSEC 改进版在 MVSEC 上也能评估，形成“泛化能力”对比

---

## 9. 阶段 E：和其他模型做公平对比

如果你要和其他模型比性能，最重要的不是“多跑几个模型”，而是保证协议一致。

你至少要固定下面这些变量：

- 数据集版本
- 数据划分
- event 表示方式
- 输入 bins / chunks
- 分辨率和 crop
- 训练 epoch
- batch size
- AMP 是否开启
- 指标定义
- 是否用 benchmark test set

### 推荐对比分层

#### 层 1：本仓库内部对比

- baseline
- 你的模块改进版
- 你的注意力改进版
- 你的剪枝版

优点：

- 可控
- 最公平
- 最适合做消融

#### 层 2：论文公开方法对比

把你最终的内部最佳版本，与论文中常见方法对照：

- STTFlowNet
- E-RAFT
- 论文里列出的其他 event optical flow 方法

注意：

- 如果这些数字来自论文原表，而不是你自己复现，需要明确注明来源
- 如果你自己复现外部模型，必须尽量统一协议

### 阶段 E 完成标准

1. 有一张内部对比表
2. 有一张与外部方法的主结果表
3. 每张表都写清楚协议

---

## 10. 后续改模型应该怎么做

你后面无论是改模块、改注意力、还是做剪枝，都不要直接冲完整训练。

推荐固定成 4 步：

### 第一步：Smoke 回归

目标：

- 确认改动没有破坏基础链路

做法：

- 重新跑单序列预处理
- 重新跑单序列 1 epoch 训练
- 重新跑单序列评估

### 第二步：小规模开发实验

目标：

- 快速判断改动有没有价值

做法：

- 固定少量序列
- 缩短 epoch
- 先看 loss 和验证指标趋势

### 第三步：完整 DSEC baseline 对比

目标：

- 确认在正式训练协议下是否真的提升

做法：

- 只改一个变量
- 其他 config 保持一致
- 记录 run id、显存、速度、参数量

### 第四步：MVSEC / 泛化验证

目标：

- 看改动是否只是记住了 DSEC，还是具有跨数据集泛化收益

---

## 11. 三类研究方向怎么推进

### 11.1 模块改进

适合做的事情：

- 更换 patch embedding
- 改 decoder / refinement head
- 增加轻量 multi-scale 分支

做法：

1. 保持训练协议不变
2. 先 smoke
3. 再完整 DSEC baseline
4. 最后做 MVSEC 泛化

重点指标：

- AEE / AAE
- 参数量
- 显存

### 11.2 注意力改进

适合做的事情：

- 稀疏 window attention
- token merge / token prune
- QK gating
- 结构化 head 剪枝

做法：

1. 先证明不掉链路
2. 再证明同精度下更快 / 更省
3. 或同资源下更高精度

重点指标：

- AEE / AAE
- 推理时延
- 显存占用
- FLOPs / 有效 token 数

### 11.3 剪枝与压缩

适合做的事情：

- channel pruning
- head pruning
- token pruning
- mixed-precision / 低比特实验

做法：

1. 先有强 baseline
2. 再做单变量剪枝
3. 需要同步记录精度和资源曲线

重点指标：

- 精度下降多少
- 参数减少多少
- 推理变快多少
- 显存是否下降

论文里真正有说服力的是 trade-off 曲线，而不是只给一个剪枝点。

---

## 12. 建议的实验表格设计

### 表 1：DSEC 主结果

列建议：

- Method
- Data protocol
- AEE
- AAE
- PE1
- PE2
- PE3
- Params
- FPS / Latency

### 表 2：内部消融

列建议：

- Baseline
- + Module A
- + Module B
- + Attention A
- + Pruning A

### 表 3：效率对比

列建议：

- Method
- Params
- FLOPs
- GPU Mem
- Latency
- AEE

### 表 4：泛化结果

列建议：

- Train on
- Test on
- AEE
- AAE

这里最自然的组合就是：

- DSEC baseline -> DSEC valid
- MDR-trained baseline -> MVSEC
- 改进版 -> DSEC + MVSEC

---

## 13. 论文研究的完成标准

如果你想把这个项目推进到“能写论文”，最低完成标准建议是：

### 最低可投稿标准

1. 有完整 DSEC baseline
2. 有至少一个稳定优于 baseline 的改进版
3. 有消融表
4. 有效率分析
5. 有 MVSEC 或其他泛化结果

### 更强的标准

1. 有官方 DSEC benchmark 提交结果
2. 有多 seed 稳定性
3. 有可视化案例
4. 有复杂度和部署分析

---

## 14. 建议你接下来按这个顺序做

### 第一优先级

完成“完整 DSEC baseline”

你现在最缺的不是更多改法，而是完整 baseline。

### 第二优先级

把单序列 smoke 固化成回归测试流程

目的：

- 以后你每次改模块，都先跑 smoke，而不是直接开完整训练

### 第三优先级

定义第一个研究改动

建议只选一种：

- 注意力稀疏化
- 轻量 refinement head
- token / head 剪枝

不要一开始把三种混在一起。

### 第四优先级

在完整 DSEC baseline 上做系统性消融

### 第五优先级

再接 MVSEC / 泛化结果

---

## 15. 当前仓库的两个重要实现备注

为了后续研究稳定，你应该记住这两件事已经发生过：

### 15.1 单序列预处理脚本已增强

文件：

- `tools/prepare_dsec_single_sequence.py`

当前支持：

- CUDA 加速
- 断点续跑
- 跳过已生成文件
- 进度显示

所以它适合做 smoke，不适合直接替代完整 DSEC 预处理流水线。

### 15.2 评估链已补过 MLflow 兼容

文件：

- `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`
- `third_party/SDformerFlow/utils/utils.py`

修复内容：

- 缺省可视化配置兼容
- 新版 MLflow 模型路径兼容

这意味着你后面用 `runid` 评估时，默认可以直接恢复模型，不需要每次手工找 `model.pth`。

---

## 16. 一句话执行摘要

你接下来的最优路径不是“立刻继续堆新模块”，而是：

1. 固化 smoke 链
2. 完整跑通 DSEC baseline
3. 明确本地 validation 与官方 benchmark 的区别
4. 用严格一致协议做模块 / 注意力 / 剪枝改进
5. 最后再做 MVSEC 泛化与论文表格

只有这样，后面的每一个改动才有研究意义，而不只是“又跑了一个实验”。
