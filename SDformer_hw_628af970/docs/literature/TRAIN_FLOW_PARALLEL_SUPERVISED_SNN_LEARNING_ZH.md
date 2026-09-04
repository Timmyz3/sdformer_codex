
# SDformerFlow 学习笔记

这份笔记用于持续记录我对 `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py` 以及后续相关入口文件的学习过程。

约定：

- 讲解按“代码块 + 行号范围”组织，但会尽量覆盖到每一行的作用。
- 每次聊天里的讲解，都会同步整理到这份文档里，后续继续追加。
- 当前默认配置文件是 `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`。

---

## 文件定位

- 目标文件：`third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
- 作用：这是 upstream SDformerFlow 的 SNN 监督训练入口。
- 主链路：解析 YAML -> 初始化模型 -> 构建优化器与损失 -> 加载 DSEC 数据 -> 训练 -> 验证 -> 记录 MLflow -> 保存最优模型

---

## 第 1 轮精读：行 1 到 153

### 1. 导入区：行 1 到 27

`1` `import argparse`

- 引入命令行参数解析器。
- 这个脚本最终支持 `--config`、`--prev_runid`、`--finetune`、`--resume` 等启动参数。

`2` `import mlflow`

- 用于记录实验参数和指标。
- 这说明脚本默认把训练结果交给 MLflow 管，而不是单纯只在本地打印日志。

`3-5`

- `os`、`time`、`torch`。
- `time` 后面主要用于统计 epoch 和 validation 耗时。
- `torch` 是训练主框架。

`6` `from torch.optim import *`

- 把优化器名字直接导进当前命名空间，例如 `AdamW`。
- 后面作者用 `eval(config["optimizer"]["name"])` 动态实例化优化器，所以这里用通配符导入。
- 工程上这不是很干净，但在这种“配置里写名字再动态构造对象”的风格里很常见。

`7` `from configs.parser import YAMLParser`

- 训练配置从 YAML 读入，并经过 `YAMLParser` 转成脚本内部统一配置结构。
- `YAMLParser` 还负责确定 `device` 和随机种子。

`8` `from loss.flow_supervised import *`

- 导入监督光流损失。
- 本文件里真正用到的是 `flow_loss_supervised`。

`9` `from tqdm import tqdm`

- 给 dataloader 外面包进度条。

`10` `import math`

- 这个文件里实际上没直接用到，属于冗余导入。

`11`

- 从 `Spiking_STSwinNet` 导入 3 个模型类。
- 默认配置里真正会用到的是 `MS_SpikingformerFlowNet_en4`。
- 脚本并没有手写 `if model_name == ...`，而是稍后用 `eval(config["model"]["name"])` 动态选择。

`12` `from utils.gradients import get_grads`

- 如果配置要求存梯度，会用它把每层梯度抓出来保存到 CSV。

`13` `from utils.runtime_backend import configure_snn_backend`

- 这是 SNN 特有步骤。
- 它会根据当前设备、神经元类型、配置项，为模型设置合适的后端实现。

`14`

- 这里导入了一组模型加载/恢复/统计/保存工具。
- 训练过程中最关键的是：
- `load_model`：载入预训练权重。
- `resume_model`：恢复优化器、scheduler、scaler、epoch。
- `save_model` / `save_state_dict`：保存最优模型和训练状态。
- `count_parameters` / `print_parameters`：打印模型规模。

`15` `from utils.visualization import Visualization_DSEC`

- 如果打开可视化，会用这个类把事件、GT、预测流可视化。

`16` `from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite`

- 这是 DSEC 光流训练集读取入口。
- 本脚本训练和验证都依赖它。

`17`

- 导入数据增强算子。
- 这个脚本实际只用了 `Compose`、`CenterCrop`、`RandomCrop`、`Random_horizontal_flip`、`Random_vertical_flip`。
- 其他项是一起导进来但没用。

`18-19`

- `log_config`、`log_results` 在这个脚本里没真正调用，属于冗余导入。
- `compute_throughput_stats` 会在每个 epoch 结束后统计吞吐。

`20-27`

- `F`、`cv2`、`random`、`collections`、`DistributedSampler` 这些在当前文件里也基本没用。
- `spikingjelly.activation_based` 这一行很关键：
- `functional` 用来重置 SNN 状态、设置 step mode。
- `neuron` 和 `surrogate` 用于按配置选择脉冲神经元与 surrogate gradient。
- `from ...Spiking_submodules import *` 是为了拿到 `GatedLIFNode`、`PSN`、`SLTTLIFNode` 等自定义神经元类。

结论：导入区已经暴露了这个文件的真实职责，它不是“单纯训一个 PyTorch 模型”，而是“用 YAML 配置驱动的 SNN 光流训练入口”。

### 2. 全局开关和训练函数入口：行 29 到 48

`29` `use_ml_flow = True`

- 全局常量，默认总是启用 MLflow。
- 所以后面训练脚本天然绑定 MLflow 工作流。

`32` `def train(args, config_parser):`

- 整个训练逻辑都包在 `train()` 里。
- 外部只负责解析命令行和构造 `YAMLParser`。

`33-38`

- 从 `config_parser` 中取出配置和设备。
- `config_parser.device` 是在 `YAMLParser.get_device()` 里决定的。
- 这里打印设备，帮助确认是 CPU 还是哪个 GPU。

`39-47`

- 如果启用 MLflow，就设置 tracking URI、experiment 名称、开始 run、记录参数。
- `mlflow.log_params(config)` 直接把当前配置字典整体打上去。
- `prev_runid` 也额外记一份，方便知道这次是不是在接着某个历史 run 训练。
- `artifact_uri[:-9]` 这类切片写法比较硬编码，目的是打印 artifact 根目录，而不是完整子路径。

`48` `config = config_parser.combine_entries(config)`

- 这一句很重要。
- `YAMLParser` 会把 YAML 顶层的 `spiking_neuron` 合并进 `config["model"]["spiking_neuron"]`。
- 后面代码访问神经元配置时，实际走的是 `config["model"]["spiking_neuron"]` 这一份。

### 3. AMP、可视化和输入尺寸修正：行 50 到 68

`50-55`

- 根据 `config['optimizer']['use_amp']` 决定是否启用混合精度。
- 开启时创建 `torch.cuda.amp.GradScaler()`，否则 `scaler = None`。
- 后面 forward 会配套使用 `autocast(enabled=scaler is not None)`。

`57-59`

- 如果可视化打开，就实例化 `Visualization_DSEC(config)`。
- 默认配置里 `vis.enabled: False`，所以一般不会走这里。

`62-68`

- 这一段在修正 Swin Transformer 的输入尺寸。
- 如果 loader 配了 `crop`，就把 transformer 的 `input_size` 设成裁剪后的大小。
- 否则退回原始分辨率 `resolution`。
- 这很关键，因为窗口划分、patch embed、后续多尺度结构都依赖输入尺寸。

### 4. 模型构建与预训练加载：行 70 到 121

`70-73`

- 这里动态实例化模型。
- 如果 `config["swin_transformer"]["use_arc"][0]` 为真，就给模型传两份配置：
- 第一份是 `model` 配置。
- 第二份是 `swin_transformer` 配置。
- 默认配置里 `use_arc[0] = "swinv1"`，是非空字符串，所以会走第一个分支。
- 默认最终等价于：

```python
model = MS_SpikingformerFlowNet_en4(config["model"].copy(), config["swin_transformer"].copy())
```

`75-76`

- 把模型搬到设备上。
- 然后显式调用 `model.init_weights()` 初始化参数。

`77-80`

- 这一段名义上是“multi-gpu”。
- 但当前代码并没有真正包 `DataParallel` 或 `DistributedDataParallel`。
- 它只是当 `loader.gpu` 是字符串时，试图解析成多卡 id 列表。
- `model.module.init_weights()` 这句只有模型已经被并行封装后才成立，所以这一段从实现上看是不完整的。

`83-84`

- 初始化恢复训练相关状态。
- `epoch_initial = 0` 表示默认从头训。
- `remap = None` 用于 finetune 时重映射某些预训练权重键。

`86-92`

- 如果传了 `--finetune`，就根据 `use_arc[0]` 决定 remap 规则。
- `swinv2 -> "v2"`，`swinv1 -> "v1"`。
- 这说明作者预期不同 Swin 主干对应不同权重命名格式，需要 remap。

`93` `model = load_model(args.prev_runid, model, device, remap)`

- 真正加载预训练权重的入口。
- 即使不是 finetune，也会执行这句，只不过 `prev_runid` 可能为空。
- 所以 `load_model` 内部必须自己处理“空 runid 不加载”的情况。

`95-100`

- 如果优化器配置里定义了 `SG_alpha`，就遍历全模型，把所有 `surrogate.ATan` 的 `alpha` 改掉。
- 这说明 surrogate gradient 的斜率不是完全写死的，而是允许从配置调。

`102` `functional.reset_net(model)`

- 对 SNN 模型很关键。
- 因为 SNN 神经元内部有膜电位等状态，训练前必须先清空。

`103` `functional.set_step_mode(model, config['data']['step_mode'])`

- 设置 SpikingJelly 的 step mode。
- 默认配置里是 `'m'`，表示 multi-step。
- 这和事件序列按时间步展开处理有关。

`106-119`

- 根据配置中的 `neuron_type` 选神经元类。
- 可选项包括 `if`、`lif`、`plif`、`glif`、`psn`、`SLTTlif`。
- 默认配置里是 `psn`，所以最终 `neurontype = PSN`。
- 最后的 `raise "neurontype not implemented!"` 写法不规范，严格来说应该抛 `Exception(...)`。

`121` `configure_snn_backend(model, device, config, neurontype)`

- 这是“把模型真正切换到目标 SNN 运行后端”的步骤。
- 到这一行，模型才算不只是“普通 PyTorch Module”，而是完成了 SNN 运行时配置。

### 5. 参数统计、优化器和恢复训练：行 123 到 153

`123-127`

- 打印模型结构、参数细节、参数总量。
- 方便先确认网络真的按预期实例化出来了。

`128-130`

- 如果开了 MLflow，再把参数总量记进去。

`132-137`

- 按配置创建优化器。
- 如果是 `AdamW`，会显式带 `weight_decay`。
- 否则只传 `lr`。
- 默认配置是：

```yaml
optimizer:
  name: AdamW
  lr: 0.0001
  wd: 0.01
```

所以默认就是 `AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)`。

`139-142`

- 如果 scheduler 设为 `multistep`，就创建 `MultiStepLR`，并把 gamma 固定成 `0.5`。
- 默认 milestones 是 `[10, 20, 30, 40, 50, 70, 90, 120]`。
- 否则 scheduler 为 `None`。

`143` `optimizer.zero_grad()`

- 在正式训练前先清一次梯度。

`144-148`

- 处理梯度累积。
- `num_acc` 如果没配就默认 1。
- 默认配置里是 1，所以等价于“不做梯度累积”。

`149-150`

- 如果传了 `--resume`，就恢复优化器、scheduler、scaler 和起始 epoch。
- 这跟 `load_model()` 不同：
- `load_model()` 更偏向载入模型权重。
- `resume_model()` 是恢复完整训练状态，继续跑没结束的实验。

`152-153`

- 实例化监督光流损失：

```python
loss_function = flow_loss_supervised(config, device)
```

- 这个对象后面会根据 `gamma` 是否为空，决定是走“多级预测 sequence loss”还是“普通监督 loss”。
- 默认配置里 `loss.gamma: Null`，所以默认训练更偏向“逐个预测结果求监督损失后平均”。

---

## 到第 153 行为止，你应该建立的心智模型

这份入口脚本到这里做完了 5 件事：

1. 把外部 YAML 和命令行参数读进来。
2. 把 SNN 模型实例化、初始化、必要时加载预训练权重。
3. 根据配置设置 surrogate gradient、step mode、neuron type 和 backend。
4. 构建优化器、scheduler、AMP scaler、loss。
5. 为正式进入 dataloader 和训练循环做好准备。

换句话说，`154` 行之前都还在做“训练系统搭建”，真正的数据流和梯度流还没开始。

---

## 下一轮建议阅读范围

下一段最值得继续的是：

- 行 `155-217`：训练/验证数据集与 dataloader 构建。
- 行 `221-387`：训练主循环，这是整份文件最核心的部分。
- 行 `397-518`：验证与学习率更新。
- 行 `527-566`：命令行入口。

---

## 第 2 轮精读：行 155 到 566

### 6. 数据增强、训练集和验证集：行 155 到 217

`155` 只是一个分隔注释，表示进入 data loader 部分。

`157-175`

- 这里定义训练和验证时的数据增强。
- 注意，增强不是写进 `DSECDatasetLite` 里，而是样本取出来之后在训练循环里手动调用。

`158-165`

- 如果是 finetune 模式，训练增强只做水平翻转和垂直翻转。
- `transform_valid = None`，说明 finetune 时验证阶段不做中心裁剪。
- 这背后的意图一般是：微调时尽量保留原始分辨率几何关系。

`166-175`

- 非 finetune 模式下，训练增强包含：
- `RandomCrop`
- `Random_horizontal_flip`
- `Random_vertical_flip`
- 验证增强只做 `CenterCrop`。
- 默认配置里 `crop: [288, 384]`，所以训练和验证都会围绕这个尺寸工作。

`180-186`

- 构建训练集：

```python
train_dataset = DSECDatasetLite(config, file_list='train', stereo=False)
```

- `file_list='train'` 表示去训练 split 里取样本。
- `stereo=False` 说明这里只训练单目光流分支，不走双目。

`190-197`

- 构建训练 dataloader。
- 默认是：
- `batch_size = config["loader"]["batch_size"]`
- `shuffle=True`
- `drop_last=True`
- `pin_memory=True`
- `num_workers = config["loader"]["n_workers"]`
- 默认配置里 `batch_size: 1`，`n_workers: 4`。

`200-206`

- 构建验证集，与训练集差别只在 `file_list='valid'`。

`210-217`

- 构建验证 dataloader。
- 与训练集区别是：
- `shuffle=False`
- `drop_last=False`
- 这符合验证阶段“固定顺序、不丢最后一批”的常规逻辑。

到这里你要意识到一个事实：这个入口脚本把 dataset 和 transform 分开了。也就是说，`DSECDatasetLite` 更像“原始样本读取器”，而不是“完整训练管线”。

### 7. 训练循环初始化：行 221 到 240

`223` `best_loss = 1.0e6`

- 用一个很大的初始值做“最优训练损失”基线。
- 后面只要 epoch loss 更小，就保存模型。

`224` `grads_w = []`

- 如果要记录梯度，就把每一步抓到的梯度塞进这里，epoch 末再统一存 CSV。

`231-238`

- 进入 epoch 循环。
- 每个 epoch 开头会：
- 打印 epoch 编号。
- 记录开始时间。
- 如果是 CUDA，就重置峰值显存统计。
- 切到 `model.train()`。
- 初始化本 epoch 的 batch 计数 `sample` 和累计损失 `train_loss`。

`240` `for chunk, mask, label in tqdm(train_dataloader):`

- 训练 dataloader 每次返回 3 个对象：
- `chunk`：事件表示。
- `mask`：有效像素掩码。
- `label`：GT 光流。

### 8. 每个 batch 的准备工作：行 241 到 252

`241` `torch.autograd.set_detect_anomaly(True)`

- 每个 batch 都打开 anomaly detection。
- 好处是反向图里一旦出现 NaN/非法梯度，更容易定位。
- 代价是很慢，所以这更像调试期开关，不太像最终高性能训练配置。

`243-244`

- 每个 batch 前都重置 SNN 状态，再设置 step mode。
- 这一步非常关键，因为 SNN 有时序状态，不能把前一个 batch 的膜电位带到下一个 batch。

`246-249`

- 把 `chunk`、`label`、`mask` 搬到 device 上。
- `chunk` 和 `label` 转成 `float32`。
- `mask` 先搬运，再 `unsqueeze` 成 `[B,1,H,W]` 风格，便于与 flow 逐像素相乘。
- 注释说明当前 `chunk` 原始形状大致是 `[B,20,2,H,W]`。

`250-251`

- 如果定义了训练增强，就把 `chunk, label, mask` 打包后一起变换。
- 这说明 crop/flip 不只是改输入，也会同步改 GT 和 mask，避免几何错位。

### 9. forward 前的输入编码与归一化：行 253 到 303

`253`

- 如果 AMP 打开，就在 `autocast` 里做 forward 和 loss。

`255-260` `encoding == 'cnt'`

- 如果输入编码是 count-based 表示：
- 可视化开时，只把时间维求和用于显示。
- 如果不开可视化且 `polarity=True`，会把 `[B,T,2,H,W]` reshape 成 `[B,2T,H,W]`。
- 这更像传统事件体素或 EV-FlowNet 风格通道展开。

`263-275` `encoding == 'voxel'`

- 默认配置就是 `voxel`，所以训练主路径看这里。
- 当 `polarity=True` 时：
- `neg = relu(-chunk)` 取负极性。
- `pos = relu(chunk)` 取正极性。
- 然后在一个新的 polarity 维上拼起来，形成 `[B, C, P, H, W]`。
- 这里的 `C` 本质上对应时间 bin，`P=2` 对应正负极性。
- 如果开可视化，还会把正负事件沿时间维求和，生成展示用张量。

这里有个关键认识：默认配置 `loader.polarity: True` 并不是“忽略极性”，而是把正负极性重新拆成单独维度交给 SNN 主干。

`278-280`

- 如果编码既不是 `cnt` 也不是 `voxel`，直接报错。
- 配置是这个训练入口的第一控制器，编码错了，后续所有张量形状都会错。

`282-296`

- 归一化阶段。
- `minmax`：仅对非零元素做最小最大归一化。
- `std`：仅对非零元素做标准化。
- 默认配置是 `minmax`。
- 只处理非零元素的原因很直接：零值在事件表示里通常意味着“没有事件”，不想被归一化步骤污染。

`300-302`

- 如果配置了 `spike_th`，就把输入硬阈值化成 0/1。
- 默认配置里 `spike_th: Null`，所以这一段通常不执行。
- 这说明脚本既支持“连续值事件体素输入”，也支持“阈值脉冲输入”。

### 10. forward、loss 和 backward：行 304 到 343

`304` `pred_list = model(chunk.to(device))`

- 执行模型前向。
- 根据模型 `MS_SpikingformerFlowNet_en4.forward()`，返回的是一个字典：

```python
{"flow": flow_list, "attn": attns}
```

- 其中 `flow_list` 是多尺度 flow 预测列表。

`305` `pred = pred_list["flow"]`

- 这里的 `pred` 不是单个 flow，而是一整个 flow list。
- 这点很重要，因为名字 `pred` 很容易让人误以为它是最终一张预测图。

`307`

- 注释写着 “backward pass only the last flow pred”。
- 但代码实际不是这样。
- 训练时传给 `loss_function` 的是整个 `pred` 列表，不是最后一级。
- 也就是说，这条注释已经过时或不准确。

`308-313`

- 根据 `config["metrics"]["mask_events"]` 决定 loss 是否再额外乘一个 event mask。
- `event_mask` 的构造方式是沿时间维和 polarity 维求和，再转 `bool()`。
- 含义是：只有有事件的像素才参与损失。
- 默认配置里 `metrics.mask_events: False`，所以一般直接用原始 `mask`。
- 损失最后还会除以 `num_acc_steps`，支持梯度累积。

`316-317`

- 如果 loss 是 NaN，直接抛异常中断训练。

`319-322`

- 如果启用了 AMP，就走：

```python
scaler.scale(curr_loss).backward()
```

- 否则普通 `curr_loss.backward()`。

`328-329`

- 如果配置了 `clip_grad`，就做梯度裁剪。
- 默认是 `100.0`，所以默认会裁剪。

`330-331`

- 如果要存梯度，就记录当前 batch 的梯度。

`332-341`

- 这是优化器 step 的触发点。
- 满足梯度累积步数，或者已经到 dataloader 最后一个 batch，就执行参数更新。
- AMP 路径下需要 `scaler.step()` 和 `scaler.update()`。
- 之后统一 `optimizer.zero_grad()`。

`343`

- 把当前 batch 的 loss 累加到 `train_loss`。
- 它乘了 `batch_size`，显然作者想做“按样本数累计”的 epoch loss。

### 11. 可视化和 epoch 结束统计：行 345 到 394

`346-355`

- 只有在 `vis.enabled=True` 且 `batch_size == 1` 时才做可视化。
- 可视化用的是最后一级 flow：`pred_list["flow"][-1]`。
- 如果 `vis.mask_events=True`，还会额外用 event mask 把无事件位置清掉。
- 然后调用 `vis.update(chunk_vis, label, mask, flow_vis, None)`。

`358` `sample += 1`

- 本 epoch 的 batch 计数加一。

`362-366`

- 如果记录了梯度，就在 epoch 末写入 `grads_w.csv`，然后清空缓存列表。

`368-379`

- 计算并打印 epoch 级别统计：
- `epoch_loss`
- `epoch_time_sec`
- `train_step_time_sec`
- `train_samples_per_sec`
- `max_gpu_mem_gib`
- 这几项里，吞吐和显存统计都很实用，说明这份脚本不只关心精度，也关心训练效率。

`381-387`

- 如果开启 MLflow，就把这些训练指标全部记录进去。

`389-394`

- 如果当前 epoch loss 优于 `best_loss`，就保存模型和训练状态。
- `save_model(model)` 最终走的是 `mlflow.pytorch.log_model(model, "model")`。
- `save_state_dict(...)` 会把优化器、scheduler、AMP scaler、epoch 一起存下来，供 resume 使用。

这里要记住：这份脚本按“训练损失最小”保存 best model，不是按 validation loss。

### 12. 验证循环：行 397 到 518

`400`

- 验证触发条件是：

```python
if epoch % config["test"]["n_valid"] == 0:
```

- 默认 `n_valid: 5`，所以 epoch `0, 5, 10, ...` 都会验证。
- 注意，epoch 0 也会验证，这一点很多人第一次看会忽略。

`401-408`

- 初始化验证计时、计数和累计 loss。
- 有个值得注意的点：
- 如果 `batch_size > 1`，模型切到 `eval()`。
- 否则反而保持 `train()`。
- 默认 batch size 是 1，所以验证默认其实跑在 `train()` 模式。
- 这通常意味着作者想保留某些层在 batch size 1 下的训练态行为，或者是为了兼容某些 SNN/BN 实现。

`411-420`

- 整个验证在 `torch.set_grad_enabled(False)` 下执行。
- 每个 batch 一样先 reset SNN 状态、设置 step mode、搬运数据、给 mask 增维。

`422-477`

- 验证 forward 前的编码、归一化、spike threshold 逻辑，和训练阶段几乎完全一致。
- 这说明作者希望训练/验证只有“是否做数据增强”和“是否反向传播”不同，输入预处理逻辑尽量一致。
- 一处明显区别是：

```python
pred = pred_list["flow"][-1]
```

- 验证阶段只取最后一级 flow 作为评估对象。

`479-488`

- 如果开可视化且 batch size 为 1，就显示验证预测。
- 同样用最后一级 flow。

`490-496`

- 计算验证损失。
- 如果 `mask_events=True`，就用 `mask * event_mask`。
- 这里传给 `loss_function` 的是 `[pred]`，即只有一个元素的列表。
- 这与训练阶段“整组多尺度 flow 都进 loss”不同。

结论：训练优化多尺度输出，验证只看最后一级输出。这是读这份文件时最重要的行为差异之一。

`498-501`

- 累计验证损失，并限制验证样本数。
- 当 `sample > config['test']['sample'] // batch_size` 时提前 break。
- 默认 `test.sample: 40`，所以不会把整个验证集都跑完，而是只跑一个子集。

`503-514`

- 计算验证平均损失、验证耗时和每 step 时间。
- 打印并写入 MLflow。

`516-518`

- 如果有 scheduler，就在 epoch 结束时 `scheduler.step()`。
- 注意这一步在验证之后，而不是验证之前。

### 13. 训练结束和命令行入口：行 522 到 566

`522-523`

- 如果启用了 MLflow，训练全部结束后显式 `mlflow.end_run()`。

`527-560`

- 定义命令行参数：
- `--config`：默认训练配置文件。
- `--path_mlflow`：MLflow tracking URI。
- `--prev_runid`：预训练或恢复训练依赖的历史 run id。
- `--save_path`：表面上是模型保存路径。
- `--finetune`：是否做 finetune。
- `--resume`：是否恢复训练状态。

其中有两个特别值得注意：

- `save_path` 在这份脚本里实际上没有被使用。
- `finetune` 和 `resume` 都不是布尔型参数，而是“只要传了就当真值”的字符串式开关。

`562` `args = parser.parse_args()`

- 解析命令行。

`566` `train(args, YAMLParser(args.config))`

- 真正启动训练。
- 执行顺序很清楚：
- 先用配置文件构造 `YAMLParser`
- 再把 `args` 和 `config_parser` 一起传给 `train()`

---

## 这份入口脚本的完整执行链

从整体看，这个文件可以压缩成下面这条主线：

1. 解析 YAML，决定设备和训练参数。
2. 实例化 `MS_SpikingformerFlowNet_en4` 这类 SNN 光流模型。
3. 设置 surrogate gradient、step mode、神经元类型和 backend。
4. 创建 AdamW、scheduler、AMP scaler、监督光流损失。
5. 从 `DSECDatasetLite` 读取训练/验证样本。
6. 每个 batch 重置 SNN 状态，整理输入编码，归一化后前向。
7. 训练时用多尺度 flow 列表算 loss 并反传。
8. 验证时只取最后一级 flow 算 loss。
9. 把 train/valid 指标和 best model 交给 MLflow 管理。

---

## 这一轮最该记住的 8 个细节

1. 默认模型不是手写死的，而是由 `config["model"]["name"]` 动态决定，默认值是 `MS_SpikingformerFlowNet_en4`。
2. `spiking_neuron` 会通过 `combine_entries()` 并入 `config["model"]["spiking_neuron"]`。
3. 每个 batch 前都会 `functional.reset_net(model)`，这是 SNN 训练里非常核心的动作。
4. 默认输入编码是 `voxel`，并且会把正负极性拆到单独维度。
5. 默认训练不会做输入脉冲阈值化，因为 `spike_th` 是 `Null`。
6. 训练时 loss 吃的是整组多尺度 flow；验证时只吃最后一级 flow。
7. best model 是按训练损失保存，不是按验证损失保存。
8. 这份脚本对 MLflow 耦合很深，`prev_runid` 既服务于加载预训练，也服务于 resume。

---

## 下一步最合适的学习路径

如果继续顺着入口往下学，我建议下一轮看下面两个文件：

- `third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py`
- `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py`

原因很直接：

- 这份训练入口的“数据从哪里来”，核心在 `DSECDatasetLite`。
- 这份训练入口的“flow 列表怎么生成”，核心在 `MS_SpikingformerFlowNet_en4.forward()`。

---

## 补充基础概念：什么是 YAML，什么是 MLflow

这一节是给只有初步深度学习基础时看的。读训练入口前，先把这两个词看懂，后面会轻松很多。

### 1. YAML 是什么

你可以把 YAML 理解成一种“专门用来写配置的文本格式”。

它不是模型，也不是框架，而是一个“把参数整齐写下来”的办法。深度学习项目里常用它来保存：

- 学习率
- batch size
- 数据路径
- 模型名字
- 训练轮数
- 是否开启 AMP
- 各种超参数

例如当前项目默认配置文件：

- `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`

里面有这样的内容：

```yaml
model:
    name: MS_SpikingformerFlowNet_en4

optimizer:
    name: AdamW
    lr: 0.0001

loader:
    batch_size: 1
    n_epochs: 60
```

这段话的意思非常直接：

- 模型名叫 `MS_SpikingformerFlowNet_en4`
- 优化器是 `AdamW`
- 学习率是 `0.0001`
- batch size 是 `1`
- 训练 `60` 个 epoch

所以 YAML 的本质作用就是：

“把训练参数从 Python 代码里拿出来，单独放到一个好改、好读、好管理的配置文件里。”

这样做的好处是：

1. 改实验参数时，不用频繁改 Python 源码。
2. 同一个训练脚本，可以配很多不同的 YAML，跑很多实验。
3. 更容易记录“这次实验到底用了什么参数”。

你可以把它类比成：

- Python 脚本 = 厨师
- YAML = 菜谱

厨师负责做菜，菜谱负责告诉厨师“用什么材料、多少火候、做多久”。

当前这个入口脚本里，YAML 是通过这里读进来的：

- [train_flow_parallel_supervised_SNN.py:566](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:566)
- [parser.py:15](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py:15)

对应关系是：

- `args.config` 给出 YAML 文件路径
- `YAMLParser(args.config)` 负责把 YAML 文件读进来
- 读出来后变成 `config` 这个 Python 字典，后面代码一直在用它

例如：

- [train_flow_parallel_supervised_SNN.py:133](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:133)

这里写：

```python
if config["optimizer"]["name"] == 'AdamW':
```

意思就是：“去 YAML 里看看优化器名字是不是 `AdamW`。”

### 2. MLflow 是什么

你可以把 MLflow 理解成一个“实验记录本 + 模型仓库”。

训练模型时，通常会遇到这些问题：

- 这次实验用了什么学习率？
- 跑了多少个 epoch？
- 哪次实验的 loss 更低？
- 最好的模型权重存在哪里？
- 这次是不是接着上一次训练继续跑的？

如果全靠手动记，很快就会乱。

MLflow 的作用就是帮你系统地记录这些内容。

它通常会记录：

- 这次实验的参数 `params`
- 训练过程中的指标 `metrics`
- 训练产物，比如模型文件 `artifacts`

你可以把它类比成：

- TensorBoard 更偏“看曲线”
- MLflow 更偏“完整管理一次实验”

在这个项目里，MLflow 主要干三件事：

1. 记录配置参数
2. 记录每个 epoch 的 train loss / valid loss / 学习率 / 吞吐
3. 保存模型和恢复训练状态

对应代码位置：

- 设置实验和启动 run：
  - [train_flow_parallel_supervised_SNN.py:41](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:41)
  - [train_flow_parallel_supervised_SNN.py:43](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:43)
- 记录配置参数：
  - [train_flow_parallel_supervised_SNN.py:44](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:44)
- 记录训练指标：
  - [train_flow_parallel_supervised_SNN.py:382](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:382)
- 记录验证指标：
  - [train_flow_parallel_supervised_SNN.py:512](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:512)
- 保存模型：
  - [utils.py:125](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:125)
- 恢复训练状态：
  - [utils.py:75](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:75)

### 3. 用“训练一次模型”的视角理解 MLflow

假设你训练了 3 次模型：

- 第 1 次：学习率 `1e-4`
- 第 2 次：学习率 `5e-5`
- 第 3 次：学习率 `1e-5`

如果不用 MLflow，你可能只能靠：

- 文件夹名字
- 终端输出
- 自己手写笔记

去分辨这三次实验。

如果用了 MLflow，它会自动帮你记：

- 第几次 run
- 这次 run 的配置
- 每个 epoch 的 loss
- 最终保存的模型

以后你想问“第 2 次实验的最好模型在哪”，它就不是靠猜，而是可以顺着 run 直接找到。

### 4. 在这份脚本里，YAML 和 MLflow 分别扮演什么角色

一句话概括：

- YAML 决定“这次怎么训练”
- MLflow 记录“这次训练发生了什么”

也就是：

- YAML 是输入给训练脚本的实验配置
- MLflow 是训练脚本输出实验记录和模型产物的地方

你可以把整个流程想成：

```text
YAML 配置
  -> 训练脚本按配置训练
  -> MLflow 记录过程和结果
```

### 5. 你现在先不用死记的部分

对于当前阶段，你先不用掌握：

- YAML 的完整语法细节
- MLflow 的服务器部署方式
- artifact_uri、runid、experiment 的所有概念

你现在只要先记住：

1. YAML 就是训练配置文件。
2. `config[...]` 里的内容大多都来自 YAML。
3. MLflow 就是实验记录和模型保存系统。
4. `prev_runid` 的意思是“拿某次历史实验当起点继续用”。

### 6. 你现在再回头看入口脚本，应该这样理解

你看到这句：

```python
train(args, YAMLParser(args.config))
```

就可以翻译成一句人话：

“把 YAML 配置文件读进来，然后按这份配置启动训练。”

你看到这几句：

```python
mlflow.start_run()
mlflow.log_params(config)
mlflow.log_metric("train_loss", epoch_loss, step=epoch)
```

就可以翻译成一句人话：

“开始记录这次实验，把配置记下来，再把每个 epoch 的 loss 记下来。”

---

## 补充基础概念：runid、artifact、checkpoint、resume

这几个词经常一起出现，因为它们都和“训练一次实验之后，怎么把结果保存下来、怎么接着之前的结果继续跑”有关。

### 1. runid 是什么

`runid` 可以理解成：

“某一次实验的唯一编号。”

只要你启动一次新的 MLflow 实验记录，MLflow 就会给这次实验分配一个独一无二的 id，这个 id 就是 `runid`。

你可以把它类比成：

- 快递单号
- 订单编号
- 学号

它的作用不是表达含义，而是“唯一定位某一次实验”。

在这个项目里，`runid` 最重要的用途有两个：

1. 找到某次实验保存下来的模型
2. 找到某次实验保存下来的训练状态，继续训练

你在入口脚本里看到的：

- [train_flow_parallel_supervised_SNN.py:45](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:45)
- [train_flow_parallel_supervised_SNN.py:149](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:149)

本质上都是在说：

“如果用户给了一个历史 `runid`，那我就去找那次实验留下的东西。”

### 2. artifact 是什么

`artifact` 可以理解成：

“一次实验跑出来的文件产物。”

只要是训练过程中产生、并且值得保存的文件，都可以叫 artifact，比如：

- 模型权重文件
- checkpoint 文件
- 日志文件
- 图片
- CSV 统计结果

在这个项目里，典型 artifact 包括：

- `model.pth`
- `training_state_dict/state_dict.pth`
- `grads_w.csv`

对应代码可以看：

- [utils.py:125](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:125)
- [utils.py:129](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:129)
- [utils.py:139](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:139)

所以 artifact 不是某一种特殊格式，它只是一个总称，意思是“实验产物文件”。

### 3. checkpoint 是什么

`checkpoint` 可以理解成：

“训练过程中的一个存档点。”

你可以把它想成打游戏时的“存档”。

为什么训练要存档？

因为训练往往很长，可能会遇到：

- 训练中断
- 机器重启
- 显存炸掉
- 想从第 20 个 epoch 接着跑，而不是重头再来

这时候如果提前存了 checkpoint，就能从存档接着继续。

在深度学习里，一个完整 checkpoint 往往不只保存模型参数，还会保存：

- 模型权重
- 优化器状态
- 学习率调度器状态
- AMP scaler 状态
- 当前 epoch

这个项目里保存训练状态的代码在：

- [utils.py:129](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:129)

里面保存了：

```python
state_dict = {
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "epoch": epoch,
    "scaler": scaler.state_dict() if scaler else None,
}
```

这就是一个典型 checkpoint。

注意，它和“纯模型权重”不是一回事。

### 4. 纯模型权重 和 checkpoint 的区别

这两个新手最容易混。

“纯模型权重”通常只关心：

- 神经网络参数本身

它主要用来：

- 做推理
- 做测试
- 做 finetune 初始化

而 `checkpoint` 关心的是“整个训练现场”：

- 模型现在学到了什么
- 优化器历史动量是多少
- 学习率调度走到哪了
- AMP 缩放器状态是什么
- 当前是第几个 epoch

所以：

- 纯模型权重像“只保存角色属性”
- checkpoint 像“保存整个游戏进度”

在当前项目里：

- `save_model(model)` 更接近保存模型本身
- `save_state_dict(...)` 更接近保存 checkpoint

### 5. resume 是什么

`resume` 的意思就是：

“从之前保存的 checkpoint 接着训练，而不是从头开始。”

例如你本来计划训练 60 个 epoch，结果训练到第 18 个 epoch 机器断了。

如果你有 checkpoint，就可以：

- 重新启动脚本
- 告诉它上一轮实验的 `runid`
- 让它把优化器、scheduler、epoch 这些状态都读回来

这就叫 `resume training`。

在当前项目里，对应入口是：

- [train_flow_parallel_supervised_SNN.py:149](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:149)
- [utils.py:75](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:75)

核心逻辑是：

```python
optimizer, scheduler, scaler, epoch_initial = resume_model(...)
```

也就是把之前那次训练的状态读回来，然后从下一个 epoch 继续跑。

### 6. finetune 和 resume 的区别

这两个也非常容易混。

`finetune`：

- 重点是“拿已有模型权重，当作新任务或新设置的起点”
- 通常只关心模型参数
- 不一定关心之前训练到第几个 epoch
- 常常会重新设学习率、重新定义训练计划

`resume`：

- 重点是“继续同一次训练”
- 不只是加载模型参数，还要恢复优化器、scheduler、epoch 等完整状态
- 训练轨迹尽量和中断前接上

简单记：

- `finetune` = 借别人的基础重新训练
- `resume` = 把自己上次没练完的继续练完

### 7. 在这份脚本里，这几个概念怎么串起来

这份脚本的逻辑可以翻译成下面的人话：

1. 启动一次训练，MLflow 会给它一个 `runid`
2. 训练过程中，loss、lr、时间这些都会被记录
3. 当效果更好时，会把模型和训练状态作为 `artifact` 保存起来
4. 以后如果你给脚本一个旧 `runid`
5. 它就能找到那次实验留下来的模型或 checkpoint
6. 然后选择：
   - 只加载模型，做 finetune
   - 或恢复完整状态，做 resume

### 8. 你当前阶段最值得记住的版本

先记住下面这 4 句就够用了：

1. `runid` 是某一次实验的唯一编号。
2. `artifact` 是某次实验产生的文件。
3. `checkpoint` 是训练过程中的存档，通常不只包含模型，还包含优化器和 epoch。
4. `resume` 是从 checkpoint 接着训练，不是从头训练。

### 9. 回到你现在正在读的代码

你看到这句：

```python
model = load_model(args.prev_runid, model, device, remap)
```

可以先翻译成人话：

“如果给了旧实验的 `runid`，就去把那次实验保存的模型权重拿回来。”

你看到这句：

```python
optimizer, scheduler, scaler, epoch_initial = resume_model(...)
```

可以翻译成人话：

“如果这是接着上次训练继续跑，那就把上次训练的完整进度一起恢复回来。”

---

## 补充基础概念：epoch、iteration、batch、optimizer、scheduler、gradient、backward

这几个词就是训练主循环的骨架。你只要把它们顺清，`train_flow_parallel_supervised_SNN.py` 里从 `for epoch in ...` 往下的代码就会顺很多。

### 1. batch 是什么

`batch` 就是一小批样本。

训练时，通常不会一次把整个数据集都丢进模型，因为：

- 数据太多，显存放不下
- 一次只看一部分样本，训练更高效

所以做法是：

- 先把整个数据集切成很多小批
- 每次拿一批样本送进模型

这一小批样本就叫 `batch`。

例如当前配置里：

- [train_DSEC_supervised_SDformerFlow_en4.yml:80](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:80)

写的是：

```yaml
batch_size: 1
```

意思就是每次只拿 1 个样本训练。

### 2. batch size 是什么

`batch size` 就是“一个 batch 里有多少个样本”。

例如：

- `batch_size = 1`：每次喂 1 个样本
- `batch_size = 8`：每次喂 8 个样本
- `batch_size = 32`：每次喂 32 个样本

batch size 越大：

- 一次看到的样本越多
- 显存压力通常越大

在当前脚本里，dataloader 就是按这个参数组织 batch 的：

- [train_flow_parallel_supervised_SNN.py:192](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:192)
- [train_flow_parallel_supervised_SNN.py:212](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:212)

### 3. iteration 是什么

`iteration` 可以理解成：

“训练循环里处理一个 batch 的一次过程。”

也就是：

1. 取一个 batch
2. 做 forward
3. 算 loss
4. backward
5. 更新参数

这完整一轮，通常就叫一次 iteration。

在这份脚本里，对应的就是这个循环体：

- [train_flow_parallel_supervised_SNN.py:240](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:240)

```python
for chunk, mask, label in tqdm(train_dataloader):
```

这个 `for` 每转一圈，基本就可以看作一次 iteration。

### 4. epoch 是什么

`epoch` 可以理解成：

“把整个训练集完整看一遍。”

如果数据集有 1000 个样本，`batch_size = 10`，那一个 epoch 里通常要跑：

- 100 次 iteration

因为每次拿 10 个样本，100 次刚好把 1000 个样本看完一轮。

在这份脚本里，对应的是：

- [train_flow_parallel_supervised_SNN.py:231](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:231)

```python
for epoch in range(epoch_initial, config["loader"]["n_epochs"]):
```

也就是说，外层循环控制 epoch，内层 dataloader 循环控制 iteration。

简单记：

- 外层一圈：`epoch`
- 内层一圈：`iteration`

### 5. forward 是什么

`forward` 就是“把输入送进模型，得到输出预测”。

例如这句：

- [train_flow_parallel_supervised_SNN.py:304](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:304)

```python
pred_list = model(chunk.to(device))
```

这就是 forward。

人话翻译：

“把这一批事件数据送进网络，让网络预测光流。”

### 6. loss 是什么

`loss` 就是“模型这次预测得有多差”的一个数字。

- loss 越大，说明预测越差
- loss 越小，说明预测越接近目标

在当前脚本里：

- [train_flow_parallel_supervised_SNN.py:311](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:311)
- [train_flow_parallel_supervised_SNN.py:313](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:313)

这里就是在算当前 batch 的 loss。

### 7. gradient 是什么

`gradient` 就是“参数应该往哪个方向改，改多少”的信号。

你可以先把它理解成：

“loss 对每个参数的责备程度。”

如果某个参数让 loss 变大很多，那它的 gradient 通常就更明显；优化器会据此决定怎么调这个参数。

不用一开始去记数学表达式，你当前阶段只要知道：

- gradient 是为了更新参数
- 没有 gradient，模型就不知道该怎么学

### 8. backward 是什么

`backward` 就是“把 loss 反向传播回去，算出每个参数的 gradient”。

在这份脚本里：

- [train_flow_parallel_supervised_SNN.py:320](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:320)
- [train_flow_parallel_supervised_SNN.py:322](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:322)

对应：

```python
scaler.scale(curr_loss).backward()
```

或者

```python
curr_loss.backward()
```

人话翻译：

“根据这次 loss，回头去算模型里每个参数该怎么改。”

### 9. optimizer 是什么

`optimizer` 就是“真正负责改参数的人”。

如果说：

- `loss` 告诉你错了多少
- `gradient` 告诉你该往哪边改

那 `optimizer` 就是执行修改动作的工具。

当前项目默认优化器是：

- [train_DSEC_supervised_SDformerFlow_en4.yml:68](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:68)

```yaml
name: AdamW
```

在代码里创建优化器的位置是：

- [train_flow_parallel_supervised_SNN.py:133](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:133)

而真正执行“更新参数”的动作在这里：

- [train_flow_parallel_supervised_SNN.py:335](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:335)
- [train_flow_parallel_supervised_SNN.py:338](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:338)

也就是：

```python
optimizer.step()
```

或者 AMP 下的：

```python
scaler.step(optimizer)
```

### 10. zero_grad 是什么

这是和 optimizer 总是一起出现的一个动作。

`optimizer.zero_grad()` 的作用是：

“把上一次 batch 留下的梯度清空。”

因为在 PyTorch 里，gradient 默认是累加的，不清空的话，新一轮 backward 会叠到旧梯度上。

当前脚本里有两处关键位置：

- [train_flow_parallel_supervised_SNN.py:143](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:143)
- [train_flow_parallel_supervised_SNN.py:341](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:341)

你可以把训练一步粗暴理解成：

```text
zero_grad
-> forward
-> loss
-> backward
-> optimizer.step
-> zero_grad
```

### 11. scheduler 是什么

`scheduler` 就是“学习率调度器”。

它不直接改模型参数，它改的是：

- 学习率 `lr`

为什么要调学习率？

因为训练时常常不是从头到尾都用同一个学习率最合适。

常见思路是：

- 前期学快一点
- 后期学慢一点，更稳定

当前配置里用的是：

- [train_DSEC_supervised_SDformerFlow_en4.yml:71](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:71)

```yaml
scheduler: "multistep"
```

代码里对应：

- [train_flow_parallel_supervised_SNN.py:139](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:139)

而每个 epoch 结束后更新 scheduler 的位置在：

- [train_flow_parallel_supervised_SNN.py:517](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:517)

```python
scheduler.step()
```

### 12. 把这些词串成一整步训练流程

你可以把一次训练 iteration 理解成下面这条链：

1. dataloader 拿出一个 `batch`
2. 模型做 `forward`
3. 算出 `loss`
4. 用 `backward` 算 `gradient`
5. `optimizer.step()` 根据 gradient 更新参数
6. `zero_grad()` 清理梯度

而：

- 很多次 iteration 组成一个 epoch
- 很多 epoch 组成整个训练过程

### 13. 对照你现在正在读的代码

在当前脚本里，对应关系基本是：

- `epoch`
  - [train_flow_parallel_supervised_SNN.py:231](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:231)
- `iteration`
  - [train_flow_parallel_supervised_SNN.py:240](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:240)
- `forward`
  - [train_flow_parallel_supervised_SNN.py:304](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:304)
- `loss`
  - [train_flow_parallel_supervised_SNN.py:311](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:311)
- `backward`
  - [train_flow_parallel_supervised_SNN.py:320](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:320)
- `optimizer.step`
  - [train_flow_parallel_supervised_SNN.py:335](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:335)
- `scheduler.step`
  - [train_flow_parallel_supervised_SNN.py:517](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:517)

### 14. 你当前阶段最应该先记住的版本

先只记住下面这 7 句：

1. `batch` 是一小批样本。
2. `iteration` 是处理一个 batch 的一次训练步骤。
3. `epoch` 是把整个训练集看完一轮。
4. `forward` 是模型从输入算出预测。
5. `loss` 是这次预测有多差。
6. `backward` 是根据 loss 算梯度。
7. `optimizer.step()` 是根据梯度真的去修改模型参数。

---

## 补充基础概念：dataset、dataloader、label、mask、tensor shape、to(device)、autocast

这一节补的是“数据怎么流进模型”的基础概念。把这些顺清后，你基本就能自己读懂当前训练脚本的大部分语句。

### 1. dataset 是什么

`dataset` 可以理解成：

“整个数据集的读取器。”

它的职责通常是：

- 知道数据放在哪
- 知道怎么按索引取第 `i` 个样本
- 知道每个样本包含什么内容

在当前项目里，dataset 是：

- [DSEC_dataset_lite.py](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py)

而在当前训练入口里，训练集和验证集都是这样创建的：

- [train_flow_parallel_supervised_SNN.py:182](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:182)
- [train_flow_parallel_supervised_SNN.py:202](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:202)

```python
train_dataset = DSECDatasetLite(config, file_list='train', stereo=False)
valid_dataset = DSECDatasetLite(config, file_list='valid', stereo=False)
```

人话翻译：

- `train_dataset` 知道怎么去训练 split 里取样本
- `valid_dataset` 知道怎么去验证 split 里取样本

### 2. dataloader 是什么

`dataloader` 可以理解成：

“把 dataset 包装成一个会自动吐出 batch 的迭代器。”

dataset 更像“一个个样本的仓库”。

dataloader 更像“一个自动发货系统”，它会帮你：

- 一次取多个样本
- 组装成 batch
- 打乱顺序
- 开多个 worker 并行读取

在当前脚本里：

- [train_flow_parallel_supervised_SNN.py:190](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:190)
- [train_flow_parallel_supervised_SNN.py:210](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:210)

就是在创建 dataloader。

所以这句话：

```python
for chunk, mask, label in tqdm(train_dataloader):
```

人话就是：

“不断从训练 dataloader 里取出一批数据，每批数据里有 `chunk`、`mask`、`label` 三部分。”

### 3. label 是什么

`label` 就是监督学习里的“标准答案”。

比如：

- 图像分类里，label 可能是“猫”或“狗”
- 目标检测里，label 可能是边界框
- 光流任务里，label 就是“真实光流”

在这个项目里：

- `label` 代表 GT optical flow，也就是真实运动场

对应代码：

- [train_flow_parallel_supervised_SNN.py:247](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:247)

注释写的是：

```python
label = label.to(device=device, dtype=torch.float32)  # [num_batches, 2, H, W]
```

其中：

- `2` 通常表示光流的两个分量：水平位移和垂直位移

### 4. mask 是什么

`mask` 可以理解成：

“告诉你哪些位置有效，哪些位置无效的掩码。”

很多视觉任务里，不是每个像素都能参与 loss。比如：

- 有些地方没有 ground truth
- 有些地方是无效区域
- 有些地方作者故意不想算损失

这时就会有一个 `mask`。

在这个脚本里：

- `mask` 会和 loss 一起使用
- 只有 `mask` 为真的位置，才真正参与损失

比如：

- [train_flow_parallel_supervised_SNN.py:311](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:311)
- [train_flow_parallel_supervised_SNN.py:313](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:313)

```python
curr_loss = loss_function(pred, label, mask * event_mask, ...)
curr_loss = loss_function(pred, label, mask, ...)
```

人话翻译：

“loss 不是全图都算，而是只在有效区域上算。”

### 5. chunk 是什么

在当前脚本里，`chunk` 是模型输入，也就是：

“一段事件数据整理后的张量表示。”

因为这是事件相机光流任务，输入不是普通 RGB 图片，而是事件流。

这些事件会先被整理成某种张量形式，比如体素表示 `voxel`，然后送进网络。

在当前脚本里，默认配置是：

- `model.encoding: voxel`

所以 `chunk` 默认代表“voxel 化后的事件输入”。

### 6. tensor 是什么

`tensor` 你可以先理解成：

“深度学习里通用的多维数组。”

它和 numpy 数组很像，但更适合给 PyTorch 用，而且可以放到 GPU 上。

例如：

- 标量：0 维 tensor
- 向量：1 维 tensor
- 矩阵：2 维 tensor
- 图像/视频/事件体素：更高维 tensor

当前脚本里的 `chunk`、`label`、`mask`，本质上全都是 tensor。

### 7. tensor shape 是什么

`shape` 就是 tensor 每个维度的大小。

例如一个 shape 为：

```python
[B, 2, H, W]
```

通常表示：

- `B`：batch size
- `2`：两个通道
- `H`：高度
- `W`：宽度

当前脚本里最重要的几个 shape 要先记住：

`chunk`

- 读取后大致是 `[B, 20, 2, H, W]`
- [train_flow_parallel_supervised_SNN.py:246](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:246)

可以先粗略理解成：

- `B`：batch size
- `20`：时间相关的 bin 或帧片段
- `2`：两种极性
- `H, W`：空间尺寸

`label`

- 大致是 `[B, 2, H, W]`
- [train_flow_parallel_supervised_SNN.py:247](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:247)

表示每个像素的二维光流。

`mask`

- 原本通常是 `[B, H, W]`
- 然后通过 `unsqueeze` 变成 `[B, 1, H, W]`
- [train_flow_parallel_supervised_SNN.py:249](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:249)

### 8. unsqueeze 是什么

`unsqueeze` 的意思是：

“在某个位置插入一个长度为 1 的新维度。”

例如：

- 原来 `mask` 是 `[B, H, W]`
- `torch.unsqueeze(mask, dim=1)` 后变成 `[B, 1, H, W]`

为什么要这么做？

因为很多后续计算默认需要 channel 维，这样它才能和 `[B, 2, H, W]` 之类的张量对齐。

### 9. to(device) 是什么

`to(device)` 的意思是：

“把 tensor 或模型搬到某个设备上。”

这个设备通常是：

- CPU
- GPU

例如：

- [train_flow_parallel_supervised_SNN.py:246](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:246)
- [train_flow_parallel_supervised_SNN.py:247](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:247)
- [train_flow_parallel_supervised_SNN.py:75](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:75)

```python
model.to(device)
chunk = chunk.to(device=device, dtype=torch.float32)
label = label.to(device=device, dtype=torch.float32)
```

人话翻译：

“模型在 GPU 上算的话，输入和标签也必须一起搬到 GPU 上。”

否则就会报设备不一致错误。

### 10. dtype=torch.float32 是什么

这表示把 tensor 转成 `float32` 类型。

深度学习里常用的数据类型包括：

- `float32`
- `float16`
- `bfloat16`
- `int64`
- `bool`

当前脚本里：

- `chunk` 和 `label` 需要参与数值计算，所以转成 `float32`
- `mask` 有时候会转成 `bool`

对应：

- [train_flow_parallel_supervised_SNN.py:246](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:246)
- [train_flow_parallel_supervised_SNN.py:419](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:419)

### 11. bool mask 是什么

`bool` 类型只有两种值：

- `True`
- `False`

mask 变成 bool 后，含义就很明确：

- `True`：这个位置有效
- `False`：这个位置无效

例如验证阶段：

- [train_flow_parallel_supervised_SNN.py:419](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:419)

```python
mask = mask.bool().to(device=device)
```

### 12. transform 是什么

`transform` 就是“对数据做变换”。

在当前脚本里，它主要做的是：

- 裁剪
- 翻转

并且它不是只变输入，还会一起变：

- `chunk`
- `label`
- `mask`

因为三者的空间位置必须保持对应。

对应代码：

- [train_flow_parallel_supervised_SNN.py:250](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:250)
- [train_flow_parallel_supervised_SNN.py:424](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:424)

### 13. autocast / AMP 是什么

这两个词是混合精度训练相关概念。

`AMP` 是 Automatic Mixed Precision，意思是：

“让部分计算自动用更低精度做，以节省显存、提高速度。”

当前脚本里如果 `use_amp=True`，就会创建 `GradScaler`，然后在 forward 时用：

- [train_flow_parallel_supervised_SNN.py:253](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:253)

```python
with torch.cuda.amp.autocast(enabled=scaler is not None):
```

你当前阶段不用深入混合精度细节，只要记住：

- 开 AMP 是为了更省显存、更快
- `autocast` 是 forward 这段自动切换精度
- `GradScaler` 是为了让低精度训练更稳定

### 14. clip_grad 是什么

`clip_grad` 是梯度裁剪。

它的目的通常是：

- 防止梯度太大
- 让训练更稳定

对应代码：

- [train_flow_parallel_supervised_SNN.py:328](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:328)

```python
torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
```

你可以先理解成：

“如果梯度太猛，就把它压一压。”

### 15. 现在把 `for chunk, mask, label in tqdm(train_dataloader)` 翻译成人话

这一小段：

```python
for chunk, mask, label in tqdm(train_dataloader):
    chunk = chunk.to(device=device, dtype=torch.float32)
    label = label.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device)
    mask = torch.unsqueeze(mask, dim=1)
```

你现在应该能把它翻译成：

“不断从训练数据里取出一批输入事件 `chunk`、有效区域 `mask` 和真实光流 `label`。然后把它们搬到训练设备上，把数值类型整理好，并把 `mask` 变成后续更容易参与计算的形状。”

### 16. 到这里，你已经能独立读懂哪些句子

如果上面的概念你都跟上了，那你现在已经应该能自己读懂这类语句：

- 创建 dataset / dataloader
- 从 dataloader 取 batch
- 把数据搬到 GPU
- 调整 tensor 类型和形状
- 做增强
- 做 forward
- 算 loss
- backward
- optimizer.step

这已经覆盖了当前训练脚本 70% 以上的基础阅读障碍。

### 17. 接下来还剩下的真正难点

对当前 `train_flow_parallel_supervised_SNN.py` 来说，剩下最难的不是训练框架词汇，而是两个任务相关概念：

1. 事件数据编码为什么会有 `cnt` 和 `voxel`
2. 模型输出为什么是 `pred_list["flow"]` 这种“多尺度 flow 列表”

这两个如果你愿意，我下一轮可以继续专门解释到“足够自己读懂这份文件”为止。

---

## 补充任务概念：事件编码、SNN 状态、多尺度 flow 输出

这一节是为了把当前文件里最“像论文代码”的部分讲成人话。

### 1. 为什么事件相机输入不是普通图片

普通相机输出的是一帧一帧图片。

事件相机输出的不是整张图，而是一连串事件，事件通常包含：

- 位置 `(x, y)`
- 时间 `t`
- 极性 `p`

也就是：

“某个像素在某个时刻变亮了还是变暗了。”

所以事件数据不能直接像 RGB 图片那样丢进网络，通常要先整理成张量表示。

### 2. cnt 和 voxel 是什么

当前脚本支持至少两种事件表示：

- `cnt`
- `voxel`

对应分支在：

- [train_flow_parallel_supervised_SNN.py:255](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:255)
- [train_flow_parallel_supervised_SNN.py:263](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:263)

#### `cnt`

`cnt` 可以粗略理解成：

“按某个时间窗口，把事件数目统计起来。”

更像是：

- 不太强调精细时间顺序
- 更像事件计数图

#### `voxel`

`voxel` 可以粗略理解成：

“把时间也切成很多格子，再把事件填进这些时间格子里。”

所以它比 `cnt` 更保留时间结构。

你可以先把它想成：

- `cnt` 更像压缩后的统计图
- `voxel` 更像带时间层次的事件小体块

而当前默认配置是：

- [train_DSEC_supervised_SDformerFlow_en4.yml:14](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:14)

```yaml
encoding: voxel
```

所以你读这份文件时，主路径优先看 `voxel` 分支。

### 3. polarity 是什么

`polarity` 表示事件的正负极性。

你可以先理解成：

- 正事件：亮度增加
- 负事件：亮度减少

所以脚本里这段：

- [train_flow_parallel_supervised_SNN.py:266](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:266)
- [train_flow_parallel_supervised_SNN.py:267](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:267)

```python
neg = torch.nn.functional.relu(-chunk)
pos = torch.nn.functional.relu(chunk)
```

就是在把输入里正极性和负极性拆开。

### 4. 为什么要把正负极性拆开

因为对事件相机来说：

- 变亮
- 变暗

通常是两种含义不同的信号。

把它们拆开后，网络能更明确地区分：

- 哪些地方在增强
- 哪些地方在减弱

所以这句：

```python
chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)
```

本质就是：

“把正事件和负事件在一个新的维度上拼起来，交给网络一起看。”

### 5. SNN 为什么要 reset_net

普通 CNN 一般没有“内部时间状态”这个麻烦。

但 SNN 往往有：

- 膜电位
- 脉冲发放状态
- 时间步累积状态

所以如果一个 batch 跑完后不清掉，下一批数据就会“继承上一批的脑子”，这肯定不对。

因此每个 batch 前都要：

- [train_flow_parallel_supervised_SNN.py:243](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:243)

```python
functional.reset_net(model)
```

你可以把它翻译成：

“每看一批新数据之前，先把 SNN 的内部状态清空。”

### 6. step_mode 是什么

SNN 常常不是“一次性吃完整输入”，而是“按时间步处理”。

`step_mode` 就是在规定：

- 这个网络按什么方式理解时间维

当前配置里：

- [train_DSEC_supervised_SDformerFlow_en4.yml:8](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:8)

```yaml
step_mode: 'm'
```

你当前阶段只要记住：

- 这里的 `'m'` 表示 multi-step
- 说明这个网络会把输入看成一段有时间结构的序列，而不是纯静态图像

### 7. pred_list["flow"] 为什么不是一张图，而是一个列表

这是当前文件最容易让人误解的地方。

看这两句：

- [train_flow_parallel_supervised_SNN.py:304](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:304)
- [train_flow_parallel_supervised_SNN.py:305](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:305)

```python
pred_list = model(chunk.to(device))
pred = pred_list["flow"]
```

虽然变量名叫 `pred_list`，但它其实是一个字典；字典里的 `"flow"` 对应的是一个列表。

这个列表里放的是：

- 不同尺度下的光流预测

为什么要多尺度？

因为很多光流网络不是只在最终分辨率预测一次，而是：

- 先在粗尺度预测
- 再逐步细化到高分辨率

这样通常更稳定，也更容易训练。

### 8. 训练为什么用整组 flow，验证为什么只看最后一级

训练时：

- [train_flow_parallel_supervised_SNN.py:311](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:311)
- [train_flow_parallel_supervised_SNN.py:313](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:313)

传给 loss 的是整个 `pred` 列表。

这说明训练时，作者希望：

- 多个尺度的输出都被监督

验证时：

- [train_flow_parallel_supervised_SNN.py:477](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:477)

```python
pred = pred_list["flow"][-1]
```

这里只取最后一级。

这说明验证时，作者只关心：

- 最终最细、最成熟的那个输出

简单记：

- 训练：多尺度一起学
- 验证：看最后成品

### 9. gamma 在这里是什么

`gamma` 在当前 loss 里控制的是：

“如果有多级预测，不同级别的损失怎么加权。”

对应实现可以看：

- [flow_supervised.py:56](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/loss/flow_supervised.py:56)

如果 `gamma` 不为空，就走 sequence loss，对较新的预测通常给更高权重。

但当前默认配置里：

- [train_DSEC_supervised_SDformerFlow_en4.yml:63](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:63)

```yaml
gamma: Null
```

所以默认不会走那套时间衰减式 sequence loss，而是走普通监督损失逻辑。

### 10. event_mask 是什么

`event_mask` 和前面的 `mask` 不完全一样。

前面的 `mask` 更像：

- 数据集告诉你的有效像素区域

这里的 `event_mask` 更像：

- 当前输入里真的发生过事件的区域

对应：

- [train_flow_parallel_supervised_SNN.py:310](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:310)

```python
event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
```

人话翻译：

“把时间和极性维都加起来，看看哪些像素位置至少有过事件；有就记作 True，没有就记作 False。”

然后：

```python
mask * event_mask
```

就表示：

“只有既在有效 GT 区域里、又真的有事件发生的位置，才算损失。”

### 11. num_acc_steps 是什么

这是梯度累积的步数。

如果：

- `num_acc_steps = 1`

那就每个 batch 更新一次参数。

如果：

- `num_acc_steps = 4`

那就相当于先连续处理 4 个 batch，把梯度累起来，再更新一次参数。

当前默认配置里：

- [train_DSEC_supervised_SDformerFlow_en4.yml:74](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml:74)

```yaml
num_acc: 1
```

所以默认不用梯度累积。

### 12. 现在把训练主循环关键段完整翻译成人话

下面这段代码：

```python
for chunk, mask, label in tqdm(train_dataloader):
    functional.reset_net(model)
    chunk = chunk.to(device=device, dtype=torch.float32)
    label = label.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device)
    mask = torch.unsqueeze(mask, dim=1)

    with torch.cuda.amp.autocast(enabled=scaler is not None):
        if config['model']['encoding'] == 'voxel':
            neg = torch.nn.functional.relu(-chunk)
            pos = torch.nn.functional.relu(chunk)
            chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)

        pred_list = model(chunk.to(device))
        pred = pred_list["flow"]
        curr_loss = loss_function(pred, label, mask, gamma=config["loss"]["gamma"]) / num_acc_steps

    curr_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

你现在应该能把它读成：

“不断从训练集取出一批事件数据、有效区域和真实光流。每次处理新 batch 之前，先把 SNN 的内部状态清空。然后把数据搬到训练设备上，整理形状。因为当前默认输入是 voxel 事件表示，所以把正负极性拆开并拼成网络需要的格式。接着把输入送进模型，模型会输出一个多尺度的光流预测列表。再拿这个预测列表和真实光流去算损失。然后反向传播，最后让优化器根据梯度更新模型参数。”

### 12A. 训练主循环逐段翻译：`231-343`

如果你想把这一段真正吃透，可以按下面的人话顺序再读一遍。

`231-239`

- 开始一个新的 epoch。
- 打印 epoch 编号，记录开始时间。
- 如果在 GPU 上训练，就把这轮的峰值显存统计清零。
- 把模型切到训练模式，并把本轮的 batch 计数器和损失累计器清零。

`240-252`

- 从训练 dataloader 里取出一批输入事件 `chunk`、有效区域 `mask` 和真实光流 `label`。
- 打开 anomaly detection，方便排查反向传播异常。
- 因为这是 SNN，每个 batch 前都要先 `reset_net(model)` 清空内部状态，再设置 `step_mode`。
- 然后把输入、标签和掩码搬到设备上，整理成后面计算需要的 dtype 和 shape。
- 如果配置了训练增强，就同时对输入、标签和掩码做变换。

`253`

- 如果开启了 AMP，就让后面的 forward 和 loss 计算自动使用混合精度；否则按普通精度执行。

`255-280`

- 这一段在把 dataloader 给出的输入张量整理成模型真正要吃的格式。
- 当前默认配置走的是 `voxel` 分支。
- 如果要区分正负极性，就先把正事件和负事件拆开，再拼成模型需要的 `B, C, P, H, W` 格式。

`282-296`

- 对输入里的非零位置做归一化。
- 默认配置是 `minmax`，也就是按最小值最大值把数值缩放到更统一的范围。
- 零值通常表示没有事件，所以不参与归一化。

`300-302`

- 如果配置了输入脉冲阈值，就把输入变成二值脉冲：高于阈值为 1，低于阈值为 0。
- 当前默认配置里 `spike_th: Null`，通常不会执行。

`304-305`

- 把整理好的事件输入送进模型，得到输出。
- 输出不是单张 flow 图，而是一个字典；其中 `"flow"` 对应的是一个多尺度光流预测列表。

`307-313`

- 开始算当前 batch 的损失。
- 如果配置要求只在有事件的位置算损失，就额外构造一个 `event_mask`；否则直接使用原始 `mask`。
- 然后把多尺度预测、真实光流和有效区域一起交给损失函数。
- 最后再除以梯度累积步数。
- 这里要记住：训练时 loss 吃的是整组多尺度 `flow`，不是只吃最后一级。

`319-322`

- 根据当前 batch 的损失开始反向传播。
- 如果开启了 AMP，就先做缩放再 backward；否则直接 backward。

`328-341`

- 如果配置了梯度裁剪，就先把过大的梯度压一压。
- 然后在满足梯度累积条件时，正式让优化器更新参数。
- 更新完之后把旧梯度清空，给下一个 batch 做准备。
- 默认 `num_acc_steps = 1`，所以通常每个 batch 都更新一次。

`343`

- 把当前 batch 的损失累加到整个 epoch 的总损失里，为后面统计这一轮 epoch 的平均损失做准备。

把 `231-343` 再压缩成一句话：

- 每个 epoch 内，脚本不断从 dataloader 取 batch；每个 batch 前先重置 SNN 状态；把输入、标签和掩码搬到设备上；按配置整理事件编码和输入范围；前向得到多尺度光流预测；结合 GT 和掩码计算损失；反向传播算梯度；必要时裁剪梯度；然后让优化器更新参数，并把本 batch 的损失累加进 epoch 统计。

### 13. 到这一步，你读当前文件还会卡在哪里

如果你已经理解到这里，那么 `train_flow_parallel_supervised_SNN.py` 本身剩下真正会卡住你的，通常只剩两类：

1. 某个具体函数内部实现没展开，比如 `DSECDatasetLite`、`flow_loss_supervised`、`MS_SpikingformerFlowNet_en4`
2. 某些 PyTorch API 名字还不熟，比如 `relu`、`cat`、`view`、`interpolate`

也就是说，你现在应该已经具备“读懂这个训练入口整体逻辑”的基础了。

---

## 基础补课 1：Python 里的函数、类、对象、模块、返回值

这一节的目标非常明确：

- 让你能看懂 `train(...)`
- 让你能看懂 `YAMLParser(...)`
- 让你能看懂 `model(...)`
- 让你知道这些写法为什么有时像“调用函数”，有时像“创建东西”

### 0. Python 代码最基本在做什么

你可以先把 Python 代码理解成：

- 一行一行执行的指令

例如：

```python
x = 1
y = 2
z = x + y
```

大致就是按顺序做三件事：

1. 把 `1` 交给 `x`
2. 把 `2` 交给 `y`
3. 计算 `x + y`，再把结果交给 `z`

所以你读代码时，第一层永远先问：

- 这一行到底在“做什么动作”

### 0A. 什么是语句

语句可以先理解成：

- “一条完整的 Python 指令”

例如：

```python
x = 1
print(x)
if x > 0:
    print("positive")
```

这些都可以看作语句。

你当前阶段可以粗略记成：

- Python 文件就是由一条条语句组成的

### 0B. 什么是赋值

赋值就是：

- “把一个值交给一个名字”

例如：

```python
train_loss = 0.
device = config_parser.device
model = load_model(args.prev_runid, model, device, remap)
```

这些都属于赋值。

你可以统一读成：

- 等号右边先算出一个结果
- 再把结果放进左边这个名字里

所以：

```python
config = config_parser.config
```

人话就是：

“先从 `config_parser` 里取出配置，再把它交给局部变量 `config`。”

### 0C. 什么是名字

像下面这些：

```python
config
model
device
epoch
train_loss
```

都可以先叫“名字”。

名字的作用就是：

- 让你后面可以反复引用某个值

所以你现在读代码时，不要把 `config`、`model` 当成抽象符号，先把它们看成“当前程序里给某个东西起的名字”。

### 0D. 什么是表达式

表达式可以先理解成：

- “能算出一个结果的东西”

例如：

```python
1
x + y
config_parser.config
YAMLParser(args.config)
count_parameters(model)
```

这些都可以看成表达式。

因为它们最终都能“得到一个值”。

这很重要，因为 Python 里很多地方并不是只能写变量名，而是能写任何表达式。

### 0E. 括号最基本有两种常见作用

你当前阶段先记最常见的两种：

1. 表示函数/类调用
2. 改变运算或表达式组合方式

例如：

```python
train(args, parser)
YAMLParser(args.config)
count_parameters(model)
```

这里括号是“调用”。

而：

```python
(x + y) * z
```

这里括号是先把 `x + y` 作为一组去算。

### 0F. Python 读代码的第一层习惯

你现在读一行代码时，先只做这三步：

1. 这是赋值，还是调用，还是判断，还是循环？
2. 右边在算什么结果？
3. 这个结果最后交给了谁？

例如：

```python
model = load_model(args.prev_runid, model, device, remap)
```

先别怕，先拆成：

1. 这是赋值语句
2. 右边是在调用 `load_model(...)`
3. 调用结果最后交给左边的 `model`

这样就不会一上来被整行压住。

### 0G. 字典是什么

字典可以先理解成：

- “按名字存东西的盒子”

最简单例子：

```python
person = {
    "name": "Alice",
    "age": 20
}
```

这里：

- `"name"` 和 `"age"` 是键
- `"Alice"` 和 `20` 是对应的值

取值时可以写：

```python
person["name"]
```

结果就是 `"Alice"`。

当前项目里最重要的字典就是：

- `config`

所以：

```python
config["optimizer"]["lr"]
```

你应该读成：

“从配置字典里，先取 `optimizer` 这一层，再取里面的 `lr`。”

### 0H. 列表是什么

列表可以先理解成：

- “按顺序排好的一串东西”

例如：

```python
arr = [10, 20, 30]
```

它和字典不一样，不是按名字取，而是按位置取。

当前项目里常见的列表有：

- `config["loader"]["crop"]`
- `config["optimizer"]["milestones"]`
- `pred_list["flow"]`

### 0I. 索引是什么

索引就是：

- “按位置取列表里的元素”

例如：

```python
arr[0]
```

表示第一个元素。

```python
arr[1]
```

表示第二个元素。

```python
arr[-1]
```

表示最后一个元素。

当前项目里很重要的两句是：

```python
config["swin_transformer"]["use_arc"][0]
pred_list["flow"][-1]
```

它们分别表示：

- 取 `use_arc` 列表里的第一个元素
- 取 `flow` 列表里的最后一个预测结果

所以你现在读：

```python
pred = pred_list["flow"][-1]
```

应该直接理解成：

“只拿最后一级 flow 预测。”

### 0J. `if` 是什么

`if` 就是条件判断。

意思是：

- 如果条件成立，就执行下面缩进的代码

例如：

```python
if x > 0:
    print("positive")
```

当前项目里最常见的 `if` 有两类：

1. 检查开关是否打开
2. 检查某个值是不是空

例如：

```python
if config["vis"]["enabled"]:
if scaler is not None:
if config['data']['spike_th'] is not None:
```

你现在可以统一先读成：

“如果这个条件满足，就走这一条分支。”

### 0K. `elif` 和 `else` 是什么

它们是 `if` 的延伸。

例如：

```python
if a == 1:
    ...
elif a == 2:
    ...
else:
    ...
```

意思是：

- 如果第一个条件成立，走第一条
- 否则再检查第二个条件
- 如果前面都不成立，就走最后一条

当前项目里很典型的一段是：

```python
if config["model"]["spiking_neuron"]["neuron_type"] == "if":
    ...
elif ... == "lif":
    ...
elif ... == "psn":
    ...
else:
    raise ...
```

人话就是：

“根据配置里写的神经元类型，选择不同的神经元类；如果都不匹配，就报错。”

### 0L. `for` 是什么

`for` 就是循环。

你可以先理解成：

- “把一组东西一个一个拿出来处理”

例如：

```python
for x in [1, 2, 3]:
    print(x)
```

会依次处理 `1`、`2`、`3`。

当前项目里最重要的三个 `for` 是：

```python
for epoch in range(...):
for chunk, mask, label in train_dataloader:
for m in model.modules():
```

它们分别表示：

- 一个个 epoch 地训练
- 一个个 batch 地取数据
- 一个个子模块地遍历模型

### 0M. `range(...)` 是什么

`range(a, b)` 可以先理解成：

- 从 `a` 开始
- 到 `b-1` 结束

所以：

```python
for epoch in range(epoch_initial, config["loader"]["n_epochs"]):
```

你就读成：

“从起始 epoch 开始，一直循环到总 epoch 数之前。”

### 0N. 真假和值是否为空

你现在读训练脚本时，会反复看到两种判断：

```python
if config["vis"]["enabled"]:
if scaler is not None:
```

这两种虽然长得不一样，但你当前阶段都可以统一理解成：

- “检查这个东西是不是处于可用/开启状态”

区别只是：

- 第一种更像判断开关是不是开着
- 第二种更像判断对象是不是存在

### 0O. 这一层 Python 基础最短总结

在讲函数之前，你现在至少要先记住这 8 句：

1. 字典是按名字取值的容器。
2. 列表是按顺序存值的容器。
3. `[0]` 是第一个元素，`[-1]` 是最后一个元素。
4. `if` 表示条件判断。
5. `elif` 和 `else` 是同一组分支的后续选择。
6. `for` 表示循环处理一组东西。
7. `range(a, b)` 表示从 `a` 到 `b-1`。
8. 读条件语句时，先问自己：这里是在判断“开没开”，还是“有没有”。

### 1. 函数是什么

函数就是：

“一段被起了名字、可以反复调用的代码。”

最简单例子：

```python
def add(a, b):
    return a + b
```

这段的意思是：

- 定义一个叫 `add` 的函数
- 它接收两个输入 `a` 和 `b`
- 返回它们的和

调用时：

```python
x = add(2, 3)
```

结果 `x = 5`。

### 2. 当前项目里，函数长什么样

当前入口文件里最重要的函数就是：

- [train_flow_parallel_supervised_SNN.py:32](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:32)

```python
def train(args, config_parser):
```

意思是：

- 定义了一个叫 `train` 的函数
- 它需要两个输入：
  - `args`
  - `config_parser`

后面真正调用它的地方在：

- [train_flow_parallel_supervised_SNN.py:566](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:566)

```python
train(args, YAMLParser(args.config))
```

人话翻译：

“把命令行参数和配置解析器交给 `train` 这个函数，让它开始执行训练流程。”

### 3. 参数是什么

函数括号里的那些名字，叫参数。

例如：

```python
def train(args, config_parser):
```

这里：

- `args` 是一个参数
- `config_parser` 是另一个参数

参数的意思就是：

“这个函数执行时，需要外面传进来什么东西。”

### 4. 返回值是什么

返回值就是：

“函数执行完之后，交回来的结果。”

例如：

```python
def add(a, b):
    return a + b
```

这里 `return a + b` 就是在返回结果。

但不是所有函数都一定要返回值。

例如当前项目里的：

```python
def train(args, config_parser):
    ...
```

这个函数主要是执行训练流程，并没有显式 `return` 什么结果。

所以它更像：

- 一个“执行任务”的函数

而不是：

- 一个“算出数值再返回”的函数

### 4A. 函数体是什么

函数定义后面，缩进进去的那一整块代码，叫函数体。

例如：

```python
def add(a, b):
    c = a + b
    return c
```

这里函数体就是：

```python
    c = a + b
    return c
```

所以在 Python 里，缩进非常重要。

你可以先把它理解成：

- 缩进进去的内容，属于这个函数
- 没缩进的内容，不属于这个函数

### 4B. 定义函数 和 调用函数 是两回事

很多初学者会把这两件事混在一起。

定义函数：

```python
def train(args, config_parser):
    ...
```

这只是告诉 Python：

- “现在有一个叫 `train` 的函数”

但这时候它还没有执行。

调用函数：

```python
train(args, YAMLParser(args.config))
```

这才是真正让它跑起来。

所以：

- `def ...` = 定义
- `train(...)` = 调用

### 4C. 函数调用时会先算括号里的东西

看这句：

```python
train(args, YAMLParser(args.config))
```

它不是一下子就直接跑 `train`。

更准确的理解顺序是：

1. 先看第二个参数 `YAMLParser(args.config)`
2. 先创建一个 `YAMLParser` 对象
3. 然后再把这个对象传给 `train(...)`

所以函数调用时，括号里的表达式通常会先求值。

### 4D. 局部变量是什么

函数里面定义的变量，通常叫局部变量。

例如：

```python
def add(a, b):
    c = a + b
    return c
```

这里的 `c` 就是局部变量。

它主要在这个函数内部使用。

当前项目里：

```python
def train(args, config_parser):
    config = config_parser.config
    device = config_parser.device
```

这里的：

- `config`
- `device`

都可以先理解成 `train` 函数内部用的局部变量。

### 4E. `print(...)` 和 `return ...` 不是一回事

这两个非常容易混。

`print(...)` 的作用是：

- 把东西显示出来

`return ...` 的作用是：

- 把结果交回给调用者

例如：

```python
def f():
    print("hello")
```

这会打印 `hello`，但不等于把 `"hello"` 返回出去。

再例如：

```python
def g():
    return "hello"
```

这不会自动打印，但它真的返回了 `"hello"`。

在当前项目里：

```python
print('device:', device)
```

只是打印设备信息。

而：

```python
return model
```

才表示把 `model` 返回给外面。

### 4F. 一个函数可以返回多个值

Python 里一个函数可以像这样返回多个结果：

```python
return optimizer, scheduler, scaler, epoch_initial
```

当前项目里就有真实例子：

- [utils.py:115](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:115)

```python
return optimizer, scheduler, scaler, epoch_initial
```

接收时也可以一次接多个变量：

- [train_flow_parallel_supervised_SNN.py:150](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:150)

```python
optimizer, scheduler, scaler, epoch_initial = resume_model(...)
```

人话翻译：

“`resume_model(...)` 一次返回 4 个结果，左边 4 个变量同时把它们接住。”

### 4G. 函数可以调用别的函数

函数体里并不是只能写普通语句，也可以继续调用别的函数。

例如当前项目里：

```python
def train(args, config_parser):
    ...
    model = load_model(args.prev_runid, model, device, remap)
```

这说明：

- `train` 自己也是函数
- 它内部还调用了另一个函数 `load_model`

所以代码阅读时你要习惯：

- 一个函数内部会继续调用其他函数

### 4H. 参数传入时，不一定只是“变量名”

调用函数时，括号里不一定非得传一个现成变量，也可以传表达式。

例如：

```python
YAMLParser(args.config)
```

这里传进去的不是单独变量 `config`，而是表达式 `args.config`。

再例如：

```python
count_parameters(model)
```

这里传进去的是变量 `model`。

你现在只要记住：

- 函数参数位置里可以放“最终能算出一个值”的东西

### 4I. 为什么有的函数前面要写 `def`，有的不用

因为：

- `def xxx(...):` 是在定义函数
- `xxx(...)` 是在调用已经存在的函数

所以当你看到：

```python
def load_model(...):
```

是在定义。

而看到：

```python
load_model(args.prev_runid, model, device, remap)
```

是在调用。

### 4J. 这一段函数基础最该记住的版本

补完这一段后，函数部分你至少要先记住下面 8 句：

1. `def ...` 是定义函数，不是执行函数。
2. `xxx(...)` 才是在调用函数。
3. 函数体就是缩进进去的那一整块代码。
4. 函数里的变量通常先当成局部变量理解。
5. `print(...)` 是显示内容，不等于返回结果。
6. `return ...` 才是把结果交回给外面。
7. 一个函数可以返回多个值。
8. 一个函数内部还可以继续调用别的函数。

### 5. 模块是什么

模块你可以先理解成：

“一个 Python 文件。”

例如：

- `train_flow_parallel_supervised_SNN.py`
- `parser.py`
- `flow_supervised.py`

它们都可以看成模块。

当你看到：

```python
from configs.parser import YAMLParser
```

意思就是：

- 从 `configs/parser.py` 这个模块里
- 导入 `YAMLParser`

所以“模块”这个词不用想复杂，当前阶段你就把它理解成：

- 一个装代码的 Python 文件

### 6. import 到底在做什么

`import` 的本质是：

“把别的模块里定义好的名字拿过来用。”

例如：

```python
import torch
from tqdm import tqdm
from configs.parser import YAMLParser
```

分别表示：

- 把整个 `torch` 模块拿进来
- 从 `tqdm` 模块里拿 `tqdm`
- 从 `parser.py` 里拿 `YAMLParser`

所以你看到一个名字时，先问自己：

- 这是当前文件里定义的
- 还是从别的文件 import 进来的

### 7. 类是什么

类可以先理解成：

“制造对象的图纸。”

例如：

```python
class Dog:
    def __init__(self, name):
        self.name = name
```

这表示定义了一个 `Dog` 类。

### 8. 对象是什么

对象就是：

“按类这个图纸造出来的具体实例。”

例如：

```python
d = Dog("Lucky")
```

这里：

- `Dog` 是类
- `d` 是对象

### 9. 当前项目里，类和对象最典型的例子

最重要的一个是：

- [parser.py:6](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py:6)

```python
class YAMLParser:
```

这说明：

- `YAMLParser` 是一个类

当你看到：

- [train_flow_parallel_supervised_SNN.py:566](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:566)

```python
YAMLParser(args.config)
```

意思就是：

“根据 `YAMLParser` 这个类，创建一个解析配置的对象。”

### 10. `__init__` 是什么

类里经常会看到：

```python
def __init__(self, ...):
```

这个叫初始化函数。

它的作用是：

“对象一创建出来，立刻执行的初始化逻辑。”

例如在：

- [parser.py:9](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py:9)

```python
def __init__(self, config):
    self.reset_config()
    self.parse_config(config)
    self.get_device()
    self.init_seeds()
```

人话就是：

“当你创建一个 `YAMLParser` 对象时，它会立刻重置默认配置、读取 YAML、确定 device、设置随机种子。”

### 11. `self` 是什么

你现在只要先记住：

- `self` 表示“当前这个对象自己”

例如：

```python
self.parse_config(config)
```

意思是：

“让当前这个对象，调用它自己的 `parse_config` 方法。”

### 12. 方法是什么

方法其实就是：

“写在类里面的函数。”

例如在 `YAMLParser` 类里有：

- `parse_config`
- `get_device`
- `init_seeds`

它们本质上还是函数，只不过属于这个类，所以通常叫方法。

### 13. 属性是什么

属性就是：

“对象身上保存的数据。”

例如在 `YAMLParser` 里：

```python
self._config
self._device
```

这些都是它的属性。

后面代码里通过：

```python
config_parser.config
config_parser.device
```

来访问这些内容。

### 14. 为什么 `config_parser.config` 看起来不像函数

因为这里用到了 `@property`。

在：

- [parser.py:20](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py:20)

```python
@property
def config(self):
    return self._config
```

这表示：

- 你访问 `config_parser.config`
- 看起来像在读一个属性
- 其实底层会调用这个方法

所以：

```python
config = config_parser.config
```

人话就是：

“从这个配置解析器对象里，拿出解析好的配置字典。”

### 15. 为什么 `model(...)` 看起来像调用函数

你会看到：

```python
pred_list = model(chunk.to(device))
```

看起来像函数调用，但 `model` 又像一个对象。

这里你先记住：

- 在 PyTorch 里，模型对象通常可以像函数一样被调用

本质上它最终会走到模型类里的 `forward(...)`。

所以：

```python
pred_list = model(chunk)
```

你就读成：

“让模型对象对输入执行前向传播。”

### 16. 当前项目里，模型类和模型对象怎么区分

例如：

```python
model = MS_SpikingformerFlowNet_en4(config["model"].copy(), config["swin_transformer"].copy())
```

这里：

- `MS_SpikingformerFlowNet_en4` 是类
- `model` 是对象

后面：

```python
model.to(device)
model.init_weights()
pred_list = model(chunk)
```

都是在操作这个对象。

### 17. 一个你现在必须建立的阅读习惯

看到一行代码时，先分辨这 3 类东西：

1. 这是普通变量？
2. 这是函数？
3. 这是类创建出来的对象？

例如：

```python
train(args, YAMLParser(args.config))
```

你现在应该拆成：

- `train`：函数
- `YAMLParser`：类
- `YAMLParser(args.config)`：创建对象
- `args`：变量

### 18. 用当前项目代码做一次完整拆解

这句：

```python
train(args, YAMLParser(args.config))
```

现在应该这样读：

1. `args` 是命令行参数对象
2. `args.config` 是配置文件路径
3. `YAMLParser` 是配置解析器类
4. `YAMLParser(args.config)` 是创建一个配置解析器对象
5. `train(...)` 是训练函数
6. 训练函数接收命令行参数对象和配置解析器对象，然后开始训练

### 19. 再拆一条你后面会反复看到的代码

这句：

```python
config = config_parser.config
```

你现在应该读成：

1. `config_parser` 是一个对象
2. `.config` 是它暴露出来的属性
3. 这个属性里是解析好的配置字典
4. 把它取出来，存进局部变量 `config`

### 20. 这一节你必须记住的最短版本

先只记住下面 8 句：

1. 函数是一段可以反复调用的代码。
2. 参数是调用函数时传进去的输入。
3. 返回值是函数执行后交回来的结果。
4. 模块可以先理解成一个 Python 文件。
5. 类是图纸。
6. 对象是按类造出来的实例。
7. `self` 可以先读成“这个对象自己”。
8. 在 PyTorch 里，模型对象常常可以像函数一样调用。

### 21. `None` 是什么

`None` 可以先理解成：

- “没有值”
- “空”
- “这里目前什么都没有”

在当前项目里它非常常见，例如：

```python
scaler = None
scheduler = None
remap = None
transform_valid = None
```

意思就是：

- 这些变量先不指向具体对象
- 后面看条件再决定要不要真正创建

所以当你看到：

```python
if scaler is not None:
```

就读成：

“如果 scaler 真的存在，而不是空的，就走这条分支。”

### 22. `True` / `False` 和条件判断

Python 里很多代码都在做真假判断。

例如：

```python
if config["vis"]["enabled"]:
```

意思是：

“如果可视化开关为真，就执行下面的代码。”

再例如：

```python
if config['data']['spike_th'] is not None:
```

意思是：

“如果确实配置了阈值，而不是空值，就执行阈值化。”

你现在读 `if` 时，先不要急着看细节，先问自己一句：

- 这里在检查“有没有”
- 还是在检查“开没开”

### 23. `is not None` 为什么不用 `!= None`

你会在代码里看到很多：

```python
if scaler is not None:
if transform_train is not None:
if config["loss"]["clip_grad"] is not None:
```

你当前阶段可以先把它理解成一种更规范的写法，意思就是：

- “这个变量不是空的”

所以它本质上就是一种“存在性判断”。

### 24. 索引：`[0]`、`[-1]`

Python 里的列表、字符串、很多容器都可以用下标取元素。

例如：

```python
arr[0]
```

表示取第一个元素。

```python
arr[-1]
```

表示取最后一个元素。

在当前项目里：

```python
config["swin_transformer"]["use_arc"][0]
pred_list["flow"][-1]
```

分别表示：

- 取 `use_arc` 列表里的第一个元素
- 取多尺度 `flow` 列表里的最后一个预测

所以验证阶段这句：

```python
pred = pred_list["flow"][-1]
```

你应该直接读成：

“只取最后一级 flow 预测。”

### 25. `.copy()` 是什么

你在入口里会看到：

```python
config["model"].copy()
config["swin_transformer"].copy()
```

`.copy()` 可以先理解成：

- “复制一份”

这样做的常见原因是：

- 不想直接改原来的字典
- 想把一份副本传给别的函数或类

你当前阶段不用深究浅拷贝和深拷贝，只要知道：

- `.copy()` 往往意味着“我不想直接动原对象”

### 26. `type(...)` 和 `isinstance(...)`

你会在文件里看到：

```python
if type(config["loader"]["gpu"]) == str:
```

和：

```python
if isinstance(m, surrogate.ATan):
```

它们都和“这个东西是什么类型”有关。

你可以先这样区分：

- `type(x) == str`
  更像“x 的类型是不是字符串”
- `isinstance(x, SomeClass)`
  更像“x 是不是某个类或其子类的实例”

当前阶段你只要先记住：

- 这两种写法都是在做“类型判断”

### 27. `for m in model.modules()`

这一句：

```python
for m in model.modules():
```

意思是：

“把模型里的所有子模块一个个拿出来遍历。”

后面配合：

```python
if isinstance(m, surrogate.ATan):
    m.alpha = config['optimizer']['SG_alpha']
```

人话就是：

“把模型里所有 `ATan` surrogate 模块找出来，并把它们的 `alpha` 改成配置里的值。”

### 28. `range(...)` 是什么

你已经看到很多：

```python
for epoch in range(epoch_initial, config["loader"]["n_epochs"]):
```

`range(a, b)` 可以先理解成：

- 从 `a` 开始
- 一直到 `b-1`

所以这里的意思是：

- 从当前起始 epoch
- 一直循环到总 epoch 数之前

### 29. `with ...:` 是什么

`with` 可以先理解成：

- “在某个上下文里临时做事”

当前文件里你会看到：

```python
with torch.cuda.amp.autocast(...):
with torch.no_grad():
with torch.set_grad_enabled(False):
```

你现在先这样读：

- `with autocast(...)`：这段里临时启用混合精度
- `with torch.no_grad()`：这段里临时不记录梯度
- `with torch.set_grad_enabled(False)`：这段里关闭梯度计算

### 30. `raise` 是什么

你会看到：

```python
raise AttributeError
raise
```

`raise` 的意思是：

- “抛出错误，立刻中断当前流程”

所以：

```python
print("Config error: Event encoding not support.")
raise AttributeError
```

你应该读成：

“配置不对，停掉程序。”

### 31. `eval(...)` 是什么

这个项目里有一句很关键：

```python
model = eval(config["model"]["name"])(...)
```

`eval(...)` 当前阶段你先把它理解成：

- “把一个字符串，当成 Python 代码里的名字来解释”

例如如果：

```python
config["model"]["name"] = "MS_SpikingformerFlowNet_en4"
```

那么：

```python
eval(config["model"]["name"])
```

就会得到这个类本身，然后再加上 `(...)` 去创建对象。

所以整句人话就是：

“根据配置文件里写的模型名，动态创建对应的模型对象。”

### 32. 点号访问：`.` 到底在干什么

你会反复看到：

```python
model.to(device)
model.init_weights()
config_parser.config
run.info.artifact_uri
```

点号 `.` 的常见意思是：

- 访问对象的属性
- 调用对象的方法

例如：

- `config_parser.config`
  是读取属性
- `model.to(device)`
  是调用方法

### 33. 一次把几种 Python 写法串起来读

看这句：

```python
if config["loss"]["clip_grad"] is not None:
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
```

你现在应该这样拆：

1. `config["loss"]["clip_grad"]`
   从配置字典里取 `clip_grad`
2. `is not None`
   检查这个值是不是空
3. `torch.nn.utils...`
   调用 `torch` 模块里的某个函数
4. `model.parameters()`
   调用模型对象的方法，拿到参数

整句人话就是：

“如果配置里确实设置了梯度裁剪阈值，就对模型参数做梯度裁剪。”

### 34. 这一轮 Python 基础你现在最该记住的版本

补完这一轮后，Python 这一块你至少要先记住下面 10 句：

1. `None` 表示空值、当前没有对象。
2. `is not None` 表示“这个东西确实存在”。
3. `[0]` 是第一个元素，`[-1]` 是最后一个元素。
4. `.copy()` 可以先理解成复制一份。
5. `type(...)` 和 `isinstance(...)` 都是在做类型判断。
6. `range(a, b)` 表示从 `a` 循环到 `b-1`。
7. `with ...:` 表示在某个临时上下文里执行代码。
8. `raise` 表示抛出错误并中断。
9. `eval("名字")` 在这里是根据字符串动态找到对应类。
10. 点号 `.` 常常表示访问属性或调用方法。

---

## 基础补课总路线

从现在开始，先不要继续硬读入口文件，而是先补齐读这个项目真正需要的最小基础。

建议按下面顺序补：

1. Python 基础：函数、类、对象、模块、返回值
2. PyTorch 基础：`tensor`、`shape`、`nn.Module`、`forward`
3. 训练循环基础：`loss`、`backward`、`optimizer.step()`、`zero_grad()`
4. 张量操作基础：`view`、`unsqueeze`、`cat`、`sum`
5. 光流任务基础：什么是 flow，为什么标签是 2 通道
6. SNN 基础：spike、membrane potential、surrogate gradient
7. SpikingJelly 基础：`reset_net`、`set_step_mode`、`neuron`、`surrogate`

目标不是系统学完整门课，而是达到“能读懂 `train_flow_parallel_supervised_SNN.py`”这个程度。

为了避免后面继续出现“聊天里讲了，但你吸收不成体系”的情况，后续固定按这条路线学最稳：

1. Python 基础
   只学函数、类、对象、字典、列表、`if`、`for`、`import`
2. PyTorch 基础
   只学 `nn.Module`、`forward`、`tensor`、`shape`、`state_dict`
3. 训练循环基础
   只学 `forward -> loss -> backward -> optimizer.step`
4. 光流任务基础
   只学 `flow`、`label`、`mask`
5. SNN / SpikingJelly 基础
   只学 `spike`、`membrane`、`reset_net`、`surrogate`

这样你每学一块，都能立刻回到当前项目代码里验证，不会散掉。

---

## PyTorch 最小基础

### 1. `nn.Module`

在 PyTorch 里，几乎所有模型都继承自 `nn.Module`。

你可以先把它理解成：

- 一个“神经网络基类”
- 里面封装了参数管理、前向传播、训练模式切换等常用能力

所以当前项目里的模型类，例如：

```python
class MS_SpikingformerFlowNet_en4(...)
```

你可以先读成：

“这是一个 PyTorch 模型类。”

#### 1A. 为什么模型要继承 `nn.Module`

因为 `nn.Module` 已经帮模型准备好了很多通用能力，例如：

- 管理可学习参数
- 支持 `.to(device)`
- 支持 `.train()` / `.eval()`
- 支持 `state_dict()`
- 支持被优化器读取参数

所以你可以先把 `nn.Module` 理解成：

- “PyTorch 规定的模型标准外壳”

只要一个类继承它，这个类就更容易被 PyTorch 训练流程接起来。

#### 1B. 模型类 和 模型对象

在当前项目里，你会看到：

```python
model = MS_SpikingformerFlowNet_en4(...)
```

这里：

- `MS_SpikingformerFlowNet_en4` 是模型类
- `model` 是按这个类创建出来的模型对象

所以：

- 类是图纸
- 对象是实际被拿来训练的模型

#### 1C. `model.parameters()` 是什么

你会在当前训练脚本里看到：

```python
optimizer = AdamW(model.parameters(), lr=...)
```

`model.parameters()` 可以先理解成：

- “把这个模型里所有可学习参数拿出来”

优化器之所以能更新模型，就是因为它拿到了这些参数。

所以这句人话就是：

“让 AdamW 优化器接管这个模型的所有可学习参数。”

#### 1D. `model.to(device)`、`model.train()`、`model.eval()`

这三个是 `nn.Module` 对象最常见的方法。

```python
model.to(device)
```

- 把模型搬到 CPU 或 GPU

```python
model.train()
```

- 切到训练模式

```python
model.eval()
```

- 切到评估模式

在当前脚本里这三个都出现了，所以你现在看到它们时，要立刻意识到：

- 这说明 `model` 是一个真正的 PyTorch 模型对象
- 不是普通字典，也不是普通函数

### 2. `forward`

`forward` 是模型真正定义“输入怎么变成输出”的地方。

例如你看到：

```python
pred_list = model(chunk)
```

本质上最终会跑到这个模型类的 `forward(...)` 方法里。

所以：

- `model(chunk)` = 让模型执行一次前向传播

#### 2A. 为什么 `model(chunk)` 不直接写成 `model.forward(chunk)`

因为在 PyTorch 里，官方习惯是：

- 写 `model(x)`
- 不直接手动写 `model.forward(x)`

你当前阶段可以先理解成：

- `model(x)` 是更标准、更完整的调用方式
- 底层最终仍然会走到 `forward(...)`

所以代码里：

```python
pred_list = model(chunk.to(device))
```

你就直接读成：

“让模型对输入做一次前向传播。”

#### 2B. `forward` 的输入和输出是什么

`forward` 最核心就是定义：

- 输入是什么
- 输出是什么

在当前项目这个模型里，最重要的是：

- 输入是整理好的事件张量 `chunk`
- 输出是一个字典，里面至少有 `"flow"`

也就是说，`forward` 不一定只能返回一个 tensor，它也可以返回：

- 字典
- 列表
- 元组

当前项目返回字典，是因为作者想同时返回：

- `flow`
- `attn`

#### 2C. 为什么前向传播是模型最核心的部分

因为它决定了：

- 输入怎么经过网络结构
- 最终生成什么预测

如果你把训练脚本看成“外层框架”，那 `forward` 就是模型内部真正干活的地方。

训练脚本只是负责：

- 准备输入
- 调用模型
- 算 loss
- 更新参数

#### 2D. 当前项目里你该怎么理解 `pred_list = model(chunk)`

你现在应该把这句固定读成：

- `model`：模型对象
- `(chunk)`：把输入喂给模型
- `pred_list`：接住模型前向传播返回的结果

然后再继续往下看：

```python
pred = pred_list["flow"]
```

这说明模型返回的不是单个 tensor，而是一个包含多个字段的字典。

### 3. `state_dict`

`state_dict` 可以先理解成：

- “模型当前所有参数的字典”

训练时，模型学到的东西，主要就体现在这些参数里。

所以保存模型时，经常会保存：

- 模型参数
- 优化器状态

这也是为什么你会在项目里看到 `save_state_dict(...)`、`load_state_dict(...)` 这类名字。

#### 3A. 为什么 `state_dict` 很重要

因为训练出来的“知识”本质上主要都在参数里。

而 `state_dict` 正是 PyTorch 用来装这些参数的标准形式。

所以：

- 想保存模型，通常离不开 `state_dict`
- 想恢复训练，通常也离不开 `state_dict`

#### 3B. 模型的 `state_dict` 和训练状态的 `state_dict` 不一样

这是当前项目里非常重要的区分。

模型的 `state_dict` 更像：

- 模型参数字典

训练状态的 `state_dict` 更像：

- 优化器状态
- scheduler 状态
- scaler 状态
- epoch

当前项目里这段代码：

```python
state_dict = {
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "epoch": epoch,
    "scaler": scaler.state_dict() if scaler else None,
}
```

就是“训练状态”的 `state_dict`，不只是模型参数。

#### 3C. `load_state_dict(...)` 是什么

如果：

```python
model.load_state_dict(pretrained_dict, strict=False)
```

你就读成：

“把已有的一组参数装回这个模型对象里。”

这常用于：

- 加载预训练权重
- 恢复之前保存的模型参数

当前项目里对应代码在：

- [utils.py:68](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py:68)

#### 3D. `strict=False` 先怎么理解

你当前阶段不用深究太多，先把它理解成：

- “允许参数名字或数量不是完全一模一样，也尽量加载能对上的部分”

这在做：

- finetune
- 结构有轻微变化的模型恢复

时比较常见。

#### 3E. 当前项目里 `state_dict` 相关代码怎么读

看这句：

```python
pretrained_dict = pretrained_model.state_dict()
```

你应该读成：

“从已经加载进来的预训练模型对象里，把参数字典取出来。”

再看这句：

```python
model.load_state_dict(pretrained_dict, strict=False)
```

你应该读成：

“把这份参数字典装回当前要训练的模型对象里。”

所以整个流程就是：

1. 先把旧模型读进来
2. 拿到它的参数字典
3. 再把这份参数装到当前模型里

### 4. `tensor` 和 `shape`

`tensor` 是多维数组。

`shape` 是它的维度大小。

例如：

```python
[B, 2, H, W]
```

你现在先固定这样读：

- `B`：batch size
- `2`：通道数
- `H`：高度
- `W`：宽度

#### 4A. 为什么深度学习代码里到处都是 tensor

因为：

- 输入数据是 tensor
- 标签是 tensor
- 模型参数是 tensor
- 模型输出也是 tensor

所以你可以先把整个训练过程理解成：

- 一堆 tensor 在不同函数之间流动、变形、计算

#### 4B. 当前项目里最重要的 3 个 tensor

你现在优先只盯住这三个：

```python
chunk
label
mask
```

它们分别代表：

- `chunk`：输入事件张量
- `label`：真实光流
- `mask`：有效区域

#### 4C. 当前项目里最重要的几个 shape

在训练脚本注释里，最值得先记住的是：

```python
chunk  # [B, 20, 2, H, W]
label  # [B, 2, H, W]
mask   # 原始大致 [B, H, W]，后面会变成 [B, 1, H, W]
```

你当前阶段先不要追问每一维的严格物理含义，只要先知道：

- `chunk` 比普通图像多了时间/极性相关维度
- `label` 是 2 通道光流
- `mask` 是有效区域

#### 4D. 读 shape 时最容易犯的错

不要一上来试图把所有维度都一次性搞懂。

正确做法是：

1. 先数一共有几维
2. 先确认第一维是不是 batch
3. 先确认最后两维是不是 `H, W`
4. 中间维再慢慢看

这对当前项目尤其重要，因为事件输入维度比普通图像更复杂。

### 5. `view`

`view(...)` 可以先理解成：

- 重新整理 tensor 形状
- 不改变数据含义，主要改变“怎么看它”

例如：

```python
chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))
```

你现在先不用推每个维度细节，只要知道它在“重排输入形状”。

#### 5A. `view` 最核心的直觉

你可以先把 `view(...)` 理解成：

- “不换数据，只换排布视角”

类似于：

- 同一堆积木
- 重新按另一种方式摆成几行几列

#### 5B. `-1` 是什么意思

在：

```python
chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))
```

里面的 `-1` 可以先理解成：

- “这一维让 PyTorch 自动帮我推出来”

所以你现在看到 `-1` 时，不要慌，先记住：

- 这表示某一维大小不手写，让系统自动算

#### 5C. 当前项目里这句 `view(...)` 在干什么

这句：

```python
chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))
```

人话可以先读成：

“保留 batch 维和空间维，把中间几维压平成一个更普通的通道维。”

也就是说，它在做的是：

- 让输入更接近常见卷积网络吃的 `[B, C, H, W]` 格式

### 6. `unsqueeze`

`unsqueeze(dim=1)` 的意思是：

- 在第 1 维插入一个长度为 1 的新维度

例如：

- `[B, H, W] -> [B, 1, H, W]`

#### 6A. 为什么要插一个长度为 1 的维度

因为很多 PyTorch 操作默认希望：

- 数据带有 channel 维

哪怕这个 channel 维现在只有 1，也先补出来，后面操作会更统一。

#### 6B. 当前项目里 `unsqueeze(mask, dim=1)` 在干什么

这句：

```python
mask = torch.unsqueeze(mask, dim=1)
```

人话就是：

“把原来的 `mask` 从 `[B, H, W]` 变成 `[B, 1, H, W]`，方便后面和 flow 这类带 channel 的张量一起算。”

#### 6C. 你现在看 `dim=1` 怎么理解

当前阶段先不要深究维度编号体系，只先记住：

- `dim=1` 表示在 batch 维后面插入一个新维度

对这份代码来说，这就够用了。

### 7. `cat`

`torch.cat(...)` 是把多个 tensor 沿某个维度拼接起来。

例如当前文件里：

```python
chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)
```

你先读成：

“把正事件和负事件沿新的极性维拼起来。”

#### 7A. `cat` 和普通加法不是一回事

`cat` 不是把数值相加，而是：

- 把两个张量接在一起

你可以先把它理解成：

- 左一块 + 右一块，拼成更长的一块

#### 7B. 当前项目里这句 `cat(...)` 的真正作用

这句：

```python
chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)
```

人话就是：

“先分别给正事件和负事件补一个极性维，然后沿这个新维度把它们拼起来。”

所以它不是在做数值融合，而是在做结构重排。

### 8. `sum`

`torch.sum(...)` 就是求和。

它常常用来：

- 沿某个维度聚合
- 构造 event mask
- 可视化时压缩时间维

#### 8A. 当前项目里 `sum` 最常见的两种用途

第一种：

- 沿时间维压缩，做可视化

例如：

```python
chunk_vis = torch.sum(chunk, dim=1)
```

可以先读成：

“把时间维加起来，得到更容易展示的一张图。”

第二种：

- 构造 event mask

例如：

```python
event_mask = torch.sum(torch.sum(chunk, dim=1), dim=1, keepdim=True).bool()
```

你可以先读成：

“先把时间维和极性维都加起来，看看哪些像素位置至少有过事件，再把结果转成真假掩码。”

#### 8B. `keepdim=True` 现在怎么理解

你当前阶段先记成：

- “求和以后，尽量保留维度结构，方便后面继续参与运算”

不用现在深究更多。

#### 8C. 这一轮张量操作最该记住的版本

补完这一轮后，PyTorch 数据流这一块你至少先记住：

1. tensor 是深度学习里的多维数组。
2. shape 是 tensor 每一维的大小。
3. `view(...)` 是重排形状。
4. `unsqueeze(...)` 是插入一个长度为 1 的新维度。
5. `cat(...)` 是把多个 tensor 沿某一维拼起来。
6. `sum(...)` 是沿某个维度聚合。
7. 当前项目里这些操作主要是在整理事件输入、掩码和可视化张量。

---

## 训练循环最小基础

### 1. 一步训练的最小骨架

你现在必须先把下面这 5 步背熟：

```text
forward
-> loss
-> backward
-> optimizer.step()
-> optimizer.zero_grad()
```

这几乎就是所有训练脚本的核心。

#### 1A. 为什么这 5 步是训练的骨架

因为无论模型复杂还是简单，训练本质上都在做同一件事：

1. 先拿输入做预测
2. 再和标准答案比较
3. 再根据误差反向算梯度
4. 再用梯度更新参数
5. 再把旧梯度清掉

当前项目虽然是事件相机 + SNN + 多尺度 flow，但骨架仍然没有变。

#### 1B. 当前项目里这 5 步对应哪几句代码

最关键的对应关系是：

- `forward`
  - [train_flow_parallel_supervised_SNN.py:304](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:304)
- `loss`
  - [train_flow_parallel_supervised_SNN.py:311](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:311)
  - [train_flow_parallel_supervised_SNN.py:313](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:313)
- `backward`
  - [train_flow_parallel_supervised_SNN.py:320](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:320)
  - [train_flow_parallel_supervised_SNN.py:322](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:322)
- `optimizer.step()`
  - [train_flow_parallel_supervised_SNN.py:335](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:335)
  - [train_flow_parallel_supervised_SNN.py:338](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:338)
- `optimizer.zero_grad()`
  - [train_flow_parallel_supervised_SNN.py:341](d:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:341)

### 2. `forward`

模型根据输入做预测。

在当前项目里：

```python
pred_list = model(chunk)
```

#### 2A. 当前项目里 `forward` 之前已经做了什么

在真正 `model(chunk)` 之前，训练脚本其实已经做了不少准备工作：

- 把 `chunk` 搬到 device
- 整理事件编码
- 做输入归一化
- 可选地做阈值化

所以当你看到：

```python
pred_list = model(chunk.to(device))
```

不要把它理解成“模型从原始数据直接开始算”，而要理解成：

“模型开始处理的是已经整理好的输入 tensor。”

#### 2B. 当前项目里 `forward` 返回的不是单个 tensor

这里：

```python
pred_list = model(chunk)
pred = pred_list["flow"]
```

说明当前模型返回的是一个字典。

所以这个项目的 `forward` 更准确地说是在做：

- 输入：事件 tensor
- 输出：带有多尺度 `flow` 的结果字典

### 2C. `forward` 结束后，训练脚本最关心什么

最关心的是：

- 能不能从输出里拿到后面算 loss 需要的东西

在当前项目里就是：

```python
pred = pred_list["flow"]
```

也就是：

“从模型输出里取出多尺度光流预测。”

### 3. `loss`

衡量预测和标准答案差多少。

在当前项目里：

```python
curr_loss = loss_function(pred, label, mask, ...)
```

#### 3A. 当前项目里 `loss` 的三个核心输入

当前项目里，算 loss 时最关键的三个量是：

- `pred`
- `label`
- `mask`

人话就是：

- 预测结果
- 标准答案
- 哪些区域有效

#### 3B. 为什么这里还要传 `mask`

因为不是所有像素位置都应该参与损失。

有些地方：

- 没有有效 GT
- 没有事件
- 不应该纳入当前监督

所以 loss 不是“全图一锅端”，而是“只在有效区域上算”。

#### 3C. 当前项目里 `loss` 还有一个容易忽略的点

训练时：

```python
pred = pred_list["flow"]
curr_loss = loss_function(pred, label, mask, ...)
```

这里的 `pred` 实际上是一个多尺度 flow 列表，不是单张 flow。

所以当前项目训练时的 loss 是：

- 用整组多尺度预测一起算

这点和很多初学者脑中“一个输入对应一个输出图再算 loss”的印象不完全一样。

### 4. `backward`

根据 loss 计算梯度。

```python
curr_loss.backward()
```

#### 4A. `backward()` 之后发生了什么

`backward()` 本身不会立刻更新参数。

它做的是：

- 把当前 loss 反向传回网络
- 在每个参数上算出梯度

也就是说，`backward()` 结束时，模型参数已经“知道自己该往哪边改”，但还没真的改。

#### 4B. 当前项目里为什么有两种 `backward`

你会看到：

```python
scaler.scale(curr_loss).backward()
```

和：

```python
curr_loss.backward()
```

区别只是：

- 如果用了 AMP，就先做缩放
- 否则直接 backward

但本质作用一样，都是：

- 计算梯度

### 5. `optimizer.step()`

根据梯度真正更新参数。

```python
optimizer.step()
```

#### 5A. `step()` 和 `backward()` 的分工

这是训练循环里最容易混的一对。

你现在要强行区分：

- `backward()`：算梯度
- `optimizer.step()`：改参数

所以：

- 没有 `backward()`，优化器不知道怎么改
- 没有 `optimizer.step()`，参数不会真的更新

#### 5B. 当前项目里为什么也有两种 `step`

你会看到：

```python
scaler.step(optimizer)
```

和：

```python
optimizer.step()
```

它们的区别仍然只是：

- 是否走了 AMP 路径

但它们共同的本质都是：

- 真正执行参数更新

### 6. `optimizer.zero_grad()`

把旧梯度清掉，为下一步训练做准备。

```python
optimizer.zero_grad()
```

#### 6A. 为什么一定要清梯度

因为在 PyTorch 里，梯度默认是累加的。

如果你不清：

- 下一次 `backward()` 的梯度会叠到上一次上面

这通常不是你想要的默认行为。

所以标准训练循环里几乎总会看到 `zero_grad()`。

#### 6B. 当前项目里 `zero_grad()` 出现在什么位置

当前项目把它放在参数更新之后：

```python
optimizer.step()
optimizer.zero_grad()
```

你可以把它读成：

“这次参数已经更新完了，现在把旧梯度清掉，给下一轮 batch 做准备。”

#### 6C. 把这 5 步再压缩成一句人话

你现在应该能把这一整套训练骨架稳定地读成：

“模型先根据输入做预测；再拿预测和标准答案算损失；再根据损失反向算出梯度；再让优化器用梯度更新参数；最后把旧梯度清空，进入下一轮。”

---

## 光流任务最小基础

### 1. 什么是 optical flow

光流可以先理解成：

- 图像里每个像素“往哪里动了”

它通常包含两个分量：

- 水平方向位移
- 垂直方向位移

所以光流标签通常是 2 通道。

这就是为什么你在代码里看到：

```python
label  # [B, 2, H, W]
```

### 2. 为什么 `mask` 很重要

不是每个像素都有有效的真实光流。

所以训练时不能全图都算 loss，通常只在有效区域上算。

这就是 `mask` 的作用。

---

## SNN 最小基础

### 1. SNN 和普通网络最大的区别

普通网络大多直接处理连续值激活。

SNN 更强调：

- 时间维
- 脉冲发放
- 神经元内部状态

### 2. spike

你可以先把 spike 理解成：

- 神经元在某个时刻有没有发放

最粗略地看，像：

- 发放：1
- 不发放：0

### 3. membrane potential

这是神经元内部积累的状态。

如果它积累到一定程度，就可能发放 spike。

### 4. 为什么每个 batch 前都要 `reset_net`

因为 SNN 有内部状态。

如果不清掉，上一个 batch 的状态会污染下一个 batch。

所以：

```python
functional.reset_net(model)
```

你应该直接读成：

“清空 SNN 内部时间状态。”

### 5. surrogate gradient

spike 的真实发放过程不适合直接做普通反向传播。

所以训练 SNN 时，常用一个“替代梯度”近似，这就叫 surrogate gradient。

你现在先只要知道：

- 它是为了让 SNN 能训练起来
- `surrogate.ATan()` 这类配置就是在指定它

---

## SpikingJelly 最小基础

### 1. 它是什么

SpikingJelly 是这个项目用的 SNN 框架工具箱。

你可以先把它理解成：

- “专门给脉冲神经网络用的 PyTorch 扩展工具”

### 2. 在当前项目里最重要的 4 个名字

`functional.reset_net`

- 清空网络内部状态

`functional.set_step_mode`

- 设置网络按什么时间步模式运行

`neuron`

- 提供不同类型的脉冲神经元

`surrogate`

- 提供 surrogate gradient

---

## 你现在应该达到的最小阅读水平

如果这一轮基础你跟住了，那么看到下面这段代码时，你至少应该知道它的大意：

```python
for chunk, mask, label in train_dataloader:
    functional.reset_net(model)
    chunk = chunk.to(device=device, dtype=torch.float32)
    label = label.to(device=device, dtype=torch.float32)
    mask = torch.unsqueeze(mask.to(device=device), dim=1)
    pred_list = model(chunk)
    curr_loss = loss_function(pred_list["flow"], label, mask)
    curr_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

它的人话就是：

“不断取一批训练数据；先清空 SNN 的内部状态；把输入、标签和掩码搬到设备上并整理形状；让模型做预测；根据预测和真实答案计算损失；反向传播得到梯度；用优化器更新参数；最后清空旧梯度。”
