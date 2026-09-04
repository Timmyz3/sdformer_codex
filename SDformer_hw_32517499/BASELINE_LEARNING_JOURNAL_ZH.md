# SDformerFlow Baseline 学习日志

这份文档专门记录“从零开始学习 baseline”的每次讲解内容。

使用规则：

- 每次学习 baseline，都往这份文档追加
- 目标不是写论文式总结，而是写成“能跟着看代码”的学习笔记
- 默认读者是：
  - 深度学习初学者
  - Python 不熟
  - 需要把入口、数据流、模块关系一点点掰开理解

---

## 第 1 课：训练入口到底怎么把 baseline 跑起来

### 1. 这一课要解决什么问题

先不看注意力、不看剪枝、不看硬件。

只解决一个最基础的问题：

**当你运行训练命令时，程序到底是按照什么顺序一步步把 baseline 跑起来的？**

如果这件事没搞清楚，后面看任何模型文件都会乱。

---

### 2. 你先把几个最小概念记住

如果你 Python 还不熟，先把下面这几个词理解成很朴素的意思：

- `函数 function`
  - 就是一段可以被调用的步骤集合
- `类 class`
  - 可以理解成“一个模块的模板”
- `对象 object`
  - 就是这个模板真正造出来的实例
- `配置 config`
  - 就是一份参数表，告诉程序该用什么模型、什么数据、什么 batch size

放到这个项目里：

- `train()` 是“训练流程函数”
- `MS_SpikingformerFlowNet_en4` 是“模型类”
- `model = MS_SpikingformerFlowNet_en4(...)` 是“真正构造出的模型对象”
- `train_DSEC_supervised_SDformerFlow_en4.yml` 是“训练配置表”

---

### 3. 整个入口文件是谁

原版 DSEC 监督训练入口文件是：

- [train_flow_parallel_supervised_SNN.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)

你可以先把它理解成：

**训练总导演**

它自己不负责实现所有细节，但它决定：

1. 读哪个配置
2. 用哪个模型
3. 用哪个数据集
4. loss 怎么算
5. 优化器怎么更新
6. 什么时候验证
7. 什么时候保存模型

---

### 4. 程序从哪里开始执行

文件最后有这段：

- [train_flow_parallel_supervised_SNN.py:527](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L527)

这里的作用是：

1. 读取命令行参数
2. 创建配置解析器 `YAMLParser(args.config)`
3. 调用 `train(args, config_parser)`

你可以把它理解成：

```text
用户给命令
  -> 程序读命令
  -> 程序读配置文件
  -> 程序进入训练主函数
```

---

### 5. 真正的训练从哪个函数开始

训练主函数在：

- [train_flow_parallel_supervised_SNN.py:32](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L32)

也就是：

```python
def train(args, config_parser):
```

你后面看这个文件，脑子里始终要有一个问题：

**这个函数是按什么顺序组织训练流程的？**

---

### 6. `train()` 一进来先做什么

最开头先拿两样东西：

1. `config = config_parser.config`
2. `device = config_parser.device`

意思就是：

- `config`
  - 全部训练参数
- `device`
  - 当前训练设备，比如 `cuda:0` 或 `cpu`

所以你可以把前几行翻译成人话：

```text
先把“训练说明书”和“训练机器”拿到手
```

---

### 7. 配置是谁解析出来的

配置解析器在：

- [parser.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py)

它最重要的类是：

- [YAMLParser](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py#L6)

这个类做的事情不复杂：

1. 打开 `.yml` 文件
2. 读出里面的参数
3. 补上一些默认值
4. 判断当前是否能用 GPU
5. 决定 `device`

你现在只要记住：

**训练器自己不直接读 YAML。它是通过 `YAMLParser` 读的。**

---

### 8. baseline 用的是哪份配置

原版 baseline 配置在：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)

这份文件里最重要的几段是：

#### 8.1 数据部分

- `data.path`
- `data.preprocessed`
- `data.num_frames`

它们决定：

- 数据从哪里读
- 读的是不是已经预处理好的数据
- 输入有多少个时间 bin

#### 8.2 模型部分

- `model.name = MS_SpikingformerFlowNet_en4`
- `model.encoding = voxel`

它们决定：

- 训练器要实例化哪个模型类
- 输入事件张量按什么表示方式处理

#### 8.3 Swin Transformer 部分

- `swin_depths = [2,2,6,2]`
- `swin_num_heads = [3,6,12,24]`
- `window_size = [2,9,9]`

它们决定：

- backbone 一共有多少个 stage
- 每个 stage 多少 heads
- attention 的局部窗口大小

---

### 9. 什么时候初始化 MLflow

在训练函数前面很早就做了：

- [train_flow_parallel_supervised_SNN.py:41](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L41)

MLflow 在这里的作用，你先不用想太复杂。

你现在就把它理解成：

**训练日志和模型记录器**

它负责记：

- 本次实验叫什么
- 用了什么配置
- 训练 loss
- 验证 loss
- 保存的模型

---

### 10. 模型是什么时候真正创建出来的

在训练器中，模型构建发生在：

- [train_flow_parallel_supervised_SNN.py:65](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L65)
- [train_flow_parallel_supervised_SNN.py:71](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L71)

这里做了两件关键事：

#### 10.1 先设置输入尺寸

如果配置里有 `crop`，就把 crop 大小传给 Swin。

也就是说：

```text
模型看到的输入空间尺寸
不一定是原图分辨率
而是训练 crop 后的尺寸
```

#### 10.2 再按类名实例化模型

训练器会读：

```python
config["model"]["name"]
```

然后执行类似：

```python
model = MS_SpikingformerFlowNet_en4(...)
```

所以这里特别重要的一点是：

**训练器不是写死某一个模型，而是由 YAML 决定模型类名。**

这也是后面做模型变体最自然的入口。

---

### 11. baseline 当前到底实例化了谁

根据配置：

- `model.name = MS_SpikingformerFlowNet_en4`

所以这里实例化的是：

- [MS_SpikingformerFlowNet_en4](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L319)

你现在先只把它当成一个黑盒模型名字，不着急深入内部。

第一节课只要求你知道：

```text
训练器不是直接写了很多层
而是构造了一个叫 MS_SpikingformerFlowNet_en4 的模型对象
```

---

### 12. 模型创建后，训练器立刻做什么

创建模型后，训练器继续做：

1. `model.to(device)`
2. `model.init_weights()`
3. 如果给了旧 run id，就尝试 `load_model(...)`
4. 配置 SNN backend

这些分别对应：

- 把模型放到 GPU 或 CPU
- 初始化权重
- 如果是恢复训练或微调，就加载旧模型
- 决定脉冲神经网络底层用什么运行后端

这里最特别的是第 4 步，因为它是 SNN 模型，不是普通 ANN。

---

### 13. 什么是 SNN backend

在：

- [train_flow_parallel_supervised_SNN.py:121](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L121)

训练器会调用：

```python
configure_snn_backend(model, device, config, neurontype)
```

你现在把这一步理解成：

**告诉脉冲神经网络：“接下来你底层该用哪种方式跑。”**

在标准 CUDA 路线里，默认偏向 `cupy`。  
在我们后来给异构卡加的兼容路径里，也可以切成 `torch`。

但这只是运行方式选择，不改变 baseline 的核心结构。

---

### 14. 损失函数什么时候创建

在：

- [train_flow_parallel_supervised_SNN.py:153](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L153)

创建的是：

- [flow_loss_supervised](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/loss/flow_supervised.py#L6)

你先把它当成：

**“预测光流”和“真实光流”之间的误差计算器。**

现在第一节课不展开公式，只要知道训练器前面已经把：

- 模型
- 优化器
- 损失函数

三大件准备好了。

---

### 15. 数据集什么时候创建

训练集创建在：

- [train_flow_parallel_supervised_SNN.py:182](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L182)

验证集创建在：

- [train_flow_parallel_supervised_SNN.py:202](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L202)

用的都是：

- [DSECDatasetLite](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py#L36)

这里你现在要记一个最重要的事实：

**每个 batch 从 dataloader 里出来时，是 3 个东西：**

```python
chunk, mask, label
```

对应：

- `chunk`
  - 输入事件张量
- `mask`
  - 哪些像素位置有效
- `label`
  - 真实光流标签

后面第二节课我们专门讲它们是怎么从 `saved_flow_data` 读出来的。

---

### 16. 训练循环从哪里开始

epoch 循环在：

- [train_flow_parallel_supervised_SNN.py:231](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L231)

batch 循环在：

- [train_flow_parallel_supervised_SNN.py:240](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L240)

也就是说真正训练时，程序会不停重复：

```text
取一个 batch
  -> 预处理输入
  -> 模型前向
  -> 算 loss
  -> 反向传播
  -> 更新参数
```

这就是深度学习训练最经典的主循环。

---

### 17. 一个 batch 进来后先做什么

进入 batch 循环后，训练器首先做：

1. `functional.reset_net(model)`
2. `functional.set_step_mode(...)`
3. 把 `chunk / label / mask` 搬到 device

这一步你要理解成：

**因为这是脉冲神经网络，所以每个 batch 开始前都要把网络状态清干净。**

普通 CNN 通常不用显式做这一步。

---

### 18. 输入在进模型前做了哪些处理

这一段在：

- [train_flow_parallel_supervised_SNN.py:252](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L252)

baseline 当前默认是：

- `encoding = voxel`
- `polarity = True`

所以训练器会做下面这些事：

#### 18.1 把正负事件分开

原始 `chunk` 里正负值混在一起。

训练器会构造：

- `pos = relu(chunk)`
- `neg = relu(-chunk)`

也就是：

- 正事件一份
- 负事件一份

#### 18.2 把它们拼成显式极性通道

得到近似这种结构：

```text
B, C, 2, H, W
```

这里的 `2` 就是极性维度：

- 0：正事件
- 1：负事件

这一步非常重要，因为后面模型看到的输入已经不是最原始的 voxel 了，而是：

**按正负极性显式拆开的张量。**

#### 18.3 对非零值做归一化

如果配置里：

- `norm_input = minmax`

那就只对非零元素做 min-max 归一化。

#### 18.4 如果配置了阈值，就做 spike 化

如果：

- `data.spike_th` 不为空

还会把输入阈值化成 0/1。

---

### 19. 真正的模型前向在哪里发生

前向调用在：

- [train_flow_parallel_supervised_SNN.py:305](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L305)

```python
pred_list = model(chunk.to(device))
pred = pred_list["flow"]
```

这两行你必须吃透：

#### 19.1 `model(...)` 返回的不是单个 tensor

它返回的是一个字典。

#### 19.2 真正拿来算 loss 的主要内容是

```python
pred_list["flow"]
```

也就是说：

**模型的输出不是一个简单的单层结果，而是带结构的输出。**

---

### 20. loss 是怎么接上的

loss 在：

- [train_flow_parallel_supervised_SNN.py:308](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L308)

如果不做 event mask，就大概是：

```python
curr_loss = loss_function(pred, label, mask, gamma=...)
```

人话翻译：

```text
把模型预测的光流
和真实光流标签
再加上有效区域 mask
一起送进损失函数
```

所以训练器自己不算光流误差，它只是把数据交给 loss 模块。

---

### 21. 什么时候反向传播和更新参数

在 loss 算完后，训练器执行：

- backward
- gradient clipping
- optimizer.step()
- optimizer.zero_grad()

这就是参数真正被更新的地方。

也就是说：

```text
前向算出预测
  -> loss 比较预测和标签
  -> backward 算梯度
  -> optimizer 用梯度更新模型参数
```

---

### 22. 什么时候保存模型

每个 epoch 结束后，训练器会比较：

- 当前 train loss
- 历史 best loss

如果当前更好，就保存。

对应位置：

- [train_flow_parallel_supervised_SNN.py:392](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L392)
- [train_flow_parallel_supervised_SNN.py:393](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L393)

这意味着当前 baseline 的 best model 判定，核心看的是：

**训练阶段的 loss**

不是你直觉里一定按验证集指标保存。

---

### 23. 第一课的最简主链总结

你现在把整条链压缩成下面这 8 步就够了：

1. 命令行启动训练脚本
2. `YAMLParser` 读取配置
3. `train()` 拿到 config 和 device
4. 按配置实例化 `MS_SpikingformerFlowNet_en4`
5. 构建 `DSECDatasetLite`
6. 训练器把 `chunk` 做 polarity split 和归一化
7. `model(chunk)` 输出 flow 预测
8. `loss_function(pred, label, mask)` 计算损失并更新参数

---

### 24. 第一课结束前你必须能回答的 3 个问题

如果你回答不上来，就说明这节还没吃透。

#### 问题 1

训练脚本是在哪一行真正进入 `train()` 的？

#### 问题 2

训练器实例化的 baseline 模型类名是什么？

#### 问题 3

一个 batch 从 dataloader 出来后，训练器拿到的 3 个核心变量是什么？

---

### 25. 下一课预告

第二课讲：

**`DSECDatasetLite` 是怎么把 `saved_flow_data` 里的 `.npy` 文件读成 `chunk, mask, label` 的。**

这一课结束后，你会真正知道：

- 输入张量的形状
- 每个 `.npy` 文件到底是什么
- 为什么 `sequence_lists/train_split_seq.csv` 必须存在

---

## 第 1 课课后检查：第一次回答与纠正

用户第一次回答：

1. `config` 应该是 `train()` 这个函数解析的
2. dataloader 拿到的是 `mask`、`chunk` 和标签
3. 模型返回的是一个字典

### 1. 哪些地方答对了

后两点是对的：

- dataloader 的核心三元组是：
  - `chunk`
  - `mask`
  - `label`
- 模型前向返回的确实不是单个 tensor，而是一个字典

训练器里能直接看到：

```python
for chunk, mask, label in tqdm(train_dataloader):
```

以及：

```python
pred_list = model(chunk.to(device))
pred = pred_list["flow"]
```

### 2. 哪个地方需要纠正

第一点不对：

`config` 不是 `train()` 自己解析出来的。

真正的顺序是：

1. 主程序先调用：
   - `YAMLParser(args.config)`
2. `YAMLParser` 负责读 YAML、补默认值、决定 `device`
3. 然后把已经解析好的 `config_parser` 传给：
   - `train(args, config_parser)`
4. `train()` 再从里面取：
   - `config = config_parser.config`

所以更准确的说法应该是：

**`config` 是由 `YAMLParser` 解析出来的，`train()` 只是把它取出来使用。**

### 3. 这一轮学习的正确答案

#### 问题 1

`config` 是谁解析出来的？

正确答案：

- `YAMLParser`

#### 问题 2

训练器从 dataloader 里每次拿到哪 3 个东西？

正确答案：

- `chunk`
- `mask`
- `label`

#### 问题 3

模型前向之后，训练器是不是直接得到一个单独的光流张量？

正确答案：

- 不是
- 它先得到一个字典
- 然后从字典里取：
  - `pred_list["flow"]`

### 4. 下一步学习重点

下一课继续：

**第二课：`DSECDatasetLite` 是怎么工作的。**

这节会重点解决：

1. `saved_flow_data` 目录里每个子文件夹是干什么的
2. 为什么 `train_split_seq.csv` / `valid_split_seq.csv` 必须存在
3. `chunk`、`mask`、`label` 分别从哪个 `.npy` 文件读出来
4. `chunk` 的形状到底是什么

---

## 第 2 课：`DSECDatasetLite` 是怎么把 `saved_flow_data` 读成训练样本的

### 1. 这一课的目标

这一课只解决一个问题：

**训练器里面出现的 `chunk, mask, label`，到底是从哪里来的？**

如果这一层不懂，后面你做：

- token 剪枝
- 稀疏化
- patch/window 改动
- 输入重编码

都会没有落脚点。

---

### 2. 这一课最重要的文件

核心文件是：

- [DSEC_dataset_lite.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py)

最关键的类是：

- [DSECDatasetLite](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py#L36)

你可以把这个类理解成：

**“样本读取器”**

它的工作不是训练模型，而是：

1. 找到数据文件在哪
2. 决定读哪一个样本
3. 把这个样本读成 Python / PyTorch 能用的张量
4. 返回给训练器

---

### 3. 先理解 `saved_flow_data` 是什么

当前 baseline 训练默认不是直接读原始 DSEC 事件流。

它默认读的是：

- 预处理后的 `saved_flow_data`

你可以把它理解成：

**“已经整理好的训练缓存目录”**

里面通常有这几块：

```text
saved_flow_data/
  gt_tensors/
  mask_tensors/
  event_tensors/
  sequence_lists/
```

每个目录都负责一种东西。

---

### 4. `gt_tensors` 是什么

`gt_tensors` 里存的是：

**真实光流标签**

也就是训练里的标准答案。

每个文件通常是一个 `.npy`，例如：

```text
zurich_city_09_a_0001.npy
```

这个文件里保存的通常是一个形状像下面这样的数组：

```text
[2, H, W]
```

其中：

- 第 0 个通道：x 方向光流
- 第 1 个通道：y 方向光流

所以你看到训练器里那个 `label`，本质上就是从这里读出来的。

---

### 5. `mask_tensors` 是什么

`mask_tensors` 里存的是：

**有效像素区域**

因为不是每个像素位置都有可靠的光流标签，所以需要一个 mask 告诉 loss：

```text
哪些位置要参与监督
哪些位置不要算
```

这个目录里的文件名跟 `gt_tensors` 一一对应。

例如：

```text
gt_tensors/zurich_city_09_a_0001.npy
mask_tensors/zurich_city_09_a_0001.npy
```

这两个文件是配套的。

所以训练器里的 `mask`，就是从这里读出来的。

---

### 6. `event_tensors` 是什么

`event_tensors` 里存的是：

**事件输入张量**

这就是训练器里的 `chunk` 的来源。

在当前 baseline 默认配置下：

- `encoding = voxel`
- `num_frames = 10`
- `polarity = True`

所以数据集会去找：

```text
event_tensors/10bins/left/<sequence>/<file>.npy
```

例如：

```text
event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy
```

这里的 `10bins`，表示这个事件样本已经被预处理成：

**10 个时间 bin 的体素表示**

你先不用纠结 voxel 的数学构造。  
现在只需要知道：

`chunk` 不是原始 event stream，而是已经做好的事件体表示。

---

### 7. `sequence_lists` 是什么

这个目录特别关键，很多人第一次看 baseline 会忽略它。

里面通常有：

```text
train_split_seq.csv
valid_split_seq.csv
```

它们的作用不是存数据本身，而是存：

**“这一轮训练该读哪些文件名”**

也就是说，这两个 csv 更像是：

**样本清单**

例如 `train_split_seq.csv` 里一行可能就是：

```text
zurich_city_09_a_0002.npy
```

这意味着：

```text
训练集要读这个样本
```

所以如果没有 `sequence_lists`，数据集根本不知道：

- 哪些文件属于 train
- 哪些文件属于 valid

这就是为什么我之前一直强调：

**没有 `train_split_seq.csv` / `valid_split_seq.csv`，baseline 跑不起来。**

---

### 8. `DSECDatasetLite.__init__()` 在干什么

现在开始看类初始化逻辑。

位置在：

- [DSECDatasetLite.__init__](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py#L37)

这个函数可以理解成：

**“当数据集对象被创建时，先把所有路径和规则准备好。”**

它主要做 6 件事：

#### 8.1 记住配置

```python
self.config = config
```

就是把总配置表保存下来，后面随时取。

#### 8.2 记住标签和 mask 路径

```python
self.flow_path = .../gt_tensors
self.mask_path = .../mask_tensors
```

这一步的意思很直白：

```text
以后 label 去 gt_tensors 里找
以后 mask 去 mask_tensors 里找
```

#### 8.3 读输入编码方式

```python
self.input = self.config['model']['encoding']
```

这决定它后面去 `event_tensors` 的哪种子目录找输入。

#### 8.4 计算 bin 数

```python
self.num_bins = self.num_frames_per_ts * self.num_chunks
```

这一步是在决定：

```text
每个样本总共有多少个时间 bin
```

在当前 baseline 里：

- `num_frames = 10`
- `num_chunks = 1`

所以：

- `num_bins = 10`

#### 8.5 决定 `events_path`

如果是预处理数据、并且编码方式是 voxel，它会去：

```python
event_tensors/10bins/left
```

或者如果不按极性拆分，会去：

```python
event_tensors/10bins_pol/left
```

这也是为什么当时我们预处理时特别在意：

- `10bins`
- `10bins_pol`

名字必须和配置对应。

#### 8.6 读取样本清单 csv

这一步非常重要。

它最后会把：

```python
self.files = pd.read_csv(sequence_file, header=None)
```

也就是说：

**真正决定“这个 dataset 有哪些样本”的，是 csv 文件，不是目录遍历。**

---

### 9. `__len__()` 做了什么

位置在：

- [DSECDatasetLite.__len__](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py#L84)

只有一句：

```python
return len(self.files)
```

翻译成人话：

```text
这个数据集有多少个样本？
答案就是 csv 里有多少行
```

所以数据集长度不由磁盘里文件数量决定，而是由 `sequence_lists/*.csv` 决定。

---

### 10. `__getitem__()` 是最关键的

位置在：

- [DSECDatasetLite.__getitem__](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py#L100)

这是 PyTorch 数据集最核心的方法。

你可以把它理解成：

**“给我一个样本编号 `idx`，我就把第 `idx` 个样本读出来。”**

训练器每次从 dataloader 里拿 batch，本质上就是在反复调用这里。

---

### 11. `__getitem__()` 第一步做什么

如果 `num_chunks == 1`，它会先从 `self.files` 里拿出当前文件名：

```python
target_file_1 = self.files.iloc[idx, 0]
```

这句话的意思是：

```text
去 csv 的第 idx 行、第 0 列
把那一行文件名拿出来
```

假设读出来的是：

```text
zurich_city_09_a_0001.npy
```

那当前样本的“主文件名”就确定了。

---

### 12. 然后它怎么读 `mask` 和 `label`

接着它执行：

```python
mask = np.load(mask_path/target_file_1)
label = np.load(flow_path/target_file_1)
```

人话翻译：

```text
同名文件
去 mask_tensors 里读 mask
去 gt_tensors 里读光流标签
```

所以一个文件名就同时索引出：

- 一个 mask
- 一个 GT flow

---

### 13. 它为什么要拆出 `seq_folder1`

然后它做：

```python
seq_folder1 = "_".join(target_file_1.split('_')[:-1])
```

如果文件名是：

```text
zurich_city_09_a_0001.npy
```

这一步会得到：

```text
zurich_city_09_a
```

也就是：

**序列名**

为什么要这么做？

因为事件输入不是直接放在 `event_tensors` 根目录，而是分 sequence 子文件夹存的。

例如：

```text
event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy
```

所以程序必须先把：

- 文件名

拆成：

- 序列文件夹名
- 具体样本名

---

### 14. 它怎么读 `chunk`

如果是预处理数据，它会执行：

```python
chunk = np.load(os.path.join(self.events_path, seq_folder1, target_file_1))
```

翻译成人话：

```text
去 event_tensors/10bins/left/序列名/样本名.npy
把事件输入张量读出来
```

这就是训练器里那个 `chunk` 的来源。

所以到这里，`__getitem__()` 已经把：

- `chunk`
- `mask`
- `label`

三样东西都凑齐了。

---

### 15. 最后返回什么

最后返回：

```python
return chunk, mask, label
```

也就是说，训练器里看到的：

```python
for chunk, mask, label in train_dataloader:
```

并不是魔法，而就是 `__getitem__()` 一次次返回出来的。

---

### 16. 到这里你应该建立的完整映射关系

你现在应该能把一个样本的来源说清楚：

假设 csv 里这一行是：

```text
zurich_city_09_a_0001.npy
```

那么：

- `label`
  - 来自：
    - `gt_tensors/zurich_city_09_a_0001.npy`
- `mask`
  - 来自：
    - `mask_tensors/zurich_city_09_a_0001.npy`
- `chunk`
  - 来自：
    - `event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy`

这三者是靠同一个文件名绑定在一起的。

---

### 17. 为什么这对后面做剪枝/稀疏很重要

因为你以后想改 baseline，至少要先知道：

#### 17.1 你改的是数据层还是模型层

例如：

- 如果你改 `prepare_dsec_single_sequence.py`
  - 你是在改输入表示
- 如果你改 `DSECDatasetLite`
  - 你是在改读取方式
- 如果你改 `Spiking_swin_transformer3D.py`
  - 你是在改模型 backbone

#### 17.2 `chunk` 还不是模型最终看到的样子

dataset 里读出来的 `chunk`，后面训练器还会继续做：

- polarity split
- normalization

所以：

**dataset 输出形状**

和

**模型真正看到的输入形状**

不是同一个概念。

这点以后做模块插入时极其重要。

---

### 18. 这一课的最小总结

第二课你只要记住下面这条链：

```text
train_split_seq.csv 里的一行文件名
  -> 决定当前样本名
  -> 去 gt_tensors 读 label
  -> 去 mask_tensors 读 mask
  -> 去 event_tensors/.../sequence/ 读 chunk
  -> 返回 chunk, mask, label
```

这就是 `DSECDatasetLite` 的本质。

---

### 19. 这一课结束前你必须能回答的 4 个问题

#### 问题 1

`train_split_seq.csv` 的作用是什么？

#### 问题 2

`label` 是从哪个目录读出来的？

#### 问题 3

`chunk` 为什么还需要先拆出 sequence 文件夹名，再去读？

#### 问题 4

dataset 返回的 `chunk`，是不是就已经等于模型最终看到的输入？

---

### 20. 下一课预告

第三课讲：

**`MS_SpikingformerFlowNet_en4` 顶层模型是怎么由 encoder、residual blocks、decoder、prediction head 拼起来的。**

---

## 第 2 课课后检查：第二次回答与纠正

用户这次直接回答了：

```text
0
```

这可以理解为：

- 还没有形成清晰答案
- 或者暂时答不上来

这在当前阶段是正常的，因为第二课第一次接触了：

- csv split
- 目录映射
- `__getitem__`
- 样本名和序列名拆分

这些对 Python 初学者来说本来就容易混。

### 1. 这 4 个问题的标准答案

#### 问题 1

`train_split_seq.csv` 的作用是什么？

标准答案：

- 它是训练样本清单
- 它决定训练集包含哪些文件名
- dataset 的长度和取样顺序都以它为依据

#### 问题 2

`label` 是从哪个目录读出来的？

标准答案：

- `gt_tensors`

#### 问题 3

`chunk` 为什么要先拆出 `zurich_city_09_a` 这个 sequence 文件夹名？

标准答案：

- 因为事件输入不是平铺在一个目录里
- 而是按序列分子文件夹存放
- 所以必须先从文件名里拆出 sequence 名，才能定位到正确路径

#### 问题 4

dataset 返回的 `chunk`，是不是就已经等于模型最终看到的输入？

标准答案：

- 不是
- dataset 只负责把事件张量读出来
- 训练器后面还会做 polarity split、归一化，必要时还会做阈值化

### 2. 当前学习建议

第二课还没吃透时，不要急着进入第三课。

先把下面这条映射关系背熟：

```text
csv 文件名
  -> 决定当前样本
  -> gt_tensors 读 label
  -> mask_tensors 读 mask
  -> event_tensors/.../sequence/ 读 chunk
```

只要这条链你能复述出来，第二课就算过关。

---

## 第二节课后补充：用户回答“0”后的超简版复盘

如果现在还答不上来，不代表学不会，只代表第二节里同时出现了太多新概念。

这一节最重要的内容，其实只有一句话：

```text
csv 里写样本名
  -> 去 gt_tensors 读 label
  -> 去 mask_tensors 读 mask
  -> 去 event_tensors/.../sequence/ 读 chunk
```

把这句话再拆成最简单版本：

### 1. `train_split_seq.csv` 是什么

它就是“训练样本名单”。

例如里面一行写：

```text
zurich_city_09_a_0001.npy
```

意思就是：
这一个文件名，代表一个训练样本。

### 2. `label` 是什么

`label` 就是标准答案，也就是真实光流。

它从这里读：

```text
gt_tensors/样本名.npy
```

### 3. `mask` 是什么

`mask` 是“哪些像素位置有效”。

它从这里读：

```text
mask_tensors/样本名.npy
```

### 4. `chunk` 是什么

`chunk` 是输入，也就是预处理好的事件张量。

它从这里读：

```text
event_tensors/10bins/left/序列名/样本名.npy
```

### 5. 为什么要先拆出序列名

因为 `chunk` 不是直接平铺在一个文件夹里。

它是按序列存的，例如：

```text
event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy
```

所以程序必须先从文件名里拆出：

```text
zurich_city_09_a
```

才能进入正确的子文件夹找输入。

### 6. 当前第二节只要记住这 3 个对应关系

```text
样本名
  -> gt_tensors 读 label
  -> mask_tensors 读 mask
  -> event_tensors/... 读 chunk
```

### 7. 当前最容易混淆的一点

dataset 返回的 `chunk`，还不是模型最后真正看到的输入。

因为训练器后面还会继续对 `chunk` 做：
- polarity split
- 归一化
- 可能的阈值化

所以要分清：
- dataset 读出来的输入
- 模型最终吃进去的输入

这两个不是完全一样的东西。

---

## 当前阶段判断：需要懂到什么程度，才能开始改 baseline

用户当前已经能大致复述主流程：

```text
先用 YAML 读配置
再决定 device 和模型
再通过 dataloader 读取 DSECDatasetLite 准备好的数据
然后做前向传播、反向传播
```

这说明“训练主链”已经开始成形。

但是当前仍然存在一个很正常的困惑：

```text
我知道大致流程了
但具体每个模块内部怎么做，我还看不懂
那我是不是得先把整个 baseline 每一行都看懂，才能开始优化？
```

答案是：

**不需要先把整个 baseline 每一行都吃透，才能开始做优化。**

正确做法不是“全看懂再动手”，而是分层理解。

### 1. 你真正需要掌握的，不是全部代码，而是 3 层理解

#### 第一层：主流程层

你已经基本掌握了：

```text
配置 -> 数据 -> 模型 -> 损失 -> 更新参数
```

这一层的目标是：
- 不再迷路
- 知道程序从哪里进、到哪里出

#### 第二层：模块职责层

这一层你接下来要继续学的是：
- dataset 负责什么
- encoder 负责什么
- decoder 负责什么
- loss 负责什么

这一层不要求你立刻会写代码，但要求你知道：

```text
如果我要改输入稀疏，就该去 dataset/输入预处理附近看
如果我要改 attention，就该去 encoder/block 里看
如果我要改多尺度预测，就该去 decoder/prediction head 看
```

#### 第三层：局部实现层

这一层才是：
- 具体某个类的 forward
- 张量维度怎么变
- 哪一行在做 QKV
- 哪一行在做 window partition

这层不需要一开始全懂。

它应该在“你已经知道自己要改哪里”之后，再定点突破。

### 2. 为什么不能一上来要求自己全懂

因为 baseline 太大了。

如果一开始就要求：
- 训练器全懂
- dataset 全懂
- encoder 全懂
- decoder 全懂
- attention 全懂
- loss 全懂

那你会一直卡在“阅读焦虑”，反而迟迟进不了真正的研究。

所以更合理的做法是：

```text
先懂骨架
再懂模块职责
最后只深挖你准备下手改的那一块
```

### 3. 对你当前目标，最小必要理解是什么

如果你后面要做：
- 注意力改进
- token 稀疏
- window 剪枝
- timestep 裁剪

你当前真正必须先搞懂的，其实只有下面这些：

#### A. 输入是怎么进模型的

你要知道：
- `chunk` 来自哪里
- 训练器在模型前又对 `chunk` 做了什么

因为很多稀疏模块可能就插在这里。

#### B. 模型最外层由哪些块组成

你要知道：
- encoder
- residual blocks
- decoder
- prediction head

分别干嘛。

这样你才知道：
- 改注意力该进 encoder
- 改输出融合该进 decoder

#### C. attention 真正在哪个文件

你要知道真正的注意力主战场在：

```text
Spiking_swin_transformer3D.py
```

不是在训练器里。

### 4. 你现在不需要立刻搞懂什么

当前先不必强求完全吃透：
- 每一层的全部数学公式
- 每个 tensor 的每一步 reshape
- 每个训练技巧的全部细节
- 全部 residual/skip connection 细节

这些后面都是可以“按需学习”的。

### 5. 当前最适合你的学习策略

不是“全看懂再改”，而是：

```text
先继续把 baseline 拆成几个大模块
再确定你后面准备改的目标模块
最后只深挖那一块
```

所以接下来的学习顺序应该是：

1. 先看模型最外层大模块
2. 再看 encoder 主干
3. 再看 attention block
4. 最后才回来看具体想改的点

### 6. 当前阶段的结论

用户现在已经不需要再怀疑自己“是不是必须全懂才能动手”。

正确答案是：

**不用全懂。**

只要先把：
- 主流程
- 模块职责
- 改动目标所在位置

这三件事搞清楚，就已经足够开始 baseline 优化准备。

---

## 复习课：前两节课回顾与概念澄清

这一部分用于帮助用户把前两节课重新串起来，并澄清一个关键问题：

```text
DSECDatasetLite 只是把文件读出来，那这算不算预处理？
```

### 1. 第一节课到底讲了什么

第一节课讲的是：

**训练主链怎么跑起来。**

最简单的主链是：

```text
运行训练脚本
  -> 读取 YAML 配置
  -> 解析 config 和 device
  -> 创建模型
  -> 创建 dataset 和 dataloader
  -> dataloader 提供 chunk / mask / label
  -> 模型前向传播
  -> loss
  -> 反向传播
  -> 更新参数
```

这里最重要的是：

- YAML 决定“用什么模型、什么设备、什么超参数”
- train 脚本负责把所有模块串起来
- dataloader 负责一批一批提供训练样本

### 2. 第二节课到底讲了什么

第二节课讲的是：

**dataloader 里的样本，是怎么从磁盘文件里读出来的。**

核心对象是：

```text
DSECDatasetLite
```

它的工作不是训练，不是算 loss，也不是做注意力。

它只负责：

1. 根据 csv 知道“这次该读哪个样本”
2. 去 `gt_tensors` 读 `label`
3. 去 `mask_tensors` 读 `mask`
4. 去 `event_tensors/...` 读 `chunk`
5. 把这三样东西返回出去

所以第二节课的最核心映射关系是：

```text
样本名
  -> gt_tensors 读 label
  -> mask_tensors 读 mask
  -> event_tensors/... 读 chunk
```

### 3. 你现在复述的大致流程，基本是对的

用户当前的理解：

```text
调用 YAML 来读取 config，决定模型和 device，
再通过 dataloader 读取 DSECDatasetLite 弄出的预处理数据，
进行前向传播，再反向传播。
```

这个理解已经抓住了大骨架，方向是对的。

更准确一点可以说成：

```text
YAMLParser 读取 YAML，生成 config 和 device
  -> train 脚本根据 config 创建模型和 dataloader
  -> dataloader 调用 DSECDatasetLite
  -> DSECDatasetLite 从 saved_flow_data 里把 chunk / mask / label 读出来
  -> train 脚本把 chunk 喂给模型做前向传播
  -> 再根据 label 和 mask 计算 loss
  -> 再反向传播更新参数
```

### 4. `DSECDatasetLite` 算不算预处理

这是当前最重要的概念澄清。

答案是：

**严格来说，`DSECDatasetLite` 本身不负责“生成预处理数据”，它负责“读取已经预处理好的数据”。**

也就是说，要区分两个阶段：

#### 阶段 A：离线预处理

这一步会把原始 DSEC 数据处理成：

- `gt_tensors`
- `mask_tensors`
- `event_tensors`
- `sequence_lists`

这些东西合起来，就是：

```text
saved_flow_data
```

这一阶段才叫真正的“预处理”。

#### 阶段 B：训练时读取

这一步就是 `DSECDatasetLite` 做的事情：

- 不重新生成数据
- 只把已经准备好的 `.npy` 文件读出来
- 拼成训练器需要的 `chunk / mask / label`

所以更准确的说法是：

```text
DSECDatasetLite 不是预处理器
它是预处理数据的读取器
```

### 5. 但是为什么你会觉得它也像“预处理”

因为它虽然不做离线预处理，但它确实做了“数据组织”：

- 根据 csv 决定样本
- 根据文件名找到序列目录
- 把文件读成数组

所以从广义上看，它是在做“数据准备”；
但从工程上更准确的术语看：

- `prepare_dsec_*.py` 这类脚本：预处理
- `DSECDatasetLite`：读取预处理结果

### 6. 当前前两节课你应该掌握到什么程度

如果现在能明白下面这两句话，就说明前两节课已经过关：

#### 第一句

```text
第一节课讲的是：训练脚本如何把配置、模型、数据、loss、优化器串成一条训练主链。
```

#### 第二句

```text
第二节课讲的是：DSECDatasetLite 如何从 saved_flow_data 里读出 chunk、mask、label。
```

### 7. 当前最容易混淆的两个概念

#### 混淆 1：dataset 和 dataloader

- dataset：定义“单个样本怎么读”
- dataloader：定义“怎么一批一批地取样本”

#### 混淆 2：预处理 和 读取预处理结果

- 预处理：把原始数据变成 `saved_flow_data`
- 读取预处理结果：把 `saved_flow_data` 读成训练输入

### 8. 当前复习后的最简总结

```text
第一节：
  讲训练主链

第二节：
  讲 DSECDatasetLite 怎么把 saved_flow_data 读成 chunk / mask / label

关键澄清：
  DSECDatasetLite 不是生成预处理数据
  它是读取预处理数据
```

---

## 第三节课前置补课：原始数据从头到尾的数据流

这一节专门回答一个关键问题：

```text
预处理到底在哪里？
原始 DSEC 数据是怎么一步一步变成模型输入的？
```

### 1. 先给出整条总数据流

最完整的数据流可以先写成：

```text
原始 DSEC 数据
  -> 离线预处理脚本
  -> saved_flow_data
  -> DSECDatasetLite
  -> DataLoader
  -> train 脚本里的输入整理
  -> 模型 forward
  -> loss
```

如果把它翻成更白的话，就是：

```text
原始数据先被提前加工
加工结果存盘
训练时再把这些加工结果读出来
然后再喂给模型
```

### 2. 第一个阶段：原始 DSEC 数据是什么

原始数据不是直接可训练的 `.npy`。

它主要包含两大类：

#### A. 原始事件数据

例如：

```text
train_events/zurich_city_09_a/events/left/events.h5
train_events/zurich_city_09_a/events/left/rectify_map.h5
```

这里面存的是：
- 事件流
- 相机校正映射

#### B. 原始光流标签

例如：

```text
train_optical_flow/zurich_city_09_a/flow/forward/*.png
train_optical_flow/zurich_city_09_a/flow/forward_timestamps.txt
```

这里面存的是：
- DSEC 官方给的光流 PNG
- 每张光流对应的时间区间

这一阶段的数据，还不能直接送进 baseline 训练。

### 3. 第二个阶段：离线预处理在哪里

这一步才是真正的“预处理”。

你可以看两个脚本：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- [prepare_dsec_full.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_full.py)

它们的职责是：

```text
把原始 DSEC 数据
加工成训练时容易读取的 saved_flow_data
```

也就是说，预处理不在 `DSECDatasetLite` 里，
而在这些 `prepare_dsec_*.py` 脚本里。

### 4. 预处理脚本具体做了哪些事

以 [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py) 为例，它主要做 3 件事。

#### 第 1 件事：把光流 PNG 解码成 `gt_tensors`

脚本里有：

```python
decode_flow_png(...)
```

它会把原始 DSEC 的 16-bit PNG 解码成：

- `flow`
- `valid`

也就是：

```text
gt_tensors/样本名.npy
mask_tensors/样本名.npy
```

其中：
- `flow` 变成训练标签
- `valid` 变成 mask

#### 第 2 件事：把原始事件流切成 voxel

脚本里有：

```python
make_voxel_chunk(...)
```

它做的事情是：

1. 根据光流时间窗 `[t_beg, t_end]`，从 `events.h5` 里取这一段事件
2. 用 `rectify_map.h5` 做坐标校正
3. 把这段事件累计成 voxel grid

最后得到：

```text
event_tensors/10bins/left/sequence/样本名.npy
```

这就是后面 dataset 读取的 `chunk` 来源。

#### 第 3 件事：生成训练/验证样本名单

脚本里有：

```python
write_split_csvs(...)
```

它会写：

```text
sequence_lists/train_split_seq.csv
sequence_lists/valid_split_seq.csv
```

这样训练时 dataset 才知道：
- 哪些样本属于训练集
- 哪些样本属于验证集

### 5. 预处理完成后，会得到什么

预处理结果统一放在：

```text
saved_flow_data/
```

里面最关键的是：

```text
saved_flow_data/
  gt_tensors/
  mask_tensors/
  event_tensors/
  sequence_lists/
```

所以你可以把 `saved_flow_data` 理解成：

**训练专用中间格式。**

也就是：
- 不是原始数据
- 也不是模型输出
- 而是介于两者之间的训练缓存格式

### 6. 第三个阶段：`DSECDatasetLite` 做什么

这一步不再生成任何新数据。

它只是读取：

```text
saved_flow_data
```

也就是说：

```text
prepare_dsec_*.py 负责“做预处理”
DSECDatasetLite 负责“读预处理结果”
```

这一步里：

- 从 `gt_tensors` 读 `label`
- 从 `mask_tensors` 读 `mask`
- 从 `event_tensors/...` 读 `chunk`

然后返回：

```python
chunk, mask, label
```

### 7. 第四个阶段：DataLoader 做什么

这一步不是新增数据内容，而是“打包样本”。

它负责：
- 一次取一个 batch
- 决定打乱顺序还是不打乱
- 决定并行读取几个样本

所以：

- dataset 定义“单个样本怎么读”
- dataloader 定义“一批样本怎么取”

### 8. 第五个阶段：train 脚本在模型前还会再处理一次输入

这个阶段很容易被忽略。

虽然 dataset 已经给了 `chunk`，
但 train 脚本里还会继续做：

- polarity split
- 归一化
- 必要时阈值化

所以：

```text
dataset 输出的 chunk
!=
模型最终看到的输入
```

这是当前必须牢牢记住的一点。

### 9. 第六个阶段：模型 forward

到这一步，输入才真正进入模型。

然后模型输出：

```text
pred_list["flow"]
```

再和：
- `label`
- `mask`

一起送进 loss。

### 10. 现在把整条数据流再压缩成一句话

```text
原始 events.h5 和 flow png
  -> 被 prepare_dsec_*.py 预处理成 saved_flow_data
  -> DSECDatasetLite 从 saved_flow_data 读取 chunk / mask / label
  -> DataLoader 打包成 batch
  -> train 脚本继续整理输入
  -> 模型 forward
  -> loss
```

### 11. 当前最重要的概念澄清

如果用户以后再问：

```text
DSECDatasetLite 算不算预处理？
```

最准确的回答应该是：

**不算真正的预处理。**

更准确地说：

- `prepare_dsec_single_sequence.py` / `prepare_dsec_full.py`：预处理器
- `DSECDatasetLite`：预处理结果读取器

### 12. 当前阶段你应该能回答的 3 个问题

#### 问题 1

原始 DSEC 数据直接就能训练吗？

标准答案：
- 不能
- 必须先经过预处理，变成 `saved_flow_data`

#### 问题 2

真正做预处理的是谁？

标准答案：
- `prepare_dsec_single_sequence.py`
- `prepare_dsec_full.py`

#### 问题 3

`DSECDatasetLite` 的职责是什么？

标准答案：
- 从 `saved_flow_data` 中把 `chunk / mask / label` 读出来

---

## 第三节课：`MS_SpikingformerFlowNet_en4` 最外层模型由哪些大模块组成

这一节的目标不是马上看懂 attention 细节，
而是先回答一个更重要的问题：

```text
baseline 模型最外层到底由哪些大模块组成？
```

如果这一步不清楚，后面做优化时就会很容易迷路：
- 改 attention 却跑到训练器里找
- 改输入稀疏却跑到 decoder 里找
- 改输出融合却跑到 dataset 里找

所以第三节课的核心是：

**先看“机器外壳”，再看“机器内部齿轮”。**

### 1. baseline 顶层模型名字是什么

当前 baseline 配置里的模型类名是：

```text
MS_SpikingformerFlowNet_en4
```

它定义在：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

你可以把这个类先理解成：

**整个光流网络的最外层壳子。**

### 2. 顶层模型最简单的结构图

把这个模型先压缩成最外层结构，可以写成：

```text
输入 chunk
  -> encoder
  -> residual blocks
  -> decoder
  -> 多尺度 prediction head
  -> 输出 flow list
```

也就是说，它不是“只有 transformer”。

它本质上是：

```text
SNN encoder + 中间残差块 + U-Net 式 decoder + 多尺度输出头
```

### 3. 顶层 `FlowNet` 实际上主要是在“包一层”

在 [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py) 里，
`SpikingformerFlowNet.forward()` 的核心逻辑很简单：

1. 取输入 `x`
2. 调用 `self.sttmultires_unet.forward(x)`
3. 拿到多尺度 `multires_flow`
4. 对每个时间维结果求和
5. 上采样回原图大小
6. 返回：

```python
{"flow": flow_list, "attn": attns}
```

所以顶层 `FlowNet` 更像是：

**总出口和总包装器。**

真正的大部分特征提取工作，是下面那个 `MultiResUNet` 做的。

### 4. 真正的大骨架在 `Spikingformer_MultiResUNet`

这个类也在：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

你可以把它理解成：

**baseline 的主机身。**

它里面真正包含了：
- encoder
- residual blocks
- decoder
- prediction layers

也就是说，后面你想做结构优化，大多数时候真正要改的，是这个层次下面的东西，不是最外层 `FlowNet` 壳子。

### 5. 这个 `MultiResUNet` 不是从零写出来的，而是“继承+替换”

`Spikingformer_MultiResUNet` 继承自：

- [SpikingMultiResUNet](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)

这里非常关键。

意思是：

```text
SNN_models.py 里先定义了一个通用的 Spiking U-Net 框架
Spiking_STSwinNet.py 再把 encoder 换成了 spike-former encoder
```

所以理解 baseline 最外层结构时，你可以这样记：

```text
先有一个通用 spiking U-Net 骨架
再把其中的 encoder 升级成 Swin/Spiking Transformer encoder
```

### 6. `SpikingMultiResUNet` 这个通用骨架负责什么

在 [SNN_models.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py) 里，
`SpikingMultiResUNet` 负责定义 U-Net 骨架。

它主要负责：

#### A. encoder 列表

```text
一层一层往下提特征
同时降低空间分辨率、增加通道数
```

#### B. residual blocks

```text
在最深层继续做特征处理
```

#### C. decoder 列表

```text
一层一层往上恢复空间分辨率
并和 encoder 的 skip features 融合
```

#### D. prediction layers

```text
每个 decoder 层都能输出一个光流预测
```

这就是为什么 baseline 最终输出不是单尺度，而是多尺度 `flow list`。

### 7. 先把 encoder / residual / decoder / prediction 想成 4 个工位

你可以先把整个模型想成一个工厂流水线：

#### 工位 1：encoder

职责：
- 从输入里提取越来越抽象的特征
- 降低分辨率
- 增加通道数

#### 工位 2：residual blocks

职责：
- 在最深层继续加工特征
- 相当于“深加工区”

#### 工位 3：decoder

职责：
- 逐步放大特征图
- 把高层特征和早期细节融合起来

#### 工位 4：prediction head

职责：
- 在不同尺度上输出光流预测

### 8. `Spikingformer_MultiResUNet` 和普通 `SpikingMultiResUNet` 的真正区别

最关键的区别在 encoder。

普通骨架里，encoder 是卷积式的 spiking encoder。

而在 `Spikingformer_MultiResUNet` 里，encoder 被换成了：

```text
spiking_former_encoder
```

这意味着：

```text
decoder 还是 decoder
residual 还是 residual
prediction head 还是 prediction head
真正变“像 Transformer”的，主要是 encoder 这一段
```

这就是为什么以后如果你想做：
- attention 改进
- window 机制改进
- token/head 稀疏

第一反应应该是：

**先去 encoder 里找。**

### 9. `spiking_former_encoder` 又是什么

这个类在：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

它本身不是整个 U-Net，
而是单独的一块：

**Transformer 风格的编码器。**

它内部最关键的是：

```python
self.swin3d = ...
```

也就是说，真正的 Swin/Window Attention 主干还在更深一层。

所以层级关系是：

```text
MS_SpikingformerFlowNet_en4
  -> Spikingformer_MultiResUNet
    -> spiking_former_encoder
      -> SwinTransformer3D
```

### 10. 这就是为什么“改 attention”不该从训练器开始找

因为训练器只负责：
- 组织流程
- 喂数据
- 算 loss

真正的 attention 主战场在更深层：

```text
FlowNet 外壳
  -> MultiResUNet 主机身
    -> encoder
      -> Swin block / attention
```

所以如果以后你想优化：
- QKV
- window attention
- token 稀疏
- head pruning

你首先该找的是：

```text
encoder -> Swin block
```

不是：

```text
train_flow_parallel_supervised_SNN.py
```

### 11. 现在把 baseline 模型再压缩成一句话

```text
baseline 是一个“用 Swin/Spiking Transformer 做 encoder、用卷积式 U-Net 做 decoder、最后做多尺度光流预测”的网络。
```

### 12. 这一节结束后你应该能回答的 4 个问题

#### 问题 1

`MS_SpikingformerFlowNet_en4` 是不是整个 baseline 的最外层壳子？

标准答案：
- 是

#### 问题 2

真正承载 encoder / residual / decoder / prediction 的主机身是谁？

标准答案：
- `Spikingformer_MultiResUNet`

#### 问题 3

`Spikingformer_MultiResUNet` 和通用 `SpikingMultiResUNet` 的关键区别主要在哪？

标准答案：
- 在 encoder 被换成了 `spiking_former_encoder`

#### 问题 4

如果以后你想改 attention，第一反应应该先去哪里找？

标准答案：
- 先去 encoder，再往下看 Swin block

---

## 第三节课补充：更清晰的“分支调用图”

如果用户仍然觉得“知道大概做什么，但不知道程序具体怎么走”，
那就不要再只讲概念，而要改成：

**按程序执行顺序，把调用链画成树。**

### 1. 训练主调用树

```text
train_flow_parallel_supervised_SNN.py
└─ main
   ├─ argparse 读取命令行参数
   ├─ YAMLParser(配置文件)
   │  ├─ 读取 YAML
   │  ├─ 生成 config
   │  └─ 生成 device
   └─ train(args, config_parser)
      ├─ config = config_parser.config
      ├─ device = config_parser.device
      ├─ config = combine_entries(config)
      ├─ model = MS_SpikingformerFlowNet_en4(...)
      ├─ loss_function = flow_loss_supervised(...)
      ├─ train_dataset = DSECDatasetLite(..., file_list="train")
      ├─ valid_dataset = DSECDatasetLite(..., file_list="valid")
      ├─ train_dataloader = DataLoader(train_dataset)
      ├─ valid_dataloader = DataLoader(valid_dataset)
      └─ epoch 循环
         └─ batch 循环
            ├─ 取出 chunk, mask, label
            ├─ 处理 chunk
            ├─ pred_list = model(chunk)
            ├─ curr_loss = loss_function(pred, label, mask)
            ├─ backward
            └─ optimizer.step()
```

### 2. 数据调用树

```text
原始 DSEC 数据
└─ prepare_dsec_single_sequence.py / prepare_dsec_full.py
   ├─ decode_flow_png(...)
   │  ├─ 读取原始光流 PNG
   │  ├─ 解码成 flow
   │  └─ 解码成 valid mask
   ├─ make_voxel_chunk(...)
   │  ├─ 从 events.h5 取一段事件
   │  ├─ 用 rectify_map 做校正
   │  └─ 转成 voxel
   └─ 写出 saved_flow_data
      ├─ gt_tensors/
      ├─ mask_tensors/
      ├─ event_tensors/
      └─ sequence_lists/

saved_flow_data
└─ DSECDatasetLite.__getitem__(idx)
   ├─ 从 csv 取样本名
   ├─ gt_tensors 读 label
   ├─ mask_tensors 读 mask
   ├─ event_tensors 读 chunk
   └─ return chunk, mask, label

DataLoader
└─ 把多个 chunk, mask, label 打成一个 batch
```

### 3. 模型调用树

```text
MS_SpikingformerFlowNet_en4
└─ forward(x)
   ├─ multires_flow = self.sttmultires_unet.forward(x)
   ├─ 对每个尺度沿时间维求和
   ├─ 上采样回原图大小
   └─ return {"flow": flow_list, "attn": attns}

self.sttmultires_unet
└─ Spikingformer_MultiResUNet.forward(x)
   ├─ blocks = self.encoders(x)
   ├─ x = blocks[-1]
   ├─ x 经过 resblocks
   ├─ 逐层 decoder
   │  ├─ 和 skip features 融合
   │  ├─ decoder(x)
   │  └─ pred(x)
   └─ return predictions

self.encoders
└─ spiking_former_encoder.forward(inputs)
   ├─ features = self.swin3d(inputs)
   └─ return outs

self.swin3d
└─ MS_Spiking_SwinTransformer3D_v2 / Spiking_SwinTransformer3D_v2
   ├─ stage 1
   ├─ stage 2
   ├─ stage 3
   └─ stage 4
      └─ 每个 stage 里有多个 Swin block
         └─ 每个 block 里有 window attention + MLP
```

### 4. 当前最实用的定位规则

如果用户后面要改某个东西，先按下面的树找：

```text
改原始数据变 voxel
  -> prepare_dsec_*.py

改训练时读取样本
  -> DSECDatasetLite

改模型最外层结构
  -> Spiking_STSwinNet.py

改 encoder / attention / token / window
  -> Spiking_swin_transformer3D.py

改 decoder / 多尺度预测
  -> SNN_models.py + Spiking_STSwinNet.py

改 loss
  -> flow_supervised.py
```

### 5. 当前阶段的关键提醒

用户现在不需要一次把整棵树每一层都看懂。

当前真正需要做到的是：

```text
我知道现在程序走到树的哪一层了
我知道我要改的东西应该落在哪个分支
```

---

## 学习方法调整：从“讲系统”改成“跟踪一个样本”

用户反馈当前困难点非常明确：

```text
知道它大概在干什么
但还是不知道具体怎么走
也不知道以后该从哪里改 baseline
```

这说明当前“按模块讲职责”的方法，对基础还比较薄弱的学习者不够友好。

后续教学改用新的方法：

### 新方法

不再同时讲整个系统。

改成：

```text
只盯一个样本
从磁盘上的一个文件名开始
一路跟到 loss
```

例如始终盯住：

```text
zurich_city_09_a_0001.npy
```

然后只回答下面这些最具体的问题：

1. 这个文件名先出现在什么地方？
2. 程序拿着这个文件名去哪个目录找什么？
3. 它什么时候变成 `chunk`？
4. `chunk` 什么时候进模型？
5. 模型什么时候吐出 `pred`？
6. `pred` 什么时候和 `label` 比较？

### 为什么这种方法更适合当前阶段

因为它不要求用户同时理解：
- YAML
- dataset
- dataloader
- encoder
- decoder
- attention
- loss

而是只要求用户顺着“一条线”往前看。

### 当前阶段的学习目标调整

不是“看懂整个 baseline”。

而是先做到：

```text
我能跟着一个样本
说清楚它从文件名
一路走到 loss 的过程
```

只要这一步吃透，后面再谈：
- attention 改动
- token 稀疏
- window 剪枝

才不会完全失去方向。

---

## 第四节课：只跟踪一个样本，从 `train_split_seq.csv` 到 `chunk / mask / label`

这一节只跟踪一个样本：

```text
zurich_city_09_a_0001.npy
```

目标不是看懂整个系统，而是只回答：

```text
这个样本是怎么一步一步被 DSECDatasetLite 读出来的？
```

### 1. 第一步：样本最开始出现在哪里

最开始，这个样本不是先出现在模型里，
也不是先出现在训练器里。

它最开始出现在：

```text
train_split_seq.csv
```

也就是说，程序最先知道的不是“一个张量”，
而只是“一个文件名”。

例如：

```text
zurich_city_09_a_0001.npy
```

这一行的意思很简单：

```text
训练集里有这个样本
```

### 2. 第二步：`DSECDatasetLite` 先拿到什么

在 `__getitem__(idx)` 里，
它做的第一件事不是读张量，
而是先去 csv 里拿文件名。

也就是先得到：

```text
target_file_1 = "zurich_city_09_a_0001.npy"
```

所以当前这一刻，程序手里还只有一个字符串。

不是图片，
不是 tensor，
不是 flow。

只是一个文件名。

### 3. 第三步：拿着这个文件名去读 `mask`

程序会去：

```text
mask_tensors/zurich_city_09_a_0001.npy
```

把它读出来，得到：

```text
mask
```

你现在可以把 `mask` 理解成：

```text
哪些像素位置是有效的
```

### 4. 第四步：拿着同一个文件名去读 `label`

程序还会去：

```text
gt_tensors/zurich_city_09_a_0001.npy
```

把它读出来，得到：

```text
label
```

你现在可以把 `label` 理解成：

```text
标准答案
也就是真实光流
```

### 5. 第五步：为什么还要拆出 `zurich_city_09_a`

程序接下来不会立刻去读 `chunk`。

它先做一件事：

从文件名里拆出序列名。

也就是：

```text
zurich_city_09_a_0001.npy
-> zurich_city_09_a
```

为什么非要拆？

因为输入事件张量不是平铺放在一个目录里的，
而是按序列分文件夹存。

也就是说，程序必须先知道：

```text
这个样本属于哪个 sequence
```

才能找到输入文件。

### 6. 第六步：拿着“序列名 + 文件名”去读 `chunk`

现在程序已经知道了两件事：

- 序列名：`zurich_city_09_a`
- 文件名：`zurich_city_09_a_0001.npy`

于是它去这里读：

```text
event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy
```

读出来的就是：

```text
chunk
```

你当前可以先把 `chunk` 理解成：

```text
输入
也就是预处理好的事件体表示
```

### 7. 第七步：到这里 `DSECDatasetLite` 才真正完成工作

它最后返回：

```python
return chunk, mask, label
```

所以 `DSECDatasetLite` 做的事情，用最白的话说就是：

```text
先从 csv 拿一个文件名
再去三个地方把这个样本配套的三样东西读出来
最后一起交给训练器
```

### 8. 这一节里一定要记住的对应关系

如果样本名是：

```text
zurich_city_09_a_0001.npy
```

那么：

```text
mask  <- mask_tensors/zurich_city_09_a_0001.npy
label <- gt_tensors/zurich_city_09_a_0001.npy
chunk <- event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy
```

### 9. 这一节你最容易弄混的地方

不要把下面两件事混在一起：

#### A. 文件名本身

```text
zurich_city_09_a_0001.npy
```

它只是一个“索引钥匙”。

#### B. 真正读出来的数据

- `chunk`
- `mask`
- `label`

它们才是真正的内容。

也就是说：

```text
文件名不是数据本身
文件名只是用来找到数据的钥匙
```

### 10. 这一节结束时，你应该能用一句话复述

```text
一个样本先在 csv 里以文件名出现，
然后 DSECDatasetLite 用这个文件名去读 mask、label，
再拆出 sequence 名去读 chunk，
最后返回 chunk, mask, label。
```

### 11. 第四节课后的检查问题

#### 问题 1

`zurich_city_09_a_0001.npy` 最开始是先出现在模型里，还是先出现在 csv 里？

标准答案：
- 先出现在 csv 里

#### 问题 2

程序先拿到的是“张量”，还是“文件名字符串”？

标准答案：
- 先拿到文件名字符串

#### 问题 3

为什么程序要从文件名里拆出 `zurich_city_09_a`？

标准答案：
- 因为 `chunk` 是按 sequence 分子文件夹存放的

#### 问题 4

`DSECDatasetLite` 最后返回哪三个东西？

标准答案：
- `chunk, mask, label`

---

## 第四节课后追问：`mask` 到底是什么、从哪来、有什么用

用户当前已经能回答：

- 样本先出现在 csv
- 程序先拿到文件名
- 拆出 sequence 名是为了去读 `chunk`
- 最后返回 `chunk, mask, label`

这说明第四节主线已经基本跟住了。

当前新的关键问题是：

```text
mask 到底是什么？
它是预处理得到的，还是原数据集直接就有的？
它到底有什么用？
```

### 1. 最白的话：`mask` 是“哪些位置的答案可信”

`mask` 可以先理解成一张“有效区域地图”。

它告诉程序：

```text
这张图里哪些像素位置的光流标签可以用
哪些位置不要拿来算误差
```

所以 `mask` 不是输入，
也不是模型预测，
它是一个“告诉你哪里可以算分、哪里不要算分”的标记图。

### 2. `mask` 是原始数据直接给的吗

严格来说：

**不是以 `.npy` 这个训练格式直接给的。**

原始 DSEC 光流标签在 PNG 里。

在预处理脚本里：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)

有一个函数：

```python
decode_flow_png(...)
```

它会把原始 DSEC 的 16-bit flow PNG 解码成两样东西：

- `flow`
- `valid`

这里的 `valid`，后面就被保存成：

```text
mask_tensors/样本名.npy
```

所以更准确的说法是：

```text
mask 的“原始来源”在官方 flow PNG 里
mask 的“训练格式”是在预处理时被解码并保存成 .npy 的
```

### 3. 为什么会有 `mask`

因为不是每一个像素位置都有可靠的光流标签。

有些位置：
- 官方没有提供有效光流
- 或者该位置不该拿来监督训练

所以不能让模型在整张图的每个像素都一视同仁地算 loss。

否则会出现一种问题：

```text
模型明明在无效区域预测错了
但这些区域本来就不应该参与评分
结果 loss 被错误放大
```

这就是 `mask` 存在的原因：

**只在有效像素上算误差。**

### 4. `mask` 在训练里具体怎么用

在 loss 里：

- [flow_supervised.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/loss/flow_supervised.py)

程序会做这种事情：

```text
先算每个像素的误差
再用 mask 把无效位置乘掉
最后只对有效像素求平均
```

所以 `mask` 的作用不是改输入，
而是：

**控制 loss 只在有效位置上计算。**

### 5. 当前最简单的记忆方式

你现在可以这样记：

```text
label 是标准答案
mask 是“哪些答案位置有效”
```

再压缩一点：

```text
label 告诉你答案是什么
mask 告诉你这份答案哪些地方能算分
```

### 6. 当前这一步最关键的结论

如果以后用户再问：

```text
mask 是预处理来的，还是原始数据集自带的？
```

最准确的回答是：

**两者都沾边。**

更完整地说：

- 原始官方 flow PNG 里本身就带“有效性信息”
- 预处理脚本把这个有效性信息解码出来
- 再保存成训练时读取的 `mask_tensors/*.npy`

所以：

```text
mask 不是凭空生成的
它来源于原始标签
但它是在预处理阶段被提取并保存成训练格式的
```

---

## 预处理究竟干了什么：原始数据和预处理后数据的区别

用户当前最想彻底弄明白的是：

```text
预处理到底干了什么？
预处理后的数据和原始数据到底有什么区别？
```

### 1. 最白的话：预处理就是“把难直接训练的数据，提前整理成训练方便的数据”

原始 DSEC 数据不是训练器一拿来就能直接用的格式。

原始数据更像：
- 原始事件流水
- 原始标签文件
- 原始时间戳

而训练器更想要的是：
- 一条样本对应一个输入文件
- 一条样本对应一个标签文件
- 一条样本对应一个 mask 文件

所以预处理做的本质工作就是：

```text
把原始复杂数据
提前整理成“一个样本对应一组整齐文件”的训练格式
```

### 2. 原始数据长什么样

原始数据主要是：

#### A. 原始事件流

例如：

```text
events.h5
rectify_map.h5
```

特点：
- 存的是连续事件流
- 不是按“一个训练样本一个文件”放好的

#### B. 原始光流标签

例如：

```text
flow/forward/*.png
forward_timestamps.txt
```

特点：
- 标签在 PNG 里
- 时间区间在 txt 里
- 也不是直接按训练器最喜欢的 `.npy` 结构给好的

### 3. 预处理到底做了哪几件具体事情

预处理脚本主要做 4 件事。

#### 第 1 件：把原始 flow PNG 解码成训练标签

原始标签在 PNG 里，不方便训练时每次现场解码。

所以预处理会把它解码成：

```text
gt_tensors/样本名.npy
```

也就是训练时直接可读的 `label`。

#### 第 2 件：把“有效性信息”提出来，变成 mask

原始 PNG 里还带着哪些位置有效的信息。

预处理会把它提出来，变成：

```text
mask_tensors/样本名.npy
```

这样训练时就能直接知道：
- 哪些位置可以算 loss
- 哪些位置不要算

#### 第 3 件：把连续事件流切成一个个 voxel 样本

原始事件流是连续的，不是天然按“一个样本”切好的。

预处理会：

1. 根据某个光流标签对应的时间窗
2. 从连续事件流里截出这一小段
3. 做 rectification
4. 累积成 voxel grid

然后保存成：

```text
event_tensors/.../样本名.npy
```

也就是说：

```text
原始事件流
-> 变成一个个离散的训练样本输入文件
```

#### 第 4 件：生成样本名单 csv

预处理还会写：

```text
train_split_seq.csv
valid_split_seq.csv
```

这样训练时程序才知道：
- 哪些样本拿来训练
- 哪些样本拿来验证

### 4. 所以原始数据和预处理后数据的最大区别是什么

可以直接对照记：

#### 原始数据

特点：
- 连续事件流
- 标签还在 PNG 里
- 时间信息单独放着
- 训练时不方便直接一条条拿来用

#### 预处理后数据

特点：
- 一个样本一个输入 `.npy`
- 一个样本一个标签 `.npy`
- 一个样本一个 mask `.npy`
- 还有现成的 train/valid 样本列表

所以一句话：

```text
原始数据更接近“传感器/官方存储格式”
预处理后数据更接近“训练器喜欢的样本格式”
```

### 5. 为什么要预处理，而不是训练时现场做

因为如果不预处理，训练时每取一个样本都要现场做很多事：
- 读 h5
- 查时间窗
- 截事件
- 做 rectification
- 做 voxelization
- 解码 PNG

这样会很慢，也很乱。

所以工程上通常会先离线做好，变成 `saved_flow_data`。

### 6. 当前阶段你最该记住的一句话

```text
预处理不是“改数据内容的意义”
而是“把原始数据整理成训练时方便直接读取的样本格式”
```

---

## 第五节课：`chunk, mask, label` 被训练器拿到后，`chunk` 怎么变成模型真正看到的输入

这一节继续只盯一个样本，不讲整个系统。

前面第四节讲到：

```text
DSECDatasetLite 最后返回 chunk, mask, label
```

现在第五节只回答：

```text
这三个东西被 train 脚本拿到以后，尤其是 chunk，会发生什么？
```

### 1. 训练器拿到 batch 时，先得到什么

在训练循环里，程序会写：

```python
for chunk, mask, label in train_dataloader:
```

这说明：

现在训练器手里已经拿到了：
- `chunk`
- `mask`
- `label`

但是注意：

**这时候的 `chunk` 还不是模型最终看到的输入。**

### 2. 第一步：先把它们搬到 device

训练器会先做：

```text
chunk -> GPU / CPU
label -> GPU / CPU
mask  -> GPU / CPU
```

这一步只是“搬位置”，还没有改变它们的含义。

### 3. 第二步：可能做数据增强

如果配置里开了 augmentation，
训练器会先对：

- `chunk`
- `label`
- `mask`

一起做裁剪或翻转。

当前你先把它理解成：

```text
输入、答案、有效区域要一起同步变换
```

### 4. 第三步：如果编码方式是 voxel，就开始整理 `chunk`

当前 baseline 默认是：

```text
encoding = voxel
```

所以训练器会进入 voxel 这条分支。

这一步里最重要的是：

#### 如果启用了 polarity

训练器会把 `chunk` 拆成：

- 正事件部分 `pos`
- 负事件部分 `neg`

然后重新拼起来。

也就是说：

```text
原来的 chunk
-> 被拆成正负两部分
-> 再组合成模型更方便处理的输入形状
```

### 5. 第四步：对输入做归一化

如果配置要求，
训练器还会继续对 `chunk` 做归一化。

例如：
- min-max
- std

你现在先不用记具体公式，
只记住：

```text
训练器在模型前还会继续整理输入数值范围
```

### 6. 第五步：必要时阈值化成 spike 输入

如果配置里设置了：

```text
spike_th
```

训练器还会继续把输入做阈值化。

也就是：

```text
大于阈值 -> 1
小于阈值 -> 0
```

### 7. 第六步：到这里，`chunk` 才真正进模型

然后训练器才调用：

```python
pred_list = model(chunk)
```

这句话很重要，说明：

```text
dataset 读出来的 chunk
并不是马上直接喂进模型
而是先经过训练器的输入整理
最后才变成模型真正看到的输入
```

### 8. 第七步：模型输出什么

模型输出的是：

```python
pred_list
```

然后训练器会取：

```python
pred = pred_list["flow"]
```

也就是说：

- `pred_list` 是一个字典
- 里面最重要的是 `"flow"`

### 9. 第八步：什么时候用到 `label` 和 `mask`

模型输出以后，
训练器会调用 loss：

```python
curr_loss = loss_function(pred, label, mask, ...)
```

这说明：

- `chunk`：拿去做模型输入
- `label`：拿去和预测结果比较
- `mask`：拿去控制哪些位置参与 loss

### 10. 第五节课最重要的一句话

```text
DSECDatasetLite 读出的 chunk 还不是模型最终看到的输入；
训练器会先对它做正负极性拆分、归一化、必要时阈值化，
然后才送进模型。
```

### 11. 第五节课后的检查问题

#### 问题 1

`DSECDatasetLite` 读出的 `chunk`，是不是马上直接进模型？

标准答案：
- 不是

#### 问题 2

训练器在模型前会不会继续处理 `chunk`？

标准答案：
- 会

#### 问题 3

`label` 和 `mask` 主要是在什么时候用上？

标准答案：
- 在 loss 计算时用上

---

## 第五节课后检查：用户回答与纠正

用户当前回答：

```text
chunk 会按照设置决定是否双极化、归一化、脉冲化，会继续处理，
label 和 mask 在 loss 时使用。
```

这个回答已经抓住了第五节课最关键的内容，整体是对的。

### 1. 当前回答里已经答对的部分

#### A. `chunk` 不会直接原样进模型

用户已经理解：

```text
chunk 还会继续被处理
```

这正是第五节课最关键的一点。

#### B. 训练器会根据配置处理 `chunk`

用户已经理解：

- 双极性拆分（更准确地说是正负极性拆分）
- 归一化
- 脉冲化 / 阈值化

这些都是训练器在模型前会做的输入整理。

#### C. `label` 和 `mask` 主要在 loss 阶段使用

这也是正确的。

### 2. 一个小纠正：更准确的说法

用户说的：

```text
双极化
```

更准确一点，可以说成：

```text
正负极性拆分
```

也就是：

```text
把输入中的正事件和负事件拆开再组合
```

### 3. 当前第五节课最应该记住的最终版本

```text
dataset 读出来的 chunk 还不是模型最终输入；
训练器会根据配置继续对 chunk 做正负极性拆分、归一化、必要时阈值化，
然后 chunk 才真正进模型；
label 和 mask 主要在 loss 阶段使用。
```

### 4. 当前阶段结论

用户已经基本过掉第五节课。

下一步可以进入：

```text
模型真正开始 forward 以后，
最外层是怎么一步一步往 encoder / decoder / pred 走的
```

---

## 第六节课：`chunk` 进模型后，最外层 forward 是怎么走到 encoder / decoder / prediction 的

这一节只回答一个问题：

```text
chunk 终于进模型以后，程序接下来到底怎么走？
```

这次仍然不讲复杂 attention 公式，
只讲“最外层 forward 路线”。

### 1. 训练器把谁真正送进模型

在训练器里，前面做完：
- 正负极性拆分
- 归一化
- 必要时阈值化

之后，才会执行：

```python
pred_list = model(chunk)
```

这说明：

```text
现在这个 chunk
才是模型真正收到的输入
```

### 2. 这个 `model` 当前是谁

在 baseline 里，
当前 `model` 一般就是：

```text
MS_SpikingformerFlowNet_en4
```

你可以把它先理解成：

**模型总入口。**

也就是说，现在程序真的进入模型了，第一站先到这里。

### 3. `MS_SpikingformerFlowNet_en4` 自己做了很多事吗

严格说：

**没有特别多。**

它更像一个总调度器。

它最重要的一步是：

```text
把输入 x 交给 self.sttmultires_unet
```

也就是：

```python
multires_flow = self.sttmultires_unet.forward(x)
```

所以你要先记住：

```text
模型总入口不是最终主战场
真正的大部分结构计算是在 sttmultires_unet 里面
```

### 4. `self.sttmultires_unet` 是谁

它可以先理解成：

**模型主机身。**

也就是前面第三节课讲过的：

```text
Spikingformer_MultiResUNet
```

现在程序走到这里时，真正开始做下面这些事：

```text
输入
  -> encoder
  -> residual blocks
  -> decoder
  -> prediction
```

### 5. 第一步：先走 encoder

在 `Spikingformer_MultiResUNet.forward(x)` 里，最开始会做：

```python
blocks = self.encoders(x)
```

这句话你先翻译成最白的话：

```text
先把输入送进编码器
得到一组不同层次的特征
```

这里的 `blocks`，
你可以暂时理解成：

```text
encoder 每一层输出的中间特征
```

它们后面还有用，因为 decoder 会拿这些早期特征做 skip connection。

### 6. 第二步：拿 encoder 最深层输出，继续往下走

然后程序会做：

```python
x = blocks[-1]
```

意思就是：

```text
先拿 encoder 最后、最深的那层特征
把它当成后续主干输入
```

为什么拿最后一层？

因为最后一层通常是：
- 最深层
- 最抽象
- 语义信息最强

### 7. 第三步：经过 residual blocks

接着程序会做：

```python
for resblock in self.resblocks:
    x = resblock(x)
```

你现在先把它理解成：

```text
在最深层特征上继续做一轮深加工
```

所以现在的路线是：

```text
输入
  -> encoder
  -> 最深层特征
  -> residual blocks
```

### 8. 第四步：进入 decoder

然后程序开始进入 decoder 循环。

这里的核心逻辑是：

```text
一层一层往回恢复空间结构
同时把 encoder 早期特征接回来
```

代码里你会看到两类操作：

#### A. 和 skip feature 融合

```python
x = self.skip_ftn(x, blocks[...], dim=2)
```

你现在不用管 `skip_ftn` 具体怎么实现，
只要知道它在做：

```text
把 decoder 当前特征
和 encoder 之前存下来的特征
拼/加 回来
```

这样做的目的，就是把早期细节带回来。

#### B. 经过 decoder 层

```python
x = decoder(x)
```

意思就是：

```text
把当前特征往输出方向再推进一层
```

### 9. 第五步：每一层 decoder 后都做 prediction

然后程序会做：

```python
pred_out = pred(x)
predictions.append(pred_out)
```

这说明：

```text
不是只有最后一层才输出预测
而是每一层 decoder 都会吐一个 prediction
```

所以后面 `Spikingformer_MultiResUNet.forward()` 返回的不是单个预测，
而是：

```text
predictions
```

也就是多尺度预测列表。

### 10. 第六步：返回到最外层 `FlowNet`

刚才 `sttmultires_unet.forward(x)` 返回的是：

```text
multires_flow
```

也就是一组多尺度预测。

回到最外层 `SpikingformerFlowNet.forward()` 之后，
它还会继续做两件事：

#### A. 沿时间维求和

```python
flow = torch.sum(flow, dim=0)
```

你现在先简单理解成：

```text
把时间维上的结果汇总起来
```

#### B. 上采样回原图尺寸

```python
torch.nn.functional.interpolate(...)
```

意思就是：

```text
把不同尺度的预测结果都拉回原始图像大小
```

这样后面训练和评估会更方便。

### 11. 第七步：最外层最终返回什么

最后最外层返回：

```python
{"flow": flow_list, "attn": attns}
```

所以训练器看到的：

```python
pred_list = model(chunk)
pred = pred_list["flow"]
```

现在你应该能理解了：

```text
model(chunk)
先得到一个字典
字典里最重要的是 "flow"
"flow" 里面是一组多尺度预测
```

### 12. 把这条路线压缩成一句话

```text
chunk 进模型
  -> 先交给 sttmultires_unet
  -> 先走 encoder
  -> 再走最深层 residual blocks
  -> 再逐层 decoder
  -> 每层 decoder 都输出 prediction
  -> 返回最外层
  -> 最外层再把多尺度结果整理成 flow_list
```

### 13. 第六节课最应该记住的 4 个事实

#### 事实 1

`MS_SpikingformerFlowNet_en4` 是模型总入口，但不是全部细节都在这里做。

#### 事实 2

真正的大部分主干计算在：

```text
self.sttmultires_unet
```

#### 事实 3

`sttmultires_unet.forward(x)` 的主路线是：

```text
encoder -> residual blocks -> decoder -> prediction
```

#### 事实 4

模型最后给训练器的不是单个 flow，而是：

```text
flow_list
```

### 14. 第六节课后的检查问题

#### 问题 1

`chunk` 真正进模型后，第一站先到谁？

标准答案：
- `MS_SpikingformerFlowNet_en4`

#### 问题 2

`MS_SpikingformerFlowNet_en4` 接下来把输入交给谁？

标准答案：
- `self.sttmultires_unet`

#### 问题 3

`sttmultires_unet.forward(x)` 的主路线是什么？

标准答案：
- `encoder -> residual blocks -> decoder -> prediction`

#### 问题 4

模型最后返回给训练器的是单个 flow，还是一个包含 flow 的字典？

标准答案：
- 一个包含 `flow` 的字典

---

## 第六节课后检查：用户回答与纠正

用户当前回答：

```text
第一站先经过 encoder，第二个不知道，主路线是 encoder res decoder，返回字典
```

这个回答已经说明：
- 用户知道模型内部真正的重要路线是 `encoder -> res -> decoder`
- 用户知道最终返回给训练器的是字典

这说明第六节课已经吃到了大半。

### 1. 当前答对的部分

#### A. 主路线基本答对了

用户说：

```text
encoder res decoder
```

这已经抓住了核心。

更完整一点可以写成：

```text
encoder -> residual blocks -> decoder -> prediction
```

#### B. 最后返回字典，答对了

模型最后不是直接返回单个 flow tensor，
而是返回一个字典，
里面最重要的是：

```text
"flow"
```

### 2. 需要纠正的地方：第一站不是 encoder

用户说：

```text
第一站先经过 encoder
```

这不算完全错，
但层级上还差一步。

更准确的顺序是：

```text
chunk
  -> 先进入最外层模型 MS_SpikingformerFlowNet_en4
  -> 再被交给 self.sttmultires_unet
  -> 然后才真正进入 encoder
```

也就是说：

#### 第一站

```text
MS_SpikingformerFlowNet_en4
```

#### 第二站

```text
self.sttmultires_unet
```

#### 第三站

```text
encoder
```

### 3. 当前最应该记住的层级顺序

```text
chunk
  -> MS_SpikingformerFlowNet_en4
  -> self.sttmultires_unet
  -> encoder
  -> residual blocks
  -> decoder
  -> prediction
  -> 返回字典 {"flow": ...}
```

### 4. 为什么用户容易把“第一站”误答成 encoder

因为从“真正有大量计算的地方”看，
encoder 确实是最先显眼的大工位。

但从“程序调用层级”看，
在到 encoder 之前，
还要先经过：

- 最外层模型入口
- 主机身 `sttmultires_unet`

所以：

```text
从结构层级看，第一站不是 encoder
从计算感觉上看，encoder 是第一大工位
```

这两种说法容易混，
现在需要把它们区分开。

---

## 第六节课复盘加强版：把执行路线和具体文件、具体类名一一对上

这一部分不新增新概念，
只做一件事：

```text
把“chunk 进模型后的执行路线”
和“具体文件 / 具体类名”
一一对应起来
```

当前只盯这一条线：

```text
chunk
  -> MS_SpikingformerFlowNet_en4
  -> self.sttmultires_unet
  -> encoder
  -> residual blocks
  -> decoder
  -> prediction
  -> 返回 {"flow": ...}
```

### 1. 第一步：`chunk` 进到哪个具体类

当训练器执行：

```python
pred_list = model(chunk)
```

这里的 `model`，当前 baseline 里通常就是：

- `MS_SpikingformerFlowNet_en4`

所在文件：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

所以：

```text
chunk
  -> 先进入 Spiking_STSwinNet.py 里的 MS_SpikingformerFlowNet_en4
```

### 2. 第二步：`MS_SpikingformerFlowNet_en4` 自己主要把活交给谁

这一步对应的更底层类是：

- `SpikingformerFlowNet`

因为：

```text
MS_SpikingformerFlowNet_en4
是它的一个变体 / 子类
```

在这个层级里，最关键的调用是：

```python
multires_flow = self.sttmultires_unet.forward(x)
```

所以这里的意思是：

```text
最外层入口类
  -> 把输入交给 self.sttmultires_unet
```

### 3. 第三步：`self.sttmultires_unet` 对应哪个具体类

它对应的主机身类是：

- `Spikingformer_MultiResUNet`

也在：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

所以现在这条线变成：

```text
chunk
  -> MS_SpikingformerFlowNet_en4
  -> Spikingformer_MultiResUNet
```

### 4. 第四步：`Spikingformer_MultiResUNet` 里的第一大工位是谁

在它的 `forward(x)` 里，最开始做：

```python
blocks = self.encoders(x)
```

这里的 `self.encoders` 对应的具体类是：

- `spiking_former_encoder`

也在：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

所以：

```text
chunk
  -> MS_SpikingformerFlowNet_en4
  -> Spikingformer_MultiResUNet
  -> spiking_former_encoder
```

### 5. 第五步：`spiking_former_encoder` 里面真正调用谁

在 `spiking_former_encoder` 里，最关键的是：

```python
self.swin3d = ...
```

也就是说，真正的 Transformer/Swin 主干还要再往下一层。

对应的具体类是：

- `MS_Spiking_SwinTransformer3D_v2`
或
- `Spiking_SwinTransformer3D_v2`

所在文件：

- [Spiking_swin_transformer3D.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)

所以现在路线继续变成：

```text
chunk
  -> MS_SpikingformerFlowNet_en4
  -> Spikingformer_MultiResUNet
  -> spiking_former_encoder
  -> MS_Spiking_SwinTransformer3D_v2
```

### 6. 第六步：encoder 之后程序回到哪里

`self.encoders(x)` 跑完以后，
会把一组特征 `blocks` 返回给：

- `Spikingformer_MultiResUNet.forward(x)`

然后这句执行：

```python
x = blocks[-1]
```

说明程序此时回到了：

- `Spikingformer_MultiResUNet`

并开始进入：

- residual blocks
- decoder
- prediction

### 7. 第七步：residual / decoder / prediction 对应哪个文件

这几个部分的大骨架，其实主要来自：

- [SNN_models.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)

你可以理解成：

```text
Spiking_STSwinNet.py
主要负责把“通用骨架 + spike-former encoder”装起来

SNN_models.py
主要定义了通用的 U-Net 式骨架：
  - residual blocks
  - decoder
  - prediction layers
```

所以如果把后半段对上文件，可以记成：

```text
chunk
  -> Spiking_STSwinNet.py 里的 FlowNet / MultiResUNet 外壳
  -> Spiking_swin_transformer3D.py 里的 encoder 主干
  -> 再回到 SNN_models.py 风格的 residual / decoder / pred 路线
```

### 8. 第八步：最外层最后把结果整理成什么

在最外层 `SpikingformerFlowNet.forward()` 里，
程序最后会把多尺度结果整理成：

```python
{"flow": flow_list, "attn": attns}
```

所以训练器看到的是：

```python
pred_list
```

然后再取：

```python
pred = pred_list["flow"]
```

### 9. 当前最应该背熟的“文件 + 类名”执行线

```text
train_flow_parallel_supervised_SNN.py
  -> model(chunk)
    -> Spiking_STSwinNet.py
      -> MS_SpikingformerFlowNet_en4
      -> SpikingformerFlowNet
      -> Spikingformer_MultiResUNet
      -> spiking_former_encoder
    -> Spiking_swin_transformer3D.py
      -> MS_Spiking_SwinTransformer3D_v2 / Spiking_SwinTransformer3D_v2
    -> 回到 Spiking_STSwinNet.py / SNN_models.py
      -> residual blocks
      -> decoder
      -> prediction
    -> 返回 {"flow": ...}
```

### 10. 当前阶段的目的

这一节不是要求用户立刻读懂这些文件里所有代码。

这一节只要求做到：

```text
我知道 chunk 进模型以后，
它依次会经过哪些“具体类”
以及这些类主要在哪个文件里
```

---

## 第七节课：把 4 个最容易混的名字彻底分开

这一节不加新知识点，
只做一件事：

```text
把下面这 4 个名字彻底分清
```

1. `MS_SpikingformerFlowNet_en4`
2. `Spikingformer_MultiResUNet`
3. `spiking_former_encoder`
4. `MS_Spiking_SwinTransformer3D_v2`

### 1. 为什么现在要专门分这 4 个名字

因为用户当前最大的困难不是“完全听不懂”，
而是：

```text
这些名字都像模型
但不知道它们谁包着谁
谁是外壳
谁是主机身
谁是真正的 encoder
谁是最深处的 Swin 主干
```

所以这一节只解决：

```text
这 4 个名字在调用层级上分别是什么身份
```

### 2. 第一个名字：`MS_SpikingformerFlowNet_en4`

最白的话：

**它是最外层总入口。**

当训练器执行：

```python
pred_list = model(chunk)
```

这里的 `model`，
当前 baseline 通常就是它。

所以它的身份可以记成：

```text
最外层壳子
模型总入口
```

#### 它不是什么

它不是最深处 attention 本体。

它也不是 decoder 本身。

它更像：

```text
整个模型的大门
```

### 3. 第二个名字：`Spikingformer_MultiResUNet`

最白的话：

**它是模型主机身。**

`MS_SpikingformerFlowNet_en4` 把主要工作交给它。

它里面真正包含了：

- encoder
- residual blocks
- decoder
- prediction

所以它的身份可以记成：

```text
主机身
大骨架
```

#### 它和最外层壳子的关系

可以理解成：

```text
MS_SpikingformerFlowNet_en4
把活交给
Spikingformer_MultiResUNet
```

### 4. 第三个名字：`spiking_former_encoder`

最白的话：

**它是主机身里面的 encoder 模块。**

也就是说，它不再是整个模型，
而只是主机身里的“编码器那一块”。

它的身份可以记成：

```text
encoder 这块
```

#### 它和主机身的关系

可以理解成：

```text
Spikingformer_MultiResUNet
里面有一个 encoders
这个 encoders 对应的就是 spiking_former_encoder
```

### 5. 第四个名字：`MS_Spiking_SwinTransformer3D_v2`

最白的话：

**它是 encoder 里面真正的 Swin 主干。**

也就是说：
- 它不是整个模型
- 也不是整个 U-Net
- 它是在 encoder 再往下的一层

它的身份可以记成：

```text
encoder 里面最核心的 Swin/Transformer 主干
```

#### 它和 `spiking_former_encoder` 的关系

可以理解成：

```text
spiking_former_encoder
里面最关键的成员
就是 self.swin3d
而这个 self.swin3d
通常就是 MS_Spiking_SwinTransformer3D_v2
```

### 6. 现在把这 4 个名字串成一条层级线

最重要的就是下面这一条：

```text
MS_SpikingformerFlowNet_en4
  -> Spikingformer_MultiResUNet
    -> spiking_former_encoder
      -> MS_Spiking_SwinTransformer3D_v2
```

如果翻译成最白的话，就是：

```text
最外层总入口
  -> 主机身
    -> 主机身里的 encoder
      -> encoder 里面真正的 Swin 主干
```

### 7. 用“房子”来帮助记忆

如果还是容易混，可以这样想：

#### `MS_SpikingformerFlowNet_en4`

```text
整栋房子的最外层入口
```

#### `Spikingformer_MultiResUNet`

```text
房子的主体结构
```

#### `spiking_former_encoder`

```text
房子里专门负责“前半段提特征”的房间
```

#### `MS_Spiking_SwinTransformer3D_v2`

```text
这个房间里最核心的机器
```

### 8. 当前阶段最重要的结论

以后如果你又看到这些名字混了，
就回到这一句：

```text
FlowNet_en4 是最外层入口
MultiResUNet 是主机身
spiking_former_encoder 是主机身里的 encoder
MS_Spiking_SwinTransformer3D_v2 是 encoder 里的 Swin 主干
```

### 9. 第七节课后的检查问题

#### 问题 1

谁是最外层模型总入口？

标准答案：
- `MS_SpikingformerFlowNet_en4`

#### 问题 2

谁是主机身、大骨架？

标准答案：
- `Spikingformer_MultiResUNet`

#### 问题 3

谁是主机身里的 encoder 模块？

标准答案：
- `spiking_former_encoder`

#### 问题 4

谁是 encoder 里面真正的 Swin 主干？

标准答案：
- `MS_Spiking_SwinTransformer3D_v2`

---

## 附：baseline 整体流程框图入口

为了帮助当前阶段反复建立整体感，单独整理了一份流程图文档：

- [BASELINE_FLOWCHART_ZH.md](/D:/code/sdformer_codex/SDformer/BASELINE_FLOWCHART_ZH.md)

建议后面每一课开始前，先快速看一下里面这几部分：

1. 最简总图
2. 只看“数据”怎么流
3. 只看“训练时一个样本怎么走”
4. 只看“chunk 进入模型后怎么走”
5. 只看“loss 怎么算”

当前阶段不要求一次记住全部，
只要求做到：

```text
听一课
就回到流程图里找到这一课对应的是哪一段
```

---

## 附：最终 baseline 到底由哪些文件组成，以及作者可能的演进路线

用户提出一个关键工程问题：

```text
工程里文件很多，
哪些才是最终 baseline 真正调用的？
这些文件是不是作者做消融和逐步优化留下来的？
能不能还原作者是怎么一步步优化的？
```

### 1. 最终 DSEC baseline 从哪份配置开始

当前原版 DSEC baseline 的主配置可以先认定为：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)

这份配置里最关键的信息是：

```text
model.name = MS_SpikingformerFlowNet_en4
model.encoding = voxel
swin_depths = [2, 2, 6, 2]
swin_num_heads = [3, 6, 12, 24]
use_arc = ["swinv1", "MS_PED_Spiking_PatchEmbed_Conv_sfn"]
window_size = [2, 9, 9]
```

所以判断一个文件是不是最终 baseline 的主路径，首先看：

```text
这份 YAML 会不会调用它
```

### 2. 最终 baseline 的主调用文件清单

#### A. 训练入口

- [train_flow_parallel_supervised_SNN.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)

作用：
- 读配置
- 建模型
- 建 dataset/dataloader
- 做训练循环
- 做 loss/backward/optimizer

#### B. 配置解析

- [parser.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/parser.py)

作用：
- 读取 YAML
- 生成 config
- 判断 device
- 合并部分配置项

#### C. 数据读取

- [DSEC_dataset_lite.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py)

作用：
- 从 `saved_flow_data` 中读出 `chunk / mask / label`

#### D. 数据预处理

当前我们本地实际用于单序列/全量预处理的是：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- [prepare_dsec_full.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_full.py)

原作者/上游仓库中也有预处理相关文件：

- [DSEC_dataset_preprocess.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_preprocess.py)
- [event_representations.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/event_representations.py)

注意：
`prepare_dsec_single_sequence.py` 和 `prepare_dsec_full.py` 是为了我们当前 bring-up 和全量整理补的更明确脚本；
baseline 训练时真正读取的是 `saved_flow_data`。

#### E. 顶层模型装配

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

最终 baseline 里面关键类：

```text
MS_SpikingformerFlowNet_en4
SpikingformerFlowNet
MS_Spikingformer_MultiResUNet
Spikingformer_MultiResUNet
spiking_former_encoder
```

#### F. 通用 SNN U-Net 骨架

- [SNN_models.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)

作用：
- 定义通用 spiking U-Net 结构
- residual blocks
- decoder
- prediction layer 的大骨架

#### G. Swin / attention 主干

- [Spiking_swin_transformer3D.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)

最终 baseline 里最关键的深层类包括：

```text
MS_Spiking_SwinTransformer3D_v2
MS_Spiking_Swin_BasicLayer
MS_Spiking_SwinTransformerBlock3D
Spiking_QK_WindowAttention3D
MS_Spiking_Mlp
MS_SpikingPatchMerging
```

这里是后续 attention / token / window / head 稀疏最重要的主战场。

#### H. SNN 子模块

- [Spiking_modules.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)
- [Spiking_submodules.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py)

作用：
- 定义 spiking conv、decoder、prediction、neuron 相关基础层

#### I. loss

- [flow_supervised.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/loss/flow_supervised.py)

作用：
- 用 `pred_list["flow"]`、`label`、`mask` 计算监督光流 loss

#### J. 保存、加载、统计、backend

- [utils.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/utils.py)
- [runtime_backend.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/runtime_backend.py)
- [train_stats.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/train_stats.py)

其中 `runtime_backend.py` 是我们为了兼容 hetero / CUDA 路径新增的 backend 选择层。

### 3. 哪些文件更像是历史版本 / 实验分支

从配置文件名和类名可以看出，仓库里保留了多个分支：

#### A. ANN/STT 路线

例如：

- [train_DSEC_supervised_STT_voxel.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_STT_voxel.yml)
- [train_MDR_supervised_STT_voxel.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_MDR_supervised_STT_voxel.yml)
- `models/STSwinNet/`

这更像非 SNN 或较早 STT baseline 路线。

#### B. SNN/SDformerFlow 路线

例如：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)
- [train_MDR_supervised_SDformerFlow.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_MDR_supervised_SDformerFlow.yml)
- `models/STSwinNet_SNN/`

这才是当前 SDformerFlow baseline 主路线。

#### C. en3 / en4 或 encoder 数变化

从类名可以看到：

```text
MS_SpikingformerFlowNet
MS_SpikingformerFlowNet_en4
STTFlowNet
STTFlowNet_4en
```

这说明作者很可能比较过不同 encoder 数或不同深度版本。

#### D. attention 变体

在 [Spiking_swin_transformer3D.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py) 中存在：

```text
VanillaAttention
SDAttention
QKAttention
Spiking_QKAttention
HammingDistanceAttention
Spiking_BN_WindowAttention3D
SDSA_WindowAttention3D
Spiking_QK_WindowAttention3D
```

这说明作者至少保留了多种 attention 思路或实验痕迹。

最终配置和 MS block 更偏向：

```text
spiking / QK / window attention
```

### 4. 能不能还原作者一步步优化路线

不能百分百还原。

因为仓库文件只能告诉我们：

```text
有哪些版本被保留下来
有哪些类和配置存在
最终 baseline 调用了哪些
```

但不能完全证明：

```text
作者真实开发时的先后顺序
每一步实验的真实结果
每个文件是否都被论文实验正式使用
```

所以更严谨的说法是：

**可以根据文件和配置推断一条可能的技术演进路线，但不能把它说成作者真实历史。**

### 5. 一个合理的“可能演进路线”推断

根据当前文件结构，可以推断出一条可能路线：

```text
普通事件光流 / U-Net 光流框架
  -> STT / Swin Transformer 版本
  -> Spiking SNN 版本
  -> 多尺度 U-Net + spiking decoder
  -> encoder 替换成 spiking Swin 3D
  -> 引入 MS shortcut / MS block
  -> en4 四级 encoder 版本
  -> QK / spiking window attention 作为最终高效 attention
```

这条路线是“代码结构上的合理推断”，不是对作者开发历史的绝对断言。

### 6. 对你后续改 baseline 最有用的结论

如果你想在最终 baseline 上做改动，不要先管所有文件。

先只盯这条主路径：

```text
train_DSEC_supervised_SDformerFlow_en4.yml
  -> train_flow_parallel_supervised_SNN.py
  -> DSECDatasetLite
  -> MS_SpikingformerFlowNet_en4
  -> Spikingformer_MultiResUNet
  -> spiking_former_encoder
  -> MS_Spiking_SwinTransformer3D_v2
  -> MS_Spiking_SwinTransformerBlock3D
  -> Spiking_QK_WindowAttention3D
  -> flow_loss_supervised
```

这就是当前“最终 baseline 主线”。

后面要优化，就围绕这条线改，不要被仓库里的其他历史分支干扰。

---

## 第八节课：只看最外层 forward 的两段代码

这一节不继续往 attention 深处走。

目标只有一个：

```text
看清楚 model(chunk) 后，为什么会先到最外层 FlowNet，
再转交给 sttmultires_unet，
再进入 encoder / residual / decoder / prediction。
```

### 1. 训练器里真正触发模型的是哪一句

训练器里有：

```python
pred_list = model(chunk)
```

这句话会自动调用当前模型的：

```python
forward(...)
```

当前 baseline 的 `model` 是：

```text
MS_SpikingformerFlowNet_en4
```

它继承自：

```text
SpikingformerFlowNet
```

所以真正看的 forward 是：

- [Spiking_STSwinNet.py:278](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L278)

### 2. 最外层 forward 的核心代码

最外层 `SpikingformerFlowNet.forward` 可以先压缩成：

```python
def forward(self, x, log=False):
    H, W = x.shape[-2], x.shape[-1]
    multires_flow = self.sttmultires_unet.forward(x)

    flow_list = []
    for flow in multires_flow:
        flow = torch.sum(flow, dim=0)
        flow_list.append(interpolate(flow, ...))

    return {"flow": flow_list, "attn": attns}
```

这段代码最重要的不是细节，
而是这句：

```python
multires_flow = self.sttmultires_unet.forward(x)
```

它说明：

```text
最外层 FlowNet 没有自己直接做 encoder / decoder；
它把 x 交给 self.sttmultires_unet。
```

### 3. `self.sttmultires_unet.forward(x)` 里面怎么走

`self.sttmultires_unet` 的 forward 在：

- [Spiking_STSwinNet.py:161](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L161)

它的核心代码可以压缩成：

```python
def forward(self, x):
    blocks = self.encoders(x)
    x = blocks[-1]

    for resblock in self.resblocks:
        x = resblock(x)

    predictions = []
    for decoder, pred in zip(self.decoders, self.preds):
        x = self.skip_ftn(x, blocks[...])
        x = decoder(x)
        pred_out = pred(x)
        predictions.append(pred_out)

    return predictions
```

### 4. 现在把这两段代码接起来看

第一段：

```text
SpikingformerFlowNet.forward
  -> self.sttmultires_unet.forward(x)
```

第二段：

```text
Spikingformer_MultiResUNet.forward
  -> self.encoders(x)
  -> self.resblocks
  -> self.decoders
  -> self.preds
  -> return predictions
```

合起来就是：

```text
model(chunk)
  -> SpikingformerFlowNet.forward
    -> self.sttmultires_unet.forward
      -> encoder
      -> residual blocks
      -> decoder
      -> prediction
    -> 整理 predictions 成 flow_list
  -> return {"flow": flow_list, "attn": attns}
```

### 5. 为什么你之前会觉得乱

因为代码里有两层 forward：

#### 第一层 forward

```text
SpikingformerFlowNet.forward
```

负责：
- 接收模型输入
- 调用 `self.sttmultires_unet`
- 整理输出

#### 第二层 forward

```text
Spikingformer_MultiResUNet.forward
```

负责：
- encoder
- residual blocks
- decoder
- prediction

如果把这两层 forward 混在一起看，就会觉得：

```text
一会儿 FlowNet
一会儿 encoder
一会儿 decoder
```

所以现在必须分清：

```text
第一层 forward 是总包装
第二层 forward 才是主机身计算
```

### 6. 这一节最该记住的一句话

```text
model(chunk) 先进入 SpikingformerFlowNet.forward，
这个 forward 只负责把输入交给 self.sttmultires_unet 并整理输出；
self.sttmultires_unet.forward 才真正执行 encoder、residual、decoder、prediction。
```

### 7. 第八节课后的检查问题

#### 问题 1

`model(chunk)` 会自动调用模型的什么函数？

标准答案：
- `forward`

#### 问题 2

第一层 `SpikingformerFlowNet.forward` 最关键的一句是什么？

标准答案：
- `multires_flow = self.sttmultires_unet.forward(x)`

#### 问题 3

真正执行 encoder / residual / decoder / prediction 的是哪一层 forward？

标准答案：
- `Spikingformer_MultiResUNet.forward`

#### 问题 4

第一层 forward 更像“总包装”，还是“真正主机身计算”？

标准答案：
- 总包装

---

## Top-Down 源码学习路线：学到“足够动手改 baseline 和做硬件协同”的程度

用户当前目标已经变得很明确：

```text
后续改进大概率在这些方面：
- 体素化改进
- 神经元改进
- 注意力改进
- 结构优化
- 剪枝 / 稀疏（kernel 级、token 级等）
- 还要考虑硬件加速器设计与发论文
```

所以当前最重要的问题已经不是“逐行看懂源码”，
而是：

```text
源码应该怎么 top-down 学，学到什么程度，才足够我真正开始做改进？
```

### 1. 一个重要结论：你不需要逐行读懂整个 baseline

对你当前目标来说，
不需要做到：

- 每个类逐行精读
- 每个 tensor reshape 全背下来
- 每个历史实验分支都看一遍

你真正需要做到的是：

```text
我知道 baseline 主线是什么
我知道我的改动属于哪一段
我知道那一段往下应该看哪几个文件
我知道改完以后怎么验证
我知道它在硬件上映射成什么运算模式
```

这才是“足够动手”的标准。

### 2. 正确的 top-down 学习顺序

不要按文件顺序读。

要按下面 6 层往下压：

#### 第 0 层：跑通和主线定位层

目标：

```text
知道最终 baseline 真正跑的是哪份 YAML、哪条训练主线、哪几个关键文件
```

当前最关键主线：

```text
train_DSEC_supervised_SDformerFlow_en4.yml
  -> train_flow_parallel_supervised_SNN.py
  -> DSECDatasetLite
  -> MS_SpikingformerFlowNet_en4
  -> flow_loss_supervised
```

如果这一层不清楚，后面所有改进都会乱。

#### 第 1 层：数据流层

目标：

```text
知道原始数据怎么变成 saved_flow_data
知道 chunk / mask / label 是怎么来的
知道训练器在模型前又对 chunk 做了什么
```

关键文件：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- [prepare_dsec_full.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_full.py)
- [DSEC_dataset_lite.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py)
- [train_flow_parallel_supervised_SNN.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)

这一层学透以后，才能碰：
- 体素化改进
- 输入稀疏
- 数据表示优化

#### 第 2 层：模型最外层骨架层

目标：

```text
知道模型大骨架是怎么拼起来的
知道最外层入口、主机身、encoder、decoder 的层级关系
```

关键文件：

- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)
- [SNN_models.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)

这一层学透以后，才能碰：
- 结构优化
- encoder / decoder 级结构调整
- 多尺度 prediction 改法

#### 第 3 层：encoder / Swin block 层

目标：

```text
知道 attention 真正在哪
知道 token / window / head 是在哪一层出现的
知道 patch embed、basic layer、block、attention 的关系
```

关键文件：

- [Spiking_swin_transformer3D.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)

这一层学透以后，才能碰：
- 注意力改进
- token 稀疏
- window pruning
- head pruning
- kernel 级计算模式优化

#### 第 4 层：神经元 / SNN 基础层

目标：

```text
知道 neuron 在哪里定义和调用
知道 spike norm、step mode、backend、spiking module 是怎么接进来的
```

关键文件：

- [Spiking_modules.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)
- [Spiking_submodules.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py)
- [train_flow_parallel_supervised_SNN.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- [runtime_backend.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/utils/runtime_backend.py)

这一层学透以后，才能碰：
- 神经元改进
- spike 行为建模
- backend / 硬件兼容

#### 第 5 层：loss / metric / profiler / hardware mapping 层

目标：

```text
知道改动后该怎么验证
知道哪些统计可以映射到硬件
知道软件模块怎样映射成硬件加速器子模块
```

关键文件：

- [flow_supervised.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/loss/flow_supervised.py)
- [profiler.py](/D:/code/sdformer_codex/SDformer/src/utils/profiler.py)
- [hw/docs/arch.md](/D:/code/sdformer_codex/SDformer/hw/docs/arch.md)
- [hw/docs/interfaces.md](/D:/code/sdformer_codex/SDformer/hw/docs/interfaces.md)
- [hw/docs/perf_model.md](/D:/code/sdformer_codex/SDformer/hw/docs/perf_model.md)

这一层学透以后，才能碰：
- 硬件加速器设计
- 论文里的软硬件协同实验

### 3. 针对你的 5 类改进，应该分别学到哪一层

#### A. 体素化改进

只需要先重点吃透：

- 第 1 层：数据流层

关键文件：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- [event_representations.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/event_representations.py)

#### B. 神经元改进

重点吃透：

- 第 4 层：神经元 / SNN 基础层

#### C. 注意力改进

重点吃透：

- 第 3 层：encoder / Swin block 层

#### D. 结构优化

重点吃透：

- 第 2 层：模型最外层骨架层
- 第 3 层：encoder / block 层

#### E. 剪枝 / 稀疏

如果是：

- token / window / head 稀疏  
  重点是第 3 层

- kernel 级 / conv 级稀疏  
  重点是第 4 层和 decoder/conv 子模块

### 4. 什么时候算“足够可以动手”

你不需要等“全懂”。

对每个改进方向，只要做到下面 4 件事，就已经可以开始：

```text
1. 我知道输入输出是什么
2. 我知道主调用入口在哪里
3. 我知道改动会影响哪条后续路径
4. 我知道改完后用什么指标验证
```

这就是当前真正的“可动手标准”。

### 5. 对你当前最推荐的源码理解顺序

接下来不要继续广撒网。

推荐顺序是：

```text
先把第 0-2 层完全搞清楚
再根据你最先想做的改进方向，进入第 3 或第 4 层
```

更具体一点：

```text
阶段 1：
  只把数据流 + 最外层模型骨架学透

阶段 2：
  决定先做哪一类改进
  （体素化 / 神经元 / 注意力 / 稀疏）

阶段 3：
  只深挖这一类改进对应的那一层源码

阶段 4：
  再把这类改进映射到 profiler 和硬件模块
```

### 6. 对论文和硬件加速器设计的直接建议

如果你的论文目标里明确有：
- 软件优化
- 硬件加速器
- ICCAD / ISCAS / BioCAS

那么最推荐的切入顺序不是“神经元先行”，而是：

```text
1. 先从 attention / token / window 稀疏切入
2. 因为这类改动最容易映射成硬件调度、稀疏计算和数据搬运优化
3. 体素化改进作为输入侧加分项
4. 神经元改进作为补充项，而不是第一主线
```

这是因为：

```text
attention / 稀疏
更容易形成“算法 + 架构 + 加速器”完整闭环
```

### 7. 当前阶段最重要的结论

用户现在不需要继续焦虑“源码还没完全懂，所以不能开始做研究”。

正确结论是：

```text
只要按 top-down 方式学到与自己改进方向对应的那一层，
就已经足够开始做 baseline 改进和硬件协同设计。
```

---

## 分层学习总路线：以后固定按层来学

用户明确要求：

```text
你带我一层层学
```

后续教学固定按下面 6 层推进，不再跨层乱跳。

### 第 0 层：最终 baseline 主线层

目标：

```text
先认清最终 baseline 到底是哪条主线
哪些文件是真正会被最终 baseline 调到的
哪些文件暂时可以先不管
```

### 第 1 层：数据流层

目标：

```text
原始数据 -> 预处理 -> saved_flow_data -> dataset -> dataloader -> chunk/mask/label
```

### 第 2 层：模型最外层骨架层

目标：

```text
chunk 进模型后
先到谁
再到谁
为什么会先有最外层 FlowNet
再有主机身 MultiResUNet
```

### 第 3 层：encoder / Swin 主干层

目标：

```text
attention、token、window、head 这些真正出现在哪里
```

### 第 4 层：SNN / neuron / 基础层层

目标：

```text
神经元、spike norm、step mode、backend、spiking 基础模块在哪里
```

### 第 5 层：loss / profiler / hardware mapping 层

目标：

```text
改完以后怎么验证
怎么做 profiler
怎么映射成硬件模块
```

---

## 第 0 层开始：先认清“最终 baseline 只有一条主线”

当前最重要的不是继续读更多代码，
而是先认清：

```text
工程里虽然文件很多
但真正的最终 baseline 主线只有一条
```

当前 DSEC 最终 baseline 主线可以先固定成：

```text
train_DSEC_supervised_SDformerFlow_en4.yml
  -> train_flow_parallel_supervised_SNN.py
  -> DSECDatasetLite
  -> MS_SpikingformerFlowNet_en4
  -> flow_loss_supervised
```

也就是说，后面学习时：

### 先重点看

- [train_DSEC_supervised_SDformerFlow_en4.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)
- [train_flow_parallel_supervised_SNN.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- [DSEC_dataset_lite.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py)
- [Spiking_STSwinNet.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)
- [SNN_models.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)
- [Spiking_swin_transformer3D.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)
- [flow_supervised.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/loss/flow_supervised.py)

### 暂时可以先不重点看

- 其他数据集分支配置
- 其他历史 STT / 非 SNN 配置
- 不是最终主线调用到的旧实验文件

### 第 0 层当前最重要的一句话

```text
不要试图同时理解整个工程；
先只理解“最终 baseline 主线真正调用到的那些文件”。
```

### 第 0 层检查问题

#### 问题 1

当前 DSEC 最终 baseline 主配置是哪一个？

标准答案：
- `train_DSEC_supervised_SDformerFlow_en4.yml`

#### 问题 2

当前学习时，应该优先盯“所有文件”，还是“最终主线真正调用到的文件”？

标准答案：
- 先盯最终主线真正调用到的文件

---

## 第 0 层检查：用户回答与纠正

用户当前回答：

```text
配置是这个脚本，应该看调用到的文件
```

这个回答已经抓住了一半重点：

- “应该看最终主线真正调用到的文件”这一点是对的

但还有一个小纠正：

### 1. 配置不是“脚本”

更准确的说法应该是：

```text
当前 DSEC 最终 baseline 主配置
是 train_DSEC_supervised_SDformerFlow_en4.yml
```

也就是：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)

它是：

```text
配置文件
```

不是脚本。

### 2. 脚本和配置的区别

当前阶段必须分清：

#### 配置文件

例如：

```text
train_DSEC_supervised_SDformerFlow_en4.yml
```

作用：

```text
告诉程序要用什么模型、什么超参数、什么输入设置
```

#### 脚本文件

例如：

- [train_flow_parallel_supervised_SNN.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)

作用：

```text
真正执行训练流程
```

### 3. 当前第 0 层的标准结论

```text
主配置是 train_DSEC_supervised_SDformerFlow_en4.yml
当前学习时应优先盯最终主线真正调用到的文件
```

---

## 第 1 层：数据流层

这一层只学一件事：

```text
原始数据
-> 预处理
-> saved_flow_data
-> DSECDatasetLite
-> DataLoader
-> chunk / mask / label
```

这一层的目标不是看模型，
而是先彻底弄清：

```text
训练器手里的 chunk / mask / label 到底从哪来
```

### 1. 数据流层的最简路线

先把整条路压缩成一句话：

```text
原始 DSEC 数据先被预处理脚本整理成 saved_flow_data，
然后 DSECDatasetLite 从 saved_flow_data 里按样本名读出 chunk、mask、label，
再由 DataLoader 打包成 batch 交给训练器。
```

### 2. 第一步：原始数据是什么

原始数据不是训练器直接吃的 `.npy`，
而是更底层的官方存储格式。

最关键的几样东西是：

#### A. 原始事件流

例如：

- `events.h5`
- `rectify_map.h5`

它们保存的是：
- 连续事件流
- 校正映射

#### B. 原始光流标签

例如：

- `flow/forward/*.png`
- `forward_timestamps.txt`

它们保存的是：
- 光流标签
- 每张标签对应的时间区间

所以原始数据还不能直接变成：

```text
chunk / mask / label
```

### 3. 第二步：预处理在这里发生

真正做预处理的，不是 dataset，
而是这些脚本：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- [prepare_dsec_full.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_full.py)

它们的工作是：

```text
把原始数据整理成训练器好读取的 saved_flow_data
```

### 4. 第三步：预处理具体做了什么

预处理主要做 4 件事。

#### A. 解码 flow PNG

把原始 PNG 解码成：

- `flow`
- `valid`

然后保存成：

- `gt_tensors/*.npy`
- `mask_tensors/*.npy`

#### B. 从连续事件流里截一段事件

根据某个光流标签对应的时间窗，
从 `events.h5` 里取出这一小段事件。

#### C. 做 rectification 和 voxel 化

把事件坐标校正后，
再转换成 voxel。

最后保存成：

- `event_tensors/.../*.npy`

#### D. 写训练/验证样本名单

保存成：

- `train_split_seq.csv`
- `valid_split_seq.csv`

### 5. 第四步：什么是 `saved_flow_data`

你可以把它理解成：

**训练专用中间格式。**

它通常包括：

```text
saved_flow_data/
  gt_tensors/
  mask_tensors/
  event_tensors/
  sequence_lists/
```

也就是说：

```text
原始数据 -> saved_flow_data
```

就是预处理这一步的结果。

### 6. 第五步：`DSECDatasetLite` 做什么

到这里，预处理已经结束了。

`DSECDatasetLite` 不再生成新数据，
它只负责：

```text
从 saved_flow_data 中把一个样本读出来
```

它会做：

1. 从 csv 取样本名
2. 去 `mask_tensors` 读 `mask`
3. 去 `gt_tensors` 读 `label`
4. 去 `event_tensors` 读 `chunk`
5. 返回 `chunk, mask, label`

### 7. 第六步：`DataLoader` 做什么

`DataLoader` 不负责决定一个样本长什么样，
它负责：

```text
把 dataset 读出的多个样本打成 batch
```

所以：

- `dataset` 决定单个样本怎么读
- `dataloader` 决定一批样本怎么取

### 8. 第七步：训练器最后拿到什么

经过 `DataLoader` 之后，
训练器最后拿到：

```python
for chunk, mask, label in train_dataloader:
```

也就是：

- `chunk`
- `mask`
- `label`

### 9. 这一层最容易混淆的两个点

#### 混淆 1

`DSECDatasetLite` 不是预处理器。

更准确地说：

```text
prepare_dsec_*.py 做预处理
DSECDatasetLite 读预处理结果
```

#### 混淆 2

`chunk` 不是原始事件流。

更准确地说：

```text
chunk 是预处理以后保存下来的事件表示
通常是 voxel
```

### 10. 第 1 层最该记住的一句话

```text
原始数据先被 prepare_dsec_*.py 处理成 saved_flow_data；
然后 DSECDatasetLite 从 saved_flow_data 中读出 chunk、mask、label；
最后 DataLoader 把它们打成 batch 交给训练器。
```

### 11. 第 1 层检查问题

#### 问题 1

真正做预处理的是谁？

标准答案：
- `prepare_dsec_single_sequence.py` / `prepare_dsec_full.py`

#### 问题 2

`DSECDatasetLite` 的职责是什么？

标准答案：
- 从 `saved_flow_data` 中读出 `chunk / mask / label`

#### 问题 3

`DataLoader` 的职责是什么？

标准答案：
- 把多个样本打成 batch

---

## 第 1 层检查：用户回答与追问

用户当前回答：

```text
做预处理的是 prepare 那个 py 文件，
lite 职责就是 dataloader，就是把 chunk，label，mask 给训练器。
```

这个回答已经抓住了大方向：
- 预处理确实是 `prepare_dsec_*.py`
- 最后训练器手里确实会得到 `chunk / mask / label`

但这里有一个需要纠正的小点：

### 1. `DSECDatasetLite` 不是 `DataLoader`

更准确的区分是：

#### `DSECDatasetLite`

负责：

```text
定义“单个样本怎么读”
```

也就是：
- 从 csv 里取样本名
- 读 `chunk`
- 读 `mask`
- 读 `label`

#### `DataLoader`

负责：

```text
把 dataset 读出的多个样本打包成 batch
```

所以正确说法应该是：

```text
DSECDatasetLite 负责读单个样本
DataLoader 负责把多个样本打成 batch
最后训练器才拿到 chunk / mask / label 的 batch
```

---

## 追问：mask 是怎么得到的、起什么作用

### 1. mask 从哪里来

mask 的原始来源，在原始光流 PNG 里。

在预处理脚本：

- [prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)

里面有：

```python
decode_flow_png(...)
```

它会做：

```python
valid = flow_16[:, :, 0].astype(bool)
```

这说明：

```text
原始 DSEC 的 flow PNG 第 0 通道
本身就带“这个像素位置是否有效”的信息
```

然后预处理脚本把这个 `valid` 保存成：

```text
mask_tensors/样本名.npy
```

所以更准确地说：

```text
mask 不是凭空造出来的
它来源于原始标签 PNG 里的有效性信息
然后在预处理时被提取出来并保存成 .npy
```

### 2. mask 的作用

最白的话：

```text
mask 是“哪些位置允许算 loss”的地图
```

它的作用不是改输入，
而是告诉 loss：

```text
这些位置的标签有效，可以算误差
那些位置的标签无效，不要算误差
```

所以：

- `label` 告诉你答案是什么
- `mask` 告诉你哪些地方能算分

---

## 追问：矫正化（rectification）到底干了什么

### 1. 最白的话

rectification 可以先理解成：

**把原始事件坐标，校正到标准图像坐标里。**

原始传感器坐标可能有几何畸变，
所以不能直接把事件点拿来当最终图像坐标。

### 2. 在代码里怎么做

在：

- [event_representations.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/event_representations.py)

里有：

```python
def rectify_events(x, y, rectify_map):
    return rectify_map[y, x]
```

这说明：

```text
程序拿原始事件的 (x, y)
去查一张 rectify_map
得到校正后的坐标
```

### 3. 为什么要做 rectification

因为后面你要把事件放进 480x640 的规则网格里：
- 做 voxel
- 和标签对齐
- 喂给模型

如果事件坐标不先校正，
后面和标签、像素网格的位置就会对不准。

所以你现在可以把 rectification 记成：

```text
把原始事件坐标先摆正
这样后面才能正确落到图像网格上
```

---

## 追问：voxel 化到底干了什么

### 1. 最白的话

voxel 化就是：

**把一小段连续事件流，装进一个有“时间层数”的三维格子里。**

这里的三维不是物理 3D 世界，
而是：

```text
时间 bin × 高 × 宽
```

### 2. 为什么要 voxel 化

因为原始事件流是：

```text
一条一条离散事件
每条事件有时间、位置、极性
```

而模型更喜欢固定形状的张量。

所以需要把连续事件流整理成：

```text
C × H × W
```

这样的规则张量。

### 3. 在代码里怎么做

在预处理脚本里：

```python
voxel = make_voxel_chunk(...)
```

它内部会：

1. 从时间窗里取出这一小段事件
2. 先做 rectification
3. 把时间归一化到多个 bin 上
4. 把每个事件按 `(t, x, y)` 放到 voxel grid 里

### 4. VoxelGrid 具体做了什么

在：

- [event_representations.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/DSEC_dataloader/event_representations.py)

里有：

```python
class VoxelGrid
```

它做的核心事情是：

```text
把每个事件按照时间、空间位置分配到一个 3D 网格里
```

更具体地说：

- 时间被切成若干 bin
- 空间是 480×640
- 每个事件按极性带正负号
- 再用插值把它分摊到邻近格子

所以它不是简单“某个 bin +1”，
而是更平滑地放进去。

### 5. 现在最简单的理解

你现在先把 voxel 化记成：

```text
把一小段事件流
变成一个固定大小的时间-空间张量
方便模型读取
```

---

## 当前这一步最应该记住的总结

```text
mask：
  来源于原始 flow PNG 里的有效性信息
  作用是控制哪些位置参与 loss

rectification：
  把原始事件坐标校正到标准图像坐标

voxel 化：
  把一小段连续事件流变成固定大小的时间-空间张量
```

---

## 继续追问：voxel 是怎么分到不同 bin 里，并且为什么要插值

用户当前已经能概括：

- `mask` 从原始图像有效性信息来
- rectification 是统一坐标
- voxel 是把事件离散化

接下来最关键的是把：

```text
事件怎么落到不同时间 bin
为什么不是简单地直接扔进一个 bin
为什么还要做插值
```

讲清楚。

### 1. 先把 voxel grid 想成什么

最白的话：

```text
voxel grid 就是一摞很多张事件图
```

比如 `num_bins = 10` 时，
你可以把它想成：

```text
第 0 张图
第 1 张图
...
第 9 张图
```

每一张图的大小都是：

```text
480 × 640
```

所以 voxel grid 不是别的，
就是：

```text
10 张 480×640 的图叠在一起
```

### 2. 一条事件原来长什么样

一条事件最关键的信息是：

- 时间 `t`
- 位置 `x`
- 位置 `y`
- 极性 `p`

也就是说，一条事件本质上是在说：

```text
在某个时间点
某个像素位置
发生了一个正事件或负事件
```

### 3. 时间是怎么分到 bin 里的

代码里会先做：

```python
t_norm = (C - 1) * (t - t[0]) / (t[-1] - t[0])
```

这里你先不用怕公式。

最白的话就是：

```text
先把这一小段事件的时间
压缩到 0 到 C-1 之间
```

如果：

- `C = 10`

那么时间就会被压到：

```text
0 到 9 之间
```

这意味着：

```text
最早的事件接近第 0 个时间 bin
最晚的事件接近第 9 个时间 bin
中间事件落在中间 bin
```

### 4. 为什么不是“直接取整扔进一个 bin”

如果一条事件的归一化时间是：

```text
3.2
```

那它其实位于：

```text
第 3 个时间 bin 和第 4 个时间 bin 之间
```

如果你粗暴地只取整，
直接把它全扔进第 3 个 bin，
就会出现问题：

```text
时间信息太生硬
边界不连续
很多事件都会被粗暴地塞进单一 bin
```

所以代码不会只取一个 bin，
而是会做：

**时间插值**

### 5. 时间插值最白的话

如果一个事件的归一化时间是：

```text
3.2
```

那你可以把它想成：

```text
它离第 3 个 bin 更近
离第 4 个 bin 稍远
```

所以更合理的做法是：

```text
给第 3 个 bin 分更多一点
给第 4 个 bin 分少一点
```

这就是插值的本质：

```text
不是把一个事件只扔给一个格子
而是按距离把它分摊给邻近格子
```

### 6. 不只是时间，空间上也在插值

代码里不只是时间 `t` 会插值，
空间位置 `x`、`y` 也会。

如果事件坐标不是刚好落在整数像素中心，
比如：

```text
x = 120.3
y = 45.7
```

那它其实位于周围几个像素之间。

所以程序也不会只把它扔进：

```text
(120, 45)
```

而是会按距离分摊到附近像素格子。

### 7. 所以一条事件会被分给几个格子

在 `VoxelGrid.convert_CHW()` 里，代码会遍历：

- `x0` 和 `x0 + 1`
- `y0` 和 `y0 + 1`
- `t0` 和 `t0 + 1`

也就是：

```text
时间上 2 个候选
空间 x 上 2 个候选
空间 y 上 2 个候选
```

合起来最多就是：

```text
2 × 2 × 2 = 8 个邻近格子
```

也就是说：

```text
一条事件最多会被分摊到 8 个 voxel 小格里
```

### 8. 插值权重是怎么来的

代码里核心权重大概是：

```python
interp_weights =
    value
    * (1 - |xlim - x|)
    * (1 - |ylim - y|)
    * (1 - |tlim - t_norm|)
```

最白的话就是：

```text
离哪个格子近
就给哪个格子更大权重
离哪个格子远
就给哪个格子更小权重
```

所以：

- 时间上更近的 bin 分得更多
- 空间上更近的像素分得更多

### 9. 极性怎么体现

代码里还有：

```python
value = 2 * p - 1
```

这意味着：

- 一种极性会变成 `+1`
- 另一种极性会变成 `-1`

所以 voxel grid 里累计的不只是“有没有事件”，
还保留了：

```text
事件是正极性还是负极性
```

### 10. 用一个最小例子理解

假设：

- `num_bins = 10`
- 某个事件时间归一化后是 `3.2`
- 空间坐标是 `(120.3, 45.7)`

那它不会被简单地塞进一个格子，
而会被分给邻近的 8 个候选组合，例如：

```text
(t=3, x=120, y=45)
(t=3, x=120, y=46)
(t=3, x=121, y=45)
(t=3, x=121, y=46)
(t=4, x=120, y=45)
(t=4, x=120, y=46)
(t=4, x=121, y=45)
(t=4, x=121, y=46)
```

并且：

- 离得近的格子分更多
- 离得远的格子分更少

### 11. 为什么要这么麻烦

因为这样得到的 voxel 表示更平滑、更连续。

如果直接粗暴取整：

- 时间会跳变
- 空间会跳变
- 输入表示很粗糙

而插值后：

```text
事件表示更平滑
时间信息更连续
空间位置也更稳定
```

这对后面的模型学习更友好。

### 12. 当前最该记住的一句话

```text
Voxel 化不是把事件简单扔进某一个时间格子；
而是先把时间压到多个 bin 的范围里，
再把每个事件按时间和空间距离分摊到邻近格子中，
这样得到平滑的时间-空间体素表示。
```

---

## voxel 追问检查：用户回答与关键澄清

用户当前回答：

```text
因为有落在 bin 与 bin 之间的数据，8 个格子，按距离分配
```

这个回答已经抓住了 voxel 插值最关键的 3 个点：

- 事件可能落在两个时间 bin 之间
- 一条事件最多会分给 8 个邻近格子
- 分配原则是按距离分配权重

这说明：

**voxel 插值这一层，用户已经抓住核心了。**

### 1. 当前还需要补一个关键澄清

用户提出的问题是：

```text
时间 bin 的选择，最终就是数据维度里的时间步 t 吗？
```

最准确的回答是：

**相关，但不完全等价。**

### 2. 为什么说“相关”

因为 voxel grid 的第一个维度，
本质上就是：

```text
时间被切成的多个 bin
```

所以从张量结构上看，
它确实像一个“时间维”。

### 3. 为什么说“不完全等价”

因为这里的 `time bin`，
本质上是：

```text
对一小段连续事件流做离散化以后得到的时间片
```

它是输入表示里的时间分桶结果。

而模型里的“时间步”有时还会和：

- SNN 的 `num_steps`
- step mode
- 训练器对输入的重排方式

一起发生关系。

所以更准确的说法应该是：

```text
voxel 的 bin 维
可以先把它理解成输入张量里的时间维
但它不是“所有时间概念”的唯一来源
```

### 4. 当前阶段最推荐的记法

现在为了不把自己绕乱，
建议先这样记：

```text
在输入表示这一层，
time bin 就可以先理解成 chunk 里的时间维
```

后面等真正学到：
- `num_steps`
- step mode
- 输入 reshape

再去区分：
- voxel 时间 bin
- SNN 时间步

### 5. 当前最该记住的一句话

```text
对当前输入表示来说，
time bin 可以先看成 chunk 里的时间维；
但从整个模型和 SNN 角度看，
它只是“时间概念”的其中一层，不是全部。
```

---

## 第 1 层补充：bin 到底怎么定、这一小段时间到底多长

用户追问：

```text
bin 的选择是怎么定的？
“一小段时间”归一化到 bin 是多长时间？
是不是原始数据流的总时间？
```

### 1. bin 的个数是谁定的
最直接的来源是配置和预处理参数。

在当前 baseline 主配置里：
- `data.num_frames: 10`
- `data.num_chunks: 1`
- `model.num_bins: 10`
- `spiking_neuron.num_steps: 10`

在我们当前的预处理脚本里：
- [prepare_dsec_single_sequence.py](D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- 默认参数是 `--num-bins 10`

所以当前单样本 voxel 的时间 bin 数，就是：

```text
10 个 bin
```

不是程序自己随便选的，是配置/参数先定好的。

### 2. “一小段时间”到底指什么
不是整条原始事件流的总时间。

它指的是：

```text
某一张光流标签对应的那个时间窗
```

这个时间窗来自：
- `forward_timestamps.txt`

在预处理脚本里，会逐个读取：

```python
for idx, (flow_png, (t_beg, t_end)) in enumerate(zip(flow_files, timestamps), start=1):
```

也就是说：
- 第 1 个样本有自己的 `(t_beg, t_end)`
- 第 2 个样本有自己的 `(t_beg, t_end)`
- ...

所以：

```text
每一个样本都有自己的局部时间窗
不是拿整段序列总时长来统一切 bin
```

### 3. 归一化到底是对谁做的
是对：

```text
这个样本时间窗里的事件时间
```

做归一化。

在预处理脚本里：

```python
t = event_data["t"][mask].astype(np.float32)
if t[-1] == t[0]:
    t = np.linspace(0.0, 1.0, t.size, dtype=np.float32)
else:
    t = (t - t[0]) / (t[-1] - t[0])
```

意思就是：

```text
先只取这个样本时间窗里的事件
再把这些事件的时间压到 0 到 1
```

后面 `VoxelGrid.convert_CHW()` 再把它映射到：

```text
0 到 C-1
```

如果 `C = 10`，那就是：

```text
0 到 9
```

### 4. 所以每个 bin 可以怎么理解
当前你可以先把它粗略理解成：

```text
这一小段样本时间窗，被平均切成 10 份
```

更精确一点：
- 不是简单硬切块
- 因为每个事件还会做时间插值

所以它不是“只属于某一个 bin”，
而是“主要落在某两个相邻 bin 之间，再按距离分配权重”。

### 5. 当前阶段最该记住的一句话

```text
bin 的个数由配置/参数决定；
每个样本只对自己的局部时间窗 `(t_beg, t_end)` 做归一化和体素化；
不是拿整条原始事件流的总时间做统一切 bin。
```

---

## 第 2 层开始：进入模型前，输入维度怎么变

用户要求：

```text
后面层的讲解需要带上数据具体的维度是怎么变化的，
过的什么操作、什么作用、底层算法是怎么样的
```

从这一节开始，后面的讲解都尽量补上：
- 输入/输出维度
- 做了什么操作
- 这一步的作用
- 底层算法直觉

### 1. dataset 读出来的 `chunk` 大致是什么形状
当前 baseline 的 voxel 输入，单个样本最直观可以先理解成：

```text
[10, H, W]
```

也就是：
- 10 个时间 bin
- 每个 bin 一张 `H x W` 的图

如果 dataloader 再加上 batch 维，先粗略看成：

```text
[B, 10, H, W]
```

这里先不把 polarity 提前掺进去，
因为 polarity 主要是在训练器里继续整理。

### 2. 训练器里做 polarity 拆分之后
在训练器的 voxel + polarity 路线里，
输入会被整理成更显式的正负极性形式。

你当前可以先把它理解成：

```text
[B, 10, 2, H, W]
```

这里的 `2` 表示：
- 正事件
- 负事件

也就是：

```text
batch × 时间bin × 极性 × 高 × 宽
```

### 3. 进入 `SpikingMultiResUNet.forward()` 后第一步做了什么
在：
- [SNN_models.py](D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)

里，先做：

```python
event_reprs = x.permute(0, 2, 3, 4, 1)
```

如果上一层是：

```text
[B, 10, 2, H, W]
```

那么这里会变成：

```text
[B, 2, H, W, 10]
```

作用是：

```text
把时间维放到最后，方便后面按 SNN 时间步重新切分
```

### 4. 然后又新建了一个张量

```python
new_event_reprs = torch.zeros(B, self.num_ch, H, W, self.steps)
```

在当前 baseline 配置里：
- `num_bins = 10`
- `num_steps = 10`
- polarity = 2

所以：

```python
self.num_ch = num_bins * 2 // self.steps = 10 * 2 // 10 = 2
```

于是这个张量的形状就可以看成：

```text
[B, 2, H, W, 10]
```

### 5. 这一步到底在干什么
代码里会循环把正负极性通道搬进去：

```python
for i in range(self.num_ch):
    start, end = i//2 * self.steps, (i//2 + 1) * self.steps
    new_event_reprs[:, i, :, :, :] = event_reprs[:, i % 2, :, :, start:end]
```

当前这个 baseline 参数下，它的直观作用可以先理解成：

```text
把“时间bin + 极性”这套输入，
重新整理成适合 SNN 按时间步展开的表示
```

### 6. 最后再 permute 一次

```python
x = new_event_reprs.permute(4, 0, 1, 2, 3)
```

于是张量会变成：

```text
[T, B, C, H, W]
```

在当前配置下，大致就是：

```text
[10, B, 2, H, W]
```

这一步非常重要。

因为到这里以后，你就可以把它理解成：

```text
模型真正开始按时间步处理输入了
```

### 7. 这一节最该记住的一句话

```text
dataset 读出来的 voxel chunk，先是“按时间bin堆起来”的输入；
训练器把它整理出正负极性；
模型最外层再把它重排成 `[T, B, C, H, W]`，
从这里开始，输入才真正变成 SNN 按时间步展开的形式。
```

### 8. 一个必须纠正的点：最终 baseline 主线路不完全走上面那条通用重排
前面这段 `[B,10,2,H,W] -> [10,B,2,H,W]`，
更贴近通用 `SpikingMultiResUNet.forward()` 的输入整理逻辑。

但当前 **最终 baseline 主线** 用的是：
- `MS_SpikingformerFlowNet_en4`
- `MS_Spikingformer_MultiResUNet`

而这个 `Spikingformer_MultiResUNet` 自己重写了 `forward()`，
它没有先走通用 U-Net 里的那段输入重排，
而是更直接地做：

```text
[B, 10, 2, H, W]
  -> encoders(x)
  -> 进入 3D Swin 主干
```

所以从“最终 baseline 主线”角度，
你当前应该优先记：

```text
训练器整理后的输入
= [B, 10, 2, H, W]
这就是送进 spike-former encoder 的输入形状
```

不要把通用 U-Net 的输入整理逻辑和最终主线路完全混成一回事。

---

## 第 2 层继续：最终 baseline 主线路里的维度怎么走

当前 baseline 关键配置：
- crop: `[288, 384]`
- num_bins: `10`
- polarity: `True`
- base_num_channels: `96`
- swin_depths: `[2, 2, 6, 2]`
- swin_num_heads: `[3, 6, 12, 24]`
- swin_patch_size: `[1, 1, 2, 2]`

### 1. 训练器送进模型前的输入
训练器把 voxel 做正负极性拆分之后，主线路输入可以先看成：

```text
[B, 10, 2, 288, 384]
```

含义：
- `B`：batch
- `10`：10 个 voxel bin
- `2`：正负极性
- `288 x 384`：crop 后的空间尺寸

### 2. 进入 encoder 的第一层 patch embedding 之后
patch size 是：

```text
[1, 1, 2, 2]
```

最直观的理解：
- 不压时间 bin
- 不压 polarity 这一维
- 只在空间上做 `2x2` 下采样

所以第一层输出可以粗略理解成：

```text
[B, 96, 2, 144, 192]
```

这里：
- 通道数变成 `96`
- 空间尺寸减半
- 那个 `2` 还保留着

### 3. encoder stage 逐层往下
后面每个 stage 主要做两件事：
- 在当前分辨率上做若干个 Swin block
- stage 结束时做 patch merging（空间减半、通道翻倍）

所以 4 个 stage 的输出可以先粗略记成：

```text
stage 0: [B,  96, 2, 144, 192]
stage 1: [B, 192, 2,  72,  96]
stage 2: [B, 384, 2,  36,  48]
stage 3: [B, 768, 2,  18,  24]
```

这就是 encoder 给 U-Net 主机身留下来的 4 级特征。

### 4. 为什么 `spiking_former_encoder` 里还要 permute
在：
- `Spiking_STSwinNet.py`

里，encoder 输出后会做：

```python
out_i = features[i].permute(2, 0, 1, 3, 4)
```

所以每一级特征都会变成：

```text
[2, B, C, H, W]
```

例如：

```text
[B,  96, 2, 144, 192] -> [2, B,  96, 144, 192]
[B, 192, 2,  72,  96] -> [2, B, 192,  72,  96]
[B, 384, 2,  36,  48] -> [2, B, 384,  36,  48]
[B, 768, 2,  18,  24] -> [2, B, 768,  18,  24]
```

最白的理解：

```text
模型把那一维 2 提到最前面，
后面 residual/decoder 就统一按 [T, B, C, H, W] 这种格式处理
```

### 5. residual blocks
主机身拿最深层那一级：

```text
[2, B, 768, 18, 24]
```

连续过 residual blocks。

这一步的作用：

```text
不改分辨率，
主要继续在最深层语义特征上做加工
```

所以这一段前后维度基本不变：

```text
[2, B, 768, 18, 24]
  -> residual blocks
  -> [2, B, 768, 18, 24]
```

### 6. decoder 逐层往回恢复
decoder 每一层都会：
- 和对应 encoder 特征做 skip 连接
- 上采样
- 输出一层 prediction

粗略维度可以先记成：

```text
decoder 0: [2, B, 768, 18, 24]  -> [2, B, 384,  36,  48]
decoder 1: [2, B, 384, 36, 48]  -> [2, B, 192,  72,  96]
decoder 2: [2, B, 192, 72, 96]  -> [2, B,  96, 144, 192]
decoder 3: [2, B,  96,144,192]  -> [2, B,  96, 288, 384]
```

### 7. prediction 每层都会吐一个 flow
每个 decoder 后面的 prediction layer 都会输出：

```text
[2, B, 2, H, W]
```

这里：
- 最前面的 `2`：前面那一维时序/深度维
- 中间的 `2`：最终光流的两个分量 `(u, v)`

所以最粗略地看：

```text
pred0: [2, B, 2,  36,  48]
pred1: [2, B, 2,  72,  96]
pred2: [2, B, 2, 144, 192]
pred3: [2, B, 2, 288, 384]
```

### 8. 最外层 FlowNet 最后做什么
最外层会对前面那一维做求和，再插值回最终输出尺寸。

所以训练器最终拿到的每一级 flow，都可以理解成：

```text
[B, 2, 288, 384]
```

然后这些 flow 组成一个 list，
最后放进：

```python
{"flow": flow_list, "attn": ...}
```

### 9. 当前阶段最该记住的一句话

```text
最终 baseline 主线路里，
训练器整理后的输入是 [B,10,2,288,384]；
encoder 先把空间分辨率一路减半、通道一路翻倍；
然后把特征改成 [2,B,C,H,W] 交给 residual 和 decoder；
decoder 再一级一级恢复空间分辨率并输出多尺度 flow。
```

---

## 重新开始：参考论文，从 patch embedding 重新讲

用户要求：

```text
你讲的太快太简略了，可以参考论文一起讲，重新从 patch embedding 开始讲
```

本次参考论文：
- [arXiv 2409.04082](https://arxiv.org/abs/2409.04082)
- [ar5iv HTML](https://ar5iv.org/html/2409.04082v1)

论文里与当前主题最相关的是：
- III-B Event Input Representation
- III-C Network Architecture
- III-C-1 Spiking Feature Generator with Shortcut Patch Embedding

### 1. 论文里怎么描述 patch embedding
论文在 III-C-1 里把前端分成两个阶段：

```text
a) SFG: Spiking Feature Generator
b) SPE: Shortcut Patch Embedding
```

论文原意可以压成：

```text
先用一个脉冲卷积前端把输入事件表示变成更适合后面处理的时空特征，
再把这些特征切成 patch，并投影成给 STSF encoder 使用的 token embedding。
```

也就是说，论文里的 patch embedding 不是“凭空直接切 patch”，
前面还有一个 **SFG 前端**。

### 2. 论文视角下，patch embedding 前输入长什么样
论文 III-B 说得很关键：

```text
For the SNN model, ... we use only one event voxel chunk.
... partition the temporal channel, containing 10 bins, into blocks
along with their corresponding polarities.
This yields an event representation of size ... with 2 time steps.
```

翻成最白的话：

```text
对当前 SNN baseline，
输入不是原始事件流，
而是一个 voxel chunk；
它里面有 10 个时间 bin，
再配上正负极性。
```

结合当前代码主线，你现在可以先把送进模型前端的输入记成：

```text
[B, 10, 2, 288, 384]
```

### 3. 论文里的 SFG 在代码里对应什么
论文说：

```text
先经过一个 spiking convolutional module，
再经过两个 residual blocks，
把分辨率降一半。
```

在当前代码里，这个“前端 + patch embedding”的实现没有用一个单独叫 `SFG` 的大类暴露出来，
而是体现在：
- `embed_type`
- `patch_embed`
- patch embed 内部用的 spiking conv 风格实现

最关键的入口在：
- [Spiking_swin_transformer3D.py](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)

这里会创建：

```python
self.patch_embed = eval(embed_type)(...)
```

而当前 baseline 配置里：

```text
use_arc: ["swinv1", "MS_PED_Spiking_PatchEmbed_Conv_sfn"]
swin_patch_size: [1,1,2,2]
base_num_channels: 96
```

所以你现在要先形成一个认知：

```text
代码里的 patch embedding，
其实已经把论文说的 “SFG + SPE 前端作用” 一起承担了
```

### 4. patch embedding 的输入、输出维度
当前主线里，进入 patch embedding 前：

```text
x: [B, 10, 2, 288, 384]
```

论文里想要达到的效果是：

```text
先把输入变成更高维的时空特征，
同时把空间尺寸先减半，
让后面的 Swin encoder 更容易处理。
```

当前配置下，第一层输出最粗略可记成：

```text
[B, 96, 2, 144, 192]
```

这里 3 个最重要的变化是：

1. 通道数从原始输入表示变成 `96`
2. 空间尺寸从 `288x384` 变成 `144x192`
3. 那一维 `2` 还保留着

### 5. 每个变化分别意味着什么

#### 通道从“输入表示”变成 96
这表示：

```text
模型不再把输入只看成“10 个 bin + 正负极性”这种原始事件表示，
而是把它投影到 96 维特征空间里。
```

最白的话：

```text
从“原始记录”变成“模型更好用的特征”
```

#### 空间尺寸减半
这表示：

```text
模型先把分辨率压低一点，
减少后面 transformer 计算量
```

最白的话：

```text
先把图缩小一点，
不然 attention 太贵
```

#### 那一维 2 还在
这表示：

```text
这里还保留了短时间/深度这一维，
后面 STSF encoder 仍然会沿这个维度做时空处理
```

### 6. patch embedding 的“底层算法”最白解释
你现在先不要把 patch embedding 想成高深的 transformer 术语。

它本质上做的是：

```text
用一个带步长的前端卷积式投影，
把原始事件表示切成更粗一点的空间块，
并把每个块投影到高维特征空间。
```

换句话说：

```text
patch embedding = 下采样 + 特征投影
```

只是这里它不是普通 ANN patch embedding，
而是脉冲化版本，并且论文里强调它带 shortcut 设计。

### 7. 论文里的 “shortcut patch embedding” 为什么重要
论文 III-C-1 说得很明确：

```text
Inspired by QKFormer, we add a deformed shortcut for the patch embedding module,
which boosts the performance.
```

最白的话：

```text
作者发现，
光靠主分支做 patch embedding 还不够，
于是额外加了一条 shortcut 支路，
帮助前端更稳定、更好地保留信息。
```

这一步对你后面做研究很重要，
因为它说明：

```text
前端 patch embedding 本身就是作者重点优化过的地方，
不是一个可有可无的小模块。
```

### 8. 当前阶段你最该记住的一句话

```text
参考论文看，SDformerFlow 的 patch embedding 不是简单“切块”；
它前面本质上还有一个脉冲特征生成前端，
整体作用就是先把事件输入下采样并投影成高维时空特征，
好让后面的 Swin spikeformer encoder 去做更强的时空建模。
```
---

## patch embedding 细化版：`swin_patch_size`、残差块、shortcut patch embedding

### 1. `swin_patch_size` 是干嘛的
当前配置是：

```text
swin_patch_size: [1, 1, 2, 2]
```

在当前真正用的 patch embedding 类：
- [MS_PED_Spiking_PatchEmbed_Conv_sfn](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)

里，最关键的其实是最后两个：

```text
[2, 2]
```

它们决定：

```text
空间 patch / 空间 stride
```

最白的话：

```text
告诉前端：
后面要按 2x2 的空间步长继续做 patch 投影
```

前两个 `1,1` 在这个具体类里没有变成“时间再压缩一次”的操作，
你当前可以先理解成：

```text
当前实现里不对时间维和极性维再做 patch 压缩
```

### 2. 当前前端到底分几步
这不是一个单层 patch embedding，
而是 4 步：

```text
head
-> conv
-> residual_encoding
-> proj
```

对应代码：

```python
xs = self.head(x)
xs = self.conv(xs)
out = self.residual_encoding(xs)
out = self.proj(out)
```

### 3. 每一步维度怎么变
从训练器送进来的输入开始：

```text
[B, 10, 2, 288, 384]
```

先整理成：

```text
[10, B, 2, 288, 384]
```

然后：

#### a) `head`
- 类：`SpikingConvEncoderLayer`
- 参数：`2 -> 48`, `kernel=3`, `stride=1`

维度变成：

```text
[10, B, 48, 288, 384]
```

作用：

```text
先把 2 通道极性输入变成 48 维特征
但不下采样
```

#### b) `conv`
- 类：`MS_SpikingConvEncoderLayer`
- 参数：`48 -> 96`, `kernel=3`, `stride=2`

维度变成：

```text
[10, B, 96, 144, 192]
```

作用：

```text
第一次空间下采样
同时把通道提到 96
```

#### c) `residual_encoding`
- 类：`MS_spiking_residual_feature_generator`
- 里面有 `num_res = 2` 个 `MS_ResBlock`

维度保持不变：

```text
[10, B, 96, 144, 192]
```

作用：

```text
在这个尺度上继续把局部特征做强
但不再改尺寸
```

#### d) `proj`
- 类：`SpikingPEDLayer`
- 主分支：`3x3 conv`, `stride=2`
- shortcut 分支：`1x1 conv`, `stride=2`

维度变成：

```text
[10, B, 96, 72, 96]
```

然后在 Swin backbone 里再重排成：

```text
[B, 96, 10, 72, 96]
```

### 4. 为什么一次下采样后还要加两层维度不变的 conv 和残差
因为作者前端的目标不是“尽快缩小图片”，
而是：

```text
先把事件特征做扎实，再交给 transformer
```

所以这里的逻辑是：

```text
先下采样一次
-> 在这个尺度上用残差块再加工两轮
-> 再做带 shortcut 的 patch embedding
```

最白的话：

```text
先把图缩小一点，
但不要立刻扔给 transformer，
先在这个尺度上再把特征揉一揉
```

### 5. 这些维度不变的卷积是 1x1 吗
不是。

这里你看到的前端 residual 块：
- `MS_ResBlock`

里面主要是：

```text
conv1: 3x3
conv2: 3x3
```

所以：

```text
前端 residual 里的主要卷积不是 1x1，而是 3x3
```

真正的 `1x1` 出现在：
- `SpikingPEDLayer` 的 shortcut 分支 `conv_res`

它的作用是：

```text
做一条低成本 shortcut 下采样支路
```

### 6. shortcut patch embedding 维度为什么又变了
因为 `SpikingPEDLayer` 自己本身就是一个：

```text
带 shortcut 的下采样投影层
```

它有两条路：

#### 主分支
- `3x3 conv`
- `stride=2`

#### shortcut 分支
- `1x1 conv`
- `stride=2`

最后两条路相加。

所以它会把：

```text
[10, B, 96, 144, 192]
```

变成：

```text
[10, B, 96, 72, 96]
```

最白的话：

```text
shortcut patch embedding 就是在 patch 投影这一步，
一边走正常卷积分支，
一边走 1x1 shortcut 分支，
两边一起下采样，最后再加起来。
```

### 7. 这一节最该记住的一句话

```text
当前 baseline 的前端不是“只下采样一次”；
它先用 head 做特征投影，
再用 stride=2 的 conv 做第一次空间缩小，
再用 2 个 3x3 残差块在这个尺度上增强特征，
最后再用带 1x1 shortcut 的 patch embedding 做第二次空间缩小，
得到真正送进 Swin 的 token 特征。
```

---

## patch embedding 检查题纠正：residual 不是 1x1，1x1 只在 shortcut 分支里

用户当前回答：

```text
前端过 head、下采样、resblock、embedding；
第一次下采样在第二步；
residual 是 1x1
```

这里前两句基本对，但最后一句要纠正：

```text
前端 residual 里的主要卷积不是 1x1，而是 3x3
```

真正的 `1x1` 出现在：
- `SpikingPEDLayer` 的 `conv_res`

它不是 residual 主体卷积，
而是 shortcut patch embedding 里的捷径支路。

---

## 用户追问：为什么 shortcut 用 1x1 stride=2，会不会“跳采样看不到全局”

### 1. 先说最短答案

```text
会丢一部分局部细节，但这条 1x1 分支本来就不是负责看全局的；
它只是 shortcut 支路，主分支仍然是 3x3 stride=2。
```

所以不能把整个 patch embedding 理解成“只用 1x1 stride=2 下采样”。

真正情况是：

```text
主分支：3x3 stride=2
shortcut 分支：1x1 stride=2
两条路最后相加
```

### 2. 为什么要有 1x1 stride=2 这条 shortcut
最白的话：

```text
它的作用不是“看更大范围”，
而是给主分支提供一条更直接、更便宜的捷径。
```

换句话说：

```text
主分支负责认真提特征
shortcut 分支负责快速把输入对齐到同样尺寸和通道，再加回来
```

这样做的好处通常是：
- 更容易训练
- 信息不完全只靠主分支
- 保留更直接的低级信息通路

### 3. 1x1 stride=2 到底会做什么
`1x1` 的意思是：

```text
它不看邻域，只看当前位置这个像素点的通道信息
```

`stride=2` 的意思是：

```text
它每隔一个位置取一次，
所以空间尺寸减半
```

所以你担心的点是对的：

```text
只靠 1x1 stride=2，确实不会主动聚合周围邻域信息
```

但是这里它不是单独工作，它只是 shortcut。

### 4. 谁来负责“看邻域”
负责看邻域的是主分支：

- `3x3 conv`, `stride=2`

这个 `3x3` 本身就会看周围局部邻域。

再往前还有：
- `head` 的 `3x3`
- 前面的 2 个 residual block 里的 `3x3 + 3x3`

所以整个前端并不是：

```text
一直在跳采样
```

而是：

```text
先用很多 3x3 做局部特征提取，
再在最后用一条 1x1 shortcut 做低成本对齐
```

### 5. 那“全局”是谁来负责
这个问题也很关键。

当前前端本来就不是负责真正“全局建模”的。

最白的话：

```text
前端主要负责局部特征提取和下采样，
真正更大范围的建模，
是后面的 Swin attention 去做。
```

所以职责分工是：

#### 前端
负责：
- 局部模式
- 低级特征
- 下采样
- 初始投影

#### 后面的 Swin encoder
负责：
- 更大感受野
- window attention
- 更强上下文建模

### 6. 为什么两次下采样不算“太过分”
因为当前前端两次下采样以后：

```text
288x384
-> 144x192
-> 72x96
```

对 transformer 来说，这其实是很常见、也很必要的。

原因很简单：

```text
如果不先把空间尺寸压下来，
后面的 attention 计算会非常贵
```

所以作者是在做一个折中：

```text
牺牲一部分最细节的空间分辨率，
换取更可控的计算量和更高层的建模能力
```

### 7. 当前阶段最该记住的一句话

```text
1x1 stride=2 的 shortcut 分支不是用来看全局的，
它只是一个低成本捷径；
真正负责局部感受野的是前端那些 3x3 卷积，
真正负责更大范围建模的是后面的 Swin attention。
```


## ??????patch embedding ?????? Swin stage ???

???????
- `1x1 stride=2` ? shortcut ????????????
- ?????????????????`3x3`
- ?????????????????? Swin attention

????????

### 1. patch embedding ??????? backbone ???
?? patch embedding ???????

```text
[10, B, 96, 72, 96]
```

??? backbone ?????????

```text
[10, B, 96, 72, 96]
-> [B, 96, 10, 72, 96]
```

?????? Swin stage ????????

?????

```text
???????????? 96 ???????? 72x96???????? 10 ???????
```

### 2. ??????????????

```text
[B, 96, 10, 72, 96]
```

???
- `B`?batch size
- `96`??????
- `10`???????? backbone ????/???
- `72 x 96`?????

????????? `10` ???????????????????????? Swin3D ?????/????

### 3. ??? stage ??????
????????????
- `MS_Spiking_SwinTransformer3D_v2`
- ????? `Spiking_Swin_BasicLayer`

????
- `D:\code\sdformer_codex\SDformer	hird_party\SDformerFlow\models\STSwinNet_SNN\Spiking_swin_transformer3D.py`

?? stage 0 ?????
- depth = 2
- num_heads = 3
- window_size = `[2, 9, 9]`

????
- ??? stage ? 2 ? Swin block
- ?? block ? 3 ? attention head
- window ?????
  - ??/??? 2
  - ??? 9
  - ??? 9

### 4. ?? stage ??????? rearrange
? `BasicLayer.forward` ??????

```text
[B, C, D, H, W]
-> [B, D, H, W, C]
```

????

```text
[B, 96, 10, 72, 96]
-> [B, 10, 72, 96, 96]
```

?????

```text
??? C ?????????? window ?????? window ?? attention?
```

### 5. ??? stage ???block ??????
?? block ?????????? 5 ??

```text
????
-> ?????
-> ??????? attention
-> ???????
-> ?? MLP
```

???
????????? attention??? baseline ????

```text
window attention
```

????
- ???????? attention
- ?????? token ????

??? Swin ?????????

### 6. window_size = [2, 9, 9] ??????
?????

```text
[B, 10, 72, 96, 96]
```

??????????????????????

```text
2 x 9 x 9
```

????
- ???/?????? 2 ?
- ??????? 9x9 ?????

??????? Swin stage ? block???????????????

```text
???????????
```

### 7. ??? stage ???????????
?????????

**? block ??????????????**

???????? stage ? block ???????????

```text
[B, 10, 72, 96, 96]
```

???????

```text
block ???????????????
?????????
```

### 8. ??? stage ??? block ????? stage ????
?? stage 0 ? depth ? 2?????? block?

? Swin ?????????
- ?? block ??? window
- ??? block ? shift ?? window

?????

```text
??? block ???????????
??? block ?????????????????????????????????
```

????? block ?????????

```text
????? attention ?????
?????????????????
```

### 9. ??? stage ?????????????
stage ??? block ?????????????? stage ????

```text
patch merging / downsample
```

? stage 0 ???????

```text
[B, 96, 10, 72, 96]
-> [B, 192, 10, 36, 48]
```

????
- ?????`96 -> 192`
- ?????`72x96 -> 36x48`
- ??/??? `10` ????

????????????

```text
block ???????
stage ??? patch merging ?????
```

### 10. ??? stage ???????

```text
patch embedding ??
[10, B, 96, 72, 96]
-> backbone ??
[B, 96, 10, 72, 96]
-> stage 0 ????
[B, 10, 72, 96, 96]
-> ?? Swin block ??? window `[2,9,9]` ?? attention
-> ??????
[B, 10, 72, 96, 96]
-> stage ?? patch merging
[B, 192, 10, 36, 48]
```

### 11. ???????
??? Swin stage ?????????????????

```text
???????????????
??????? attention ?????????????
????????????????????? stage?
```

?????
- patch embedding?????????????
- stage 0??????????? attention ???
- patch merging????????

### 12. ??????
1. patch embedding ???backbone ???????????????
2. ??? Swin stage ? block ???????????????????
3. `window_size = [2, 9, 9]` ??????????
4. ??? stage ??? patch merging ?? `96, 72x96` ?????

## 第九课：第一个 stage 里的 Swin attention 具体怎么算

用户问题：
- 带 Swin 的注意力具体怎么做
- 内部怎么计算
- 维度一步步怎么变化
- 用第一个 stage 举例说明

### 1. 先说清楚：当前 baseline 用的不是“标准 softmax QKV attention”
在当前最终 baseline 主线里：
- stage 用的是 `MS_Spiking_Swin_BasicLayer`
- block 用的是 `MS_Spiking_SwinTransformerBlock3D`
- block 里的 attention 模块实际是：
  - `Spiking_QK_WindowAttention3D`

所以这不是你在 ViT 教程里常见的：

```text
QK^T -> softmax -> 乘 V
```

当前这个实现更接近：

```text
先用 Q 生成一个 token gate
再去调制 K
最后再投影回输出
```

也就是说：
- 它保留了 window-based Swin 的外壳
- 但 attention 核心算子已经不是标准 softmax attention 了

### 2. 第一个 stage 的输入从哪里来
patch embedding 之后，backbone 接到的是：

```text
[B, 96, 10, 72, 96]
```

含义：
- `B`：batch
- `96`：通道
- `10`：时间/深度轴
- `72 x 96`：空间尺寸

进入第一层 `BasicLayer.forward` 后，先重排成：

```text
[B, 10, 72, 96, 96]
```

也就是：

```text
[B, D, H, W, C]
```

### 3. 第一个 stage 的 block 配置
第一层 stage 当前配置是：

```text
depth = 2
num_heads = 3
window_size = [2, 9, 9]
```

最白的话：
- 这一层有 2 个 block
- 每个 block 分成 3 个 head
- 每个 window 看 `2 x 9 x 9` 的局部时空区域

### 4. 第一个 stage 里，先做 padding
这里有个很重要但容易漏的细节：

- `D = 10`，能被 `2` 整除
- `H = 72`，能被 `9` 整除
- `W = 96`，**不能**被 `9` 整除

所以程序会先把宽度从：

```text
96 -> 99
```

这样才能整齐切成 `9` 宽的小窗口。

所以切窗前的实际内部尺寸是：

```text
[B, 10, 72, 99, 96]
```

### 5. 一个样本会被切成多少个 window
按 window_size = `[2, 9, 9]` 来算：

- 时间上：`10 / 2 = 5`
- 高度上：`72 / 9 = 8`
- 宽度上：`99 / 9 = 11`

所以每个样本总窗口数是：

```text
5 x 8 x 11 = 440
```

也就是说：

```text
一张样本在第一个 stage 里，会被切成 440 个局部时空 window
```

### 6. 每个 window 的原始 token 数是多少
一个 window 大小是：

```text
2 x 9 x 9
```

所以 token 总数是：

```text
2 x 9 x 9 = 162
```

这就是当前 stage 里一个局部 window 的 token 数。

### 7. 切窗后，attention 模块真正接到什么维度
`window_partition_v2` 返回的不是常见的 `(B*nW, N, C)`，
而是：

```text
[T, B_, H_w, W_w, C]
```

在当前 stage 0 下就是：

```text
[2, B*440, 9, 9, 96]
```

这里：
- `T = 2`
- `B_ = B * 440`
- `H_w = 9`
- `W_w = 9`
- `C = 96`

也就是说，attention 模块每次处理的是：

```text
一个 window 内的 2 帧、9x9 空间、96 通道特征
```

### 8. 进入 `Spiking_QK_WindowAttention3D` 后先做什么
attention 一开始会先做：

```text
x = self.proj_sn(x)
```

最白的话：

```text
先过一次脉冲神经元，让输入特征先脉冲化/门控一下
```

然后分别生成：
- `q = linear_q(x)`
- `k = linear_k(x)`

注意：这个类里没有标准的 `v` 分支。

### 9. Q 和 K 的维度怎么变
输入还是：

```text
[2, B*440, 9, 9, 96]
```

先经过线性层后，形状不变：

```text
q: [2, B*440, 9, 9, 96]
k: [2, B*440, 9, 9, 96]
```

然后代码把它们 reshape 成多头形式。

当前：
- `num_heads = 3`
- `head_dim = 96 / 3 = 32`

所以：

#### q 变成

```text
[2, B*440, 3, 81, 32]
```

为什么是 `81`？
因为这里先把一个时间片里的 `9x9` 拉平：

```text
9 x 9 = 81
```

#### k 变成

```text
[B*440, 3, 162, 32]
```

为什么是 `162`？
因为 k 这里把整个 window 的：

```text
2 x 9 x 9 = 162
```

都拉平了。

### 10. 当前这个 attention 到底怎么算
这里不是标准 `QK^T`。

代码做的是：

```python
att_token = q.sum(dim=-1, keepdim=True)
att_token = self.sn2_q(att_token)
attn = k.mul(att_token.reshape(B_, self.num_heads, -1, 1))
```

最白的话：

#### 第一步
把 `q` 在 head_dim 上求和：

```text
[2, B*440, 3, 81, 32]
-> [2, B*440, 3, 81, 1]
```

意思是：

```text
给每个 token 生成一个“强弱权重”
```

#### 第二步
再把这 2 个时间片的 token 权重拼成：

```text
[B*440, 3, 162, 1]
```

#### 第三步
用这个 token 权重去乘 `k`：

```text
k:    [B*440, 3, 162, 32]
gate: [B*440, 3, 162, 1]
-> attn: [B*440, 3, 162, 32]
```

最白总结：

```text
当前 attention 不是先算 token-token 相关性矩阵，
而是先从 Q 里提一个 token gate，
再用这个 gate 去调制 K。
```

### 11. 为什么这和标准 attention 不一样
标准 attention 大概是：

```text
QK^T -> 得到 [N, N] 的注意力矩阵
```

当前这个实现没有显式形成：

```text
[162, 162]
```

这样的 token-token 矩阵。

它更像一种：

```text
轻量 token gating / QK 调制
```

这也是为什么它更适合后面谈：
- 低成本 attention
- 硬件友好
- 稀疏化和简化算子

### 12. attention 输出后怎么变回 window 特征
当前 `attn` 还是：

```text
[B*440, 3, 162, 32]
```

然后代码把它 reshape 回：

```text
[B*440, 3, 2, 9, 9, 32]
```

再 permute + reshape 成：

```text
[2, B*440, 9, 9, 96]
```

也就是：

```text
重新回到“一个 window 的时空特征”格式
```

然后再过：
- `attn_sn`
- `proj`
- `proj_bn`

最后变成：

```text
[B*440, 162, 96]
```

这是一个 window 展平后的输出形式。

### 13. 怎么从 window 拼回整张特征图
block 里会先把：

```text
[B*440, 162, 96]
```

reshape 回：

```text
[-1, 2, 9, 9, 96]
```

然后用 `window_reverse(...)` 把很多小窗口拼回完整特征图。

拼回以后得到：

```text
[B, 10, 72, 99, 96]
```

如果之前有 padding，就再裁掉补出来的部分，回到：

```text
[B, 10, 72, 96, 96]
```

### 14. block 里除了 attention 还有什么
attention 结束后，block 还会做两件事：

#### 1. 残差相加

```text
x = attention(x) + shortcut
```

#### 2. MLP

MLP 不改整体形状，输入输出还是：

```text
[B, 10, 72, 96, 96]
```

最白的话：

```text
attention 负责做局部时空信息混合，
MLP 负责在每个位置上继续做通道维特征变换。
```

### 15. 第一个 stage 的两个 block 是怎么配合的
当前 stage 0 有两个 block：

- block 0：不 shift
- block 1：shift 一半 window

所以：

```text
第一个 block 先在固定窗口里做局部 attention；
第二个 block 再把窗口错开，
这样原本不同窗口里的位置也能间接交流。
```

### 16. 第一个 stage 的完整维度线

```text
输入 backbone:
[B, 96, 10, 72, 96]

-> stage 0 内部重排
[B, 10, 72, 96, 96]

-> padding
[B, 10, 72, 99, 96]

-> window partition
[2, B*440, 9, 9, 96]

-> q:
[2, B*440, 3, 81, 32]

-> k:
[B*440, 3, 162, 32]

-> token gate from q:
[B*440, 3, 162, 1]

-> gated k / attn:
[B*440, 3, 162, 32]

-> reshape back:
[2, B*440, 9, 9, 96]

-> merge windows:
[B, 10, 72, 99, 96]

-> remove padding:
[B, 10, 72, 96, 96]

-> residual + MLP:
[B, 10, 72, 96, 96]

-> stage 0 end patch merging:
[B, 192, 10, 36, 48]
```

### 17. 这一节最重要的结论
当前 baseline 第一个 stage 里的 attention：

```text
不是标准 softmax(QK^T)V，
而是局部 window 内的 spiking QK-gating 形式。
```

它的好处是：
- 结构更轻
- 没有显式的大 token-token 注意力矩阵
- 更适合你后面研究：
  - 注意力改进
  - 稀疏化
  - 硬件加速器映射

### 18. 这一节检查题
1. 当前第一个 stage 里的一个 window token 数是多少？
2. 为什么宽度 `96` 在切 `9x9` window 前要 padding 到 `99`？
3. 当前 baseline 第一层 attention 为什么说它不是标准 softmax attention？
4. stage 0 结束后，维度为什么会从 `96,72x96` 变成 `192,36x48`？

## 第十课：从多头 Q/K 开始，重新讲第一个 stage 的 attention

用户反馈：
- 前一节没太搞懂 `q/k` 那里怎么算注意力
- 对多头注意力本身也不熟
- 需要从“分多头 q”开始重新详细讲

### 1. 先不看这份代码，先讲“标准多头注意力”最白版本
假设现在一个窗口里有：

```text
N 个 token
每个 token 的特征维度是 C
```

那么输入可以先想成：

```text
[N, C]
```

标准 attention 会做三件事：

```text
Q = XWq
K = XWk
V = XWv
```

也就是：
- 从同一份输入特征 `X`
- 各自线性投影出 `Q`
- 各自线性投影出 `K`
- 各自线性投影出 `V`

最白的话：

```text
Q：我想看什么
K：我有什么可以被看
V：真正被拿来汇总的内容
```

### 2. 什么叫“多头”
假设：

```text
C = 96
num_heads = 3
```

那就会把 `96` 维特征拆成 3 份：

```text
96 = 3 x 32
```

也就是说：
- 第 1 个 head 看 32 维
- 第 2 个 head 看 32 维
- 第 3 个 head 看 32 维

最白的话：

```text
多头就是：
不要只用一种“看问题的方法”，
而是让 3 组子空间并行地各看各的，
最后再拼回来。
```

### 3. 标准多头 attention 的形状怎么变
假设输入是：

```text
[N, 96]
```

做完线性层后：

```text
Q: [N, 96]
K: [N, 96]
V: [N, 96]
```

然后按 3 个 head 拆开：

```text
Q -> [3, N, 32]
K -> [3, N, 32]
V -> [3, N, 32]
```

再做：

```text
QK^T
```

每个 head 都会得到一个：

```text
[N, N]
```

的注意力矩阵。

意思是：

```text
第 i 个 token 会去看第 j 个 token 多大程度上和自己相关
```

最后再用这个矩阵去加权 `V`。

### 4. 当前 baseline 这份代码和标准 attention 最大的区别
当前 baseline 在第一个 stage 里，用的不是标准：

```text
softmax(QK^T)V
```

而是一个更轻的变体：

```text
先从 Q 提一个 token gate
再用它去调制 K
```

所以你一定要先分清：

```text
“多头”这个概念是共通的，
但“多头之后怎么计算注意力”这一点，
当前代码不是标准 transformer 教科书写法。
```

### 5. 现在回到第一个 stage 的真实输入
第一个 stage 切窗之后，attention 模块真正拿到的是：

```text
[T, B_, H_w, W_w, C]
```

当前 stage 0 下具体就是：

```text
[2, B*440, 9, 9, 96]
```

含义：
- `T = 2`：这个窗口在时间/深度上有 2 层
- `B*440`：总窗口数
- `9 x 9`：空间窗口大小
- `96`：通道数

最白的话：

```text
现在 attention 处理的不是整张图，
而是很多个局部小立方体；
每个小立方体大小是 2 帧 x 9 x 9 空间。
```

### 6. `q` 是怎么从输入里来的
代码先做：

```python
q = self.linear_q(x)
```

这一步不会先改整体形状，只是把每个位置上的 96 维特征做一次线性投影。

所以先还是：

```text
q: [2, B*440, 9, 9, 96]
```

### 7. 为什么要把 q reshape 成多头
当前配置：

```text
num_heads = 3
C = 96
head_dim = 96 / 3 = 32
```

所以代码做完多头拆分后，实际上是在说：

```text
原来每个位置的 96 维特征，
现在拆成 3 组，
每组 32 维。
```

这时 `q` 变成：

```text
[2, B*440, 3, 81, 32]
```

为什么是 `81`？

因为这里是：

```text
把一个时间片里的 9x9 空间拉平
9 x 9 = 81
```

所以这一维的含义是：

```text
一个时间片里的 81 个空间 token
```

你现在可以把它读成：

```text
[时间片数, 窗口批次, 头数, token数, 每头维度]
```

也就是：

```text
[2, B*440, 3, 81, 32]
```

### 8. `k` 为什么长得不一样
代码里 `k` 最后变成：

```text
[B*440, 3, 162, 32]
```

为什么这里不是 `81`，而是 `162`？

因为 `k` 这里把整个 window 的时间和空间一起拉平了：

```text
2 x 9 x 9 = 162
```

所以：

```text
k` 的 token 维 = 整个窗口的所有 token
q` 这里先保留了“两个时间片各自的 81 个 token”
```

这就是这份代码为什么看起来不那么像标准 attention 的原因之一。

### 9. 现在最关键一步：从 q 里提 token gate
代码做：

```python
att_token = q.sum(dim=-1, keepdim=True)
```

这里 `dim=-1` 指的是最后那个 `32`，也就是每个 head 的特征维。

所以：

```text
[2, B*440, 3, 81, 32]
-> 在 32 维上求和
-> [2, B*440, 3, 81, 1]
```

最白的话：

```text
现在程序不再保留每个 token 的完整 32 维描述，
而是把它压成一个单独的“强弱值”。
```

你可以把这个值先粗暴理解成：

```text
这个 token 当前有多“活跃”
```

然后它还会再过一次脉冲神经元：

```python
att_token = self.sn2_q(att_token)
```

意思就是：

```text
再对这个 token 强弱值做一次脉冲门控
```

### 10. 为什么要把 `att_token` reshape 成 162
当前 `att_token` 还是：

```text
[2, B*440, 3, 81, 1]
```

它代表：
- 第 1 帧有 81 个 token
- 第 2 帧也有 81 个 token

代码下一步会把这两帧拼起来，变成：

```text
[B*440, 3, 162, 1]
```

最白的话：

```text
把窗口里 2 帧的 token gate 合成一个完整窗口的 162 个 token gate
```

### 11. 然后怎么和 k 结合
此时：

```text
k:         [B*440, 3, 162, 32]
att_token: [B*440, 3, 162, 1]
```

代码做：

```python
attn = k.mul(att_token)
```

这一步不是矩阵乘，而是逐元素乘。

最白的话：

```text
让每个 token 自己的 gate，去放大或压低这个 token 在 k 里的 32 维特征。
```

也就是说：

```text
某个 token gate 大
-> 这个 token 的 k 特征被保留得更多

某个 token gate 小
-> 这个 token 的 k 特征被压得更弱
```

所以这一层注意力的核心不是：

```text
token A 去看 token B 的相关性矩阵
```

而更像：

```text
每个 token 先给自己一个重要性门控，
再去调制自己的特征
```

### 12. 这和标准 attention 最大的认知差别
标准 attention 的核心问题是：

```text
token 和 token 之间谁更相关？
```

所以会显式出现：

```text
[N, N]
```

的关系矩阵。

当前这份实现更像是在问：

```text
这个 token 自己该被放大还是缩小？
```

所以它更像：

```text
token-wise gating
```

而不是标准的：

```text
token-token relation matrix
```

### 13. 为什么它还叫 attention
因为它依然在做：

```text
对 token 的选择性强调
```

只是这个“选择性”的实现方式，不是传统 softmax 矩阵，而是：

```text
从 Q 提 gate，再调制 K
```

### 14. 这一节你最该记住的一句话

```text
多头的意思是把 96 维拆成 3 个 head、每个 head 32 维；
当前 baseline 第一个 stage 里，
q 先被拆成 `[2, B*440, 3, 81, 32]`，
再在 32 维上求和得到 token gate，
最后这个 gate 去逐元素调制 k，
所以它不是标准 softmax(QK^T)V，而是轻量的 QK-gating attention。
```

### 15. 这一节检查题
1. 当前第一个 stage 里，为什么 `96` 会被拆成 `3 x 32`？
2. `q.sum(dim=-1)` 这一步在最白的话里是什么意思？
3. 当前实现里，`att_token` 是怎么作用到 `k` 上的？
4. 为什么说它不是标准的 token-token 注意力矩阵？

## 第十一课：当前 attention 里 token 相关性、位置编码、shift window 到底在哪里

用户追问：
- 现在看起来像“用自己的 q 给自己打分”，那 token 间相关性怎么来？
- 论文里还有位置编码，这里是不是还没讲？
- 后面是不是还有“改变位置继续 Swin”的操作？

### 1. 先给结论
当前 baseline 第一个 stage 里的 `Spiking_QK_WindowAttention3D`：

```text
没有显式计算标准的 token-token 相关性矩阵
```

也就是说，它没有真的形成：

```text
[N, N]
```

那种“第 i 个 token 看第 j 个 token 多相关”的完整矩阵。

所以如果你问：

```text
它有没有标准 transformer 意义上的 pairwise token relation？
```

答案是：

```text
没有显式地算出来
```

这正是这个注意力变体“更轻”的代价和特点。

### 2. 那它现在到底在做什么
当前实现更像：

```text
每个 token 先根据自己的 q 生成一个 gate
再用这个 gate 去调制这个 token 对应的 k 特征
```

所以你刚才说的：

```text
相当于用自己的 q 给自己打分
```

这个直觉是对的。

但我帮你修正一处：

```text
不是简单地“1 保留 / 0 清零”这么死板，
更准确地说是：
用一个 token gate 去控制这个 token 的特征强弱。
```

在脉冲版本里，你可以先把它近似理解成二值/门控行为；但从工程上说，它本质是：

```text
token-wise gating
```

### 3. 那 token 间相关性到底从哪里来
这是关键。

既然这个 attention 核心里没有显式：

```text
QK^T
```

那 token 间交互主要不是靠“一个大关系矩阵”来完成的，而是靠下面几件事**间接形成**的。

#### A. 局部 window 这个外壳本身
虽然当前 attention 核心没显式算 token-token 矩阵，但它仍然发生在：

```text
一个局部 window 里
```

也就是说：
- 一次只处理 `2 x 9 x 9` 这一小块
- 所有 token 都被组织在同一个局部块中

最白的话：

```text
它不是全图乱看，而是在局部小房间里做门控和特征处理。
```

#### B. K 里带了位置编码
代码里有这一步：

```python
positional_encoding = self.positional_encoding.reshape(T, 1, H, W, C)
k = k + positional_encoding
```

也就是说：

```text
位置编码是加在 k 上的
```

不是加在 q 上，也不是最后再补。

这步的作用是：

```text
让不同位置的 token 即使特征值很像，
也不会被完全当成“同一个位置”的东西。
```

最白的话：

```text
告诉模型：
“你不光要看这个 token 长什么样，
还要知道它站在窗口里的什么位置。” 
```

所以虽然没有显式 token-token 相似度矩阵，位置编码仍然在帮助模型区分：
- 这是左上角 token
- 这是中间 token
- 这是后一帧还是前一帧

#### C. 后面的 MLP 和残差
attention 后面不是马上结束，还会：

```text
残差相加
-> MLP
```

MLP 虽然不是 token-token 混合，但会继续改变每个 token 的通道特征。

#### D. 更重要的是：shift window
这一步就是你问的“后面是不是还有改变位置继续 Swin 的操作”。

答案是：

```text
有，而且这就是 Swin 的关键操作之一。
```

在第一个 stage 里：
- block 0：`shift_size = (0, 0, 0)`
- block 1：`shift_size = (1, 4, 4)`

为什么是 `(1, 4, 4)`？
因为 window size 是：

```text
[2, 9, 9]
```

程序会取一半：

```text
[1, 4, 4]
```

所以第二个 block 不是在和第一个 block 完全相同的位置切窗，而是：

```text
把窗口往时间、空间方向都平移一半，再切一次
```

最白的话：

```text
第一个 block 先在固定小房间里看；
第二个 block 再把房间挪一挪，
让原来不在一个房间里的 token，
下一次有机会进到同一个房间里。
```

这就是当前 baseline 里“token 间间接交流”的一个主要来源。

### 4. 所以这里的 token 相关性该怎么理解
你现在不要再用“有没有标准 attention 矩阵”这个单一标准去理解它。

更准确地说：

```text
当前这份实现里，
显式的 pairwise token-token 相似度没有被完整算出来；
token 间交互主要是通过：
局部窗口约束 + 位置编码 + shifted windows + 后续层堆叠
间接建立起来的。
```

这是一个重要结论。

因为你后面如果想做：
- 注意力改进
- token 稀疏
- 硬件加速器

就必须先接受：

```text
这个 baseline 的“attention”已经不是标准 transformer 语义下的 attention 了
```

### 5. 位置编码在当前模块里到底长什么样
当前模块里有：

```python
self.positional_encoding = nn.Parameter(
    torch.zeros(size=(1, num_heads, window_size[0] * window_size[1] * window_size[2], head_dim))
)
```

对于第一个 stage：
- `num_heads = 3`
- `window_size = [2,9,9]`
- `window token 数 = 162`
- `head_dim = 32`

所以它本质上是一份：

```text
[1, 3, 162, 32]
```

的可学习位置编码。

然后在 forward 里 reshape 成：

```text
[2, 1, 9, 9, 96]
```

再加到 `k` 上。

最白的话：

```text
它给窗口里每个时间位置、空间位置都配了一份可学习的位置特征，
然后把这份位置信息加到 k 上。
```

### 6. shift window 是怎么做的
在 block 里，如果 shift size 不为 0，就会先做：

```python
shifted_x = torch.roll(x, shifts=(-shift_d, -shift_h, -shift_w), dims=(1, 2, 3))
```

最白的话：

```text
先把整张特征图沿时间、纵向、横向轻轻挪一下
```

然后再切窗口。

等窗口 attention 做完、窗口拼回去以后，再反向 roll 回来。

所以 shift window 不是“乱移位置”，而是：

```text
为了让不同窗口边界附近的 token 也能在下一层接触到彼此
```

### 7. 为什么还要有 attn_mask
一旦做了 shift，窗口边界就会变复杂。

所以程序会先计算：

```text
attn_mask
```

作用是：

```text
防止 roll 之后本来不该在同一个有效窗口里的位置，
被错误地混到一起。
```

最白的话：

```text
窗口挪了以后，为了不让“跨错边界”的 token 乱交流，
程序会加一个掩码把不该连通的位置挡掉。
```

### 8. 这一节最重要的结论
你现在要记住下面这句：

```text
当前 baseline 的第一个 stage 里，
attention 核心不是标准 token-token softmax 矩阵，
而是 token-wise QK-gating；
位置信息通过加到 k 上的 positional encoding 注入；
token 间更大范围的交流主要靠 shifted windows 在相邻 block 中逐步建立。
```

### 9. 这一节检查题
1. 当前这份 attention 里，有没有显式算出标准的 `[N, N]` token-token 相似度矩阵？
2. 位置编码是加在 `q` 上、`k` 上，还是输出最后再加？
3. 第一个 stage 的第二个 block 为什么要 shift window？
4. 你现在该怎么理解“token 间相关性”的来源？

## 第十二课：第一个 stage 里每一步到底在哪个模块、哪几行代码实现

用户反馈：
- 需要把“每一步在代码里怎么实现的、哪个模块实现的”讲清楚
- 不能只讲概念

这一节只盯第一个 stage，也就是：

```text
[B, 96, 10, 72, 96]
```

进入第一个 `BasicLayer` 以后发生的所有事情。

### 1. stage 入口在哪里
真正进入第一个 stage 的代码在：
- `Spiking_swin_transformer3D.py:1223-1233`

这里最关键的是：

```python
1225: x = self.patch_embed(x)
1230: x = rearrange(x, 't b c h w -> b c t h w')
1232: for i, layer in enumerate(self.layers):
1233:     x, out_x = layer(x.contiguous())
```

最白的话：

```text
patch embedding 先输出 [T,B,C,H,W]
再重排成 [B,C,D,H,W]
然后逐层送进每个 Swin stage。
```

所以第一个 stage 接到的是：

```text
[B, 96, 10, 72, 96]
```

### 2. 第一个 stage 自己的入口在哪里
第一个 stage 对应的是：
- `Spiking_Swin_BasicLayer.forward`
- `Spiking_swin_transformer3D.py:1065-1088`

关键代码：

```python
1071: B, C, D, H, W = x.shape
1073: x = rearrange(x, 'b c d h w -> b d h w c')
1074: Dp = int(np.ceil(D / window_size[0])) * window_size[0]
1075: Hp = int(np.ceil(H / window_size[1])) * window_size[1]
1076: Wp = int(np.ceil(W / window_size[2])) * window_size[2]
1077: attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
1078: for blk in self.swin_blocks:
1079:     x = blk(x, attn_mask)
1083: if self.downsample is not None:
1084:     x_out = self.downsample(x)
1087: x_out = rearrange(x_out, 'b d h w c -> b c d h w')
1088: return x_out, x
```

这段代码说明 stage 的总流程是：

```text
[B,C,D,H,W]
-> 变成 [B,D,H,W,C]
-> 算 attn_mask
-> 依次过两个 block
-> 再做 patch merging
-> 返回下一层输入 x_out，以及当前层输出 out_x
```

### 3. 为什么要先把 `[B,C,D,H,W]` 变成 `[B,D,H,W,C]`
代码：

```python
1073: x = rearrange(x, 'b c d h w -> b d h w c')
```

输入：

```text
[B, 96, 10, 72, 96]
```

输出：

```text
[B, 10, 72, 96, 96]
```

作用：

```text
把通道放到最后，
方便后面切 window、做线性层、做 attention。
```

### 4. padding 和 mask 是在哪实现的
#### 4.1 mask 在哪算
代码：
- `Spiking_swin_transformer3D.py:980-993`

函数：

```python
981: def compute_mask(D, H, W, window_size, shift_size, device):
```

然后 stage 里调用：

```python
1077: attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
```

#### 4.2 这里 `Dp/Hp/Wp` 是什么
代码：

```python
1074: Dp = int(np.ceil(D / window_size[0])) * window_size[0]
1075: Hp = int(np.ceil(H / window_size[1])) * window_size[1]
1076: Wp = int(np.ceil(W / window_size[2])) * window_size[2]
```

当前第一个 stage：
- `D = 10`, `window[0]=2` -> `Dp = 10`
- `H = 72`, `window[1]=9` -> `Hp = 72`
- `W = 96`, `window[2]=9` -> `Wp = 99`

所以这里就解释了：

```text
为什么宽度要从 96 补到 99
```

目的就是：

```text
让窗口尺寸能整除当前特征图
```

#### 4.3 attn_mask 有什么用
最白的话：

```text
第二个 block 做 shift window 时，
不让本来不该在一起的 token 错误混到一起。
```

### 5. 一个 block 在哪实现
block 类在：
- `Spiking_swin_transformer3D.py:720`

真正执行是在：
- `Spiking_swin_transformer3D.py:819-914` 这一带

block 的两个关键函数是：

#### A. `SSA(...)`
- `Spiking_swin_transformer3D.py:781`

它负责：

```text
norm
-> padding
-> shift
-> partition window
-> attention
-> reverse window
-> reverse shift
```

#### B. `forward(...)`
- `Spiking_swin_transformer3D.py:866` 附近

它负责：

```text
attention 分支输出 + shortcut
-> MLP 分支 + shortcut
```

### 6. shift window 是在哪行实现的
在 `SSA(...)` 里：

```python
805: if any(i > 0 for i in shift_size):
806:     shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
807:     attn_mask = mask_matrix
```

还有 reverse shift：

```python
821: if any(i > 0 for i in shift_size):
822:     x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
```

所以：
- 第一个 block：`shift_size = (0,0,0)`，不滚动
- 第二个 block：`shift_size = (1,4,4)`，先 roll 再切窗，最后再 roll 回来

### 7. window 切分是在哪行实现的
切窗函数在：
- `Spiking_swin_transformer3D.py:100-113`

```python
100: def window_partition_v2(x, window_size):
```

在 `SSA(...)` 里调用：

```python
813: x_windows = window_partition_v2(shifted_x, window_size)
```

当前 stage 0 下：
- 输入 `shifted_x` 近似是 `[B, 10, 72, 99, 96]`
- 输出 `x_windows` 是：

```text
[2, B*440, 9, 9, 96]
```

这一步的作用：

```text
把整张特征图拆成很多个局部 2x9x9 的时空小窗口
```

### 8. attention 核心模块在哪
attention 核心模块是：
- `Spiking_QK_WindowAttention3D`
- `Spiking_swin_transformer3D.py:605-717`

真正 forward 在：
- `661-717`

### 9. attention 内部每一步代码怎么对应
#### 9.1 输入尺寸读取

```python
667: T, B_, H, W, C = x.shape
```

对应当前：

```text
[2, B*440, 9, 9, 96]
```

#### 9.2 先过一个脉冲门

```python
670: x = self.proj_sn(x.float())
```

作用：

```text
先做一次脉冲神经元门控
```

#### 9.3 生成 q

```python
671: q = self.linear_q(x)
672-674: bn + spiking neuron
```

这时 q 还是：

```text
[2, B*440, 9, 9, 96]
```

#### 9.4 生成 k，并加位置编码

```python
675: k = self.linear_k(x).float()
676-677: bn
678: positional_encoding = self.positional_encoding.reshape(T, 1, H, W, C)
679: k = k + positional_encoding
680: k = self.sn_k(k)
```

这一步很关键：

```text
位置编码就是在这里加到 k 上的
```

#### 9.5 分多头 reshape

```python
687: q, k = q.reshape(...), k.reshape(...)
```

当前 stage 0：
- `num_heads = 3`
- `head_dim = 96/3 = 32`

所以：

```text
q -> [2, B*440, 3, 81, 32]
k -> [B*440, 3, 162, 32]
```

#### 9.6 从 q 提 token gate

```python
692: att_token = q.sum(dim=-1, keepdim=True)
693: att_token = self.sn2_q(att_token)
```

也就是：

```text
[2, B*440, 3, 81, 32]
-> [2, B*440, 3, 81, 1]
```

#### 9.7 用 gate 去调制 k

```python
694: attn = k.mul(att_token.reshape(B_, self.num_heads, -1, 1))
```

这一步说明：

```text
当前实现没有显式 QK^T，
而是 token gate 逐元素调制 k
```

#### 9.8 attention 输出 reshape 回窗口特征

```python
709: x = (attn).reshape(B_, self.num_heads, T, H, W, C // self.num_heads)
710: x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C).float()
711: attn = self.attn_sn(x)
712-714: proj + bn
715: x = x.reshape(B_, N, C)
```

所以 attention 模块最后输出：

```text
[B*440, 162, 96]
```

### 10. 从窗口拼回整图在哪实现
回到 block 的 `SSA(...)`：

```python
817: attn_windows = attn_windows.view(-1, *(window_size + (C,)))
818: shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
```

作用：

```text
把窗口级输出重新拼回完整特征图
```

然后：

```python
821-824: reverse shift
826-827: 去掉 padding
```

这时又回到：

```text
[B, 10, 72, 96, 96]
```

### 11. 残差和 MLP 在哪实现
在 block 的 `forward(...)` 里：

```python
902: shortcut = x
...
910: x = self.sew_function(self.drop_path(x), shortcut, self.cnf)
914: x = self.sew_function(self.mlp(...), x, self.cnf)
```

最白的话：

```text
attention 分支做完后先和原输入相加；
再过 MLP；
MLP 输出再和当前结果相加。
```

所以一个 block 完成后，形状仍然是：

```text
[B, 10, 72, 96, 96]
```

### 12. patch merging 在哪实现
stage 末尾 downsample 在：
- `Spiking_swin_transformer3D.py:953-978`
- 类：`MS_SpikingPatchMerging`

在 stage forward 里调用：

```python
1083: if self.downsample is not None:
1084:     x_out = self.downsample(x)
```

这一层会把：

```text
[B, 10, 72, 96, 96]
-> [B, 10, 36, 48, 192]
```

然后再重排成：

```text
[B, 192, 10, 36, 48]
```

### 13. 第一个 stage 的完整“代码路径图”

```text
MS_Spiking_SwinTransformer3D_v2.forward
  -> patch_embed
  -> rearrange 到 [B,C,D,H,W]
  -> 第一个 layer = Spiking_Swin_BasicLayer.forward
       -> rearrange 到 [B,D,H,W,C]
       -> compute_mask
       -> 第一个 block = MS_Spiking_SwinTransformerBlock3D.forward
            -> SSA
               -> shift(第一个 block 无 shift)
               -> window_partition_v2
               -> Spiking_QK_WindowAttention3D.forward
               -> window_reverse
            -> residual
            -> mlp
       -> 第二个 block = MS_Spiking_SwinTransformerBlock3D.forward
            -> SSA
               -> shift(第二个 block 有 shift)
               -> window_partition_v2
               -> Spiking_QK_WindowAttention3D.forward
               -> window_reverse
               -> reverse shift
            -> residual
            -> mlp
       -> MS_SpikingPatchMerging
       -> return x_out, out_x
```

### 14. 这一节最重要的结论
如果你以后要改第一个 stage 的 attention、位置编码、shift window、token 稀疏，入口优先顺序就是：

1. `Spiking_QK_WindowAttention3D.forward`  
2. `Spiking_SwinTransformerBlock3D.SSA`  
3. `Spiking_Swin_BasicLayer.forward`

也就是：

```text
先看 attention 核心
再看 block 里怎么切窗/shift
最后看 stage 里怎么串两个 block 和 downsample
```

### 15. 这一节检查题
1. 第一个 stage 的总入口是哪个类的 `forward`？
2. 位置编码具体是在 `Spiking_QK_WindowAttention3D.forward` 的哪一步加到 `k` 上？
3. shift window 是在 block 的哪个函数里做的？
4. patch merging 是在 stage 的哪一步调用的？

## 第十三课：MLP 层和 patch merging 到底是什么、起什么作用

用户问题：
- MLP 层是什么
- 起什么作用
- patch merge 又是干什么的

### 1. 先讲 MLP 是什么
在 transformer 里，MLP 可以先最白地理解成：

```text
对每个位置自己的特征，再做两层全连接变换
```

它不是卷积，不是在看周围邻居。  
它更像是在说：

```text
这个位置自己手里的特征，
再重新加工一遍。
```

### 2. 当前 baseline 里的 MLP 在哪
代码在：
- `D:\\code\\sdformer_codex\\SDformer\\third_party\\SDformerFlow\\models\\STSwinNet_SNN\\Spiking_swin_transformer3D.py`

类是：
- `Spiking_Mlp`
- `MS_Spiking_Mlp`

其中当前最终 baseline 的 block 用的是：
- `MS_Spiking_Mlp`

因为：
- `MS_Spiking_SwinTransformerBlock3D`
- 里面设置了：
  - `mlp_module = MS_Spiking_Mlp`

### 3. MLP 在 block 里的哪一步被调用
在 block 的 `forward(...)` 里，attention 做完、残差相加以后，会做：

```text
MLP
```

也就是：

```text
attention 分支先处理一遍
-> 和 shortcut 相加
-> 再过 MLP
-> 再和当前结果相加
```

所以你可以把 block 粗略看成：

```text
attention 子层
+ shortcut
-> MLP 子层
+ shortcut
```

### 4. MLP 内部具体做什么
`Spiking_Mlp` 里主要是：

```text
fc1
-> bn/norm
-> spiking neuron
-> fc2
-> bn/norm
-> spiking neuron
```

最白的话：

```text
先把通道维拉大一点，
做一次非线性变换，
再压回原来的通道数。
```

### 5. MLP 为什么有用
attention 主要负责：

```text
不同 token/位置之间的信息混合
```

MLP 主要负责：

```text
每个位置自己内部的通道特征重组
```

最白的话：

```text
attention 负责“跟别人交换信息”；
MLP 负责“把自己拿到的信息再消化一遍”。
```

### 6. MLP 会不会改整体尺寸
不会。

在 block 里，MLP 输入输出整体形状保持不变。

比如第一个 stage 的 block 里：

```text
[B, 10, 72, 96, 96]
-> MLP
-> [B, 10, 72, 96, 96]
```

也就是说：
- 不改时间/深度轴
- 不改空间尺寸
- 不改最终通道数

它主要只是：

```text
改每个位置内部的特征表示
```

### 7. 再讲 patch merging 是什么
最白的话：

```text
patch merging = 把相邻的小块合并起来，
让空间尺寸变小、通道数变大。
```

你可以把它理解成：

```text
下采样 + 通道扩展
```

### 8. 当前 baseline 里的 patch merging 在哪
代码在：
- `D:\\code\\sdformer_codex\\SDformer\\third_party\\SDformerFlow\\models\\STSwinNet_SNN\\Spiking_swin_transformer3D.py`

类是：
- `SpikingPatchMerging`
- `MS_SpikingPatchMerging`

当前最终 baseline 主线实际用的是：
- `MS_SpikingPatchMerging`

它是在每个 stage 末尾被调用的。

### 9. patch merging 在 stage 里的哪一步调用
在：
- `Spiking_Swin_BasicLayer.forward`

里有：

```python
if self.downsample is not None:
    x_out = self.downsample(x)
```

这里的 `self.downsample`，对当前主线来说就是：

```text
MS_SpikingPatchMerging
```

所以 stage 的整体流程是：

```text
先过若干个 block
再在 stage 末尾做 patch merging
```

### 10. patch merging 内部到底做什么
它会把相邻的 2x2 空间位置取出来：

```text
x0 = 左上
x1 = 左下
x2 = 右上
x3 = 右下
```

然后把这四块在通道维拼起来。

最白的话：

```text
原来是 4 个相邻位置各有一份特征；
现在把这 4 份特征收拢到一个位置上，
所以空间尺寸减半，
但通道信息会变多。
```

### 11. 为什么 patch merging 后通道数会翻倍，而不是变成 4 倍
因为拼接之后，临时确实会变成：

```text
4C
```

但后面它还会再过一个线性层：

```text
4C -> 2C
```

所以最终结果是：

```text
空间减半
通道翻倍
```

这就是你看到的：

```text
96 -> 192
72x96 -> 36x48
```

### 12. patch merging 起什么作用
最白的话：

```text
让后面的层看到更粗、更抽象的特征，
同时减少空间 token 数量，降低后续计算量。
```

所以它的两个主要作用是：

1. **降计算量**  
空间尺寸更小，后面 attention 更省

2. **提语义层级**  
把邻近位置合并后，特征更偏高层、更抽象

### 13. MLP 和 patch merging 的区别
这个你一定要分清：

#### MLP

```text
发生在 block 里面
不改整体尺寸
作用是重组每个位置自己的通道特征
```

#### patch merging

```text
发生在 stage 末尾
会改空间尺寸和通道数
作用是下采样并进入下一层更粗的特征层级
```

### 14. 第一个 stage 里你可以这样理解

```text
block 1:
  attention + MLP

block 2:
  shifted attention + MLP

stage end:
  patch merging
```

所以：

```text
MLP 是 stage 内部每个 block 都有的“通道加工器”
patch merging 是 stage 结束时的“下采样器”
```

### 15. 这一节最重要的一句话

```text
MLP 负责在不改变整体尺寸的情况下，
把每个位置自己的通道特征再加工一遍；
patch merging 负责把相邻空间位置合并，
让空间尺寸减半、通道数翻倍，
把特征送进下一层更深的 stage。
```

### 16. 这一节检查题
1. MLP 更像是在“和别的 token 交换信息”，还是“加工当前位置自己的通道特征”？
2. patch merging 是发生在 block 内部，还是 stage 末尾？
3. patch merging 为什么会让空间减半、通道翻倍？
4. 第一个 stage 里，MLP 和 patch merging 的职责有什么本质区别？
