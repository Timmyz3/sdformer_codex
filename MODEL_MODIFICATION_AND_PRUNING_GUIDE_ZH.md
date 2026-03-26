# SDformer 模型改动与剪枝指南

## 1. 这份文档解决什么问题

这份文档专门回答下面这些问题：

- 如果要改注意力，应该从哪里插入
- 如果要增加模块，应该改哪一层
- 如果要替换 patch embed，接口在哪里
- 如果要改脉冲神经元，哪些地方要同步改
- 如果要做剪枝或稀疏，推荐从什么层级下手

这份文档不讲实验流程本身。  
实验流程请看：

- [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
- [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)

如果你想先把 baseline 模型从输入到输出彻底看懂，再决定怎么改，请先看：

- [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)

## 2. 模型总入口

训练配置里决定模型结构的关键项在：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)

关键字段：

- `model.name`
- `swin_transformer.use_arc`
- `swin_transformer.swin_depths`
- `swin_transformer.swin_num_heads`
- `swin_transformer.window_size`
- `swin_transformer.mlp_ratio`
- `spiking_neuron.neuron_type`

真正把模型拼起来的入口在：

- [Spiking_STSwinNet.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

其中最重要的是：

- [Spikingformer_MultiResUNet](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L90)
- 编码器构建位置：[L136](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L136)
- forward 主路径：[L161](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L161)

## 3. 代码结构图

当前这份模型可以粗分成 5 段：

1. 事件输入编码
2. Patch embedding
3. Swin-style spiking encoder
4. Residual + decoder
5. Prediction head

对应代码：

- Patch embedding：
  - [MS_PED_Spiking_PatchEmbed_Conv_sfn](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L1710)
- Attention：
  - [Spiking_QK_WindowAttention3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L605)
- Swin block：
  - [MS_Spiking_SwinTransformerBlock3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L888)
- Swin stage：
  - [MS_Spiking_Swin_BasicLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1128)
- Backbone：
  - [Spiking_SwinTransformer3D_v2](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1132)
- Decoder residual block：
  - [MS_ResBlock](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L880)
- Prediction head：
  - [MS_SpikingPredLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L607)
- Spiking neuron wrapper：
  - [Spiking_neuron](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L26)

## 4. 如果你要改注意力

### 4.1 最推荐的改法

不要直接改原始 `Spiking_QK_WindowAttention3D`。  
推荐做法是：

1. 新建一个 attention 类
2. 新建一个 block 子类，把 `attn_module` 指向你的新类
3. 新建一个 stage 子类，把 `swin_block_type` 指向你的 block
4. 新建一个 backbone 子类，把 `swin_layer_type` 指向你的 stage
5. 新建一个整网类，把 encoder/backbone 接进去
6. 新建一份 config，把 `model.name` 改成你的新类

### 4.2 当前注意力入口

当前默认 attention 类是：

- [Spiking_QK_WindowAttention3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L605)

当前 block 绑定 attention 的位置是：

- [MS_Spiking_SwinTransformerBlock3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L888)

这里最关键的一行是：

- `attn_module = Spiking_QK_WindowAttention3D`

### 4.3 你必须保持不变的接口

如果你只想替换注意力逻辑，下面这些接口不要改：

- attention 输入仍然是窗口形式
- attention 输出仍然返回 `(x, attn)`
- block 的输入输出 shape 不变
- stage 的输出通道数不变

如果你把接口也改了，后面 patch merging、decoder、prediction head 都会一起被牵连。

### 4.4 适合做的改动

- 改 `QK` 交互方式
- 改局部窗口 attention 公式
- 增加 gate / mask / local bias
- 改 attention 内部的脉冲激活形式

### 4.5 不建议一开始就做的改动

- 引入复杂全局 attention
- 大量动态 gather/scatter
- 改得让输入输出 shape 不再兼容现有 block

## 5. 如果你要插新模块

### 5.1 推荐插入位置

最推荐的三个位置：

1. `encoder` 输出之后
   - 位置：[Spikingformer_MultiResUNet.forward](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L161)
   - 适合加全局增强、融合模块、adapter

2. `decoder` 每层前后
   - 位置：[L173-L179](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L173)
   - 适合加 refinement、skip gating、多尺度融合

3. `patch embed` 内
   - 位置：[MS_PED_Spiking_PatchEmbed_Conv_sfn](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L1710)
   - 适合改输入事件编码、前端卷积、局部残差

### 5.2 插模块时先看什么

先确认你的模块输入是：

- `T,B,C,H,W`
- 还是 `B,D,H,W,C`

如果你的模块天然吃卷积特征图，优先插在 decoder/residual 路径。  
如果你的模块天然吃 token/窗口表示，优先插在 Swin block/stage 路径。

## 6. 如果你要换 patch embed

这是最干净、最容易实验的一类改动。

当前 patch embed 构建位置在：

- [Spiking_SwinTransformer3D_v2](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1176)

这里通过：

- `self.patch_embed = eval(embed_type)(...)`

来实例化 patch embed 类。

这意味着你只要：

1. 在 [Spiking_modules.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py) 新增一个 patch embed 类
2. 保证类名可被当前模块 `eval(embed_type)` 找到
3. 在 config 里把 `swin_transformer.use_arc[1]` 改成你的新类名

就能切换前端。

## 7. 如果你要换脉冲神经元

统一包装入口在：

- [Spiking_neuron](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L26)

如果你只是想在已有类型之间切换，可以直接改 config：

- `spiking_neuron.neuron_type`

如果你要新增新的 neuron type，不只要改 `Spiking_neuron`，还要同步改：

- [train_flow_parallel_supervised_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- [eval_DSEC_flow_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/eval_DSEC_flow_SNN.py)

因为训练和评估脚本都会根据 `neuron_type` 选择 `neurontype`，再交给：

- [runtime_backend.py](/root/private_data/work/SDformer/third_party/SDformerFlow/utils/runtime_backend.py#L30)

如果你忘了同步改 train/eval 分支，模型可能能 import，但训练会直接挂。

## 8. 如果你要改输出头

当前输出头类是：

- [MS_SpikingPredLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L607)

它适合做这些改动：

- 更深的 flow head
- depthwise / separable head
- uncertainty head
- multi-task head

这类改动通常风险比改 backbone 小，适合先做快速试验。

## 9. 如果你想继承 baseline 参数继续训练

这份工程支持在已有 run 的参数上继续训练，入口在：

- [train_flow_parallel_supervised_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- [utils.py](/root/private_data/work/SDformer/third_party/SDformerFlow/utils/utils.py)

关键机制有两种：

1. `--prev_runid`
   - 加载旧模型参数作为新的起点
   - 适合微调、改小模块、改注意力但保留大部分主干结构

2. `--resume 1`
   - 不只恢复模型参数，还恢复 optimizer、scheduler、scaler、epoch
   - 适合训练中断后断点续训

### 9.1 代码里是怎么实现的

加载旧参数的入口在：

- [train_flow_parallel_supervised_SNN.py#L92](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L92)

真正执行加载的函数在：

- [utils.py#L38](/root/private_data/work/SDformer/third_party/SDformerFlow/utils/utils.py#L38)

这里最关键的是：

- `model.load_state_dict(pretrained_dict, strict=False)`

这意味着：

- 名字和 shape 对得上的层会加载
- 新增层会保留随机初始化
- 被删掉或改名的旧层会被跳过

所以“结构改了还能不能继承参数”不是二元问题，而是：

- 完全兼容：大部分层名和 shape 不变，基本等于在 baseline 上继续训
- 部分兼容：主干大部分能加载，新模块随机初始化，这是最常见情况
- 几乎不兼容：大量关键层 shape 改了，只能加载很小一部分，实际接近重新训练

### 9.2 哪些改动通常还能继续加载

下面这些改动通常适合继续继承 baseline 参数：

- 在 decoder 里插新模块
- 在 block 外围加 adapter / gate / residual branch
- 新增输出头
- 替换 attention 内部逻辑，但保持输入输出维度不变
- 替换 patch embed 内部计算逻辑，但保持输出通道不变

### 9.3 哪些改动通常只能部分加载

下面这些改动往往会导致大面积 shape 改变：

- 改 `base_num_channels`
- 改 `swin_depths`
- 改 `swin_num_heads`
- 改 stage 数
- 改 patch embed 输出维度
- 做真实的 channel / head / block 结构化剪枝

这类改动不是完全不能加载旧参数，而是通常只能部分加载。

### 9.4 剪枝场景怎么理解参数继承

如果你只是做 mask pruning，不改原张量 shape，通常还能继续加载原权重。

如果你做的是“物理裁剪”：

- 把 channel 从 `96` 裁到 `64`
- 把某些 head 真的删掉
- 把某些 block 真的移除

那旧权重 shape 已经变了，通常需要：

- 从头训练新结构
- 或自己写权重迁移逻辑
- 或只保留对得上的那部分层

### 9.5 最实用的训练方式

1. 加载已有模型权重，重新开始一轮新实验

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow \
  --prev_runid <OLD_RUN_ID>
```

2. 从已有 run 断点续训

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow \
  --prev_runid <OLD_RUN_ID> \
  --resume 1
```

### 9.6 实际研究中的建议

如果你想减少重训成本，优先做“结构局部改动、维度保持不变”的设计。

如果你准备大改 backbone 宽度、head 数、stage 深度或做结构化剪枝，就不要默认自己是在“微调 baseline”，而要把它当成：

- 部分加载旧参数的再训练
- 或新的结构重新训练

### 9.7 warm start 后多久能看出趋势

可以，但要分“看趋势”和“看最终结论”。

如果你是：

- 在 baseline 上插一个小模块
- 改注意力内部逻辑但保持维度不变
- 增加一个 adapter / gate / head

而且是用 `--prev_runid` 继承已有参数重新训练，那么通常：

- `0.5` 到 `1` 个 epoch：能看出训练是否稳定、loss 是否异常
- `1` 到 `3` 个 epoch：通常能看出相对 baseline 的早期趋势
- `3` 到 `5` 个 epoch：通常足够判断“值得不值得继续 full train”

如果你改动更大，比如：

- 改 `base_num_channels`
- 改 `swin_depths`
- 改 `swin_num_heads`
- 做结构化剪枝

那就不要指望“几个 step”就得出结论。  
这类改动即使加载了部分旧参数，也更接近新结构再训练，通常要更长的预算才能稳定看趋势。

最实用的经验法则是：

1. 先跑 smoke，看代码和 loss 是否正常
2. 再跑 dev subset，看 `1` 到 `3` 个 epoch 的相对趋势
3. 只有趋势明显不差，才去跑 full DSEC

不要拿 `1` 个 epoch 的 full 训练结果直接当论文结论。  
早期 epoch 更适合做“方向筛选”，不是做最终排名。

### 9.8 warm start 研究时该怎么判定是否值得继续

后续每次改模型时，建议至少固定比较下面这些量：

- `train_loss`
- `valid_loss`
- `AEE / AAE / PE`
- 每 step 时间
- 每 epoch 时间
- GPU 显存

如果是 warm start 训练，最该看的是：

- 同样训练预算下，validation loss 是否下降更快
- 同样训练预算下，AEE/AAE 是否优于 baseline
- 是否在速度或显存上更省

如果改动后出现下面任一情况，通常就不用继续 full train：

- loss 收敛更慢
- valid 指标没有改善
- 显存更高且速度更慢
- 结构更复杂但没有换来更好的趋势

## 10. baseline 性能应该怎么看

你后续会同时遇到 3 种“baseline 数字”，不要混用。

### 10.1 smoke baseline

这是你当前已经跑通的单序列 smoke 结果，只用于链路回归，不用于论文对比。

当前 smoke 结果是：

- `AEE = 6.170446`
- `AAE = 57.41096`

对应结果文件：

- [metrics_2.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/results_inference/89cc9fdf5c93495fa2ab5f0071edc41d/metrics_2.yml)

这组数字只能回答：

- 代码链路通不通
- 改完模型会不会直接炸

它不能回答：

- 模型论文级性能怎样
- 和其他方法是否公平可比

### 10.2 你自己的本地 full baseline

这是你后续最重要的比较对象。

原因很简单：

- 你改模块时用的是同一套代码
- 用的是同一份预处理数据
- 用的是同一套训练/eval 脚本
- 这是最公平的内部对比口径

所以后续真正做研究时，第一优先不是拿外部论文数字比，而是先拿：

- `你的 full DSEC baseline`
- `你的改进版模型`

在同一口径下做对比。

### 10.3 论文 / 官方 benchmark baseline

这是外部参考数字，用来判断你的结果是否接近文献水平。

根据 DSEC 官方 benchmark 当前公开页面，`SDformerFlow` 的 all-sequence averages 是：

- `1PE = 37.576`
- `2PE = 17.123`
- `3PE = 10.051`
- `EPE = 1.602`
- `AE = 4.871`

根据论文公开结果，MVSEC `dt = 1 frame` 上 `SDformerFlow_v2` 的平均结果是：

- `AEE = 0.66`
- `Outlier = 1.57`

我这里说的是“当前公开页面/论文里的参考值”，不是你本地已经复现得到的值。

### 10.4 为什么你的本地结果不一定和论文一模一样

即使模型名一样，本地结果也可能和论文或官方 benchmark 不完全一致，常见原因包括：

- 数据切分不同
- 用的是本地 validation，不是官方隐藏 test
- crop / resolution 不同
- epoch 数不同
- warm start 或从头训策略不同
- 预处理细节不同
- MLflow / eval 脚本口径不同

所以你后续应该把“自己的 full baseline”当成主比较对象，把“论文/benchmark 数字”当成外部参考上界或目标区间。

## 11. 后续该如何对比

后续对比时，至少固定下面 4 个层次，不要混淆：

### 11.1 层次 1：回归测试对比

- 单序列 smoke
- 目的：确认代码没坏

### 11.2 层次 2：开发期趋势对比

- dev subset
- 预算：短 epoch 或固定 step
- 目的：快速筛方向

### 11.3 层次 3：本地正式对比

- full DSEC train + local validation
- 这是你论文消融最该依赖的主表

### 11.4 层次 4：外部结果对齐

- DSEC 官方 benchmark
- MVSEC 论文口径评测
- 用于对外展示和跨论文对照

### 11.5 最低限度的公平比较要求

如果你要说“我的改进优于 baseline”，至少要保持：

- 相同数据集
- 相同 split
- 相同预处理
- 相同训练预算
- 相同 eval 脚本
- 相同输入分辨率 / crop

如果其中任意一项变了，你就不能把数字直接写成“优于 baseline”，而应该写成：

- 在某种新协议下的结果
- 或补充说明这不是严格一一对应比较

### 11.6 论文表格建议怎么组织

建议你把结果表拆成 3 张：

1. `内部开发表`
   - smoke / dev subset / full local baseline / variant

2. `正式精度表`
   - DSEC local validation
   - DSEC benchmark
   - MVSEC

3. `效率表`
   - 参数量
   - FLOPs
   - 显存
   - step time / FPS
   - 稀疏率 / 能耗估计

## 12. 如果你要做轻量化

最推荐先做“配置级轻量化”，不要一开始就做物理剪枝。

优先改这些配置：

- `base_num_channels`
- `swin_depths`
- `swin_num_heads`
- `window_size`
- `mlp_ratio`

位置在：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml#L29)

这种方式优点是：

- 结构一致
- 训练脚本不用改
- 更容易做基线对比
- 对硬件也更友好

## 11. 如果你要做剪枝

### 11.1 推荐顺序

1. 先做配置级缩小
2. 再做结构化剪枝
3. 最后才考虑非结构化稀疏

### 11.2 优先推荐的剪枝类型

- channel pruning
- head pruning
- block pruning

### 11.3 风险更高的类型

- 非结构化权重稀疏
- 运行时动态不规则稀疏

### 11.4 为什么优先结构化

因为这份工程后续还有硬件设计需求。  
结构化剪枝更容易映射到：

- 更小的矩阵维度
- 更规则的 tile
- 更稳定的访存
- 更简单的量化和 RTL 接口

## 13. 如果你要做稀疏

要先区分：

- `activation/spike` 天然稀疏
- `weight` 人工稀疏

推荐路线：

1. 先统计 spike 稀疏率
2. 再做规则化 token/head/channel 缩减
3. 最后再看是否值得加 weight sparsity

如果你的目标包含硬件实现，不建议优先走非结构化稀疏。

## 14. 最稳的开发方式

永远不要直接把 baseline 类改坏。

推荐固定做法：

1. 新建类，不覆盖旧类
2. 新建 config，不覆盖旧 config
3. 先跑 smoke
4. 再跑 dev subset
5. 通过后再进 full DSEC

这样你始终保留：

- baseline 结果
- baseline 代码
- baseline 配置

回滚成本最低。

## 15. 改模型时最常见的坑

### 15.1 只写了新类，训练脚本却找不到

训练脚本通过 `eval(config["model"]["name"])` 实例化模型。  
如果你新增整网类，只写到模型文件里还不够，还要保证训练/评估脚本能 import 到它。

### 15.2 改了 head 数，维度却除不尽

注意力里有：

- `head_dim = dim // num_heads`

如果 `dim % num_heads != 0`，结构就不合法。

### 15.3 改了接口 shape，decoder 路径一起坏掉

如果你改注意力或中间模块时改变了张量 shape，后面的：

- patch merging
- decoder
- pred head

都可能连锁失效。

### 15.4 只改模型，不改配置

很多切换点是配置驱动的，不改 config 等于新模块根本没被用到。

## 16. 一句话操作建议

- 改注意力：从 [Spiking_swin_transformer3D.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py) 下手
- 改前端：从 [MS_PED_Spiking_PatchEmbed_Conv_sfn](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L1710) 下手
- 插模块：从 [Spikingformer_MultiResUNet.forward](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L161) 下手
- 改输出头：从 [MS_SpikingPredLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L607) 下手
- 改神经元：同时改 [Spiking_neuron](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L26) 和 train/eval 分支
- 做轻量化：优先改 config
- 做剪枝：优先结构化，不优先非结构化
