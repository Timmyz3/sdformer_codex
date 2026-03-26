# SDformerFlow Baseline 模型走读

## 1. 这份文档解决什么问题

这份文档用于把当前 baseline 从底层梳理清楚，目标是让你后续做：

- 注意力改进
- 新模块插入
- patch embed 替换
- 神经元替换
- 剪枝 / 稀疏

时，不只是知道“改哪一行”，而是知道这份 baseline 整体是怎么工作的。

这份文档默认讲的 baseline 是：

- `MS_SpikingformerFlowNet_en4`

对应训练配置：

- [train_DSEC_supervised_SDformerFlow_en4.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)

## 2. 先记住这 4 个核心结论

1. 这不是纯 CNN，也不是纯 Transformer，而是：
   - `spiking patch embed + 3D Swin-style encoder + conv decoder + multi-res flow head`
2. `en4` 表示这条 baseline 用 `4` 个 encoder stage。
3. 当前 baseline 的主干是 `MS` 变体，不是最原始的 `SEW` 变体。
4. 模型内部有两套常见张量布局：
   - `T,B,C,H,W`
   - `B,D,H,W,C`

后面看代码时，大部分困惑都来自这两套布局在切换。

## 3. baseline 由哪几个文件组成

### 3.1 训练配置

- [train_DSEC_supervised_SDformerFlow_en4.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)

### 3.2 模型装配总入口

- [Spiking_STSwinNet.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)

### 3.3 UNet 父类与 decoder/pred 组织方式

- [SNN_models.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)

### 3.4 patch embed / decoder / residual / pred / neuron

- [Spiking_modules.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)

### 3.5 Swin-style spiking encoder

- [Spiking_swin_transformer3D.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)

### 3.6 训练时输入是怎么变形的

- [train_flow_parallel_supervised_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)

## 4. 配置是怎么映射到模型结构的

配置里最关键的几个字段在：

- [model](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml#L12)
- [swin_transformer](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml#L29)
- [spiking_neuron](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml#L45)

它们分别决定：

- `model.name = MS_SpikingformerFlowNet_en4`
  - 选整网类
- `base_num_channels = 96`
  - 基础通道宽度
- `swin_depths = [2,2,6,2]`
  - 4 个 stage 各有多少 block
- `swin_num_heads = [3,6,12,24]`
  - 4 个 stage 的 head 数
- `window_size = [2,9,9]`
  - 局部窗口的时间/空间大小
- `spiking_neuron.neuron_type = psn`
  - 当前 baseline 用 `PSN`

## 5. 训练时输入张量是怎么来的

从 dataloader 出来的 `chunk` 是预处理后的事件张量。  
在训练脚本里，voxel 路径的处理在：

- [train_flow_parallel_supervised_SNN.py:262](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L262)

如果 `polarity=True`，它会做：

1. `neg = relu(-chunk)`
2. `pos = relu(chunk)`
3. `chunk = cat(pos, neg, dim=2)`

于是输入从单极性体素张量变成：

- `B, C=num_bins, P=2, H, W`

也就是：

- `B, 10, 2, H, W`

然后才送进模型：

- [train_flow_parallel_supervised_SNN.py:303](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L303)

## 6. 顶层模型是怎么搭起来的

最顶层类在：

- [MS_SpikingformerFlowNet_en4](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L319)

它本身很薄，关键只是：

- `unet_type = MS_Spikingformer_MultiResUNet`
- `num_en = 4`

也就是说，真正主体在：

- [MS_Spikingformer_MultiResUNet](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L239)

## 7. 整网 forward 的真实路径

### 7.1 顶层 forward

顶层 forward 在：

- [SpikingformerFlowNet.forward](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L278)

它做的事情很少：

1. 记住输入原图大小 `H,W`
2. 调用 `self.sttmultires_unet.forward(x)`
3. 得到多尺度 flow
4. 对每个 flow 在时间维 `dim=0` 上求和
5. 再插值回原图大小
6. 返回 `{"flow": flow_list, "attn": attns}`

这说明一个关键点：

- UNet 内部输出的是带时间维的多尺度 flow
- 顶层模型最后把时间维累加成最终光流

## 8. UNet 主体是怎么工作的

UNet 父类组织方式在：

- [SpikingMultiResUNet](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L12)

它定义了四类组件：

1. `encoders`
2. `resblocks`
3. `decoders`
4. `preds`

这些组件的构建函数在：

- [build_encoders](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L101)
- [build_resblocks](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L118)
- [build_multires_prediction_decoders](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L145)
- [build_multires_prediction_layer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L131)

### 8.1 但 baseline 并不是直接用父类默认组件

在：

- [MS_Spikingformer_MultiResUNet](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L239)

它覆盖了几种组件类型：

- `encoder_block = MS_spiking_former_encoder`
- `ff_type = MS_SpikingConvEncoderLayer`
- `res_type = MS_ResBlock`
- `upsample_type = MS_SpikingDecoderLayer`
- `transpose_type = MS_SpikingTransposeDecoderLayer`
- `pred_type = MS_SpikingPredLayer`

也就是说，baseline 用的是 `MS` 版本的一整套模块。

## 9. baseline 的数据流怎么走

### 9.1 输入到 patch embed

在：

- [Spiking_SwinTransformer3D_v2.forward](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1223)

第一步就是：

- `x = self.patch_embed(x)`

### 9.2 patch embed 在做什么

当前 baseline 的 patch embed 是：

- [MS_PED_Spiking_PatchEmbed_Conv_sfn](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L1710)

它的作用不是简单一层卷积，而是：

1. 把输入事件表示按时间步重排
2. 做一个 `head` 卷积
3. 做一个 `conv` 下采样
4. 做两个残差块 `residual_encoding`
5. 最后做 `proj`

所以它本质上是一个小前端，而不是单纯 patchify。

### 9.3 patch embed 前后 shape

输入到 patch embed 前：

- `B, 10, 2, H, W`

在 patch embed 内部重排后：

- `T, B, C, H, W`

输出后在 backbone forward 中又被转成：

- `B, C, T, H, W`

见：

- [Spiking_swin_transformer3D.py:1229-1230](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1229)

## 10. 3D Swin backbone 是怎么组织的

整个 backbone 类在：

- [Spiking_SwinTransformer3D_v2](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1132)

它按 stage 组织：

- `self.layers = nn.ModuleList()`

每个 stage 用：

- [Spiking_Swin_BasicLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L995)

在 baseline 里，实际用的是：

- [MS_Spiking_Swin_BasicLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1128)

## 11. 一个 stage 里面是什么

每个 stage 里面有：

1. 多个 Swin block
2. 可选的 patch merging

其中 block 列表在：

- [Spiking_Swin_BasicLayer:1038-1058](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1038)

也就是：

- `self.swin_blocks = nn.ModuleList([...])`

如果不是最后一个 stage，还会有：

- `self.downsample`

## 12. 一个 block 里面是什么

当前 baseline 的 block 是：

- [MS_Spiking_SwinTransformerBlock3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L888)

它继承自：

- [Spiking_SwinTransformerBlock3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L720)

block 里核心有两部分：

1. `attn`
2. `mlp`

对应：

- `self.attn`
- `self.mlp`

它的 forward 是：

1. `SSA`
2. 残差连接
3. `mlp`
4. 再残差连接

## 13. 当前 baseline 的注意力到底是什么

当前 baseline attention 是：

- [Spiking_QK_WindowAttention3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L605)

这份 attention 不是标准 softmax attention。  
它更像：

- 先生成 `q`
- 生成带位置编码的 `k`
- 用 `q` 的聚合 token 去调制 `k`
- 再投影回来

关键步骤在：

- [L670-L717](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L670)

所以如果你后面改注意力，不要默认把它理解成标准 `QK^T softmax V` 那一套。

## 14. 为什么这是一个 `MS` baseline

`MS` 的意思在这份代码里，不是单纯“multi-scale”四个字，而是很多模块采用了不同于原始 `SEW` 的 shortcut / spiking 顺序。

你可以对比：

- [MS_SpikingConvEncoderLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L298)
- [MS_ResBlock](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L880)
- [MS_SpikingTransposeDecoderLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L461)
- [MS_SpikingPredLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L607)

共同特点是：

- 先经过 spiking neuron
- 再经过 conv / deconv / pred
- shortcut 的位置也不同

所以如果你要做“baseline 对比”，一定要明确你对比的是：

- `SEW` 风格
- 还是当前 `MS` 风格

## 15. decoder 是怎么工作的

decoder 组织在：

- [build_multires_prediction_decoders](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L145)

每一层 decoder 输入通道数是：

- `2 * input_size + prediction_channels`

原因是它把：

1. 当前 decoder feature
2. 对应 encoder skip
3. 上一层 prediction

按通道拼起来。

真正拼接逻辑在：

- [SpikingMultiResUNet.forward:207-213](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py#L207)

skip 函数定义在：

- [skip_concat](/root/private_data/work/SDformer/third_party/SDformerFlow/models/model_util.py#L14)

所以你后面如果改 decoder，要注意它不是“单纯上采样 + conv”，而是带：

- encoder skip
- previous prediction skip

的多尺度解码。

## 16. prediction head 是怎么工作的

prediction head 在：

- [MS_SpikingPredLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L607)

它很轻：

1. 先 `Spiking_neuron`
2. 再 `1x1 Conv2d`

每层 decoder 后面都会接一个 flow head。  
所以整个模型是：

- `multi-resolution flow prediction`

而不是只有最后一层才出 flow。

## 17. 为什么顶层 forward 最后要按时间维求和

在：

- [SpikingformerFlowNet.forward:291-301](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py#L291)

每个多尺度 flow 都会：

- `torch.sum(flow, dim=0)`

也就是把时间维聚合成最终的连续值光流。

所以如果你后面改 head 或改 temporal aggregation，这是一个很关键的切入点。

## 18. loss 是怎么接这个输出的

训练脚本里：

- [train_flow_parallel_supervised_SNN.py:303-312](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py#L303)

这里：

- `pred_list = model(chunk.to(device))`
- `pred = pred_list["flow"]`

也就是说 loss 吃的是顶层模型输出的 `flow list`。

因此如果你改了顶层 forward 的输出格式，loss 也可能跟着要改。

## 19. 你后面看 baseline 的推荐阅读顺序

建议按这个顺序读代码：

1. [train_DSEC_supervised_SDformerFlow_en4.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml)
2. [train_flow_parallel_supervised_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
3. [Spiking_STSwinNet.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)
4. [SNN_models.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py)
5. [Spiking_swin_transformer3D.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)
6. [Spiking_modules.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)
7. [model_util.py](/root/private_data/work/SDformer/third_party/SDformerFlow/models/model_util.py)

## 20. 你后面改动时最重要的 5 个接口

1. `patch embed`
   - [MS_PED_Spiking_PatchEmbed_Conv_sfn](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L1710)
2. `attention`
   - [Spiking_QK_WindowAttention3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L605)
3. `block/stage/backbone`
   - [MS_Spiking_SwinTransformerBlock3D](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L888)
   - [MS_Spiking_Swin_BasicLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py#L1128)
4. `decoder + pred`
   - [MS_SpikingTransposeDecoderLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L461)
   - [MS_SpikingPredLayer](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L607)
5. `spiking neuron`
   - [Spiking_neuron](/root/private_data/work/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py#L26)

## 21. 一句话总结

当前 baseline 可以理解成：

- `事件体素输入`
- `前端 spiking patch embed`
- `4-stage 3D spiking Swin encoder`
- `MS 风格 residual + decoder`
- `多尺度 flow heads`
- `时间维求和得到最终 flow`

如果你把这条主线吃透了，后面无论你改注意力、加模块还是做轻量化，都会知道自己改动的是哪一层抽象，而不是只在代码里“试错”。
