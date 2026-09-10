独立只读结论：`evaluate_coarse_readout.py` 选择的是已有第三个完整光流预测 `preds.2`；时间求和和位移单位正确。它没有使用最终头、GT 或未来帧来生成输出。双线性插值是一项明确的读出变更，不能称为原 ep34 最后输出的数值等价实现。

原模型 `Spiking_STSwinNet.py:161` 按 `decoder → pred → append` 依次生成四个预测；上一个预测作为下一层输入通道，但不存在将四个 flow 残差相加的步骤。因此 `preds.2` 本身就是完整位移估计，不缺 `preds.0/1` 的后处理累加。`MS_SpikingPredLayer.forward`（`Spiking_modules.py:643`）先执行神经元，再以 1×1 卷积输出 `[T,B,2,H,W]`，保留了实际 θg 对读出的影响。

顶层 `Spiking_STSwinNet.py:291–302` 对**每个**尺度执行 `sum(dim=0)`，再插值到输入分辨率，不乘空间倍率。此处 `preds.2` 为 `[10,1,2,120,160]`，求和后为 `[1,2,120,160]`，插值到 `480×640`。原代码省略 `mode`，使用 nearest；新代码的 nearest 分支复现这项读出，bilinear 分支显式使用 `align_corners=False`。GT 预处理将 DSEC 两个分量解码为 `(uint16−32768)/128` 后直接保存 FP32（`DSEC_dataset_preprocess.py:62–75`），没有在粗尺度另换位移单位。

**粗头已被直接监督。** 根代理读取的实际 ep34 配置为 `loss.gamma: null`（169 行）、`flow_scaling: 1`（226 行；该远端配置本地没有副本）。训练入口传给损失的是完整四项 `pred_list['flow']`（`train_flow_parallel_supervised_SNN.py:344–353` 及 H9 `entrypoints/train.py:508–523`）。`loss/flow_supervised.py:90–104` 在 gamma 为 None 时逐项计算相同 GT 的有效像素误差，再除以预测数。因此 `preds.2` 有直接监督，权重为四头平均中的一项；“backward pass only the last flow pred” 是失真的注释。即使启用 H9 angular 包装，它也保留逐头基础监督（`h9_losses.py:67–82`）。

真实执行边界也成立：hook 在 `preds.2` 返回时抛出 `CoarseReady`，尚未进入 `i=3` 的 skip 拼接，更不会执行 `decoder3.sn/deconv/norm_layer` 或 `preds.3`。保留的成本包括完整编码器、两个残差块、前三层解码器及其预测头；额外读出仍需 T10 求和、两通道插值和稠密输出写入。现有脚本核验 AEE，没有测出硬件周期。前级为更深编码提供数据的计算不能随最后 skip 一起删除；最多另优化那份 skip 的保留寿命。

同一 10 帧的已完成结果（`dense_heads/summary.json`，均为逐帧 AEE 平均）：

| 读出 | AEE | 训练 |
|---|---:|---|
| 原最后头 | 1.171570143 | 原 ep34 |
| `preds.2` + nearest | 1.057634958 | 无新增训练 |
| `preds.2` + bilinear | 1.013058746 | 无新增训练 |
| 新 1940→16→32/PixelShuffle | 5.078823027 | 32 帧、64 更新、batch 2 |
| 同上增加 16→16 的 3×3 | 4.679355255 | 同上 |

两个新头在有限更新内尚未拟合，结果不否定末级表征重训思想。但已有粗头明显更强、更便宜，应先完成其 full825 验证。若它满足既定 AEE 门，`decoder.3` 就不应继续作为新机制必须攻克的昂贵分母；小波/细节稀疏头降级，除非能相对“已有粗头＋双线性”给出额外精度或成本收益。删除一个已监督尾部是强算法简化对照，这项发现本身不增加硬件创新评分。

本次未运行 GPU、未修改模型或评价脚本；full825 结果由根代理单独汇总，不把上述 10 帧改善外推。
