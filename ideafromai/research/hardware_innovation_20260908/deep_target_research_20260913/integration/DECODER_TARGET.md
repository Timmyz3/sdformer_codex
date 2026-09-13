# decoder2 的实际挂点与分解边界

## 为什么重新纳入

当前 matched-dense stage320、粗头 preds.2、单帧 `zurich_city_09_a_0001.npy` 的调用表中，decoder2 输入为 `[10,386,60,80]`，权重为 `[386,96,3,3]`，输出为 `[10,96,120,160]`。原始输入非零数为 2,853,376 / 18,528,000，即 15.4003%；每个输入最多扇出 `96×9` 个乘加，粗估活跃贡献约 2.465 G。r0 conv2 的对应粗估为 2.301 G。这只能用于选取测量对象，尚未扣边界、计路由、取数或状态访问，不能叫同资源服务量。

源：[profile.json](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json)；[计数程序](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile_current.py)。计数程序第 36–39 行对转置卷积用 `input.numel()*Cout*Kh*Kw`，**没有把 stride=2 插入的零再计一次**。原记录中泛称的“transpose-zero taps”不宜解释为该项还有约四倍去零收益。

## 真实消费者边界

模型实现中 `MS_SpikingTransposeDecoderLayer.forward` 的顺序是 `sn → deconv → norm`。虽然输入拼接了前一层的两通道 flow，但整个拼接结果先经过神经元；不能将 deconv 的一部分输入写成未发放的连续 flow。AT-LIF 的静态 θ 可吸收进 deconv 权重，源仍为二电平支撑。

`MS_SpikingPredLayer.forward` 又执行 `sn → 1×1 conv`。因此 decoder2 的 96 通道结果与两通道 flow 之间存在非线性神经元，不能直接把 deconv 权重乘上最后的 1×1 权重，宣称 96 通道降为 2 通道是无损融合。

`preds.1` 在 decoder2 之前已经完成，可以提供同一前向中的粗运动信息。用它选择 decoder2 的细化区域不存在“当前最终 flow 先知”依赖；用 `preds.2` 反过来决定是否执行 decoder2 则不成立。具体 norm 是否已固定，应以当前实例核对；这里只确认源码消费者顺序。

源码：[Spiking_modules.py](../../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)、[Spiking_STSwinNet.py](../../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py)。

## 必须完整保留的先验 A

3×3、stride 2 转置卷积可按输出坐标奇偶分成四个相位，分别含 1、2、2、4 个 kernel tap；具体相位顺序随 padding 而变。Chang 与 Chang 在 ISCAS 2020 已给出四子核、输入复用及输出地址拼接，EcoFlow 给出产品分组、PE 映射、广播和局部部分和的数据流。这些构成去插零的强基线，不能再次当作新机制。[1][2]

EcoFlow 的公开 `sasiml` 包含模拟器和映射工具，不是公开 RTL。其 README 明确提醒数据分发阶段用经验估算，不能把该部分直接当作地址和握手闭合的 RTL 模型。[3]

WaveletVFI 已实现从粗尺度信息选择稀疏高频细节，前置卷积还使用 `dilate3` 扩展 mask，避免漏掉邻域输入。因此“粗流 + 稀疏细节 + halo”也已经是可借的完整算法底座，不是新增贡献。[4]

## 剩余可试的 X

**候选，不是结果：**利用已有 `preds.1`、事件时间分布与空间可观测性，联合决定“哪些细化区域值得保留”与“哪些完整 T10 / 多相请求组可以不发”。对没有事件却缺乏可靠运动信息的区域，保留廉价全图预测及必要细化；不能把没有事件简单解释为运动为零或无需输出。

新增点若仅是把 WaveletVFI 移植到 SNN 并补齐 PSN/BN，不够强。需要证明：事件可观测性和请求组费用共同训练，能在相同 AEE 下比普通 flow 梯度 mask、原 WaveletVFI 式粗细阈值、等预算空间块选择减少更多**实际事务或周期**。非因果 PSN 和多相地址是必须完成的接口，不自动等于创新。

第一项算法对照应包含 `preds.1` 直接上采样。如果简单删除整个 decoder2 已满足质量且优于复杂选择器，选择性细化必须以更好的质量—费用曲线证明必要性。第一份 RTL 应覆盖 mask 的实际生成、相位地址、有限缓存、所有有效源的完整通道归约、norm、后继神经元和 pred 输出；不能只跑一个预先给好 mask 的执行叶。

## 来源

1. K.-W. Chang, T.-S. Chang. [Efficient Accelerator for Dilated and Transposed Convolution with Decomposition](https://arxiv.org/abs/2205.02103), ISCAS 2020，§II-C/D，图 6、9；arXiv 上传时间为 2022，不是会议年份。DOI: [10.1109/ISCAS45731.2020.9180402](https://doi.org/10.1109/ISCAS45731.2020.9180402)。
2. L. Orosa et al. [EcoFlow: Efficient Convolutional Dataflows for Low-Power Neural Network Accelerators](https://arxiv.org/abs/2202.02310)，作者全文 §4，尤其 §4.1.1/4.1.2。期刊版本 IEEE Transactions on Computers，DOI [10.1109/TC.2023.3272282](https://doi.org/10.1109/TC.2023.3272282)。
3. CMU-SAFARI. [sasiml 官方实现及限制](https://github.com/CMU-SAFARI/sasiml)。
4. L. Kong et al. [Dynamic Frame Interpolation in Wavelet Domain](https://arxiv.org/abs/2309.03508)，IEEE Transactions on Image Processing 32, 5296–5309, 2023，§III-C、算法 1、图 4；[官方代码](https://github.com/ltkong218/WaveletVFI)。
