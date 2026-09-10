现有加载器能作为真实 coarse-flow 恢复的底座，但不能直接对当前评估接口调用 backward。下列判断来自本地源码；本轮没有启动GPU或恢复训练。

1. `algorithm/nrv_cost_probe/run_probe.py:load_system` 先 `model.requires_grad_(False)`，这只冻结参数，并不禁止对输入反传。因子U/V必须改为独立 `nn.Parameter`，显式交给优化器；当前数值adapter中的普通Tensor不是训练参数。V仍需乘固定连接mask，禁止优化出组外连接。
2. 同加载器的preds.2 hook使用 `output.detach().sum(0)`，会切断最终损失。可在preds.2加 `prepend=True` 的只捕获hook，保存 `output.sum(0)` 到另一键；原停止hook照常抛 `CoarseReady`。无需重写完整网络、执行decoder3或修改已有加载器。删除评估主循环的 `@torch.no_grad()`；整个模型保持eval及同固定BN状态，不能直接model.train()改变推理函数。
3. 当前因子adapter和根评估器的T10门是布尔比较，U/V到Conv2在这里完全断梯度。最少要对该r1.sn2门加surrogate，前向仍严格返回同一 `theta*(margin>=0)`。例如自定义Function前向直接返回已有hard值，反向按固定AT-LIF三角窗或既有ATan导数；不能把有效判决阈值tau替换成theta。完整母轴先用同完整A，conditional轴还需给许可分支的软反向，但其前向必须保留已保存硬许可。
4. 下游原生 `ATLIFTernaryPSN.forward` 已调用自定义Binary/OfficialATLIFSurrogate，在eval下也保留输入梯度；`no_grad`块主要记录发放统计，不需统一重写全网神经元。其余冻结卷积、FC2、动态BN2和skip保持正常autograd。Swin残差可传梯度，因此不能断言所有整数MLP都会让整个网络梯度为零。
5. 但要让真实消费者分支也参与导数，需要处理现有整数接口：`evaluate_stage2_deployment.install_saved_consumers` 的12个FC1后门比较；`probe_direct_code_integer.integer_codes` 的round/long/3bit地址；`train_class_shift_probe.install_class_consumers` 再次将source转 `!=0/long` 并索引E，随后B→tau又做硬比较。仅给source输出加STE不够，后续重新编码会再次切断。六S2最好整体替换为前向同整数结果的可微wrapper：保存真正hard code，反向从三判决概率得到8类概率→固定mapping→E期望→真实W/B→门surrogate→原FC2/BN2。其他六MLP只需compiled margin的门surrogate。输入量化可用round/clamp STE，前向仍保留原signed12范围及int64分类函数；缩放/surrogate宽度须固定并说明，不能宣称精确梯度。

最小入口（不是已运行训练）如下：

```python
system = probe.load_system(args)
model, modules, _, current, *_ = system
probe.install_sources(system, saved_source_parameters)
# 装原四fixed patch BN；安装可训练U/V与r1.sn2前向不变的surrogate。
def keep_graph(module, inputs, output):
    current['train_flow'] = output.sum(0)
handle = modules['sttmultires_unet.preds.2'].register_forward_hook(keep_graph, prepend=True)
functional.reset_net(model)
try:
    model(x)                       # 不放在no_grad里
except CoarseReady:
    flow = current.pop('train_flow')
    current.pop('flow', None)       # 丢掉原评估hook的detach副本
pred = F.interpolate(flow, (480, 640), mode='bilinear', align_corners=False)
error = pred.permute(0, 2, 3, 1)[valid] - gt.permute(0, 2, 3, 1)[valid]
loss = torch.sqrt(error.square().sum(-1) + 1e-6).mean()
loss.backward()                     # 先单帧只核U/V梯度，暂不optimizer.step
```

Y-MSE与按通道平衡的门BCE未考虑真实Conv2权重、遮挡和有效光流像素，可能约束与flow无关的差异；这只是损失目标不一致，尚不能把现有精度退化归因于它。可先同布局/同预算比较真实flow恢复，而不继续扫描局部代理权重。若蒸馏teacher flow，其teacher前向可no_grad并缓存小2通道流；学生后继不能no_grad。

显存：r1一个T10×C96×240×320 FP32特征约294.9MB，R56潜变量约172.0MB；反传还需保存下游神经元、Swin和decoder中间量，冻结权重并不会免掉输入梯度状态。最小试验一帧、保持原分辨率/完整T，不直接承诺80GB内峰值；先量实测max_memory_allocated。原Swin `use_checkpoint` 分支含 `checkpoint.checkpoint(self.mlp(x), x)`，不能为省显存直接打开而未核对。需要时对返回Tensor且无CoarseReady副作用的模块增加正确checkpoint，完整forward/门先核同值。train输入、teacher流、保存统计不得长期挂住旧计算图。

建议根先做：前向与现评估同帧0门差/同flow；再单帧loss.backward核U/V有有限非零梯度，并分别检查原生与整数消费者分支。达到这一点才进行固定预算真实消费者/GT恢复，不把“能反传”当训练有效。
