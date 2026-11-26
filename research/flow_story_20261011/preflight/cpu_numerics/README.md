# 原生 MLP 局部执行与 BN 校准工具

2026-10-11，在 `ic.ismd-nemo` 使用既有 CPU 环境执行；未训练、未加载 checkpoint、未启动 GPU。不同于上一轮 NumPy 结构探针，本轮直接 import **未修改的原生 `MS_Spiking_Mlp`**，使用真实 PyTorch/SpikingJelly 算子。【[结果](native_mlp_results.json)、[源码](native_mlp_probe.py)】

## 已实际通过的检查

- 50 个冻结 BN 的执行用例，包含 FP64/FP32、T=10、9/15 空间块、C=96/384；还包含全幅 v2 的 stage0 `120×160,C96` 与 stage2 `30×40,C384` 真实特征形状，两个形状均测9和15块。不是完整光流网络前向。【[逐例结果](native_mlp_results.json)】
- 对全开、全关、棋盘格、末端边界块、首块五种 mask，分别比较“完整算 MLP 后屏蔽”的数值参考与“仅对选中块调用 MLP”的执行。所有用例最大绝对误差为 **0**；容差在脚本中预先规定为 FP64 `1e-11`、FP32 `2e-5`。FC hooks 检查实际处理的输入行数与被选像素×T相同；全关时 MLP 调用与两层 FC 行数均为0。【[源码/结果](native_mlp_probe.py)】
- 同权重、同输入改为 current-stat BN 后，局部执行与完整参考不相同，十个配置均得到非零差异。实际安装的 SpikingJelly 0.0.0.0.14 中，multi-step BN 与沿 T/B/H/W 的手工归一化数值一致；不再只是猜测统计域。【[结果](native_mlp_results.json)】
- 新 BN 校准工具通过不同样本数、不同均值的合成统计及28项guard检查；合并方差与直接拼接计算一致，而直接平均 batch variance 的负对照不一致。原current-stat参考的状态/策略/输出不变，校准后的部署副本仍通过局部执行等价。拒绝评价成员、未知/重复target、未绑定forward、B2、缺失BN、异常/非有限统计和共享源模块安装；失败安装原子回滚。【[校准检查](bn_calibration_test_results.json)、[校准工具](bn_calibration.py)】

**这些结果只证明声明条件下的实现和依赖合同。** 权重/输入是固定种子的合成数据；为了避免全零神经元使测试失去意义，脚本明确调整合成 PSN 的初始化。没有测 AEE、训练收敛、GPU速度、周期或功耗。不能由50例推出所有浮点输入永远逐位相同；阈值附近的舍入敏感性仍需真实权重、定点格式与全网络检查。【证据限制】

## 可复用的部署步骤

`bn_calibration.py` 提供 `BNCalibrationCapture`，输入允许的训练侧 calibration target 集、评价 target 集和显式 `sample_batch_axis`；两集交叉立即报错。逐B1 target前调用 `begin_sample`，每次root forward必须重新绑定，只记录完整路径的BN输入。使用逐通道pooled population moments，包含不同批次均值差带来的方差项；转换FP64前先切有界块，避免复制整幅非连续激活。收集结束后只对**部署副本**调用 `install`，原参考不变。【工具接口】

调用者必须先声明采集源图是论文 current-stat 协议、公开代码 running-stat 协议还是某次冻结部署图；必须完整执行所有分支并按原生样本规则 reset。工具不会自行猜测源图、偷偷关闭 gate 或判断一份文件的真实训练来源。成员 guard 只验证提供的清单，不能替代实际数据绑定审计。【实现边界】

所有层一次采集并冻结后，上游统计改变可能使下游输入分布改变。允许另行记录一次“冻结后的完整图重新采集”，但不能把无数次校准或验证集调参当免费步骤。实际校准集选择和停止规则见 [复现协议](../reproduction_protocol/PROTOCOL_BN_CLOSURE.md)。同日后续已在旧服务器核齐共同成员数据并取回4个核验样本，见 [数据审计](../data_protocol/ARCHIVE_DATA_AUDIT.md)；本轮依然只通过了合成数据上的校准**实现**检查，没有运行真实模型校准或取得AEE。【计划与边界】

## 窗口9到15的加载机制

[window_checkpoint_transition.py](window_checkpoint_transition.py) 用原生QK的合成state-dict检查位置编码 `[1,3,162,32]` → `[1,3,450,32]`。直接 `strict=False` 仍拒绝尺寸不匹配；调用现有 `load_pretrained_interpolate` 后 `strict=True` 成功，其他参数和源模型均不变。模型虽为v2，代码架构标记 `swinv1` 对应 `remap="v1"` 路径；这是既有插值实现，没有修改旧loader。【[结果](window_checkpoint_transition_results.json)】

该检查仅证明机械加载兼容，没有真实checkpoint、训练或AEE；全幅9与论文全幅15仍是不同协议。可在同CPU环境单独运行该脚本复核。

## 复现

环境：`/opt/anaconda3/envs/pytorch310_cpu/bin/python`，Torch 2.7.1+cpu、SpikingJelly 0.0.0.0.14、timm 0.6.13。该 CPU 环境原先缺 einops，本轮仅在本目录 `deps/` 安装 einops 0.8.1；未升级共享环境。它是当前探针依赖，不宣称论文原始完整环境已恢复。【[版本记录](native_mlp_results.json)】

```bash
cd /home/zhumd/work/sdformer_codex/research/flow_story_20261011/preflight/cpu_numerics
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/anaconda3/envs/pytorch310_cpu/bin/python native_mlp_probe.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/anaconda3/envs/pytorch310_cpu/bin/python verify_bn_calibration.py
```

`deps/` 未纳入源码交付；在另一台相同 CPU 环境中缺该包时可用 `python -m pip install --no-deps --target ./deps einops==0.8.1` 重建探针局部依赖。验证命令覆盖自己的 JSON。模型源码 SHA256 及脚本哈希保存在原生执行结果中。【可复现入口】
