# R8 到 I24：真实合同、数据和质量

固定上轮Q1/Q2/output_scale，本目录给出新的真实BN/identity合同和独立质量证据；不训练、不重选rank/格式。所有模型运行在已有A800 Python3.12.3 / torch2.2.2+cu121 / cupy13.6.0，旧目录只读。

全部825已完成并逐帧配对：新Q40/Q20消费者AEE **1.3276350226079938**，原整数R8/浮点消费者 **1.3276465686026133**，同A800环境原NB0 **1.447936665574317**。历史NB0 **1.44535253468097**单独保留；新部署低于新旧两个质量门。三侧均825帧/48,152,523有效像素。见 [QUALITY_REPORT.md](QUALITY_REPORT.md)、[quality_validation.json](quality_validation.json)、[825逐帧配对](quality_paired825.csv)。

## 固定的真实出口

真实MS_ResBlock为 `identity=x; sn1→conv1→norm1→sn2→conv2→norm2; output=identity+branch`。r0输出直接进入r1.sn1。当前r0 BN2是固定eval（不是动态patch.proj BN）：training=false、track_running_stats=true、有运行均值/方差、eps=1e-5。参数与验证见 [bn_runtime.json](bn_runtime.json)、[consumer_contract.json](consumer_contract.json) 和 [consumer_first8.npz](consumer_first8.npz)。

原输出先按float32卷积、BN、ADD，再在实际LiteralForward.reader做I24 signed24/f14 RNE/sat。新出口固定为：

```
a[o] = RNE(output_scale[o] * BN_gain[o] * 2^40)   signed32
b[o] = RNE(BN_offset[o] * 2^20)                  signed32
J    = sat32(RNE(identity_float32 * 2^20))
wide = int64(p)*a + ((int64(J)+b) << 20)
I24  = sat24(RNE(wide / 2^26))
```

a范围540863…2113405、b范围−1009544…338446；宽积加和按冻结p界和任意J32仍安全。详细定义及小fixture见 [consumer_integer_definition.json](consumer_integer_definition.json)、[consumer_integer_first8.npz](consumer_integer_first8.npz)。这项BN/scale融合与J/a/b量化改变原float32舍入步骤，必须独立质量；不是原float消费者bittrue。

原网络后续r1浮点卷积/BN/add是当前部署helper忽略的影子计算，真实后继读取I24。评价把I24/16384精确编码为float32，在当前reader逐帧核对I24全相等。保留的其它浮点网络与原PSN/头不能据此称全网bittrue。

## 硬件数据接口

帧固定zurich_city_09_a_0001，来自本轮同A800前向。原生源为 [first_source_words.npy](first_source_words.npy)，uint16低10位、形状[C96,H240,W320]。输出tile按120×160行主序，每tile包含[T10,N96,2,2]。三份完整数组均int32[19200,10,96,2,2]：[raw_p_full.npy](raw_p_full.npy)、[identity_q20_full.npy](identity_q20_full.npy)、[i24_new_full.npy](i24_new_full.npy)。对应first64文件是tile ids128…191，跨行边界。

[validate_full_fixture.py](validate_full_fixture.py) 用纯NumPy int64逐值独立重算全部73,728,000个新I24，全部一致；8个小块与64块也和完整数组一致。见 [full_fixture_contract.json](full_fixture_contract.json)。新增实际IEEE FP32 identity入口数据见 [identity_fp32_full.npy](identity_fp32_full.npy) 和 [identity_fp32_contract.json](identity_fp32_contract.json)。因原先仅落盘J，额外执行唯一首帧前缀并停在r0输入；全部73,728,000个J转换值经远端和本机独立检查一致，未重跑质量或更改格式。

完整首帧新旧I24有468,058个值相差1，最大差1；十帧AEE有可见变化，不能将该差异描述为任务上无损。最大尾帧两函数各两次交替复测逐位重复，具体误差、9个r1门翻转及仅参数容器training标志的源码审计见QUALITY_REPORT。

## 独立执行审阅

[REVIEW_CONSUMER.md](REVIEW_CONSUMER.md)另核对最终FP32入口与连续退休，5,332项守恒检查全过。[REVIEW_PACKED.md](REVIEW_PACKED.md)从原生源独立核对双P打包的3,808项计数及53,760个gold；带宽/异步MAC时序边界明确单列。[REVIEW_CONSUMER_PACKED.md](REVIEW_CONSUMER_PACKED.md)补核完整FP32出口与真实全帧双P需求，2,772项消费者守恒及72项源需求检查全过。

[REVIEW_FUSION.md](REVIEW_FUSION.md) 核对相邻支持/差分/窗口执行：280记录、5,600计数检查全过；旧mode6/7共112记录×11字段完整复现。审阅仅认可既定tile与端口的功能/周期证据；额外支持位、寄存器窗口和选择逻辑均计资源，物理时序未验证。

复现入口为 [run_r8.py](run_r8.py)、[evaluate_deployed.py](evaluate_deployed.py)、[evaluate_nb0_valid825.py](evaluate_nb0_valid825.py)。远端使用既有env312并设置R0_STREAM_BASE、R0_STREAM_REPO为远端实验根/代码根，所有输出仍在本目录对应镜像；本机纯整数审阅入口固定`/opt/anaconda3/bin/python3.12`。耗时仅为评价作业时长，不是网络速度比较。

## Git中的冻结参数

[frozen_parameters.json](frozen_parameters.json)保存实际Q1/Q2、output_scale、consumer a/b，以及bias/theta、BN gain/offset；每项包含shape/dtype/实值。整数原值与float64十进制JSON往返均逐项array_equal验证。`expanded`按`q2.astype(int64) @ q1.astype(int64)`精确恢复，无需重复保存82,944个系数。

重建执行所需NPZ字段（输出到独立目录，不覆盖现有文件）：

```python
from pathlib import Path
import json, numpy as np
v = json.loads(Path('frozen_parameters.json').read_text())['arrays']
a = {k: np.asarray(x['values'], dtype=x['dtype']).reshape(x['shape']) for k, x in v.items()}
out = Path('reconstructed_parameters'); out.mkdir(exist_ok=True)
f = {k: a[k] for k in ('q1', 'q2', 'output_scale', 'bias', 'theta')}
f['expanded'] = a['q2'].astype(np.int64) @ a['q1'].astype(np.int64)
np.savez(out/'factors.npz', **f)
np.savez(out/'consumer_coefficients.npz', **{k: a[k] for k in
    ('a_q40', 'b_q20', 'q1', 'q2', 'output_scale', 'BN_gain', 'BN_offset')})
```

原NPZ的first_original/second_original/first_scale只是量化前SVD诊断档案，不属于冻结执行函数，也不是上述运行入口所需字段；保留本地原文件。模型、完整source/identity/gold等大数据继续由本地/远端镜像保留，不因这个小JSON就声称可从Git单独重建完整模型或验证集。既有runners未修改。
