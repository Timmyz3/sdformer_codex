# GCM-P投影阈值与最小定点合同

**日期**：2026-07-13  
**静态结构审计**：通过  
**网络精度签核**：未完成，仍需真实valid825投影量化推理

## 1. 结论

1. H67与H68的12个attention block均使用标量K事件阈值，且checkpoint中全部精确为1.0。GCM-P无需为token、时间或channel携带事件幅值；该阈值折叠在当前两条线中是恒等操作。
2. 每个投影都是带bias的C乘C Linear，后接eval态BatchNorm2d。BN可静态折叠为一组有效权重和有效偏置，运行时不需要独立BN除法或平方根。
3. 本审计只证明checkpoint结构、BN折叠公式、int8权重与int32累加的数值代理。它不等于valid825网络精度，也不允许据此冻结最终位宽。
4. GCM-P和direct基线必须使用同一组折叠后权重码、偏置码、累加位宽和末端舍入；这样架构消融只比较数据流，不混入量化差异。

## 2. 静态合同

| 设计 | block数 | K阈值 | head维 | BN | 结构结果 |
|---|---:|---|---:|---|---|
| H67 | 12 | 1..1 | 32 | BatchNorm2d | 通过 |
| H68 | 12 | 1..1 | 32 | BatchNorm2d | 通过 |

折叠公式为：

```text
alpha[o] = gamma[o] / sqrt(running_var[o] + eps)
W_fold[o,i] = alpha[o] * W[o,i]
b_fold[o] = alpha[o] * (b[o] - running_mean[o]) + beta[o]
```

GCM-P对每个`(block, final_gate_code, global_input_channel)`生成一次
`gate_code乘W_fold[:,i]`，但每个token-output仍独立累加。若接口使用head内局部lane，则地址必须
显式携带head并转换为全局input channel；不同block权重不同，不能跨block合并。跨window复用只能
使用最终Q1.7 gate code，不能使用归一化前score class。

## 3. CPU最小int8代理

gate固定为无符号Q1.7码0到256，权重分别采用整tensor对称int8和逐输出通道对称int8，bias换算到对应累加尺度并使用int32。合成输入覆盖2%、10%和50%事件密度；这些输入只用于检查算术范围与量化误差，不代表真实光流分布。

| 设计 | 权重量化 | 最大权重相对L2 | 最大合成输出相对L2 | 理论累加上界 | int32理论余量 | scale条目 |
|---|---|---:|---:|---:|---:|---:|
| H67 | 整tensor int8 | 0.011345 | 0.011736 | 4,906,782 | 437.7倍 | 12 |
| H67 | 逐输出通道 int8 | 0.007686 | 0.007798 | 7,802,394 | 275.2倍 | 4,416 |
| H68 | 整tensor int8 | 0.011318 | 0.011448 | 4,887,942 | 439.3倍 | 12 |
| H68 | 逐输出通道 int8 | 0.007682 | 0.007730 | 7,816,202 | 274.7倍 | 4,416 |

## 4. RTL冻结边界

首版RTL可以冻结：

- K事件payload为1 bit，当前H67/H68的K阈值1.0不占运行时接口；
- gate为9 bit无符号Q1.7码，范围0到256；
- 投影先做BN静态折叠，再将同一份权重镜像供direct和GCM-P模式使用；
- 乘积为9乘8 bit有符号结果，token-output使用至少32 bit有符号累加；
- bias在所有活动输入累加后加一次，不能随class或K事件重复加入；
- int32安全性由所有输入channel取最大gate码时的逐输出通道绝对和上界证明，不依赖随机样本；
- 最终重标定、舍入、饱和和后续ATLIF输入格式在网络量化验证后再冻结。

当前不能冻结：

- 最终选择整tensor还是逐输出通道int8；
- 输出截位位宽、逐层scale SRAM格式和bias码；
- int8投影后的valid825 AEE、AAE、spikes变化；
- SRAM宏、乘法器和多播网络的真实DC/SAIF PPA。

## 5. 最小补跑清单

1. 在H67 epoch19与H68 epoch19的冻结dyadic部署图中加入12层BN折叠投影int8仿真，跑valid825。
2. 同时记录逐block输出相对L2、最大误差、下一ATLIF事件翻转率和最终AEE/AAE/spikes。
3. 优先比较FP32投影、整tensorint8、逐输出通道int8三档；若逐通道scale控制开销过高，再评估按stage或按16输出通道组共享scale。
4. 接受门槛暂定为AEE相对当前dyadic部署退化不超过0.5%，AAE退化不超过0.1度，spikes变化不超过1%，且无NaN/Inf。超过门槛时不进入DC主线。

## 6. 证据边界

本报告的静态结构数据来自真实checkpoint；量化误差来自确定性CPU合成输入；网络精度与真实workload的最终结论仍等待valid825和ordered profile。论文中必须分开标注，不能将本报告的合成误差写成任务精度。
