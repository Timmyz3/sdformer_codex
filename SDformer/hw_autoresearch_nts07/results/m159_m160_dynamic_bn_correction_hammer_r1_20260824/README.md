# M159/M160 动态 BN correction overlay 独立打铁复核 r1

## 结论

overlay 的核心裁决成立：冻结 H67/Motion ep35 profile 不是 running-stat BN 推理，而是 `no_running`、eval batch 1、当前批次统计。M159 的算术热点可以保留，M160 r1 的静态 BN fold 只能保留为 constructor-default counterfactual，不能代表冻结推理。

我用冻结配置、production profile loader 和 ep35 checkpoint 在 CPU 上重新实例化模型。checkpoint load 为 `missing=0/unexpected=0`；profile protocol 前 78/78 BN 有 running buffers，调用 `configure_batch_norm_evaluation(..., "no_running")` 后 78/78 均为 `track_running_stats=false` 且 running mean/var 为空。模型共有 12 个 `MS_Spiking_Mlp`，其中 24 个 FFN BN 全部为 multi-step；每个 FFN 的 `sn2` temporal weight/bias 都是共享的 `[10,10]`/`[10,1]`，不是按 hidden channel 复制。

综合评分：overlay 语义纠正 `92/100`，M161 r1 DSE `81/100`，当前硬件 admission readiness `43/100`。没有 P0 虚假 admission；P1 主要是强基线、公平性、rank-3 checkpoint/精度、定点与可执行 buffer/BN2 schedule 尚未收口。

## 源码与冻结协议复核

- 配置原始训练 loader batch size 是 2，但 `load_config()` 强制 profile loader 为 1，主入口 DataLoader 也显式为 1；`test.eval_batch_size=1`、`test.bn_policy=no_running`。
- `build_model()` 先安装 105 个 ATLIF、完整加载 checkpoint、设置 `step_mode=m` 并 `model.eval()`；主 profile 随后才应用 `configure_batch_norm_evaluation()`。M160 r1 停在前一步，正是误判根因。
- SpikingJelly `BatchNorm2d(step_mode=m)` 通过 `seq_to_ann_forward()` 将 `T` 与 `B` 展平后调用普通 BN。因此 FFN BN1/BN2 每通道的统计域是 `T*B*H*W`，不是每个 timestep 单独一组统计。四个 stage 的每通道 population 分别是 192000、48000、12000、3000。
- 正确 FFN 顺序是 `sn1 -> dropout1(p=0) -> fc1 -> BN1(current batch) -> sn2 -> dropout2(p=0) -> fc2 -> BN2(current batch) -> residual`。DropPath 只在 attention residual。
- `ATLIFTernaryPSN.forward()` 先把 timestep 以外的维度全部 flatten，然后由同一 `[T,T]` 或 `L[R]` temporal matrix 左乘所有列；因此 shared temporal parameters 不能随 hidden channel mask 删除。

## 保留/撤销边界

M159 可保留：12 blocks、120 个 s10 groups、159,784,111 Linear activity cycles、45,600,000 已计入的 FFN-local ATLIF issue cycles、合计 205,384,111、占 620,302,905 envelope 的 33.1102933%，以及 BN1/BN2/residual 的 350,208,000 / 87,552,000 / 87,552,000 element extents。

M159 必须撤销或降级：running-stat 语义、FFN DropPath 节点、按 hidden channel 删除 sn2 temporal 参数；5,472,000 只能叫 96-lane 一遍扫描的 ideal vector-issue quotient，不能叫完整 BN/residual cycles。

M160 r1 可保留：12-module checkpoint parameter census，以及“假设 running-stat BN”时的静态 affine 代数。冻结推理必须撤销：静态 BN1/BN2 fold、静态 zero path、`176640 -> 17904` temporal-bias storage、437,760,000 elements/frame 的静态 no-materialization 资格。

## 动态 BN 与 rank-R temporal transform 的精确代数

令 `x[t,p,j]` 是 fc1 输出，`p=(b,h,w)`；`M=T*B*H*W`。PyTorch 当前批次 BN forward 使用 biased variance：

`S[j]=sum(t,p) x[t,p,j]`

`Q[j]=sum(t,p) x[t,p,j]^2`

`mu[j]=S[j]/M`

`var[j]=Q[j]/M-mu[j]^2`

`a[j]=gamma[j]/sqrt(var[j]+eps)`

`c[j]=beta[j]-a[j]*mu[j]`

于是 `y[t,p,j]=a[j]x[t,p,j]+c[j]`。对 `W=L*R`，定义 `u[r,p,j]=sum(t)R[r,t]x[t,p,j]` 和常数 `r1[r]=sum(t)R[r,t]`，则：

`R*y = a[j]*u + c[j]*r1`

`h = L*(a[j]*u+c[j]*r1)+psn_bias`

所以 barrier 后的正确 rank-state 修正是 `u_bn=a*u+c*(R*1_T)`，不是 `a*u+c`。12 个实际 BN1 的随机张量 manual miter 最大误差为 `9.54e-7`；对每个实际 sn2 的 rank-3 SVD effective matrix，直接 BN 后投影与上述 rank-state correction 的最大误差同为 `9.54e-7`。故意漏掉 `R*1_T` 时，12 个模块的最大误差范围为 `0.0883..1.0763`，该项必须进入 RTL。

## 何时可以把 T=10 materialization 换成 R=3 state

必须同时满足：

1. 部署模型实际使用并验证 `W=L*R`；不能只在硬件中把冻结 dense W 临时截断。
2. BN affine 系数在 temporal axis 上共享。当前 multi-step BN 展平 `T*B`，这一条成立；若改成 BNTT/per-timestep BN，则不成立。
3. 在丢弃 raw `x` 前，同一次 ingest 已完整累计 `S/Q` 和 `R*x`。
4. moment barrier 后才允许输出，并包含 `c*(R*1_T)`、PSN bias、left projection 和 threshold。
5. BN1 与 sn2 之间没有 nonlinear consumer；当前 topology 满足，dropout 为 p=0 identity。
6. rank-state 精度、SRAM 宽度/端口、context/tag、backpressure 和 fixed-point rounding 已由 VCS/reference miter 接纳。

当前 ep35 checkpoint 的 12 个 FFN sn2 全部仍是 dense `temporal_factor_rank=0`。对其做未经训练的最佳 rank-3 SVD，relative Frobenius error 为 `0.4138..0.5412`，因此 R=3 只是等待 PAFT/valid825 的部署候选，不是冻结 checkpoint 已有性质。

## M161 r1 的可保留部分与公平基线修正

M161 的实数代数、`R*1_T` 常数项、T10/R3/16-lane/96-product-slot 几何、32 raw values/issue 和 1.667x sn2 局部乘积数候选均可保留；其所有 cycle/system/RTL/VCS/PPA admission 为 false 也正确。

但 `4.203x` Q24 bit-movement 只相对“3 次 raw + 2 次 Q8”的五移动显式物化基线成立。强 dense baseline 可以在 fc1 输出时同步累计 moments 并写 raw 一次，barrier 后读 raw 一次、边 normalize 边直接送入 dense PSN，不需要 stats reread 和 normalized write/read。按 M161 自己的逐模块 raw width 重算：

- 强 dense streaming baseline：10,395,648,000 bits/frame；
- Q24 rank-3 write+read：5,042,995,200 bits/frame；
- 公平 Q24 reduction：`2.0614x`；
- Q8 rank-3（仍需训练）：1,680,998,400 bits/frame，公平 reduction `6.1842x`。

因此 `4.203x/12.610x` 只能标注为“相对显式五移动实现”，DATE 对标与 paper-safe 句子应优先使用 `2.061x/6.184x` 强基线数字，直到 SRAM transaction model 给出新的公平 A/B。

## 不能省略的成本

- BN1 每通道 `sum/sumsq/count`、biased variance、epsilon、variance nonnegative clamp 和 reciprocal-sqrt/affine coefficient generation；
- 完整模块 moment barrier；32 square lanes 还需要解决同通道多更新的 bank/RAW conflict，不能仅由 issue quotient 宣称零额外周期；
- `R*x` product/add、R=3 state buffer、地址/tag、barrier 后 read/correction、`R*1_T` 常数、L reconstruction 和 threshold；
- BN2 的 `sum/sumsq`、barrier，以及 raw fc2 output 的 buffer+replay、或 fc2 recompute；
- residual tensor 的保留/重读、对齐、add 和 commit；
- coefficient/moment/state traffic、双缓冲、端口空泡和固定点保护位。

## M161 RTL/VCS 最小可执行切分

先只做 `BN1-current-batch + rank3-right-state + correction + left/threshold` island；BN2/residual 另做 commit island。不要在 BN2 完成前将其称为完整 FFN。

RTL 至少需要 `IDLE -> INGEST -> MOMENT_BARRIER -> COEFF -> RANK_REPLAY -> EMIT` 状态，输入 tag 包含 module/context/timestep/spatial-column/hidden-lane。VCS/reference miter 必须覆盖：

- 四种真实 stage geometry、12 个实际模块、batch 1、T=10；
- PyTorch `no_running` BN，使用 biased variance；all-zero、constant、near-zero variance、正负 gamma/beta、epsilon；
- `R*1_T` 为零/非零/正负混合，另设故意漏项的 negative test；
- rank-0 checkpoint 必须拒绝进入 rank-3 mode，PAFT factor identity 必须 exact-SHA 绑定；
- accepted-beat 精确计数、duplicate/missing/out-of-order timestep、跨 context 污染、reset/epoch、任意 backpressure；
- sum/sumsq overflow、variance cancellation/clamp、rsqrt 与 alpha/offset 的 fixed-point 顺序；
- 同通道双 timestep 更新的 moment-bank collision 与 forwarding；
- BN2 buffer/replay 或 recompute、residual 在 stall 下的 tag 对齐、commit 前无可见输出。

## P0/P1/P2

P0：0。现有 overlay/M161 均保持 fail-closed，没有把 rank-3、cycle、system 或 PPA 当作已接纳结果。

P1：

1. 当前 ep35 checkpoint 是 dense rank-0，未经训练 rank-3 SVD 误差很大；必须等 PAFT/valid825。
2. M161 的 4.203x 使用弱五移动 baseline；强 streaming baseline 下是 2.061x。
3. 32-lane moment path 尚无 conflict-free bank/port/forwarding schedule 和 RTL。
4. fixed-point moment/rsqrt/correction/threshold 次序尚未证明。
5. BN2 replay/recompute 与 residual commit 尚未成为可执行 schedule。

P2：

1. overlay identity 尚未固定 SpikingJelly `layer.py`/`functional.py`，而它们决定 `T*B` flatten 语义。
2. 当前 bit-movement 数字未计 coefficient、moment、tag 和 residual traffic，必须继续叫 local intermediate candidate。
3. M161 应把 `M=T*B*H*W`、biased variance 和 `R*1_T` 写入最终 RTL contract/SVA，而不只留在 DSE prose。

## 最终裁决

`PASS_SEMANTIC_CORRECTION_REVISE_M161_BEFORE_RTL_ADMISSION`。可以继续写独立 BN1-rank3 bridge RTL；不得以 4.203x 对标强基线，也不得在 PAFT/valid825、fixed-point、SRAM schedule、VCS 和 BN2 commit 完成前声明完整 FFN 或系统速度优势。
