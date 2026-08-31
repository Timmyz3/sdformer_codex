# M232 独立打铁评审

结论：**76/100，P0/P1/P2 = 2/6/2**。M232的exact-SHA、12块几何咎coef-service递推算术通过，但只能作条件性cycle screen，不能作BN/FFN/system性能或PPA。

## 独立复算

| 项 | 结果 |
|---|---:|
| FFN blocks / BN phases | 12 / 24 |
| BN1 / BN2 channel coefficient pairs | 17,664 / 4,416 |
| total pairs / alpha+offset scalars | 22,080 / 44,160 |
| II16 serial no-overlap | 353,280 cycles |
| ping-pong exposed, **conditional on Lfirst=16** | 21,504 cycles |
| II31 / II32 all-phase rate match | pass / fail |
| BN1 / BN2 channel tiles | 1,104 / 46 |
| M167 PREFOLD issue floor omitted by screen | 7,728 |

M232的`channels*II`填tile公式隐含首结果延迟`Lfirst=II`。更一般的公式是`Lfirst+(channels-1)*II`；当II=16而Lfirst=64时，24 phase的coefficient-only暴露是22,656 cycles，不是21,504。steady-state的II31/32边界不变。

## 必须纠正的存储基线

`184,320 bits = (3,072 BN1 + 768 BN2) * 48`的算术无误，但BN1和BN2是顺序互斥phase，基线也可复用同一buffer。公平的最大单phase payload是`3,072*48=147,456 bits`，所以两个96-channel tile的`9,216 bits`对应**16x** local coefficient-payload机会，不是20x。这不包含tag/ECC/banking/control或PREFOLD结果。

## 硬件缺口

1. 单路系数引擎尚未实现：rsqrt/divide、`alpha=gamma*invstd`、`offset=beta-mean*alpha`的乘加、首延迟、II、RNE/饱和咎误差都未绑定。
2. channel-major双tile需要地址化rank/raw state buffer、row-to-channel reorder、bank端口咎反压；当前只有无stall recurrence。
3. BN1每个16-channel tile需M167的7个PREFOLD issue，共至少7,728 issues。PREFOLD咎BACK共用同一96-product pool，不能在连续BACK中免费隐藏。
4. BN2的96-lane消费者只是理想issue边界；fc2 raw位宽、affine乘加、residual source read、raw buffer replay咎commit都未建模。

## 接纳口径

允许引用：精确通道数；在`pair II=16, Lfirst=16, channel-major, no-stall`条件下的353,280/21,504 cycle recurrence；II31/32稳态边界；纠正后的16x payload-only机会。

禁止引用：20x公平存储节省；已实现II16 rsqrt引擎；21,504为完整BN cycles；M167 PREFOLD/BN2 residual已组成；任何accuracy、PPA、energy、FFN/network/system speedup。

下一里程碑应直接做M233：用A800真实mean/variance/gamma/beta/epsilon咎激活范围定位宽，实现带valid/ready的系数对引擎咎两个有限96-channel bank，显式加入PREFOLD、BN2 residual咎存储器stall，然后跑exact-SHA VCS/SVA与新思DC。
