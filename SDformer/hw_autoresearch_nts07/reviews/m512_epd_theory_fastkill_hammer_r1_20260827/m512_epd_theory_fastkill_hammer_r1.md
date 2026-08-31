# M512｜EPD 理论 headroom 独立 fast-kill

日期：2026-08-27  
结论：`NO_GO__KILL_PHASE_BALANCED_MULTI_SOURCE_EPD_SCHEDULER_BEFORE_RTL`  
评分：**98/100**  
P0/P1：**0/0**  
运行 GPU/VCS/DC/DSE：**否**

## 一句话裁决

原 EPD 的 `phase-balanced multi-source issue` 在 product-issue 轴上没有任何
理论 headroom：四层 Cout=`384/192/96/96` 全是 96 整倍，强 A1 对每个
source-event×legal-tap 都能以完整 96-output vector 满发，因此
EPD/A1 product-issue 上界严格是 **1.0x**。

这条可直接快杀 RTL，无需等 exact S10。S10 仍然值得抓，但用途改为检验
memory stall 和新的 memory-centric decoder 机制。

## 1. Product-issue 上界证明

对层 `l`，令 `q_l=Cout_l/96`；某 active source event 的合法 spatial tap 数为
`b_e∈{4,6,9}`。则：

```text
P_l = Σ_e b_e × Cout_l
    = 96 × Σ_e b_e × q_l
```

任意 96-lane 机器至少需要 `P_l/96` 个 issue cycle。强 A1 的构造式调度是：
按 source、legal tap、96-channel slice 三重遍历，每拍发一个完整 vector。
它恰好用 `Σ_e b_e q_l=P_l/96` 拍，每拍 96/96 useful lane。

所以 multi-source bundling 只能改顺序，不能再减拍。M510 的
`170.62--183.47M cycle/frame` 只是 aggregate 下的 `P/96` 范围，不是
exact trace；但“任意实际 bitmap 的 useful lane utilization=100%”是不依赖 bitmap 的
整除证明。

### boundary 不会救 EPD

`4/6/9` 只决定每个 source 有几个完整 Cout vector，不会造成 lane tail：

| 层 | Cout/96 | boundary source 最少 issue 拍 | interior source issue 拍 |
|---|---:|---:|---:|
| D0 | 4 | 16 | 36 |
| D1 | 2 | 8 | 18 |
| D2 | 1 | 4 | 9 |
| D3 | 1 | 4 | 9 |

即使最短的 D2/D3 boundary event 也可让一个单-source expander 连续发 4 拍；
小 FIFO 就能让 descriptor decode 跟上，不需要并发解码多个 source。

## 2. 哪些只是 memory stall

| 轴 | 强 A1 的合法能力 | EPD 还有什么 |
|---|---|---|
| Weight fetch | 同端口下每拍预取一个 96-weight vector；与 EPD 共享同等 cache/reuse | 只能隐藏 latency，不能突破端口吞吐 |
| Psum update | deterministic parity banking + RAW forwarding + 同容量 FIFO/cache | 只能重排尚未被 A1 消除的 stall |
| Commit | exact output vector 集合固定 | source 顺序不能减少 commit |
| Bank conflict | A1 可以用相同 parity mapping | 不得把只给 EPD 的 banking 当创新倍速 |

若 EPD 把 A1 的 stall `S` 全部消掉，而不改变 product issue `I`，则最好
倍速是 `1+S/I`。要过 `1.30x` 门，A1 原本至少 23.08% 的总周期必须是可全消除
stall；要到主性能所需的 `1.66--1.80x`，可全消除 stall 必须占 A1 总周期
39.76%--44.44%。这些都只能由 memory-aware S10 simulator 证明。

更关键的是，若强 A1 同样使用 deterministic parity banking，原 EPD scheduler 的
差异就只剩通用 multi-source/OOO。FireFly-T 已明确包含 concurrent nonzero-spike
decode、load balancing、weight dispatch 和 OOO bank-conflict elimination。因此即使真有
stall 收益，也不足以保住当前 scheduler novelty。

## 3. 与前作的边界

- [Sparse convolution SNN accelerator](https://arxiv.org/abs/2203.12437)：已有 compressed
  spike queue、self-timed scheduling 和 memory interlacing。
- [FireFly-T](https://arxiv.org/abs/2505.12771)：已有多 nonzero-spike 并发解码、
  weight dispatch、load balance 和 OOO bank-conflict elimination。
- [Prosperity](https://arxiv.org/abs/2503.03379)：若试图用相似模式复用结果
  来减 product，必须按 product sparsity 直接对标。
- [Transposed-convolution decomposition](https://arxiv.org/abs/2205.02103)：polyphase/跳过插零和
  相对 naive 的倍速已是前作。

所以 A0 dense 对 bit-sparse 的 `4.48--4.81x` 只能留在 opportunity 列，严禁
替换 EPD/A1。

## 4. 最多两个可由 exact S10 筛选的 decoder 机制

### C1｜Parity-Gather Psum Residency（PGPR）

把 source-scatter 反转为四个 output-parity gather stream。对一个 96-channel output
slice，在小 RF 中常驻 psum，遍历该 output 的 `1/2/2/4` 个空间 source 位点及
active Cin，最后只 commit 一次。

它不减 product，而是减 psum SRAM RMW。四层每帧只有 `1,104,000` 个
96-channel output-vector commit；Acc24 纯 commit 下限约 `303.22 MiB/frame`。这个数不是
DRAM 实测，也不包含 bitmap/weight 交通。

S10 必测：每 parity/output 的 contribution histogram、bitmap scan 成本、weight cache
hit、RF lifetime、总 psum 字节和总周期。基线必须是已有 parity banking 和同容量
psum cache 的强 source-centric A1；还要补一个常规 output-stationary A1。若 PGPR 与后者
代数同构，或 memory-aware cycles `<1.30x`，立即 KILL。

这不直接复制 SNE 的 source queue、FireFly-T 的 OOO 或 Prosperity 的 product
reuse；但 output-stationary 本身是常识，只有“exact bitmap completion × transposed-parity
gather × vector-psum residency”的窄缝有候选新意。

### C2｜Exact T10 Temporal-Delta Recurrence（TDR）

固定权重下：

```text
y0 = W*x0 + b
yt = y(t-1) + W*(xt - x(t-1)),  t>0
```

ATLIF binary source 的 XOR 位置产生 `+/-` delta event。这只有在 value scale、累加位宽和
次序全部固定时才是 exact。S10 必须报：`x0+Σ XOR` 的 boundary-weighted
product、正负 delta、逐层/逐 sample/逐 timestep exact miter，以及 state SRAM 读写后的总周期。

状态不能免费：前一 timestep 的四层 packed source bitmap 合计约 `0.830 MiB`；
前一 output state 若四层同时保留，INT16/Acc24 约 `20.215/30.322 MiB`；
按层复用 SRAM 时，D3 峰值仍约 `14.063/21.094 MiB`。

若 `P_delta/P_A1 >= 0.7692`，即使 state 零代价也达不到 `1.30x`，立即 KILL；
否则仍必须在收取 SRAM 后达到 `>=1.30x`。

这不与 SNE/FireFly-T/Prosperity 直接重合，但与
[DeltaCNN](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html)
直接相邻。可能的窄缝只是“T10 binary ATLIF-fed ConvTranspose2d、精确
parity boundary、全额状态收费”，不可声称发明 temporal delta。

## 5. 最终 GO/NO-GO

- 原 `phase-balanced multi-source EPD scheduler`：**NO-GO / KILL**。
- 为该 scheduler 写 RTL/VCS/DC：**NO-GO**。
- exact S10 decoder bitmap capture：**GO**，但仅用于 PGPR/TDR 与 memory-stall 测量。
- PGPR/TDR RTL：**NO-GO**，各自先过上述离线 fast-kill。
- A0 dense/bit-sparse opportunity 写成 EPD 创新倍速：**永久禁止**。
