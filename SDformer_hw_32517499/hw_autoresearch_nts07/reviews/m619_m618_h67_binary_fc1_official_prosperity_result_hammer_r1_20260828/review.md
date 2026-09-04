# M619：M618 H67 binary-FC1 × 官方 Prosperity 结果独立打铁评审

## 裁决

**PASS，97/100；P0=0、P1=0、P2=1。** M618 在冻结 H67 ep35 的 100 组 exact-binary FC1 输入上，将官方 Prosperity 的 product-sparsity 与同配置 bit-sparsity 比较，聚合周期比 **2.3728887908×**。独立复算、三 stage 六次官方 CPU 小样本重放和双 seal 均通过。

这个数字只准写成 **external official-artifact iso-workload opportunity**：不是本项目 RTL、不是与 M481 同资源、不是全网/系统倍速，也不是能量、PPA、FPS 或精度结果；不得除以 M481，不得与任何自研倍率相乘。

## 独立核验

- 官方仓库固定在 `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`，源码 SHA 与 M618 一致，评审前后工作树为空；未修改官方仓库。
- M51 manifest、GPU payload-validation receipt、100 个 payload 的 SHA/字节数/little-bit tail/popcount/exact-binary 谓词全部一致。人口为 10 samples × 10 FC1 modules = 100 records；stage0/1/2/3 = 2/2/6/0，无 stage3 FC1，也未拿 FC2 替代。
- 全部输入为 829,440,000 elements、112,213,979 active、103,680,000 packed bytes；FC1 subset SHA 为 `d4792d07dc9e4bcdce578e696d7f0912daf9882774ae0b3184386b68e8310931`。
- 映射正确：冻结 `[T,B,H,W,C]` 先变为 `[T,BHW,K]`，官方 `run_fc` 再转为 `[BHW,T,K]`；实际层 `(K,N)` 分别为 stage0 `(96,384)`、stage1 `(192,768)`、stage2 `(384,1536)`。K/N 均整除官方 `K_tile=16/N_tile=128`，M 尾块由 `cur_tile_size_M` 收费。
- 100 records × 2 modes 的 N=128 expansion 逐 counter 检查通过；官方 initial/middle memory 方程、`compute=max(issue,preprocess)`、total、stall、support 和 N 倍数关系均零 mismatch。M618 的六项 direct full-N 检查也全部零 mismatch。
- 21 个聚合桶（overall、10 modules、10 samples）全部独立重加一致。复算得到 bit/product cycles = **757,894,814 / 319,397,528**，聚合比 **2.3728887907986564×**，逐记录 geomean/min/max = **2.3389121639148795× / 1.8646697243687171× / 2.735905491218319×**，product support reduction = **63.9664644634%**。
- 独立官方 CPU smoke 只跑 sample0 的 stage0/1/2 各一条、bit/product 共 6 次 N=128 调用，全部 exact match：`(3018886,1145756)`、`(465183,195592)`、`(586202,239988)`；没有进行被禁止的全量官方复跑。
- 官方 modeled traffic：DRAM 两模式均为 27,695,185,920 bit = **3.461898240 decimal GB**；global-buffer bit/product 为 1,517,095,827,456 / 1,032,513,395,712 bit = **189.636978432 / 129.064174464 decimal GB**。这些是模型 traffic counter，不是硅后 DRAM 流量或能量。

## P2

M618 Markdown 中“`K=16、M=256、N=128`”容易被读成真实层维度。论文及后续 admission 必须改写为 `K_tile=16, M_tile=256, N_tile=128`，并并列给出上述实际层 K/N；GB 必须标注为 `modeled traffic, bits/8e9`。

## 授权边界

允许下一作者创建 admission record，但仅限把该点用于“官方 Prosperity 框架在同一冻结 binary-FC1 workload 上，product-vs-bit 为 2.37×”的外部机会/捕获差对标。`headline_admitted=false`；不得把 2.37×写成 ours 或 DATE 系统主结果。

资源回执：official full-100 rerun=0；official CPU smoke=3 records/6 modes；GPU=0、EDA=0、remote=0、official repo write=0。`docs/359` SHA 未变。
