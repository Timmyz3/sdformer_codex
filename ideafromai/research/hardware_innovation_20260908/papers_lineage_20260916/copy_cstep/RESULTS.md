# C-STEP 邻域公共脉冲：逐通道 AND（K=2）

不是整向量相等（`copy_hotspot` 横 **1.07%**）。论文 Fig.2(a) 的 82.7% 是 **K=2 邻 token 上每个通道 bit 的交**。

身份：AT-LIF `{0,θ}` 吸入 W；实测非零幅值恒为 1，bit = `|x|>0`。10 帧 GT。挂钩 `r0.conv2` 输入（T×96×240×320）和 `layers.1.swin_blocks.0.mlp.fc1` 输入（T×192×60×80）。横向邻像素/token。

定义：`common=|A∧B|`，`union=|A∨B|`，`jaccard=common/union`。复用精度 = 左像素通道 c 发放时右像素同通道也发放的比例。

| 层 | nnz | Jaccard | 复用精度 | 独立基线 Jaccard `p/(2-p)` | 论文 K=2 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| r0.conv2 | 4.37% | **0.224** | 0.366 | 0.022 | 0.827 | **WEAK**（<0.3） |
| L1.b0.fc1 | 5.03% | **0.353** | 0.522 | 0.026 | 0.827 | **MID**（不到 0.5，不 KEEP） |

邻域相关存在（约 10× 独立），但远低于 C-STEP 分类 ST 的 82.7%。公共项只算一次相对 **发放位** 的算术份额：conv2 18%、FC1 26%——不是周期、不是硅 PPA。

**去留：** 两层都不到 KEEP 门（Jaccard>0.5）。r0 局部公共复用弱；L1 略高仍不是 C-STEP 量级。公共子集继续走已有 Prosperity/FireFly 产品稀疏，不要当新 X。

收据：`results/channel_and.json`。
