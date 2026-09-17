# 阶段性结果：跳零/浅网/空 SSA 全部叠上，仍然不是成倍（2026-09-16）

10 帧 `zurich_city_09_a` GT，ep34 全网。不是 valid825，不是 RTL 净服务。身份：AT-LIF 吸入 W。

## 质量（可叠）

| 配置 | AEE |
|---|---:|
| 全跑 preds.2 | 0.716 |
| 只用 P1（跳 decoder2） | 0.709 |
| Jung：≥90% 空 Q 则关该块 SSA（42% 块） | 0.713 |
| **P1 + Jung 同时** | **0.710** |
| NB0 门 | 1.45 |

质量站得住。decoder2 和空 SSA 可以一起关。

## 系统名义 MAC（596.5 G）把成立的抄法都加上

| 抄法 | 全网省（名义） | 质量 | 是不是 2× |
|---|---:|---|---|
| r0 元素跳零（nnz≈4%） | ~20.5% | 无损 | 这一段几乎跳完，系统不够 |
| MLP bitmap G=32 | 4.9% | 无损 | Stage0 FC1 不省 |
| 注意力线性 bitmap G=32 | 5.4% | 无损 | L1.proj 可到 0.03× |
| 跳 decoder2 | 2.7% | P1 更好 | 砍错贵段 |
| Jung 关空 SSA | SSA 块 42% | 叠上 0.710 | SSA 比 linear 便宜 |
| **粗加** | **~33%** | 过门 | **≈1.5× dense MAC，不是 2×** |

不能把 GPU ms、叶子证书 34%、Zhang 5× 再乘进来。r0 的 20% 只有在 **bitmap/AAC 硬件跳零几乎免费** 时才成立；GPU gather 已测慢 4.6×。

## 新挖的两刀（多方法）

**C-STEP 逐通道邻域 AND**（不是整向量 1.1%）：r0 Jaccard **0.22**，L1.fc1 **0.35**，论文 0.83。相关有，远不够公共脉冲成倍。停 KEEP。

**BitL 8×8 权重量化 7bit：** 相对 **bit-serial 天真 7 拍**，L1.fc1 均 4.86 拍（**0.69×**）。这是相对位串行，不是相对 dense MAC。发放已是 1bit，再咬权重比特最多再挤这一层的三成位串行，全网仍是百分点。

## 跳完之后还贵什么

1. **Stage0 MLP FC1**（两层各 7 G）：几乎全非零、行几乎互异，bitmap ≈ dense，Phi 码本 AEE 变差。  
2. **r1 / stem / 其余 ~30%** 还没按行跳零测完。  
3. MLP 非零行上的连续 PSN（Codex 证书整链 −4.3%）。

## 判定（停抄稀疏家族当 2× 标题）

在本网、本 10 帧、本分母上：**把顶会里「跳零 / 浅层 / 空注意力」能抄的都抄上，系统大约 1.5× 名义 MAC，质量不炸。** 成倍要换学生（更浅 encoder/更少 T）或换执行契约（证书让别的消费者停），不是再抄一张 skip 卡。

同刊 ESTU 已占 SSA skip；Jung 只当 A。

## 若还做，只留两条

1. **valid825** 上复现 P1+Jung（10 帧可能偏于 zurich）。  
2. **Stage0 FC1 的同资源 RTL**（已证明 GPU/bitmap 都榨不出），或 Codex 证书接到这条 FC1 的 PSN，而不是再量 r0 静默。

收据：`copy_stack/results/summary.json`、`copy_stack/ledger_preview.json`、`copy_mlp/results/bitmap_all.json`、`copy_cstep/results/channel_and.json`、`copy_bitl/results/bitl.json`。
