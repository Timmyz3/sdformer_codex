# M1540｜ep34 剪枝、稀疏与跳过候选总表

日期：2026-08-31  
状态：**PASS_CANDIDATE_REGISTRY；只授权三条 fast-kill，不授权新 RTL 或新性能数字。**

## 1. 总结

本轮把无损、有损和训练侧候选合并后，不再按“idea 数量”推进。筛选尺子只有一个：判定必须发生在被省资源之前，并至少真正关掉 `weight fetch / compute / psum update` 中的一项；只减少 FLOP、event 或非物理 operation count 的点不能晋级。

用户指定暂不加入的 M501 `27.5167%` source-event reduction 已退出实现队列。该数不是时间，且 ExSpike 是直接 prior。

当前只值得立刻做三条 fast-kill：

1. **S1 ABCG**：打 ep34 新出现的非二值 patch ingress / analog residual，在 source 发出前按贡献预算门控；
2. **S2 CCBS**：用一个紧凑 block metadata 在 fetch 前整块拒绝 weight、compute 和 psum update；
3. **TSBG**：无损地把相邻 token 的同一 weight row 读一次、广播给多个 typed signed destination context。

三条共享一次轻量增量 capture。S1/S2 最多选一条有损机制进入正文；TSBG 若只有 weight bytes / energy 收益，作为 C2 memory specialization，不新增第四贡献。

## 2. 最终候选分层

| 档位 | 候选 | 类型 | 当前数据 | 真正可能省的物理项 | 当前决策 |
|---|---|---|---|---|---|
| T0 | S1 ABCG | 有损，零预算 exact | raw ingress activity `26.506%`、sampled non-binary；analog residual 存在 | weight fetch + compute + psum update | 首测；先防该层 Amdahl 太小 |
| T0 | S2 CCBS | 有损，零预算 exact | retained C1/decoder 可先做 local bound；FC/patch 缺 group payload | entire weight block + compute + psum update | 次测；必须证明不是旧 G11 换名 |
| T0 | TSBG | 无损 | FC1/FC2 activity aggregate 已有，真实 bundle overlap 缺失 | weight-row fetch，可能只省能量 | 同一次 capture 后 B2/B4/B8 fast-kill |
| T1 | ACES | 无损 | FC2 activity `3.154%`，但没有 per-group payload | descriptor / SRAM / NoC bytes | 只作 C2 transport/energy |
| T1 | LBWC | 无损 | ep34 FP32 有；正式 INT8 无 | weight-store bytes；全零 block 才少算 | 等 M1526 Q1--Q4 |
| T1 | S3 ARPE | 有损 | 同样被 INT8 阻塞 | refinement-plane fetch/compute | 等量化桥，低优先级 |
| T2 | S4 PSTP | 新 checkpoint | 45 张 10x10 matrix 没有 exact zero，phase 质量近均匀 | aligned phase fetch/multiply/add | 训练 screen only；静态先验不乐观 |
| T2 | S5 SD-N:M | 新 checkpoint | r1 静态审计 + 独立打铁 | compact weight + pruned destination update | 仅 baseline / ablation；先做 row-local r2 |

## 3. N:M 的真实结论

M1537 的 per-tensor contiguous FP32 storage-order oracle top-N 数学可复现：

- `4:8` 删掉 `50%` weight count 时，删除 `23.00%--25.00%` 的 L1 权重质量；
- `8:16` 同样删 `50%` count，删除 `21.23%--23.01%` 的 L1 权重质量。

它只证明“天然无损 N:M 不存在”。M1538 还发现两个必须修复的布局问题：patch 类混入 `600/466872` 个 ATLIF temporal coefficient；patch 的 flat group 有少量跨 reduction/fetch row。下一版必须分离 patch Conv 与 ATLIF，并按真实 hardware row 分组。未经该 r2、重训和 paired AEE，不能启动 N:M RTL，也不能写 `2x`。

## 4. 统一门槛

### S1 / S2 有损

- `epsilon=0` 逐位返回 exact baseline，local bound violation 为 0；
- official `Delta-AEE <= 0.02`，每 sequence `<=0.03`；
- same-resource local cycles `>=1.15x`，或者周期退化不超过 `5%`、weight bytes `>=30%` 且 memory energy `>=20%`；
- metadata、debt、bank/port conflict、queue、tail 和 final commit 全收费；
- S2 另要求 block metadata 至少比旧 G11 per-source metadata 小 `8x`，并出现动态同块 keep/drop witness。

### TSBG 无损

- contributor multiset、Acc24 与 output 0 mismatch；
- baseline 有相同容量 ordinary row buffer；
- FC1+FC2 cycles `>=1.15x`，或周期退化不超过 `5%`、weight bytes `>=30%`、memory energy `>=20%`；
- bundle builder、多个 destination context、bank conflict、tail 和 completion 全收费。

### 训练侧 S4/S5

S4 只有 aligned phase mask 让 C3 local cycle `>=1.25x` 且 AEE 过门才保留。S5 必须是新 checkpoint，`N=M` 为 exact baseline，并同时过 `weight bytes>=30%` 与 local cycle `>=1.20x`；否则只进 related-work baseline。

## 5. 一次增量 capture

不要重做 M1458。只在同一 ep34 checkpoint、同一 40 sample 顺序上补：

- FC1/FC2/patch per-token/channel support bitset；
- nonzero fixed-point code、sign、non-unit 标记；
- token/window/spatial/global order；
- weight block identity、address、bank key；
- ordinary row-buffer baseline address key；
- S1 的 magnitude histogram 与 `beta*|x|` debt。

该包只提供 fast-kill 输入，不自动授权 speedup、energy 或 AEE。

## 6. 论文收口

贡献结构保持三条：C1 constrained product capture、C2 typed signed source service、C3 exact temporal/system closure。新机制只允许嵌回 C1/C2/C3：

- S1/TSBG/ACES -> C2 frontend / memory / transport；
- S2 -> C1/C2 fetch gate；
- LBWC/S3 -> C1/C2 memory / precision ablation；
- S4 -> C3 training ablation；
- S5 -> structured-pruning baseline。

最终最多一个有损模式进入主表，并与 exact C1/C2/C3 分表。当前没有新增 cycle、speedup、traffic、energy 或 AEE 数字。

## 7. Prior 边界

- Bishop：error-constrained pruning；
- Phi：pattern + residual + PAFT；
- ELSA / SNE / SpikeX：bundled event、Gustavson 与跨窗口 weight reuse；
- FireFly-T / ESDA：bitmap decoder 与 sparse token interface；
- DeltaCNN / ExSpike：temporal delta 与 adjacent event compression；
- N:M co-design：structured sparse training / accelerator。

因此可主张的是 H67 的 non-binary event-optical-flow 对象、typed signed C2 协议、fetch-before-compute 决策和 240-KiB / Acc24 约束，不主张发明 threshold pruning、N:M、AER、Gustavson 或 bit-plane execution。
