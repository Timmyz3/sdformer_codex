# M328：token-dynamic cumulative-B CPU fast-kill

最终决定：`NO_GO_TOKEN_DYNAMIC_CUMULATIVE_B_AFTER_CPU_FAST_KILL`。

候选在算法上确实不是静态 beta mask：存在大量同一 `module/source/destination-group` 跨 token 时而 drop、时而 keep 的 dynamic witness；但选择开销吞掉了全部理想 K8 收益，且朴素动态 metadata 是一位静态 mask 的 24 倍。因此停止 GPU modified-forward、RTL 和 Synopsys 推进。

## 冻结范围

- FC1：10 个 exact-binary 模块、100 条 M51 payload、5,520,000 tokens、112,213,979 个 active source。
- Conv：`patch_embed.residual_encoding.resblocks.1.conv1.0`，10 条 payload、452,261,964 个 im2col source contribution，最大 active fan-in 448。
- checkpoint：冻结 H67 ep35；分析器使用 SHA-pinned constrained torchless loader，并与 M287/M293 的 group4/group16 beta48/beta96 census 完全对账。
- 网格：`G={4,16}`，`B={0,16,32,64,128,256,512,1024}`。

## 正确性

- `B=0`：所有 scope 完全 exact，drop=0、raw integer error=0、selector overhead=0。
- 全网格 cumulative-bound violation：0。
- 每个 destination accumulator 都核对了 signed INT8 raw error：`|sum dropped Wq| <= sum beta <= B`。
- 非零预算 policy 累计发现 5,148,082 个 witness 计数；最佳组合点自身有 819,815 个 source/group witness。该累计值跨 budget 重复计数，不是唯一 pair 总数。

## 最佳组合点

最佳理想点为 group4、`B=1024`：

| 指标 | 结果 |
|---|---:|
| dropped source/group tasks | 45.4944% |
| ideal K8 speedup | 1.7643x |
| 96-candidate/cycle full-domain scan 后 | 0.9167x |
| 8-active/cycle active-list lookup 后 | 0.7773x |
| 16-byte/cycle metadata stream 后 | 0.1000x |
| 最大 raw signed INT8 error | 992 |
| 最大 cumulative bound | 1024 |
| bound violations | 0 |

这个点的理想 K8 baseline/candidate 分别为 6,681,676,272 和 3,787,146,191 cycles；加入极乐观的 96-wide scan 后是 7,289,226,191 cycles，已经比 baseline 慢。

## Metadata

在当前 FC1 + 单 Conv group4 scope：

- source/group pairs：997,632；
- 一位静态 mask：124,704 bytes；
- 朴素 `uint8 beta + uint16 ordered source ID`：2,992,896 bytes；
- 比例：24.0x。

metadata stream 模型还没有计 sorter/bucket queue、bank conflict、descriptor FIFO 或 commit stall，因此 `0.1000x` 已是乐观结果。

## 分 scope 攻击

- FC1-only group4/B1024 的理想 K8 为 `1.9194x`，96-wide scan 后仍有 `1.1053x`；但 active-list 仅 `0.8171x`、metadata stream 仅 `0.1346x`，24x footprint 门也失败。
- 所选高 fan-in Conv group4/B1024 理想为 `1.4623x`，full scan/active-list/metadata 分别只有 `0.6382x/0.6913x/0.0603x`。
- 较小预算不会补救：组合 group4/B256 理想仅 `1.1812x`，full scan 后 `0.7295x`。

因此不能通过“只报理想 K8”保留该候选，也不能把 FC1-only 的 scan 敏感性包装为可执行硬件收益。

## 冻结晋级门结果

失败项：

1. `FULL_DOMAIN_SCAN96_NOT_FASTER`；
2. `METADATA_STREAM16B_NOT_FASTER`；
3. `NAIVE_DYNAMIC_METADATA_RATIO_EXCEEDS_GATE`；
4. `NO_POLICY_PASSES_ALL_FROZEN_GATES`。

active-list 模型已经忽略 runtime sorter/bucketing，仍然没有任何净加速，因此不存在值得继续实现的执行路径。

## Claim boundary

M328 只承认冻结 payload 上的 CPU fast-kill、dynamic witness、整数误差界和 ideal K8 proxy。它不承认 accuracy、modified forward、executable hardware cycles、RTL、Synopsys、energy、system speedup 或 headline。

本次未跑 GPU、未写 RTL，`docs/359` SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
