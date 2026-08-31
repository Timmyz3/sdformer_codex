# M224 H67 FC1 parent-delta 与 bank-service 强基线筛选

M224 对 M51-s10 的 10 个 exact-binary FC1、100 份原始 bitpack 做逐文件 SHA 校验和 packed-bit 精确复算。Stage 0/1/2 输入均为 binary；stage 3 的两个 FC1 由冻结 `operator_runtime.csv` 证明不是 exact binary，继续走 conventional path。

## 结论

当前 FC1 的同向量 multi-source K-bank 线为 NO-GO。K-bank 虽减少 source group 数，但每个 source 同拍可覆盖的 destination 数下降，FC1 的 4C expanded output 迫使更多 destination slice，最终所有 K2/K4/K8 均慢于同 lane-family 的强 K1。

真正有收益的是 spatial parent-delta：112,213,979 个 raw source event 降到 87,209,538，source-work 为 `1.286717×`；把 current/candidate scan、2-bit choice、parent-output seed、service 和 96-lane commit 全部收费后：

| 固定 datapath family | 最好合法点 | 完整 serial ratio vs raw K1 | 10样本范围 |
|---|---|---:|---:|
| 96 product lanes | spatial K1/D96 | 1.190252× | 1.162498–1.217234× |
| 128 product lanes | spatial K1/D128 | 1.176055× | 1.149697–1.201765× |

Temporal parent-delta 的 raw source-work 仅 `1.041933×`，完整 serial ratio 为 `1.035006×`（96 lanes）/`1.031610×`（128 lanes）。

## 为什么 K-bank 没有变成加速

以 96-lane family 为例，raw K1/D96 为 1.000×；K2/D48、K4/D24、K8/D12 分别只有 0.900952×、0.780774×、0.640354×。128-lane family 同样如此。FC1 不是 M218 弱 source-owned K1 的可复制场景，不能外推 4.952×。

M224 因此不进入 multi-source RTL。下一候选应是跨相邻 context 广播同一 96-lane weight vector：把 spatial parent-delta 的 signed residual 与 K-context destination accumulation 结合，并用同面积/同带宽 K1 重新定价。M63 的历史 K4 opportunity 只能作为假设，不能直接继承倍率。

本结果是 trace premodel，不是 RTL throughput、完整 FC1/FFN、系统加速、SRAM/PPA 或能耗。论文正文和 `docs/359` 均未修改。
