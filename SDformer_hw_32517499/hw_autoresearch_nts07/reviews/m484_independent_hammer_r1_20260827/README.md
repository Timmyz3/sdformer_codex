# M484 独立打铁评审 r1

结论：**68/100，NO-GO（性能准入、RTL 与论文性能主张）**。M484 的封存数据、身份和算术自洽，FC1 的 C-last 在线连续性也可保留为调度事实；但在相同八条 signed source lane、相同八输入归约树、相同 Acc32 行驻留状态和相同端口边界下，强 K8-resident baseline 已经能做相同的八源合并。因此 M484 对三类负载的增量周期收益都是 `1.0000x`，流量还因 bundle header/padding 略增。

本评审只读生产输出，没有导入或执行 M484 生产脚本，也没有重放 160 个 bitpack。复算使用最终双重 SHA 封存的 CSV/JSON、M51 manifest 和 M22 事务；审计方法按数据粒度、唯一键、分母公平性和 fail-closed 边界逐项检查。

## 独立复算结果

| N=8 类别 | occupancy | wait p50/p95/p99 | K1 resident cycles | K8 resident cycles | M484 cycles | K1→K8 资源扩展 | M484 vs 同资源 K8 | 流量变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conv | 93.3801% | 7/7/7 | 1,822,763,079 | 301,270,563 | 301,270,563 | 6.0503x | **1.0000x** | **-0.3804%** |
| Conv→ATLIF | 93.4832% | 7/7/7 | 1,472,084,729 | 268,039,306 | 268,039,306 | 5.4920x | **1.0000x** | **-0.3313%** |
| FC1 | 85.3135% | 7/7/7 | 117,780,501 | 25,862,339 | 25,862,339 | 4.5541x | **1.0000x** | **-0.0379%** |

负百分比表示 M484 比强 K8-resident baseline 流量更多。`6.0503x/5.4920x/4.5541x` 只能称为 K1 到 K8 的资源扩展参考，不能称为 M484 机制加速。

复算恒等式均通过：

- `bundles = full_bundles + remainder_bundles`；
- `padding = 8 × bundles - selected_sources`；
- `occupancy = selected_sources / (8 × bundles)`；
- Conv/FC1：`K1 = sources + 2 × nonempty_rows`，`K8 = M484 = bundles + 2 × nonempty_rows`；
- Conv→ATLIF 在上述两式中都加入同一份 `rows` 直接 handoff 成本；
- baseline traffic 为驻留状态 R/W、weight、共享 event metadata 与 K8 row header 之和；candidate 用 bundle header 替代 row header并加入 padding；
- 三类 full-bundle 比例分别约 87.46%、87.64%、71.61%。非 full bundle 的 wait 不超过 6，full bundle 的 wait 为 7，所以报告的 p50/p95/p99 全为 7 与聚合计数自洽。该 wait 单位是 accepted event，不是墙钟周期。

## 身份与完整性

- 最终生产 `SHA256SUMS` SHA256 为 `6f2ac16c00b4013984b3ca039278a6b935fa049b97beb18879593369e5c123d6`，其 seal 和 6 个被封文件全部复核通过。
- `docs/359_DATE终局冻结_20260813.md` 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- M51 目标集合为 170 条：70 Conv + 100 FC1。10 条缺失 payload 全部是 `sttmultires_unet.preds.3.conv.0`，本地 ledger 恰为其余 160 个唯一 `(sample_id,name)`：60 Conv + 100 FC1。
- 40 个 Conv→ATLIF 记录确为 60 个 Conv 的重叠子集，并非额外 40 个算子；dual-line reconciliation 160/160 为零差异。
- M22 独立宽度检查：160 个 operator Acc32 write 和 40 对相邻 ATLIF 的 read/write（80 项）全部匹配 `rows × out_channels × 4 byte`，0 mismatch。

## P0 强基线攻击

FC1 的 C-last 输入在同一输出行内连续。只要 K8 baseline 与 M484 都拥有八条 signed lane 和同一八输入 adder tree，baseline 就可在 Acc32 行 context 中持续累加，到行边界才提交；这与 M484 的周期式完全相同。若 baseline 只有 K1，则其 lane/tree/吞吐资源与 M484 不同，不能作为“同资源新机制”分母。

M218 RTL 中已有跨 group 保留并更新 `ctx_q` 的实现先例。M218/M219 还给出一个仅供面积敏感性参考的 FC2 Acc24 结果：K8 为 88,851.042296 µm²、cropped K1 为 76,857.858437 µm²，K8 高 15.604369%，两者都保留 18,432-bit context。但该结果是 TSMC28 3 ns、ideal-clock、ZeroWireload、0 macro 的独立 service-island，且对象是 FC2 Acc24，不是 M484 Conv/FC1 Acc32；不能冒充 M484 PPA、SRAM 端口代价或总加速器面积。

因此本评审通过的是“数据屏与排序事实”，不是新硬件优势：

- `GO`：封存身份、160 条粒度、算术、M22 Acc32 宽度；
- `GO_AS_ORDERING_FACT_ONLY`：FC1 在线原序 C-last 可形成 N=8 pack；
- `RECORD_ONLY_NOT_ONLINE`：Conv destination-major 离线 oracle；
- `NO-GO`：M484 相对同资源 K8 的性能优势、新 RTL、系统/论文性能准入。

## 在线与论文边界

FC1 online exact 行与 offline N=8/slot1 数值一致，但相对强 K8 仍是 `1.0x`，流量为 `-0.0379%`。Conv 与 Conv→ATLIF 的 online original-order 只是一源一包 safe lower bound，分别为 `1.0x/-12.02%` 和 `1.0x/-10.44%`；离线 zero-stall 仅来自 destination-major 每次只有一个 live row，不能证明 reorder frontend、容量或 backpressure。

当前还只有一个 Zurich sequence 的 10 个窗口，没有 M484 RTL/VCS/DC/PTPX、SRAM macro、端到端 cycles、FPS 或 energy。建议关闭 M484 独立 RTL 分支，把高 occupancy 当作共享 K8 source engine 的 workload justification；若要重新开门，必须提出并测到强 K8 baseline 不具备的结构收益，例如减少 lane/tree/port/metadata 资源或在同面积/同带宽下提高吞吐。

## 评分

| 项目 | 得分 |
|---|---:|
| 身份、粒度与封存 | 19/20 |
| 算术和独立事务核验 | 20/20 |
| 强基线与口径公平性 | 20/20 |
| 在线可实现性证据 | 8/15 |
| M484 RTL/PPA/宏证据 | 0/10 |
| 增量创新与性能优势 | 1/15 |
| **总分** | **68/100** |
