# M1127D｜Decoder D3 与 LB-FUSE 第一性原理只读审计

结论：**STOP 当前三行/有限行 LB-FUSE，不开 RTL，不重复 M722。** D3 确实是热点，但现有证据说明普通 line buffer 解决的是容量/lifetime，不会自动减少强 A1 的真实 backing traffic。

## 1. M1111Dr2 活前缀：D3 是单样本热点

只读观察 PID 4122290 的前四条完整记录（`interlaken_01_a`, sample 0），没有发信号、启动 runner 或消费任何 namespace。第五条仅是 sample 1 的 D0 append-progress 检查，不进入比值。

| layer | cycles | cycle share | traffic bytes | traffic share |
|---|---:|---:|---:|---:|
| D0 | 17,863,747 | 14.1805% | 4,417,016,768 | 14.2387% |
| D1 | 18,592,651 | 14.7591% | 4,602,661,664 | 14.8371% |
| D2 | 20,355,467 | 16.1585% | 5,017,975,200 | 16.1760% |
| D3 | 69,162,219 | **54.9019%** | 16,983,549,632 | **54.7482%** |

D3 的 psum read/write 各为 4,979,194,272 B，各占 D3 traffic 的 29.3177%；合计占 D3 的 **58.6355%**，也占这四层总 traffic 的 **32.1019%**。D3 每个 dense output commit 对应 22.5115 次 psum update。这证明热点存在，不证明 line buffer 能消掉它，更不能从单样本升级为多序列、系统或速度结论。

## 2. 三行容量与精度

Acc24 三行容量按 `3*W*Cout*3 B` 计算：D0/D1/D2 都是 138,240 B；D3 是 276,480 B。加 13,824 B weight 和 8,192 B control 后，D3 为 298,496 B，超过 240 KiB **52,736 B**。

- D3 48-channel 两遍 Acc24 虽只需 153,344 B，但现有公平模型为 17.468B cycle，对 A1 只有 0.73536×，即慢 35.99%。
- D3 全 96-channel Acc16 需 206,336 B。M722 的完整选定 S3x10 local-INT8 trace 上，最终范围 `[-6601,4804]`，所有顺序绝对前缀界 7288，确实 fit signed16；但全二值输入形式界为 98,552，超过 signed16。它不是 universal/final-checkpoint/accuracy proof。
- 强 A1 不需要溢出：Acc24 空间 stripe `[0,256]`、`[256,320]`，只重叠一个 source column，总容量 243,200 B，余 2,560 B，off-chip psum spill 为 0。

因此用“未分片、会 spill 的 A1”作分母是弱基线错误。

## 3. Line buffer 是否减少真实 backing traffic

M1111Dr2 的 psum 是 6-bank、48 B/bank、288 B/vector 的 1RW SRAM；每次 update 都有 read→compute→write 依赖，dense commit 还走公共 external 1RW。**Line buffer 本身就是 psum backing**：只改地址为三行，不会消除 RMW；必须先在更小 frontier/register 中合并贡献才会少写。

封存的 M722r2/M732 已经做过候选有利的完整 fast-kill：

- D3：LB on-chip RMW 比 A1 少 10.4818%，但 off-chip spill 两边都是 0；直接 group 增至 1.3428×，LB 仍慢 2.0075%，commit 8.84736 GB 不变。
- D0+D2+D3：LB group 为 A1 的 1.4134×，on-chip RMW 为 1.1519×，整体慢 8.2738%。

根因不是容量，而是 source-order direct issue 破坏 A1-OSG 跨 source 的 destination packing。恢复足够 destination-keyed state 会回到 A1-OSG/PIDP 的已审计交换。

## 4. 与 prior、PIDP、M522/M523 的边界

三行 buffer、stride-2 polyphase 和 deconvolution reordering 都有经典 prior（GANAX、Chang & Chang 等）。M522 已实现精确地址 mapper，M523 已验证 4/6/9-tap descriptor bundling；二者都不提供 psum residency 性能。

H67 的合法对象差只是 binary ATLIF descriptor、奇偶相位 contributor、signed local-INT8/Acc24、K8/96-wide 和 240-KiB 约束。PIDP 是 destination-major：D3 的 13 个 weight tile fit 16-entry cache，所以局部机会真实；但 D0-D2 weight thrash 杀死 full PIDP。LB-FUSE 为保 weight reuse 留在 source-major，因此保留 row/frontier psum 税。没有同资源收益时，这个对象差不足以成为新贡献。

## 5. 一天 CPU 门

M722r2 已经失败，不得原样重跑。只有结构上不同的 `frontier-combine` 才可重新 author CPU contract：必须在保住 A1-OSG group packing 的同时减少串行 1RW psum 事务。

主坐标固定为 96 lane、K8、240 KiB、Acc24、相同 dense commit、强 A1 Acc24 spatial stripe。Acc16/48ch 只能单列 sensitivity。GO 条件：D3 同资源 `A1/candidate >=1.20x`；或 cycle 不比 A1 差超过 5%，且 on-chip RMW 至少降低 30%，同时 off-chip spill、descriptor+weight traffic、commit 均不增加，group 数不超过 A1 的 1.05×，exact mismatch 为 0。即使过门，也只授权独立 CPU hammer，不授权 RTL/EDA/论文速度。

当前状态：**STOP。**
