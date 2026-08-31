# M361r4：k32/k64 宽分区 exact-pattern 机会筛选

使用 M73 全 18 个 DSEC train sequence 拟合嵌套 q16/32/64/128 catalog，
再在不相交的 M248 S10 runtime trace 上做 exact signed residual replay。
NumPy r4 只替换前三版不可接受的标量执行时间；catalog 目标、tie-break、
exact fallback 和数据身份不变。

| partition | q | train exact-work | disjoint S10 exact-work | full PWP capacity |
|---:|---:|---:|---:|---:|
| 32 | 16 | 1.371811x | 1.374535x | 17.25 MB |
| 32 | 32 | 1.469137x | 1.476186x | 34.50 MB |
| 32 | 64 | 1.579207x | 1.592538x | 69.01 MB |
| 32 | 128 | 1.696467x | 1.714495x | 138.02 MB |
| 64 | 16 | 1.225253x | 1.226243x | 9.29 MB |
| 64 | 32 | 1.284563x | 1.287258x | 18.58 MB |
| 64 | 64 | 1.347339x | 1.350725x | 37.16 MB |
| 64 | 128 | 1.406052x | 1.406600x | 74.32 MB |

宽分区在最高 q 下仍低于既有 k16/q128 的 disjoint exact-work
`2.04394x`，且 k32 需要 13-bit PWP、k64 需要 14-bit PWP。故宽分区不进入
构造性 cycle/RTL；主 exact 路径继续 k16，并把工程资源放到 q32/O4 的有限
队列执行与吞吐/面积上。

这些都是 vector work，不是 cycle、energy、系统倍速或 headline。
