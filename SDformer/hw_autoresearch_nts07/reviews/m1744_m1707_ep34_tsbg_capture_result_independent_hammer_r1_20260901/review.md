# M1744｜M1707 ep34 TSBG capture 独立全量结果打铁

结论：**PASS capture payload，仅授权下一步 M1727 只读分析；不授权性能、硬件量化、模型 bit-exact、S2/TSBG 准入或论文 headline。** 评分 **98/100**，P0=0、P1=0、P2=2。

## 全量验证

- canonical 目录恰有 10 个 regular non-symlink 文件；`SHA256SUMS` 的 8 个成员和外层 seal 全部复算一致。
- checkpoint、config、profile、M1707 source/contract、M1709 release 均从远端当前实体重新 SHA；40 个 sample、32 层（FC1/FC2/PATCH=12/12/8）身份与顺序闭合。
- 对 `fc_frames.bin` 的 **11,040 个 frame / 44,640,000 token** 全量解析，不抽样：逐帧校验 header、sample/layer/frame/token 顺序、维度和 extent；每个 zlib stream 均校验 EOF、unused/unconsumed tail、raw length 和 CRC32；support/sign/nonunit bitset、uint16 nnz 与 row-major signed int8 code 全部一致。
- PATCH 压缩流的 320 行也全量校验 zlib extent、sample/layer/output-tile 顺序、histogram 与 debt schema。
- 最终 hammer 运行 100.435 秒；capture tree 写入 0，GPU/EDA/M1727 analyzer 运行均为 0，未发生 capture retry。

## 关键统计与边界

- FC nonzero code：872,855,874；zero token：9,932,663；FC raw payload：7,528,535,874 B；压缩 payload：582,905,536 B。
- 所有非零 FC diagnostic code 都是 `-1/+1`，`nonunit_codes=0`。这不妨碍 TSBG 的 weight-fetch reuse 机会分析，因为该复用不依赖 source value；但绝不能把这份 payload 说成非单位 typed-value witness 或硬件量化模型等价证据。
- 当前远端已不保留 consumed M1306 runtime tar；其 SHA 由 sealed M1709 release 与 M1707 child receipt 双向交叉绑定，本评审没有冒充重新哈希了该实体。

## 处置

M1727 可以在**拉取当前已审 source/release 后**对这棵 capture 做一次只读 B4/B8 TSBG 与受限 S2 分析。任何 cycle、traffic、energy、speedup、AEE、RTL/EDA 与 paper claim 仍必须由 M1727 自己的结果和下一轮独立结果评审决定。
