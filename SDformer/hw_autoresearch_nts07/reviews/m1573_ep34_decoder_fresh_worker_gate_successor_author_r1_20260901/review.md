# M1573｜ep34 decoder fresh-worker gate successor 作者收据

日期：2026-09-01（Asia/Shanghai）  
裁决：`PASS_SOURCE_AUTHORING__INDEPENDENT_HAMMER_REQUIRED__NO_ACTUAL_EXECUTION`

M1573 保持 M1556 的三条非 product 配置、96 lanes、Acc24、3 ns、192 B/cycle、240 KiB 分区、请求顺序、依赖、cache、端口和 commit 语义不变。修改仅位于 host 运行边界：未来每个配置必须由 fresh-exec worker 执行，并在原有 gate 上同时记录 `/proc/self/status` 的当前 `VmRSS` 和 Linux `ru_maxrss` 高水位；两项都继续受严格 8 GiB 上限约束。

作者测试把 dual-RSS gate 前后的 frozen M1556 合成结果按 configuration、resource digest、total cycles、request count、kind counts、byte counts、transaction-address SHA 与 commit SHA 做 exact miter，全部一致。测试和 preflight 没有打开 actual pilot、没有重跑 M1570、没有运行 EDA/GPU/RTL，也没有产生可引用周期。

M1570 的唯一 attempt 仍是 consumed failure。M1573 不能自行执行；下一步必须先独立 hammer 源码和攻击 fresh-worker 边界，然后另写、另审、另命名 exactly-once runner。M1572 的 compact numeric engine 仍是后备优化，本收据没有声称已经实现 compact engine。

## 已验证

- CPython compile：通过；
- source tests：`PASS M1573 tests=9 actual_execution=false`；
- author CLI preflight：通过；
- synthetic hardware projection：exact；
- forbidden product branch：保持拒绝；
- M1570 retry：false；
- 论文性能、Table A、系统倍速、能量、PPA：全部 false。

## 固定 SHA

- source：`f26203424c4034230ee696ecf3b6d95685ed21647f41eb0c38b6961f0c83d02c`
- tests：`ad8f0f60f26dcb6ac3cf98d73193667fb290399c9440d3d0b76936c0e2211d6c`
- contract：`6ab5397d50de8a3bc036856af87a40be78ce017829549c8eee7459f8ae152c41`

