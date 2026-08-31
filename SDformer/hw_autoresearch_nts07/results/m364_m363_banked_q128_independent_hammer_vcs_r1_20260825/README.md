# M364：M363 balanced-banked q128 独立 fail-closed 打铁

结论：**95/100，P0/P1/P2 = 0/0/3；GO exploratory logic-only DC。** 该 GO 只允许对 M363 做 TSMC28、3.000 ns、ideal-clock/zero-wire、pre-macro logic-only 探索；完整 PWP Conv、系统性能、paper PPA 和 DATE headline 仍是 NO-GO。

## Fresh exact-SHA VCS

M364 没有读取 M363 作者 receipt 作为 PASS 输入，也没有复用作者 `simv`。冻结 M348、M356、M363、作者 SVA、独立 M364 SVA/TB、filelist、M363/M364 contract 和 `docs/359` 后，在新目录使用 Synopsys VCS V-2023.12-SP1 重新编译并仿真。compile/simulation rc 均为 0，runtime assertion failure 为 0，五个独立 coverpoint 全部非零；RUN_MANIFEST 与二级 seal 均校验通过。

## M363 与 M356 q128 语义

独立 reference 对每个输入重新遍历 128 个 pattern，按 Hamming distance、最低 center ID 选择 winner，再重算 population、`1+distance < population`、plus/minus 和 fallback。M363 与单独实例化的 M356 各自接收并退休同一批 3000 个输入，分别承受背压：两边对 reference 都是 0 mismatch，M363 对 M356 逐事务逐字段也是 0 mismatch。定向重复 pattern 的 winner 为最低 ID 2。

源码审计与动态证据一致：M356/M348 是 128-stage global-advance recurrence，M363 改成四级 balanced 4-way elastic tree；catalog、距离、最低 ID、PWP threshold、signed residual 和 exact fallback 没变。这里核对的是冻结 q128 functional mapping，不是任意 catalog 的形式等价证明。

## Elastic 与 fail-closed 攻击

- M363：3000 accepted/retired，565 个 output-stall cycle、527 个 input bubble，最长连续 accept/retire 均为 321；minimum latency=4，背压 maximum=67。
- 四个 elastic slot 同时 full 的 cover 为 2404；16-cycle long-stall cover 为 57；bubble-refill cover 为 453。stall 中 payload/order 不变。
- 配置请求在流水非空时连续阻塞 27 cycle；四个旧结果按序 drain 后，pipeline 真空才发生唯一一次 group0 handshake。
- active catalog 上首拍 bad reload 进入 sticky quarantine，bad beat 没有修改任何 catalog entry。
- error 后连续 64 个 cfg/input 攻击周期，external cfg handshake=0、input handshake=0；全部 128 个 pattern、`cfg_next_group_q`、`cfg_active` 和 pipeline valid 均无 mutation。
- 四个 slot 全满时 reset，4 个旧 token 全部丢弃，catalog/control 清零，之后 10 cycle 无 stale output；重新配置后的 sentinel 以 4-cycle handshake latency 正确返回。

## 三个 P2

1. 3000 个输入、一个冻结 catalog 的动态对拍不是对全部 65,536 输入与任意 catalog 的 Formality/形式等价。
2. reset 与 reload fault placement 是强定向覆盖，但没有枚举随机 pipeline 边界，也没有覆盖“先合法写一部分 reload、再 bad beat”的所有位置。
3. catalog/next-group/active 的强 freeze 检查位于 M364 层次化 verification harness；M363 shipped SVA 尚未 bind 同等内部状态断言。

## 口径

M363 的 4-cycle 是无背压 latency，不是 32×、4×或任何系统 speedup。II1 也不是 Fmax、throughput/mm²、完整执行器性能或 DATE 头条。下一步 DC 必须与 M362/M356 在相同 library/constraint 下报告总/组合/时序面积、flop、setup/hold 和 throughput-per-area；不能只比较 latency。
