# M474 fused parent dual-update micro-RTL 独立预审

日期：2026-08-26  
结论：**GO，93/100；仅准许冻结 exact-SHA VCS 合同，不准许性能/PPA/系统声明。**

## 冻结候选身份

- RTL：`30fdf778e5baea959c793c7b2f9d9e332364b84717f9ffd2f8ad74d85280d57c`
- SVA：`ee039ba832f0a3b62035543e64253ffa932a18690e86161e39102ead9995695b`
- TB：`b9e2edbbcbc16b557ed7fab52066c6834931df0366f0ba6734ec308b4b3bd1da`
- filelist：`5443d7b5281a34266f9003034b1238b6e65c99015a98a643326a3cd48bf2d6a1`
- `docs/359`：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

本评审明确废弃开发过程中更早的 RTL/TB 哈希；只评以上最终候选。

## 独立 Synopsys VCS 回归

使用 VCS `V-2023.12-SP1_Full64`，独立临时 build，不修改 producer：

```text
vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -f dc_handoff/filelists/date_m474_fused_parent_dual_update_vcs.f \
  -top tb_m474_fused_parent_dual_update_pipeline
```

编译和仿真均返回 0。结果：

```text
PASS M474 directed issues=6 rows=5 forward=2 reads=2 stalls=5
b2b=2 oneahead=1 exact=2 partialbeats=2 overflow_attacks=1
simulation time = 61.5 ns
```

九个 cover 均非空：forward 15、macro-read 14、exact-parent 2、
partial-parent 2、output-stall 3、overflow-atomic 1、stall-counter 9、
back-to-back completion 2、one-ahead macro read 1。临时 compile/run log SHA 分别为
`66a2e1c839d8ab2651a7cda2dbca8154d31d6482f23b29a3ab4edb6fc8062bb3`、
`e40652a68b8465d890fe72e2df35eeaa2451532e35560fb53912bafb7b7b62a4`；
它们只是本次 peer development evidence，不是正式 exact-SHA 收据。

## 核心判断

1. **真实 one-ahead 成立。** `scratch_read_enable` 在请求边沿锁存地址；下一周期
   `read_pending_q` 与 registered macro Q 直接构成 `parent_ready/source`，dependent
   issue 可在紧邻下一边沿接受，不再先 capture 再多等一拍。SVA cover 非空为 1，
   TB 的逐 lane 数值检查同时排除了“时序命中但拿错数据”。
2. **pending 消费没有重复入 buffer。** pending Q 被 final parent issue 消费时，
   `read_pending_q && !consume_parent_w` 为假；若是多 beat 行的 first issue，Q 会按需
   落入单项 buffer 供后续 final beat 使用，两个分支语义正确。
3. **同地址 RAW forwarding 成立。** final scratch write 与同地址 next-parent prefetch
   同拍时 suppress macro read，并把 `scratch_write_data` 放入 parent buffer；TB 有两次
   forward，包含连续 completion。final issue 同拍产生 scratch write、signed19 psum
   write 与 `row_complete`，没有额外 completion/read 拍。
4. **consume+prefetch 控制无组合冲突。** `consume_parent_w` 在同拍释放 prefetch
   空间；forward 分支的后写优先级正确。TB 已覆盖 row1 final consume
   buffered parent 同拍发出非同地址 macro read，row2 first 在下一拍直接消费
   registered Q，因此此前缺少的 non-forward consume+prefetch 分支已非空。
5. **overflow 已原子 fail-close。** final signed12/signed19 overflow 同时阻断
   `issue_ready`、`psum_write_valid`、scratch write 和 completion，并在该边沿锁住 fault；
   不再存在“先写截断值、下一拍才报错”。metadata/parent 协议攻击也不会写状态，fault
   sticky 已断言。
6. **数值宽度在 M473 域内可成立。** 每行最多 16 个唯一 source，实际 beat 应是
   sign-extended INT8 weight vector，因此任一前缀和位于 `[-2048,2032]`，13-bit row
   accumulator 足够，signed12 final scratch 的理论边界也闭合；signed19 psum 依赖冻结的
   checkpoint/order bound。模块端口本身允许任意 signed12 beat，故该 INT8/最多16-source
   条件必须成为正式合同或 assumption，不能把接口泛化成任意 signed12 stream。
7. **ready/valid 边界是 fail-close scheduler interface。** parent 尚未 ready 时拉高
   `issue_valid` 会锁 fault，而不是弹性等待；这是可接受的严格 scheduler 合同，但必须在
   上层集成时保持，不能描述成通用 decoupled input。

## 正式收据前的剩余 P1

producer 已在最终候选中补齐 overflow atomic directed attack/SVA、ID 相关
one-ahead assertion/cover，以及 consume+nonmatching-prefetch 的正常 macro-read 路径。
剩余唯一必须冻结的数值边界是：每 beat 为 sign-extended INT8，每行最多
16 个唯一 source，并引用 signed19 prefix-order bound；否则当前 accumulator 不能对任意
12-bit 输入作通用保证。该条应写入 exact-SHA contract，但不阻断当前 M473
intended-domain micro-functional GO。

## 声明边界

本结果只支持 M473 fused 假设在该 directed micro-RTL 中功能可实现。它不证明完整
descriptor/topological scheduler、真实 SRAM macro、3 ns timing、面积、功耗、能量、M473
全 population RTL 周期、全网络或系统加速。macro-Q 到 96-lane row/psum dual adder 是长组合
路径，必须由后续 DC/STA 与目标 1R1W macro timing 决定；当前 `performance_admitted=false`、
`ppa=false`。
