# M506 synthesis-portable onehot independent preflight + result hammer

## Verdict

`PASS_M506_RESULT_HAMMER__UNLOCK_M496_R2_DC`，99/100。

两套冻结 M492/M497 VCS 已顺序重跑并由本审查独立复验，M506 portability gate
关闭，允许进入 exact-SHA M496 r2 DC。M506 本身仍不形成性能、面积、功耗、
系统或论文 headline 证据。

## 独立检查结果

1. M342 旧源码可从当前源码唯一反变换并重建出合同记录的旧 SHA256
   `309759bfa6eeb303143e707bd3df269eddcd31e34e79ed662d507c363ba4d904`。
   真正的 RTL 语义改动只有：新增显式 `onehot8`，并把两个 `$onehot` 调用替换为
   `onehot8`；其余源码逐字不变。
2. 对全部 256 个 two-state 八位 mask 穷举，显式 predicate 与
   `popcount(mask)==1` 的 mismatch 为 0，且恰有八个 true mask：
   `01/02/04/08/10/20/40/80`。
3. 对含 X/Z 的 65,280 个四态 mask 做语义穷举：普通 SystemVerilog `case`
   只与八个纯 0/1 literal 精确匹配，因此所有含 X/Z mask 都落入 `default=0`。
   这满足合同“X/Z 非法”，并比 synthesis-visible `$onehot` 更 fail-closed。
4. 两个 runner 各有且只有三类变化：新增输出目录环境变量、M342 SHA 更新、
   原 suite contract pin 替换为 M506 contract pin。将三类变化反变换后，所得
   SHA 分别精确等于旧封存 runner SHA：M492 `8c2a006a...`、M497
   `5c61a386...`。因此 TB、filelist、top、seed、PASS 文本、周期行、SVA cover、
   mismatch 门和 receipt 生成逻辑均未改变。
5. 当前 runner 内 20/20 个 exact-SHA 输入均存在且匹配；旧 M492/M497 结果包的
   `SHA256SUMS` 与外层 seal 均复验通过；docs/359 仍为冻结 SHA
   `dedde7ce...`。
6. 组合结果 `results/m506_fc2_synth_portable_onehot_vcs_r1_20260827` 的两层 inner
   manifest/seal 与 outer manifest/seal 全部复验通过。outer seal 文件 SHA256 为
   `a8257c15c4267dad6807c2fb2fbcf8485444f4643db53721b5778b6425658955`。
7. M492/M497 compile.rc 与 sim.rc 均为 0；M492 编译 0 warning/0 error；M497
   恰有合同允许的一个 BTNL bind warning、0 error。两套 sim/assert 日志均无
   failed/offending/error/fatal/watchdog。
8. 两套新的 JSON receipt 分别与旧封存 receipt **byte-identical**，SHA 仍为
   `beec54d7...` 与 `65b36a63...`。所有 PASS 计数、五档周期行、零 mismatch、
   protocol attack 和 SVA cover 逐项不变；独立 cover 复核缺失数为 0。

## 唯一剩余风险

冻结 testbench 没有专门向 `onehot8` 注入 X/Z，因此 X/Z 结论属于对 plain
`case` 的静态语义证明，而不是这两套 directed VCS 的动态覆盖。这不阻塞 M496
r2 DC，因为合同已经把 X/Z 定义为非法，且当前实现对它们确定地 fail closed。

## M496 解锁边界

只解锁已另行冻结的 M496 r2 exact-SHA DC；不得从 M506 推导速度、面积、功耗或
系统收益。M496 仍须遵守自身三轴同资源门、no-overwrite、工具互斥和独立结果
hammer，失败必须 fail closed。
