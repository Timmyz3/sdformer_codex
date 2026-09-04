# M1637｜M1595 D0/call0 三进程 diagnostic 结果独立评审

状态：**PASS diagnostic-only，99/100；`paper_result=false`、`system_speedup=false`。**

结果目录是严格平坦双封结构：五个 manifest 成员、一个内层 manifest 和一个外层 seal，没有额外、嵌套、符号链接或特殊成员。三份 child envelope 按 `DENSE_TYPED_K8`、`BIT_EQUAL_SERVICE_K1X8`、`BIT_TYPED_K8` 排列，PID 分别为 3690416、3993068、4145605，均不同于 parent 3690376；三个 process ticket 也互不相同。每个 child 的 canonical result SHA 与根 `results[]` 完全一致。精确源代码控制流中 attempt 写入先于配置循环，循环内只有一个 launcher call site，child 内只有一个 M1583 worker call site；attempt marker 与最终 receipt 均记录 `automatic_retry=false`。

共同身份成立：资源 SHA、payload FD SHA、commit sequence SHA 在三配置一致；每个配置均有 48,000 个 commit、18,432,000 commit bytes、12,000 destinations、10 timesteps，request count 非零并等于八类 kind count 之和，RSS gate 调用非零且 current/HWM 严格低于 8 GiB。checkpoint SHA 不是 child row 字段，而是由精确 M1583→M1573→M1556→M1539 源链绑定为 ep34 `4bbaf7fc...`；这是来源链推断，不能改写成 child 自报 checkpoint。

独立 ratio-of-sums 重算：

- Dense→Bit-equal：1.678295×，时间减少 40.4157%，modeled transaction bytes 减少 21.2988%。
- Dense→Bit-K8：1.813827×，时间减少 44.8680%，bytes 减少 21.6201%。
- Bit-equal→Bit-K8：1.080756×，时间减少 7.4722%，bytes 减少 0.4083%。

三配置 modeled transaction bytes 分别为 64,137,300,768、50,476,836,000、50,270,750,592 bytes，总计 164,884,887,360 bytes。上述均是单个 D0/call0 diagnostic，不是完整 decoder、全网、能量、RTL 或 EDA 结果。

同一 hammer 在 CPython 3.6 和 3.10 下均拒绝 13 类攻击：extra、nested、symlink、重复 manifest、重复 JSON key、非有限值、PID/identity、cycle、byte、worker/root claim、nested alias claim 和 forbidden product config。评审没有打开 payload、没有重跑、没有启动 child/GPU/EDA，也没有修改原结果。

