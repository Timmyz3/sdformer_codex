# M1045 独立打铁：M1046 C2 mapped-gate SAIF release

结论：**GO，但只授权 exact M1046 mapped-gate VCS/SAIF 一次尝试。** 不授权 PT、PTPX、DC、GPU 或自动重试。

## 独立实跑

在私有临时目录，用冻结 tiny TB/Tcl 和当前许可证路由实际执行：

`vcs -full64 -sverilog -debug_access+r -lca`

编译和仿真返回码均为 0，生成 2106 B SAIF；DURATION 为 24 ns，top 与 DUT 层级均存在。临时实跑未创建或消费 M1046 正式 namespace，未记录许可证值、编译器或仿真器输出。

## Fail-closed 攻击

missing debug、missing `-lca`、UCLI power 失败、SAIF missing/empty、错误 top/DUT hierarchy、zero duration、错误 release/runner/source seal/status 以及 namespace collision 共 18 类攻击，均在 `mkdir ${attempt}` 前被拒。

M1044 release 的 `launch_now=false` 与 frozen runner 第 189 行要求的 `launch_now == false` 一致。M1033 保持 DO_NOT_RETRY，完成 gate case=0、SAIF=0。

## Claim boundary

本收据仅证明 release 链与 tiny UCLI-power 能力。没有运行 M1046 生产三轴，没有生产 SAIF，也没有功耗、能量、系统倍速或 paper-PPA claim。
