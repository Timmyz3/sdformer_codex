# M833 request：M831 C1 R19 source fresh hammer

请由与 M831 source author 不同的 fresh independent reviewer 执行。禁止 VCS、simv、许可证查询和所有 EDA；禁止创建正式 result、attempt 或 release；禁止修改 M831 source artifacts、R18 result 或 `docs/359`。

## 必锤项目

1. 重验 runner、source contract、closed candidate、四个 source test 的成员与外层 seal，重算所有 SHA；核对 M827、M829 与永久消费 R18 result 的完整双封。
2. 确认 top RTL r2、SVA r2、TB r8、macro adapter、binding plan、foundry UNIT_DELAY 模型及 13 normal + P2 + held-final + 六攻击未变，所有功能、coverage、resource、terminal、double-seal 门均未放松。
3. 确认生产 simv 行唯一为 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`；`/usr/bin/timeout` SHA 为 `2d5662...`。300 s 只能是 fail-closed wall-clock，不得解释成 cycle/performance/RTL timeout。
4. 确认 rc124/137 且 `sim.log` 无 HDL token 时分类为 `infrastructure_timeout_before_verilog_time`，失败 receipt 双封，TERM→KILL 与 tee 完成后无 orphan；其它 rc 或已有 HDL token 不得误归该类。
5. 确认没有新增或猜测 telemetry 环境变量；`-no_save` 只写成避免 VCS 文档中的 ASLR re-exec，不宣称关闭 telemetry/外部通信。
6. 用 `/usr/libexec/platform-python3.6` 独立重跑：TB r8 source-static；closure 正例与 delete/rename/inject-stale 三负变异；timeout fake-simv 的 fast/TERM/KILL/tee/receipt-seal；pre-mkdir dry-run。要求 pre-mkdir rc86 且 VCS identity/license/compile/simv/result mkdir 全为 0。
7. 静态枚举 94 条 `require_regular_sha` lower-hex edge，并结合 exact pre-mkdir dry-run确认 live；严格 JSON duplicate-key 检查；确认 prospective result、source review、candidate hammer、release、final hammer 均未先占。
8. 确认 closed candidate 的 `launch_now=false`、`authorization_effective_now=false`，本轮只能在 PASS100/P0=P1=P2=0 时授权下一位 reviewer 做 candidate hammer，仍不得 launch 或 author release。

## 输出

固定目录：`reviews/m833_m831_c1_r19_wallclock_fail_closed_source_fresh_hammer_r1_20260829/`

输出 `review.json`、`review.md`、`mechanical_checks.txt`、`RUN_COMPLETE.txt`、`SHA256SUMS`、`SHA256SUMS.seal.sha256`。只有 100/100 且 P0/P1/P2=0/0/0 才允许 PASS；否则 fail-closed 并只授权 additive source repair。
