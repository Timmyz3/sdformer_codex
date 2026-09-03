# M1901｜M1900 C2 clean-env 双轴 hold-repair runner 独立打铁

**FAIL CLOSED，74/100，P0=0，P1=2，P2=1。** 本次只对 SHA256 为 `bc7a1911216aeaf43622a20e98bcb631c5ca2eb1ce74405c97819e2ef3fc02fd` 的 M1900 runner 做静态审阅和无 EDA 的合成攻击；没有查询 license、没有创建 attempt、没有运行 DC/PT/Formality/VCS/PTPX，也没有修改 runner、Tcl、前序证据或 `docs/359`。M1902 release 不得据此创建，M1900 不得启动。

## 已修正的 M1897 五项

1. runner 使用 clean-env shebang，license 与 DC 都由绝对路径在 `env -i` 下执行；固定循环只允许 K8、K1x8 两轴和两次 DC。
2. attempt 先在 `WORK/attempt_stage` 中封存并无替换发布，发生在 license 之前；普通非零退出会把剩余 WORK 封进固定 no-retry quarantine。
3. 两轴都要求 setup/hold `MET` 且零 violation；五类 design-rule section 必须各自出现 `This design has no violated constraints.`，同时拒绝 `(VIOLATED)` 行。
4. 两轴分别绑定 M1811 DDC/SDC、设计名、基线面积及 5% ceiling；冻结 Tcl 保持 3.000/0.200 ns，按 0.070 ns 优化后恢复 0.050 ns 报告合同。
5. 成功发布后会验证 source 消失、destination 是非 symlink 目录并通过双封；raw receipt 仍明确 Formality/PT/power/PPA/system-speedup 全为 false。

## 阻断项

### P1-01｜四个 positional digest 仍是 caller-self-pin

第 6--8 行从调用者接收 runner/review/release/audit SHA；第 70--90 行虽然做了 JSON 语义检查，但 M1901 与 M1903 没有被任何独立、不可由同一调用者替换的根固定。合成攻击用完全调用者构造的最小 M1901/M1902/M1903 对象（M1901/M1903 甚至没有 schema）通过了原样语义断言。更严重的是，修改过的 runner 可以带自己的新 SHA、配套伪 review/release/audit 一起启动，因此 exact-runner 门本身也可被自洽重写。

修复要求：下一版必须消费一个**不由本次调用参数提供**的 launch root。至少把 exact runner/review/release/audit 四元组固化进另一个只接受零参数的受审 launcher/authority，并由独立审阅明确发布唯一命令；禁止再次从环境变量或位置参数接收这些 SHA。M1901 与 M1903 还应检查 exact schema、milestone、reviewer 分离和完整 identity，而不只是 status/count/部分 identity。

### P1-02｜信号 trap 可把已消费 attempt 变成无终态

`trap finish EXIT INT TERM HUP` 共用 `finish()`，而 `finish()` 用入口 `$?` 判断是否失败。无 EDA 实测表明，SIGTERM 在 bash builtin/循环边界到达时，信号 handler 看到 `rc=0`；于是第 50--53 行不会封 quarantine，handler 清除 traps 并 `exit 0`。若信号在第 109 行 attempt 已发布后、下一外部命令之外的 shell 区段到达，会留下 sealed ATTEMPT 和未封 WORK，但既无 FAILURE 也无 RESULT，违反 partial-attempt 必须有终态和失败无重试的合同。

修复要求：给 INT/TERM/HUP 分别传入强制非零码（如 130/143/129），EXIT handler 显式接收 `$?`，并让 handler 幂等；任何 `ACTIVE=1` 后的信号都必须封存固定 failure quarantine 后再非零退出。

### P2-01｜未来 review/audit 的语义检查字段不足

M1901 只检查 status、三个计数和 runner SHA；M1903 只检查 status、三个计数和三项 identity。schema、milestone、reviewer identity/different-author、scope/authorization 均未检查。这不是在已有独立 launch root 下必然改变运行参数，但会让最小伪对象冒充完整审阅，并放大 P1-01。

## 结论

工程门已有实质进步：clean env、attempt-before-license、五节 DRC、双轴时序/面积和成功发布双封均成立；但 authority 自 pin 与信号零码 trap 都是 launch-blocking。唯一授权是继续写 additive successor 并重新独立审阅；license/attempt/DC 均为 0。
