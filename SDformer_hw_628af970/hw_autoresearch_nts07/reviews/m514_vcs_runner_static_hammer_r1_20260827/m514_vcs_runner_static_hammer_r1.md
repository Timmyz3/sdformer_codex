# M514 exact-SHA VCS chain 独立静态打铁 r1

日期：2026-08-27  
结论：`NO_GO__MISSING_ASSERT_REPORT_IS_FAIL_OPEN`  
评分：**80/100**  
P0：**1**  
P1：**4**  
VCS / DC / DSE / simulator 实际执行：**否**

## Literal verdict

当前 runner SHA：

```text
b55e8189f821a5148ab1e8f8ae39ced87528eda8e0641de4e31b5d03832bdaba
```

结论是 **literal NO-GO**。即使 M496 DC 结束，也不得以该 SHA 启动 VCS；必须
修复 assertion evidence P0、产生新 runner SHA 并重新静态审查。当前 M496 DC
正在运行时，本 runner 的三次 resource gate 本应因 `dc_shell` 命中而在 attempt
之前退出；本评审没有用实际执行去验证该门。

## P0｜`assert.report` 缺失会被当成成功

runner 在仿真后执行：

```bash
! grep -Eiq 'failed at|Offending|...' sim.log assert.report
```

但没有先要求 `assert.report` 是存在、regular、non-symlink 的文件。GNU grep 在
文件缺失时返回 2，Bash 的 `!` 会把任何非零状态反转成 0，因此缺失 report 与
“存在且无 assertion failure”完全等价。本轮只用一个不存在路径重构 shell
语义，确认 `missing_assert_report_is_accepted_by_negated_grep=true`；没有运行
simulator。

影响：即使 `-assert report=...` 没生效、report 路径漂移或 assertion reporting
根本未产出，runner 仍可凭 TB `$fatal` 和 exact PASS 行发布
`DIRECTED_FUNCTIONAL_COMPLETENESS`，这不满足 contract 的 `sva_enabled` 与
`assertion_coverage_enabled` 门。

必修：仿真后先要求 `sim.log`、`assert.report` 均为 regular non-symlink 文件；
`assert.report` 允许为空（只有 passing assert、无 cover property 时可以为空），
但不能缺失。之后分别运行 failure grep，并区分 grep rc=0/1/2：只有 rc=1
“无匹配”可继续，I/O/语法错误必须失败。修订后重审新 SHA。

## 已通过项

### 身份与 review chain

- contract `60e4fe59...`、filelist `0a0dbfb3...`、RTL `90c44fc9...`、TB
  `6c283bf9...`、VCS binary `0735e4b8...`、static-review SHA256SUMS
  `20eb76fa...`、`docs/359` `dedde7ce...` 均在 attempt 前检查。
- static review 的全部四个 members 由 `sha256sum -c` 复核；contract/filelist
  内容与 runner 使用路径一致。
- caller 必须提供 `M514_EXPECTED_RUNNER_SHA256`；本评审若将来升级为 GO，只会
  授权经过新一轮审查的字面 SHA，禁止动态 `sha256sum` 代入。
- 输入在 resource gate 前后、仿真后、work seal 后多次重验；`input_sha256.txt`
  又提供 start/end 同一性。

### DC/VCS/DSE exclusion 与 dormant simv

- 三次、间隔 5 秒检查 commit/memory/swap/cgroup；failcnt、under_oom、oom_kill
  必须全为 0。
- `dc_shell[-t]`、fm/pt、vcs/vcs1、vlogan/vhdlan、common-shell 子进程以及项目
  `analyze/independent/sweep/dse/simulate_m*.py` 都会阻断 attempt。
- 同一用户的任何 `simv` 一律阻断；foreign `simv` 只有 state 以 S/I 开头、
  CPU≤0.5%、RSS≤256 MiB 才作为 dormant 例外，其余阻断并写入 preflight log。
- resource gate 失败时 `m514_attempt_live=0`，work 被移入 preflight quarantine，
  fixed one-shot attempt 不会被消耗。

### compile/simulation/PASS 主门

- compile 和 sim 都记录真实 rc，分别要求 rc=0；compile 还要求 executable
  `simv`。
- 当前 VCS 版本在盘上实际 compile logs 使用 `Warning-[...]`，现有 regex 能捕获
  该主格式以及 `Error-[`、line-start Error/Fatal。
- TB 所有 scoreboard/cover/timeout 失败都用 `$fatal(1,...)`；sim failure grep
  又覆盖 VCS assertion 的 `failed at` 与 `Offending` 标准报告形式。
- PASS regex 要求恰好一行：43 taps、非零 stalls/replacements、精确
  phases `6/10/10/17`、protocol_attack=1；Python 使用 fullmatch 再解析。

### work→canonical 与失败 quarantine

- work 与 canonical 位于同一 `results` 文件系统，最终 `mv -T` 是原子 rename；
  publish 前 work 全部 regular files 被 SHA256SUMS 提交并内外复验。
- attempt 之前失败进入 preflight quarantine；attempt 之后失败进入
  failed/incomplete quarantine，并追加明确禁止引用的 marker。
- canonical 在 publish 前必须不存在。SIGKILL 发生在 rename 后只会留下已经
  presealed 的 work 对象，不会留下部分复制目录。

## P1

1. **compile warning regex 不覆盖所有文字形式。** `Warning:` 的合成行不会被
   `Warning-\[` regex 命中。当前 pinned VCS 常见格式是 `Warning-[...]`，所以
   非 P0；修订时应把 `Warning:`/plain line-start Warning 与 grep rc=2 一并
   fail closed。
2. **attempt population 未 exact-check。** initial seal 提交
   ATTEMPT_CONSUMED/identity，但 runner 没有要求 attempt 实际文件集合恰好为
   这四个文件；identity 也只含 runner+contract，不含 VCS/RTL/TB/filelist/review/
   docs。work receipt 虽完整绑定这些输入，durable one-shot forensic receipt
   仍偏弱。
3. **`m514_complete=1` 早于 canonical rename。** 若最终 `mv -T` 失败，EXIT trap
   因 complete 已置位而不会给 work 添加 failure marker或移入 quarantine；不会
  产生错误 canonical，但会永久消耗 attempt 且留下难路由 work。应在 rename
   成功后置 complete。
4. **claim boundary 少了 contract 的 `formality=false`。** receipt 其余负声明
   完整且不会产生 formality 正主张，因此不阻断；修订时应要求 receipt claim
   key set 与 contract 完全一致。

## Claim boundary

即使修复后 VCS PASS，M514 也只证明 standalone mapper 的 directed address、
phase、ready/valid、replacement、stall 与 fault-drain completeness。它不证明
full decoder numeric equivalence、cycle speedup、area/timing/energy、Formality、
system speedup、paper PPA 或 DATE headline。

`docs/359` 未修改。
