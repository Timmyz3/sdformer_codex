# M797：M795/C1 R16 函数闭包 fresh hammer

## 裁决

**FAIL_SOURCE_GATE，98/100，P0=1、P1=0、P2=0。R16 不得 release 或启动。**

主源码本身的函数闭包修复是成立的：完整实跑得到 31 个定义、230 个保守调用、0 个未定义调用、0 个重复定义；删除定义、重命名定义、注入 R15 旧短名三种负例均按预期失败；20 个外部命令的 regular-file exact-SHA 白名单也全部通过。76 条 `require_regular_sha` 边全部实时匹配，R15 通过 M794 永久撤销，UNIT_DELAY、环境、resource、coverage 和 terminal 合同未发现弱化。

但强制的 `test_m795_r16_runner_premkdir_dry_run.py` 在项目固定的 Python 3.6.8 上无法运行。它调用了 `subprocess.run(..., text=True)`，而 `text` 是 Python 3.7+ 参数；实际执行在创建子进程前即报：

```text
TypeError: __init__() got an unexpected keyword argument 'text'
```

因此本次没有到达 exact runner stub、没有观察到 rc86，也没有得到五事件 trace。失败发生在 `Popen` 之前，所以 runner/VCS/license/simv/result 的副作用计数均为 0。

## 已通过的证据

- runner/source/candidate/handoff SHA 分别为 `c26a3cab...`、`823343...`、`635a277...`、`50e4fd...`，与任务输入一致。
- runner、source contract、candidate 和撤销的 R15 release 均通过内外双封。
- M794：`PASS_FAILURE_AUDIT` 100 分；R15 `PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE`；attempt 未消费，result 不存在。
- 函数闭包正例与三类变异负例完整实际执行通过。
- 20/20 external command exact-SHA 通过；unknown/unused/mismatch 均为 0。
- 76/76 SHA literal 为 64 位小写十六进制且 live regular non-symlink 匹配。
- `+define+UNIT_DELAY` 保留；`+notimingcheck`、`+no_notifier` 不存在；coverage minimum 仍为 `minima=1 normal_covers=13`。
- docs/359 SHA 仍为 `dedde7ce...`；R16 result、release 均不存在。

## 必须的最小修复

把 dry-run harness 的 `text=True` 改成 Python 3.6 兼容的 `universal_newlines=True`。由于测试 SHA 已绑定在 runner、source contract、candidate 与 76 条边中，不能原地把当前 M795 评成 PASS；应使用新的加法身份重封，并重新执行完整 fresh hammer。只有新 hammer 实际得到 rc86、严格五事件序列和五类副作用 0，才可产生 source/candidate 100 分 PASS 并继续 release 链。
