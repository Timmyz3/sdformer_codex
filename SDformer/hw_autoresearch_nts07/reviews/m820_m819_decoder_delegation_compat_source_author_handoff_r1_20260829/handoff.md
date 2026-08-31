# M820｜M819 decoder delegation-compat source author handoff

M819 是 M817 P1 的 additive 修复，不修改 M809/M815/M817。正式 flat attempt 保留 M819 schema 与 runner/driver/candidate/release SHA 身份，但 status 使用冻结 M809 production body 唯一接受的精确 parent token：`CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY`。

纯临时 preproduction traversal 已真正进入冻结 SHA `2b273d...6736d0` 的 `M809.run_production()`：attempt receipt 校验通过，并在其 `output.mkdir` 调用点受控停止；0 schedule row、output absent、没有 `attempt receipt identity drift`。临时 validator 在异常路径通过 `finally` 恢复。

M815 的 failure-trap 修复完整保留：publish → started → phase → postcheck → consumed preflight → production。独立 postpublish 故障注入仍得到 0 row、result absent 和精确四成员双封 quarantine；collision、destination symlink 与 log symlink 都 no-clobber。

Python 3.10 与 Python 3.6 均通过 compile、self-test、12/12 tests 和 candidate validation；`bash -n` 通过。没有 true release、正式 attempt/result/failure、production cycle、VCS/EDA/GPU/remote。

下一步只允许 M821 receipt-blind fresh source hammer。PASS100 后仍只能另行创建 true release，不能直接生产。
