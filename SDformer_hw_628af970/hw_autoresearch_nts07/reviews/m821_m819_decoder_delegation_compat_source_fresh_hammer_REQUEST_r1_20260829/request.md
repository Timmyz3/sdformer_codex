# M821 request｜M819 decoder delegation-compat fresh source hammer

Receipt-blind 审阅 M819 additive repair。只允许读取源码、运行纯临时 source tests 并双封 review；不得创建 true release、调用 formal runner、消费正式 attempt、生成生产周期或调用 VCS/EDA/license/GPU/remote。

决定性攻击是用纯临时 fixture 真正进入精确 SHA 的冻结 `M809.run_production()`：M819 attempt 必须保留外层 schema/SHA 身份，同时 status 精确等于 parent-compatible M809 token。穿透必须通过 parent receipt check，并在 `output.mkdir` 处受控停止，0 schedule row、output absent、无 `attempt receipt identity drift`。

还必须保留 M815 修好的 publish → started → phase → postcheck → consumed preflight → production 时序，以及四成员双封 failure quarantine、collision/symlink no-clobber、严格 JSON 和 Python 3.6 闭合。

M817 必须继续保持 NO-GO/release=false。M809、M815、M798 attempt 与 docs/359 不得修改。

唯一 PASS 是 `PASS100_M819_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY`；即使 PASS，仍不能直接生产。
