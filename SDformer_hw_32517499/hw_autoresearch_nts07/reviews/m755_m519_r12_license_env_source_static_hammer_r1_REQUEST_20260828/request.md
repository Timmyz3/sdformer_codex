# M755：M519 R12 license-env source/static hammer request

请做 fresh independent、只读、NO_EDA 打铁。R12 是 additive 新身份；R11 的唯一 attempt 已消费且 quarantine 永久为 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`。

重点不是证明许可一定能 checkout，而是确认以下 fail-closed 顺序：冻结身份与 exact clean env → K1 资源 preflight → double-sealed `lmutil lmstat` server/`Design-Compiler`/`DC-Ultra` 状态 → 只有全部明确通过后才发布 R12 attempt → 第一次真实 DC launch。任何不确定、不可达、无法解析或没有空闲 seat 都必须 `NO_ATTEMPT_CONSUMED`。

禁止连接 license server、禁止运行 runner live/DC/VCS/Formality/PT/PTPX/remote，禁止创建 `launch_now=true` release。可以运行 `bash -n`、`jq`、SHA/double-seal 检查和 runner 的显式 NO_EDA full-path self-test；该 self-test 必须在 resource/license preflight、attempt 和工具之前退出。

逐项要求见 `request.json`。只有 fresh review PASS 且 P0=P1=0 后，主代理才可另建最终 release 并再次打铁。
