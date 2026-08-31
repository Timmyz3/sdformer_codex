# M613 fresh independent static-hammer request

请由非 M612 作者审阅，不得运行 formal analyzer，不得创建 result/attempt/auth。

必须攻击：

1. RESULT、ATTEMPT、CONSUMED、runner staging、adapter internal staging 分别及同时为 live/dangling symlink、
   FIFO、socket、普通文件、含嵌套特殊项目录；每次必须只留下一个 exact-member 双封 qfinal，不得遗留 canonical、
   raw 或 qstage，且 symlink target 不得被修改。
2. 在 lexical path 中间和末端放 live/dangling symlink，确认 runner、adapter、authorization、static identity 全部在
   resolve 前拒绝。
3. 人工放置 stale M612 raw/qstage，确认 preflight 阻断。
4. 回归 M607 已确认关闭的 M604 项：完整 result schema/identity/equations/CSV/RUN_COMPLETE、terminal five-member
   receipt、post-publish/post-consume rehash、adapter/result/attempt no-replace。

PASS 门为严格 `100/100, P0=0, P1=0`；仅可授权后续作者起草 M614，不可在本评审中运行。
