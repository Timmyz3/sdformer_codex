# M620 fresh r5 runner static hammer request

请 fresh、independent、read-only 评审 M617 r5 runner/launcher、source contract 和 candidate，只验证 M616-P0-01/P0-02 是否机械关闭。允许 source inspection、lineage preflight 和临时 synthetic/static 故障注入；严禁 formal analyzer、runner `--execute`、正式 result/attempt/consumed、GPU、EDA、remote。

重点证明：未来 authorization 必须精确携带并验证 M615 true-release、M616 FAIL evidence 和实际 M620 PASS 的 review/manifest/outer SHA；在任何 analyzer subprocess 前，attempt 已 exclusive 创建、双封并以 `RENAME_NOREPLACE` 永久发布为 consumed。crash、failure 或 signal 后同 authorization 均不可重试，qfinal 和所有 canonical/staging/quarantine 类型均以 `lexists/lstat` fail-close。

PASS 需 score≥95、P0=0、P1=0，输出 request.json 指定的 exact schema/status/identity/one-shot 字段和双封。PASS 只允许后续作者生成 M621 admission；不授权本轮 formal execution。
