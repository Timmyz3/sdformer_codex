# M864：M863/C1 R21 fresh source hammer request

请由新的独立 reviewer 对 M863/R21 source-only 包做 100 分静态打铁，输出到 `reviews/m863_m533_r21_unit_delay_source_static_hammer_r1_20260829/`。

必须固定 M860 的 review/manifest/outer seal，重跑 exact r9→r10 diff、synthetic event-order（正例与四个负变异）、runner closure/timeout/pre-mkdir 自检，并证明 R21 result/attempt 不存在。禁止 VCS、simv、license query 和所有 EDA；本请求不授权 release 或 launch。
