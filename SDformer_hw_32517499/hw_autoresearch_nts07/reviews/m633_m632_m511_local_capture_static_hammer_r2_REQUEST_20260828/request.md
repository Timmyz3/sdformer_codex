# M633｜M632/M511 本机 exact capture 信任链 r2 fresh hammer 请求

请完整读取 M631 NO-GO、`contracts/m632_m511_local_capture_trust_chain_r2_contract_20260828.json`、当前 M511 runner、M632 wrapper、M511 producer/contract 与 payload verifier，独立判断 M631 两个 P1 是否关闭。

必须重点攻击：

1. caller 动态自算 runner SHA 是否还能绕过“唯一 admissible wrapper”并最终通过 payload verifier；
2. wrapper 内 runner literal、host/GPU literal 与 caller wrapper literal 的顺序；
3. Python real executable 是否在 pre/post、attempt exact-seven identity 与 verifier literal 中闭合；
4. package version pre/post、dangling symlink、三次七字段资源门、one-shot 和 quarantine；
5. 40-record/87,030,000B payload verifier 语义是否因改 identity 被削弱。

本轮仍只允许静态 source audit、`bash -n`、Python `compile()`，以及保证在 canonical output/attempt 创建前退出的负控。禁止生产 capture、checkpoint/model、CUDA、payload verifier生产运行、VCS/DC/DSE。

输出新的 r2 review，必须给 GO/NO_GO、score、P0/P1/P2；只有 P0=0 且 P1=0 才可给唯一字面 wrapper 命令。不得修改 `docs/359`。
