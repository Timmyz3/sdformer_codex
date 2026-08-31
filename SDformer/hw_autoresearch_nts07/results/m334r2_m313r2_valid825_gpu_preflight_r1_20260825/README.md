# M334r2：M313r2 valid825 GPU 启动前独立复核

结论：新 launcher（SHA256 `7b5c3be...`）获得一次 `GO_REMOTE_CANONICAL_PATH_ONLY`。M334 的 exact-byte contract clone P1 已闭合；克隆现在在 baseline 解析、wrapper import 和 GPU 之前由 canonical path gate 拒绝。

评分 `96/100`，`P0=0, P1=0, P2=2`。两个 P2 是下游仍转发原始 path spelling 的小型 symlink race，以及 terminal seal 后未对全部 runtime identity 做完全对称的最后一次重哈希；现有 nested identity、precommit rehash、候选后验核验和双 replay 足以阻止错误结果封存，不阻塞这一轮运行。

本地完整 main 在 M312r2 baseline receipt 的远端绝对路径处拒绝，这是预期 relocation guard。封存路径逐项映射确认：当且仅当远端仓库根为 `/root/private_data/work/sdformer_codex/SDformer` 时，receipt/launch 中的 contract、receipt、profile、per-frame 绝对路径全部吻合。

只能使用 review JSON 中 `remote_only_allowed_sequence.argv` 的 11 对固定顺序参数。启动前必须再次核 contract/launcher SHA、M312r2 manifest 与 seal，并确认候选 result root 不存在。成功后也只有四项候选 artifact 加 manifest 和 seal 全部存在且 replay 通过，才算有效结果。

本次没有导入 wrapper/evaluator，没有启动 GPU，没有修改 contract、baseline 或 docs/359。
