# M612 author handoff｜energy exact runner r4

状态：**launch_now=false；release=false；NO FORMAL RUN；fresh M613 required。**

M612 保留 M606 已关闭的完整结果 verifier、terminal identity/member map、terminal rehash 与三处
`RENAME_NOREPLACE`。本轮只修 M607 的一个 P1 和一个 P2：

- canonical 为 symlink、FIFO、socket 或含特殊成员的目录时，不再直接把特殊项放进 qstage。runner 先用 no-replace
  从 canonical 命名空间移走，再将 `lstat` 元数据、regular-file SHA 或 symlink target 原始字节编码为普通 JSON，
  最后 no-follow 删除特殊项。唯一 qfinal 只含两个普通 JSON 成员和双封。
- preflight 阻断 stale `.m612_energy.failed_quarantine.staging.*` 和 `.m612_energy.failed_raw.*`。
- caller-visible identity path 先按 lexical components 逐级 `lstat`，发现 live/dangling symlink 立即拒绝；之后才允许
  realpath 一致性比较。

作者临时测试覆盖同时存在的 dangling symlink、live symlink、FIFO、Unix socket、adapter staging symlink，最终
canonical 全 absent、无 raw/qstage、只有一个双封 qfinal；外部 live-symlink target 保持未修改。缺授权执行以 70
fail-close，未创建正式坐标。

作者不得自评。M613 必须重新攻击 M607 两项，并回归 M606 已关闭项；只有 `100/100, P0=P1=0` 才允许另行起草
M614。docs/359 未修改。
