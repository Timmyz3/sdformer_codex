# M855：M849/C1 R20 conditional true-release author handoff

M855 已生成并双封一次 conditional true-release，精确绑定 fresh independent M852 source hammer 与 M853 candidate hammer。两者的 review/manifest/outer-seal 身份分别为 `e1cf9591...` / `6857efec...` / `ff7a616a...` 和 `85c42423...` / `5c9b13c...` / `3c0e85e6...`。

release 的结构性 `launch_now=true` 是冻结 runner 的必需字面量；在不同 reviewer 完成 M856 fresh final hammer PASS100 前，`authorization_effective_now=false`，禁止启动 live runner、VCS、simv、license 或任何 EDA。

M855 没有修改 runner、TB、RTL、SVA、macro、binding、foundry model 或 candidate；没有创建 result/attempt。`docs/359_DATE终局冻结_20260813.md` 仍为 `dedde7ce...`。

下一步唯一合法动作：不同 reviewer 消费 M856 request，重新做静态 closure/timeout/pre-mkdir 打铁，在冻结 final-hammer 路径发布 PASS100；随后才允许 root 执行 exactly one no-argument R20 functional VCS+simv attempt。
