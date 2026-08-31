# M850 request：M849 C1 R20 source fresh hammer

请由与 M849 source author 不同的 fresh independent reviewer 执行。禁止 VCS、simv、许可证查询和所有 EDA；禁止创建正式 result、attempt 或 release；禁止修改 M849 source artifacts、R19 result 或 `docs/359`。

必须独立重构 TB r8→r9 diff，确认仅三行 P2 epoch consumer 从 3 改为 14：`build_reference`、`load_task`、`wait_done`。三者必须一致，normal frontier 13 后 P2=14 严格单调，禁止插 reset。必须明确 M847 文字漏列 `wait_done`；若仍等待 epoch 3，则未来仿真必然 watchdog，不能 PASS source hammer。

重跑 exact TB test、closure 正例与 delete/rename/stale 三负变异、fake timeout fast/TERM/KILL/tee/双封和 pre-mkdir dry-run。要求 13 normal cover、P2 两项、held-final、六攻击、P2 token 和 final PASS token 全保留；RTL/SVA/macro/binding/foundry 全冻结；102 条 `require_regular_sha` exact edge 全活。

输出固定在 `reviews/m850_m849_c1_r20_p2_epoch_triplet_source_fresh_hammer_r1_20260829/`。只有 PASS100、P0/P1/P2=0/0/0 才允许后续 candidate hammer；即使通过也不得 launch 或 author release。
