# M1136r6 M1133r6 final launch hammer

结论：**PASS；仅授权 root 在外部 preflight 通过后执行一次 r6 零参数 launcher。** 不授权 agent 执行，不授权自动重试。

独立 hammer 共 682 checks、24 attacks。launcher 的真实 hardcoded-authority、零参数、精确 `env -i`、stopped-r5、fresh-r6、memory/commit headroom 与 same-UID collision 路径全部执行；仅最终 engine child 被替换为一次受控 fake child，精确 argv/clean env/返回码传播均通过。没有 patch 掉 authority 或 resource gate。

双封后，M1133r6 engine 的 `static_gate()` 对真实 M1136r6 seal 成功返回，确认 authority 链完整且无环。测试前后 r5/r6 canonical attempt/result/work/failure/lock 均为空，未运行真实 launcher、engine main、VCS、DC 或 EDA。docs/359 SHA 保持 `dedde7ce...`。

唯一授权命令：

```text
/usr/bin/env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/pytorch310/bin/python3.10 /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_m1133r6_c2_authority_schema_repair_authorized_launch_r1.py
```

root 必须先确认无同 UID EDA 进程、资源门通过、r5/r6 namespace 仍符合合同。命令只可执行一次；若 attempt 被创建后失败，r6 namespace 不得重试。
