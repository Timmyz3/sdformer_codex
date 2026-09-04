# M719 / M533 r9 fresh source-static hammer request

请对唯一 r9 source-only repair 做 fresh、独立、只读评审。不得运行作者 runner、VCS、simv 或任何 EDA，不得创建 result/attempt/candidate/release，不得修改旧身份或 `docs/359`。

必须独立完成：

1. 验证 runner、source contract、author handoff、r8 失败包、M717 的全部 SHA 和双封；
2. 验证 RTL/TB/SVA/macro/binding plan 与 r8 byte-exact；
3. 运行 `bash -n`，但不得执行 runner；
4. 做 wrong-old-runner negative：新 contract 只接受 r9 SHA `27f2d7c0...604`，必须拒绝 r8 SHA `176c14d3...746e`；
5. 用隔离 shell 独立复现 old same-local RC=127 与 new split-local RC=0；
6. 验证 monitor function 除 local split 外语义不变，VCS compile 至 terminal tail 与 r8 byte-exact；
7. 验证新 result path、attempt marker、candidate、release 和未来 review path 全部不存在；
8. 验证 r9 runner 在 preflight 前及 atomic mkdir 前再次硬校验 r8 失败包和 M717；
9. 检查所有授权仍为 0，PASS 也不得授权 launch。

PASS 输出必须为：

```text
reviews/m719_m533_r9_source_static_hammer_r1_20260828/review.json
schema = m719_m533_r9_source_static_hammer_v1
status = PASS_M719_M533_R9_SOURCE_STATIC_HAMMER
score = 100
P0/P1/P2 = 0/0/0
decision.vcs_launch_authorized_now = false
```

review 的 `static_selftest` 必须含：

```json
{
  "wrong_old_runner_negative_pass": true,
  "old_same_local_rc": 127,
  "new_split_local_rc": 0,
  "new_result_path_absent": true
}
```

source-static PASS 后仍需另行建立 candidate、candidate hammer、`launch_now=true` release 和 final hammer；本 request 不是运行授权。

