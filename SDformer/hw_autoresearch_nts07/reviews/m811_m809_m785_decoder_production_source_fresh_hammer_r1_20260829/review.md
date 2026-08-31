# M811｜M809 decoder production recovery fresh source hammer

## Decision

`NO_GO_M809_TRUE_RELEASE__P1_1__AUTHOR_ADDITIVE_RUNNER_REPAIR_REQUIRED`，92/100，P0/P1/P2 = 0/1/0。

M809 的平铺 attempt、完整 consumed-attempt 预检和四成员失败回执主体均通过独立 source-only 测试，但当前 runner 仍存在一个 true-release blocker，因此不能签发请求要求的 PASS100，也不得运行 formal one-shot runner。

## P1 blocker

Runner 第 187 行先用 `mv -T --no-clobber` 将已经双封的 attempt 发布到永久 consumed 路径；第 188–192 行随后执行仍可能失败的目录/成员后检查；直到第 194 行才设置 `m809_started=1`。EXIT trap 又只在 `m809_started==1` 时生成 failed-or-incomplete 回执。

因此，如果 attempt 发布后的后检查因文件系统错误或竞争修改失败，attempt 已永久消费，但 trap 仍把它视为“尚未开始”，不会生成要求的四成员隔离回执。这与 recovery contract 的“every post-consumption failure”矛盾，也会重现 M798 的证据缺口。

最小修复不是改算法或周期模型，而是创建 additive runner identity：成功发布 attempt 后立刻设置 `m809_started=1` 和 post-consumption phase，再执行任何可能失败的后检查；同时加入一个 post-mv/postcheck 注入失败测试，要求 0 schedule row、canonical result 不存在、且产生双封的 `failure.json + driver.log + SHA256SUMS + SHA256SUMS.seal.sha256`。

## Passed evidence

- request、candidate、recovery contract、driver、runner、tests 及全部双层 sidecar SHA 均通过。
- Python compile、`bash -n`、driver self-test、9/9 source-only unit tests、candidate validation 全通过。
- 平铺 attempt 的递归 population 精确为 `{attempt.json}`，旧层级布局精确报 `sealed population mismatch`。
- 完整临时 `validate_consumed_attempt` 夹具在 0 schedule row 下通过；错误 release SHA、错误 attempt runner SHA、错误 release source SHA 均 fail closed。
- pre-stage failure receipt constructor 生成精确四成员双封目录，并在 destination collision 下保持已有目录 byte-identical。
- M785 的 M686 40 + M699 120、T10、D1 非 headline、96 lane、245760 B、Acc24、3 ns、192 B/cycle 与唯一合法 K8/equal-service 比值均保持不变。
- 旧 M798 attempt 前后递归快照 SHA 都为 `5c49be48a0930f2534a86bb8ca20b414de5a7485f8b0f1c0e197fda0eb3ef607`；没有 rerun 或改写。
- M809 true release、canonical attempt/result、failed-or-incomplete 均保持 absent；没有 production、EDA、GPU 或 remote 行为。
- `docs/359_DATE终局冻结_20260813.md` 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## Claim boundary

本审阅只有 source evidence。没有 decoder cycle、speedup、decoder-complete、full-network、Table-A、RTL/VCS/EDA/PPA/energy 准入。
