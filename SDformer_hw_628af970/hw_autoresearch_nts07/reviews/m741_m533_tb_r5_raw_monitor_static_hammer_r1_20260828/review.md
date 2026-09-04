# M741：M533 TB r5 RAW monitor 静态打铁

## 裁决

**FAIL（需小修后重新静态打铁），94/100。** r5 的修复方向正确：相对 r4 只修改了 testbench 的 RAW monitor、coverage 字段和唯一 PASS/coverage token；top r2、SVA r2、macro adapter 与 checksum-identical foundry 模型均保持 M737/M738 冻结 SHA。无 fairness 依据的 8 个墙钟周期 RAW fatal 已删除，而每任务 20,000-cycle watchdog、攻击 watchdog 和全局 watchdog 均保留。

exact recovery 的核心逻辑也成立：RAW token 只有在 exact `(epoch, consumer, parent)` 的 direct forward，或 exact `(consumer, parent)` 且同 execution epoch 的 macro read response 到达时才在正常执行路径清除；macro-read request 本身不算交付。direct-forward 与 response recovery 新计数均为独立必达项，原 arithmetic、queue identity/data、conservation、六类 attack 和 fail-closed PASS 门没有被删除。

但当前源码还不能用于制作 r12 runner 候选，原因是两处静态合同偏差：

1. M738 明确要求原 `cov_stalled_raw_recovery` 继续只表示 direct-forward。r5 在 matching response 分支也递增该旧计数，改变了旧 coverage 字段语义。虽然新增的两个分项计数已经保证 direct/response 各自必达，这个 aggregate 仍应只在 direct-forward 分支递增，response 分支只递增 `cov_stalled_raw_response_recovery`。
2. M738 要求 observation trace 分开记录两路 sink ready。r5 的 `RAW_OBS` 只输出派生的合并字段 `sinks_ready`；它不能区分 `psum_write_ready=0` 与 `row_complete_ready=0`，而且该派生信号在非 final beat 时并不等价于两路 ready。应在同一 observation-only `$display` 中分别增加 `psum_write_ready`、`row_complete_ready`，可保留合并字段。

这两项都是 TB-only 小修，不支持改 top/SVA/macro，也不支持直接启动 VCS。修正后应产生新的 TB 身份并接受一次新的独立静态 hammer；本评审不授权 r12 runner 候选，更不授权 VCS/EDA 启动。

## 逐项核查

- r4 SHA256：`320901a07f9b01cb9cef334982a293cabfbd6e8f8b528cffd769e71a3c427c82`。
- r5 SHA256：`994818ce1bba9dde9b4280af8cbd2b12b5c7098ce044110bf21f47ab55cee0c4`。
- r4→r5 diff：单文件 52 行差异（44 insertions、8 deletions），仅涉及 monitor comments/trace、两个 coverage counter、coverage/PASS token；无 DUT/SVA/macro 修改。
- top r2、SVA r2、macro adapter、foundry slow `.v` SHA 分别为 `726039db...`、`b9f66feb...`、`8fd008a3...`、`8343acf0...`，与 M737/M738 身份一致。
- exact forward 分支检查 epoch/consumer/parent；exact response 分支检查 pending consumer/parent 和 execution epoch；read request 没有提前清 token。
- `stalled RAW timeout`、`raw_age >= 8`、`raw_age > 8` 均已不存在。
- task timeout（20,000 cycle）、attack fault watchdog（20 cycle）及 3,000,000-time-unit global watchdog仍存在。
- 原 cleanroom arithmetic、queue response identity/data、per-epoch conservation、6 个 attack 及 error/fatal gate均未在 diff 中改变。
- direct-forward 与 response recovery 分项均被纳入 minima；但旧 aggregate 语义需恢复。
- `RAW_OBS` 已含 token identity/age、current、slot0/slot1 identity、pending identity、reserved、forward、macro-read、read-response；缺两路独立 sink-ready。
- `$display/$fatal` 新格式串与参数数量静态人工核对一致；未运行任何编译器、仿真器或 EDA，因此这不是语法编译结论。

## P0/P1 与授权

- P0：0。
- P1：2（旧 coverage 字段语义漂移；RAW_OBS 缺两路独立 ready）。
- P2：0。
- 允许：作者制作一个新的 TB-only 修复身份，再交独立静态 hammer。
- 不允许：基于当前 r5 制作 r12 runner 候选；启动 VCS/simv/DC/PT；修改 top/SVA/macro/foundry；把任何结果写成 RTL/timing/PPA/speedup 结论。

## 冻结检查

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
