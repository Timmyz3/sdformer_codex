# M743：M533 TB r6 RAW monitor 独立静态打铁

## 裁决

**FAIL（单点小修后重打），96/100。** r6 已经完整闭合 M741 指出的两项偏差：旧字段 `cov_stalled_raw_recovery` 只在 exact direct-forward 分支递增；response 只递增独立字段 `cov_stalled_raw_response_recovery`。`RAW_OBS` 也已经分别输出 `psum_write_ready` 和 `row_complete_ready`，同时保留派生的 `sinks_ready`。

r4→r5→r6 的累计差异仍严格限于 testbench monitor、coverage 计数/门和 coverage/PASS token。top r2、SVA r2、9×128 macro adapter、binding plan 和 checksum-identical foundry Verilog 的当前 SHA 均与 M737 冻结身份一致。exact response credit 需要 consumer、parent 和 execution epoch 全部匹配；macro-read request 本身不计 recovery。direct-forward 与 response recovery 各有独立 minima。原 per-task/global watchdog、cleanroom arithmetic/queue oracle、per-epoch conservation、P2 foundry-response identity 检查、六个 protocol attack 和唯一 PASS 门均保留。

但 r6 仍有一个新的阻塞性因果优先级问题，当前不允许制作 r12 runner 候选：当 RAW token pending 的同一拍同时出现 **matching macro response** 和 **unrelated direct forward** 时，835--874 行先进入无条件 `if (expected_forward)`，随即因 forward identity 与 token 不同而 fatal；合法且 exact 的 matching response 分支因为 `else if` 永远到不了。这不是虚构组合：本设计明确支持 prior response + same-cycle forward 的 dual enqueue，cleanroom 在 971--997 行按 response 后 forward 的顺序建模，`cov_pending_plus_forward` 也要求覆盖该组合。一个 unrelated forward 不应被当成旧 RAW token 的 credit，但也不应阻止同拍 matching response 完成该 token。

最小修复应仍为 TB-only：先分别计算 `forward_matches_token` 和 `response_matches_token`，仅 matching event 可清 token；若 response matching 而 forward unrelated，response 必须仍可完成 token。unrelated forward 单独发生时既不 credit 也不 fatal，token继续等待；若 token 到任务 drain 仍未恢复，现有 `RAW recovery escaped task` fatal 已负责 fail-closed。修复后生成新 TB 身份并重新静态打铁。不得改 top/SVA/macro/foundry，也不得只靠测试向量“碰不到”这个组合来放行。

## 机械复核

- r4 SHA256：`320901a07f9b01cb9cef334982a293cabfbd6e8f8b528cffd769e71a3c427c82`。
- r5 SHA256：`994818ce1bba9dde9b4280af8cbd2b12b5c7098ce044110bf21f47ab55cee0c4`。
- r6 SHA256：`10fb3f30d96932621033608ebd807900d782952007076a6913023620bc584507`。
- r4→r6：43 行增加、8 行删除；差异仅位于 coverage 声明、RAW observation/recovery monitor、coverage minima/打印和 PASS token。
- top r2 SHA256：`726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1`。
- SVA r2 SHA256：`b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b`。
- macro adapter SHA256：`8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783`。
- macro binding plan SHA256：`db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983`。
- foundry Verilog SHA256：`8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d`。
- M738 与 M741 双 seal 重新校验均通过。
- `cov_stalled_raw_recovery` 和 `cov_stalled_raw_forward_recovery` 仅在 exact forward 分支递增；response 分支只递增 `cov_stalled_raw_response_recovery`。
- response credit 静态检查：`expected_read_response` + exact consumer + exact parent；epoch 不同立即 fatal。
- `RAW_OBS` 分别打印 `psum_write_ready`、`row_complete_ready`，并保留 slot、pending、reserved、forward/read/response 全部观察字段。
- direct 与 response 分项 minima 均为 `>=1`；旧 aggregate 也保留。
- task watchdog 仍为 20,000 cycle，attack watchdog 仍为 20 cycle，全局 watchdog 仍为 3,000,000 time unit。
- `$fatal`/`$error` 数量相对 r4 均未减少（分别 27/9）；六类攻击计数和门未发生差异。
- 本评审没有运行 VCS、simv、iverilog、Verilator、DC、PT 或任何 EDA，因此不形成语法、功能、时序、PPA 或性能结论。

## P0/P1 与授权

- P0：1。RAW recovery 的 forward-first `if/else if` 会在合法 dual-enqueue 组合上屏蔽 matching response 并产生 false fatal。
- P1：0。M741 的两个 P1 均已闭合。
- P2：0。
- 允许：作者制作一个新的 TB-only 修复身份，将 forward/response 的 token matching 解耦，再交一次独立静态 hammer。
- 不允许：基于 r6 制作 r12 runner 候选；启动 VCS/simv/DC/PT；修改 top/SVA/macro/foundry；形成 RTL verified、timing、PPA、speedup 或论文 headline 结论。

## 冻结检查

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
