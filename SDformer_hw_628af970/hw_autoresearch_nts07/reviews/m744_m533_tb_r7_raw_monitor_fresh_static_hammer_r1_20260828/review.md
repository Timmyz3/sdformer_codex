# M744：M533 TB r7 RAW monitor 独立静态打铁

## 裁决

**PASS，100/100。** r7 以 TB-only 方式闭合了 M743 的唯一 P0：`forward_matches_raw` 与 `response_matches_raw` 分别要求 pending token、当前 execution epoch、consumer 和 parent 全部精确匹配；清 token 的优先级只发生在两个 matching predicate 之间，而不再发生在原始 event 之间。因此，同拍出现 matching macro response 与 unrelated direct forward 时，unrelated forward 不获得 credit，也不会屏蔽 matching response。

旧字段 `cov_stalled_raw_recovery` 仍只在 exact direct-forward 分支递增；response 只递增独立的 `cov_stalled_raw_response_recovery`。direct-forward 与 macro-response 两条 recovery 路径都保留独立 `>=1` minima。macro-read request 本身不计 recovery，unrelated/cross-token event 既不 credit 也不清 token；若 token 到任务 drain 仍未恢复，`RAW recovery escaped task` 继续 fail-closed。

r4→r7 的累计差异仍限定在 testbench 的 RAW monitor、observation-only trace、coverage counter/minima 和 coverage/PASS token。原 cleanroom 算术、queue pop→response→forward 顺序、queue identity/data、foundry response identity/data、per-epoch conservation、六类 protocol attack、20,000-cycle task watchdog、20-cycle attack watchdog及 3,000,000-time-unit global watchdog均未改。top r2、SVA r2、9×128 macro adapter、binding plan 和 checksum-identical foundry Verilog SHA 也保持 M737 冻结身份。

本评审允许作者制作**唯一一个 r12 runner 候选**，但不直接授权 VCS/simv/EDA 启动。r12 仍必须经过 exact-SHA runner source contract、独立 candidate hammer、launch release 和 final release hammer；运行后只能形成 `functional_vcs_only` 结论，slow-macro 时序仍未验证。

## 关键静态证明

1. `forward_matches_raw` = token pending ∧ expected forward ∧ token epoch=execution epoch ∧ token consumer=lookahead consumer ∧ token parent=lookahead parent。
2. `response_matches_raw` = token pending ∧ expected read response ∧ token epoch=execution epoch ∧ pending-response consumer=token consumer ∧ pending-response parent=token parent。
3. recovery 分支为 `if (forward_matches_raw) ... else if (response_matches_raw) ...`。所以 unrelated forward 令第一谓词为假，不会阻止第二个 exact response 谓词；unrelated response 同理不获 credit。
4. direct 分支递增 `cov_stalled_raw_recovery` 与 `cov_stalled_raw_forward_recovery`；response 分支只递增 `cov_stalled_raw_response_recovery`。三者在最终 coverage gate 中分别检查，direct/response 两条路径都不能被另一条冒充。
5. response 没有独立 epoch 字段，但 cleanroom 同时只允许一个 active execution epoch；token 又要求等于该 epoch，且 task drain 对未清 token 直接 fatal。因此 response 不可能跨任务被记账。
6. `RAW_OBS` 保留 token identity/age、current、两个 queue slot identity、pending identity、reserved、两路独立 sink-ready、合并 sinks-ready、forward、macro-read 和 read-response；只观察，不生成 oracle 状态。

## 机械复核

- r4 SHA256：`320901a07f9b01cb9cef334982a293cabfbd6e8f8b528cffd769e71a3c427c82`。
- r7 SHA256：`d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2`。
- r4→r7：47 行增加、15 行删除；逐 hunk 人工核查均属于 RAW monitor/trace/coverage/token。
- top r2 SHA256：`726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1`。
- SVA r2 SHA256：`b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b`。
- macro adapter SHA256：`8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783`。
- macro binding plan SHA256：`db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983`。
- foundry Verilog SHA256：`8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d`。
- M738、M741、M743 双 seal 重新校验均通过。
- r7 的 `$error` 数与 r4 相同（9）。RAW monitor 内删除了错误的 unrelated/cross-task immediate fatal，因而 `$fatal` 数由 27 降为 26；这不是删除数据/queue/attack/watchdog 检查，unrelated credit 由 exact predicate 禁止，跨任务 escape 仍由 task-drain fatal 捕获。
- 未运行 VCS、simv、iverilog、Verilator、DC、PT、CPU/GPU 实验或远端任务；因此本裁决仅是 source-static admission，不是语法编译、功能、时序、PPA 或性能结论。

## P0/P1 与授权

- P0：0。
- P1：0。
- P2：0。
- 允许：制作一个绑定 r7/top r2/SVA r2/adapter/foundry frozen SHA 的 r12 runner 候选，随后走独立 release chain。
- 不允许：凭本静态评审直接启动 VCS/simv/EDA；修改 top/SVA/macro/foundry；把 r11 改写成 PASS；声称 RTL/timing/PPA/speedup/headline 已闭合。

## 冻结检查

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
