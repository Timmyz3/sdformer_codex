# M610｜M579 production true-v4 contract + one-shot true release author handoff

日期：2026-08-28  
状态：**TRUE RELEASE AUTHORED；STILL NOT EXECUTED；fresh M611 hammer required。**

## 产物

- production contract：`contracts/m601_m579_paft_control_single_port_product_capture_execution_contract_r4_20260828.json`
  - SHA256 `29a471dc489da4895e38b01700a4e101a5055bcbfd37323025a0762958011bb0`
  - outer-seal-file SHA256 `faf54c358b28967b33823ef53c95c03293e20790f8248e7ef268882c55299d79`
- one-shot true release：`contracts/m610_m579_paft_control_single_port_product_capture_true_release_r1_20260828.json`
  - SHA256 `b26bcb2ed9665e561ea84cad8038ff97f2406ac3b33be90538c88d4240c7c1f6`
  - outer-seal-file SHA256 `baa860bdcf6c9143348ff0f645a80b2ab893408f5ebec6ede5328645f32b5e52`

两者同步生成并双封，均绑定 M601/M603/M605/M609 exact identities。contract 使用 production analyzer 唯一接受的
true-v4 schema，权限为 `launch_now=true/run_cpu=true/max_attempts=1`；release 仍明确
`still_not_executed=true`，作者没有调用 runner `--execute`。

## 作者机械验证

仅直接调用 analyzer `--validate-contract-only`：15/15 inputs 与 80/80 packed payload 重哈通过，
`formal_trace_records_processed=0`，没有 result/attempt。未调用 runner execution。

author 资源现场做了三次、间隔 2 秒的只读快照，最小值为：

- commit headroom `83,647,820 KiB`，门为 `50,331,648 KiB`；
- MemAvailable `416,265,720 KiB`，门为 `134,217,728 KiB`；
- SwapFree `57,212,156 KiB`，门为 `33,554,432 KiB`；
- session/user cgroup `failcnt/under_oom/oom_kill=0`；同 UID collision=0。

现场过门，所以 release 保持 true。但是冻结 M601 runner 不实现 memory/collision gate；M611 PASS 后，root 必须在
调用前重新做 live check 和 exact runner `--preflight-only`，不得拿作者快照替代。

## 冻结执行与边界

唯一未来 invocation 固定为 3 workers、80 records、同一 result/attempt/consumed 坐标。runner 的 lexists/no-symlink、
same-parent quarantine、terminal 15-input/80-payload rehash、member/outer seals 与
`renameat2(RENAME_NOREPLACE)` 保持。一次 attempt 成功后 consumed 坐标阻止第二次；失败则进入双封 quarantine。

accuracy 必须同列：valid825 单 seed +0.5730215096601543%；十帧 5 win/5 loss；完整 64 帧 PAFT 退化
1.0189020311889285%；无 Pareto。九行容量仍是 213,376 B / 245,760 B，macro PPA/energy open。

当前正式 result/attempt/consumed/quarantine/PID staging 全部 absent；formal result、RTL/VCS/PPA/energy/system
headline 全 false。raw result 即便生成仍需 fresh independent result hammer。

`docs/359` 未改，SHA256 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

