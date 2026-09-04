# M849：C1 R20 P2 epoch triplet source author handoff

## 结论

M849/R20 source-only package 已完成，尚未运行 VCS、simv、许可证查询或任何 EDA。R19 继续永久 `FAILED_DO_NOT_CITE`；本包只允许 fresh independent source hammer。

TB r8 到 r9 的可执行语义差严格为三行：

```text
build_reference(16'd3) -> build_reference(16'd14)
load_task(16'd3)       -> load_task(16'd14)
wait_done(16'd3)       -> wait_done(16'd14)
```

M847 的 minimal-successor 文字只列了前两行，漏掉 `wait_done`。照抄两行会让 DUT 正确返回 epoch 14，而 TB 永远等待 epoch 3，最终触发 20000-cycle watchdog；因此 R20 按可执行语义把三个 P2 epoch consumer 同步为 14。没有插 reset，正常序列仍为 1/2/4/10/11/12/13，P2=14 严格单调。

RTL r2、SVA r2、macro adapter、binding plan、foundry UNIT_DELAY 模型、13 normal covers、P2 两项门、held-final、六攻击和最终 PASS token 均冻结。生产 simv 命令仍唯一为 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`。

Source tests 已通过：exact TB reconstruction；35 个 custom function/281 个 call/21 个 host command closure；delete/rename/stale 三个负变异；fake simv fast/TERM/KILL/tee/双封且无 orphan；pre-mkdir rc86 且 VCS identity、license、compile、simv、result mkdir 全为 0。

下一步只能由 fresh reviewer 按 M850 request 做 source hammer。只有 PASS100 且 P0/P1/P2=0/0/0 才能进入 candidate hammer，仍不能直接 launch。

`docs/359` 未修改，SHA 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
