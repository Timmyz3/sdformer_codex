# M209 独立打铁评审 r1

结论：**60/100，`FAIL_P0_M207_BANK48_DEADLOCK__RETAIN_NONTRUNCATING_OPPORTUNITY_MODEL`**。

我没有导入生产 analyzer，改用 `numpy.unpackbits(bitorder=little)` 解全部 120 份冻结
H67 FC2 payload，并以另一份 list queue + two-slot 状态机、4 workers、4096-token chunk
逐 token 重算。软件递推数字全部复现：5,580,000 tokens、143,894,510 events、
18,869,376 descriptors、6,523,707 windows、92,878,814 cycles；四 stage 为
23,685,015 / 15,610,475 / 38,740,976 / 14,842,348，tail census、
3,716,056 次 terminal collapse、queue max 7 和 descriptor hold max 672 全部一致。

但这只能证明**非截断机会模型**。M207 的 `descriptor_bank_sum` 只有 5 bit，合法的
descriptor4 packet 可在单 bank 放入 `4*12=48` 个 event。Synopsys VCS 反例中 48 被截为
16，bank ledger 先归零后下溢为 224，最终 bitmap 尚有事件而 `candidate_count=0`，跑了
192 groups 仍 `token_done=0`。因此 M209 的 92,878,814 不再是 M207 RTL-semantic
cycles；仓库里的 revocation 是正确的。1.234355878× 还混用了 analytic S1/F1/W1
分子和另一保真度的控制分母，即便没有死锁也不能称 speedup。

stage0 的差额得到精确拆分。M207 非截断模型比 M203 多 2,266,533 cycles；其中
1,694,275 个 two-window token 在 release edge 已有 closed successor，可由 M210 当前
handoff mux 直接回收；199,969 个 three-descriptor token 的第二个 partial window 此时
尚未 done-close；另有 372,289 个 one-descriptor partial token。三项之和恰为
2,266,533。故 M210 只能回收 1,694,275（stage0 到 21,990,740），不是
1,894,244；要收完后两项必须把 partial-done close/load bypass 一起做。

正在生成的 M211 已落盘为 91,184,539 cycles，恰好只比 M209 机会模型少
1,694,275，完全印证上述独立预测。M210 已用 6-bit packet sum 和 96-event/window 守卫
关闭功能 P0，并在 bank48 VCS 中完成 192 groups 和一次 token done；但另一评审发现当前
bank48 TB 与 r2 input manifest 的 hash 不同，下一次准入前仍应按当前 exact input 重跑并
重封。

最大 hold 672 的冻结 witness 是 stage3
`sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.fc2`、
`calls/s04_m28_c151.activation.le.bitpack` 的 token 480：32 descriptors、1,281 token
cycles。synthetic calibration 的 hold 最大仅 40，建议把这个 exact token 单独喂给 VCS。

机器可读裁决见 `m209_independent_hammer_review_r1.json`；独立全量重算见
`independent_replay_m209.json`；stage0 mask/cycle 拆分见 `stage0_gap_census.json`。
`docs/359` 未修改，SHA-256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
