# M842：M831/R19 admission-integration candidate hammer

**兼容 authority PASS，100/100，P0/P1/P2 = 0/0/0。** 本文件由 M842 release integrator 写入 runner 硬编码的 legacy candidate-hammer 路径；它不是新的独立 hammer。真正独立的 source authority 仍是 M836/`318d913a...`。原 M831 closed candidate 与 byte-frozen R19 runner、M836 exact-edge-count PASS100 及固定路径 compatibility source authority 的身份一致。

生产命令仍为 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`；foundry `UNIT_DELAY`、资源门、clean environment、functional-only claim boundary 均未改变。目标 result/attempt 仍不存在。

该 compatibility PASS 只允许生成一次 conditional true release。release 在 M843 请求对应的不同 reviewer fresh final hammer PASS100 前无效；本文件不授权 VCS、simv、license 或任何 EDA。
