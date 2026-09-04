# M842 C1 R19b conditional-release author handoff

M842 已生成一次 conditional true release，绑定 byte-frozen M831/R19 runner、原 M831 source/candidate、M835 exact-edge repair 与独立 M836 PASS100。

由于 runner 固定引用 legacy `source_static_hammer` 和 `candidate_hammer` 路径，M842 release integrator 在这些路径写入了 compatibility authority。它们明确不是新的独立 hammer；唯一独立 source authority 仍是 M836。

生产命令保持 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`。本作者包没有调用 live runner、VCS、license 或 EDA，没有创建 result/attempt。release 在不同 reviewer 完成 M843 fresh final hammer PASS100 前不生效。
