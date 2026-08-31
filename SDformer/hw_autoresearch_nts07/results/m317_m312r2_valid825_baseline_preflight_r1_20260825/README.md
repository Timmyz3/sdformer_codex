# M317：M312r2 valid825 baseline 独立 preflight

结论：M312r2 获得一次 GPU baseline 启动 `GO`；M313 仍为 `WAIT`，必须等 baseline 封存后生成新 exact-SHA contract 并再次复审。

本次未导入通用 wrapper/evaluator，也未启动 GPU。正确参数路径只运行到人为设置的 import sentinel。

M316 的 M312 启动级问题已经闭合：

- launcher hard-code 完整 M312r2 contract SHA，旧合同克隆在 wrapper import 前失败；
- 命令行只能是固定顺序的 8 个 `--option value` 对；
- 33 个 equals-form、重复、paired、未知、额外、乱序、缺值攻击全部拒绝；
- 下游 `sys.argv` 由 launcher 重新构造，不再直接转发调用者参数；
- result root 必须不存在，per-frame 必须严格为 `result_root/per_frame.csv`；
- receipt/profile 使用 strict JSON 和有限 Decimal AEE；
- receipt identity、admission、tau0 zero counters、825 行 ordered population、profile checkpoint/config/load audit 均完整核验；
- manifest 恰含 receipt、profile、per-frame、launch receipt 四项，manifest 后和 seal 后各 replay 一次。

启动必须保持如下参数顺序：

```text
--contract <pinned-m312r2-contract>
--config <pinned-config>
--checkpoint <pinned-checkpoint>
--path-results <non-existing-result-root>
--distance-threshold 0
--max-samples 0
--bn-policy running
--dump-per-frame <same-result-root>/per_frame.csv
```

评分 `97/100`；`P0=0, P1=0, P2=2`。两个 P2 是 raw path spelling 的极小 symlink race，以及 terminal seal 阶段未再次重复全部 runtime identity 哈希；现有 nested identity、precommit rehash 和双输出 replay 足以防止错误结果被封存，不阻塞本次 baseline 启动。

baseline 完成并通过 launcher 自身 seal 后，才能生成 M313 contract；M317 不准入任何 M313 candidate 运行。
