# M836 request：M835 C1 R19b exact-edge-count source fresh hammer

请由与 M835 author 不同的 fresh reviewer 执行。禁止 VCS、simv、许可证查询、HDL compile、result、attempt、release 与全部 EDA；禁止修改 M835/M831 源和 `docs/359`。

## 必锤项

1. 重算 runner、M835 contract/candidate/parser/handoff、M833 negative review 的全部成员 SHA 和外层 seal，并做严格 JSON duplicate/nonfinite 拒绝。
2. 用 Python 3.6 编译并运行 continuation-aware parser 的 synthetic self-test；对 exact runner 必须得到 95 logical / 95 unique / 94 single-line / 1 multiline / 0 duplicate。
3. 唯一 multiline edge 必须是 runner 1125–1126 行、SHA 为 `dedde7ce...` 的 `docs/359` 调用；证明旧 94 只能作为 single-line 子计数，不能再称 total。
4. 重跑 TB r8 source-static、34/266/21 closure 正例与 delete/rename/stale 三负变异、fake timeout fast/TERM/KILL/tee/receipt、pre-mkdir rc86 dry-run。
5. 确认 runner、RTL/TB/SVA/foundry、`/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`、13 normal、P2、held-final、六攻击、资源与终态双封门均未改变或放松。
6. 确认 prospective review/downstream admission/release/result/attempt 均未先占，且本轮 side effects 继续为 0。

只有 100/100、P0/P1/P2=0/0/0 才 PASS。即使 PASS，也只授权下一位 reviewer 设计 additive admission integration；不直接授权 candidate hammer、release 或 launch。

固定输出：`reviews/m836_m835_c1_r19b_exact_edge_count_source_fresh_hammer_r1_20260829/`，包含 `review.json`、`review.md`、`mechanical_checks.txt`、`RUN_COMPLETE.txt`、`SHA256SUMS`、`SHA256SUMS.seal.sha256`。
