# M836：M835 / C1 R19b exact-edge-count fresh source hammer

## Verdict

**PASS，100/100，P0/P1/P2 = 0/0/0。** 本轮只审计 M835 对 M831/R19 `require_regular_sha` 计数的加法修复；没有执行 VCS、simv、许可证查询、HDL compile、result/attempt/release 或任何 EDA。

M835 的修正成立：exact runner 中有 **95 个 logical / 95 个 unique edge = 94 个单行调用 + 1 个两行 continuation**，重复为 0。唯一 multiline edge 是 1125--1126 行的 `docs/359` SHA 绑定。旧的“94 total”已撤回，94 只保留为 single-line 子计数。

## 独立证据

- Python 3.6 编译、synthetic self-test、exact runner 重放均通过；parser 排除了 heredoc payload，并合并反斜杠 continuation。
- 未导入 M835 parser 的独立状态机交叉得到同一组 `95/95/94/1/0`。
- 四个负例均拒绝：94 edges、96 edges、duplicate edge，以及保持 95 logical edges 但将 `docs/359` continuation 压成单行的 missing-multiline 变异。
- M831 runner、top r2、SVA r2、TB r8、macro adapter/binding、foundry UNIT_DELAY model 和 `/usr/bin/timeout` 身份全部重算一致。
- TB source-static 通过；保留 13 个 normal minima、P2 `(1,2)`、held-final、六个 protocol attacks 及其相位顺序。
- function closure 为 34 definitions / 266 calls / 21 commands；delete、rename、stale 三个变异全部拒绝。
- fake simv 得到 fast `(0,0)`、TERM `(124,0)`、TERM-to-KILL `(137,0)`、tee `(0,7)`，无 orphan，timeout receipt 双封通过。
- pre-mkdir dry-run 以 rc86 停在 live probe 边界，VCS identity、license、compile、simv、result mkdir 全部为 0。
- M833 的 98/100、P2=1 negative review 继续作为绑定 authority；M835 只修复该 exact-count finding。

## Authorization boundary

本 PASS 只授权下一位 fresh reviewer 做 additive admission integration review。它**不直接授权** candidate hammer、release、VCS/simv launch 或任何 EDA，也不产生可引用的 RTL、cycle、speedup、PPA、energy 或 system headline。

