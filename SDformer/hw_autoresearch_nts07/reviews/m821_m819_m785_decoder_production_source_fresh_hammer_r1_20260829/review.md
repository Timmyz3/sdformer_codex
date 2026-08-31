# M821｜M819 decoder delegation-compat fresh source hammer

结论：`PASS100_M819_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY`，100/100，P0/P1/P2 = 0/0/0。

M819 的 additive compatibility repair 闭合了 M817 的唯一 P1：正式 attempt 保留 M819 outer schema 和 runner/driver/candidate/release SHA 绑定，但 status 精确使用冻结 M809 body 唯一接受的 `CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY`。独立临时 traversal 真正进入 exact-SHA M809 `run_production()`，通过 parent receipt check，并在 `output.mkdir` 前受控停止；0 schedule row、output absent、无 identity drift。controlled stop 和 delegated exception 两条路径都在 `finally` 中恢复 validator。

Python 3.10 与 3.6 均通过编译、self-test、12/12 tests、candidate validation；3.6 使用固定 SHA 的 dataclasses backport。runner 顺序为 no-clobber attempt publish → started=1 → explicit postconsumption phase → fallible postcheck → consumed-attempt preflight → production。独立 postpublish failure 注入得到四成员双封 quarantine；collision、destination symlink、stdout/stderr log symlink 均拒绝且不改目标。

wrong source SHA、canonical drift、old M815 token、alternate token、伪造 M817 PASS/release=true、future-release presence、pre-attempt absence 均 fail-closed。40+120、T10、96 lanes、245760 B、Acc24、3 ns、192 B/cycle、D1 charged/nonheadline 与唯一合法 K8/equal-service K1x8 headline 均未变。M811/M817 继续是负权威；M809/M815、M798 consumed attempt 和 docs/359 未改。

本 review 只授权作者创建一个绑定本双封 review 的 true release。它不授权 formal runner、production replay、正式 attempt/result/failure、周期/倍速、Table-A、VCS/EDA/license/GPU/remote。
