# M834 request: independent C2 R21 Unicode source hammer

请对 M833 R21 做 fresh、receipt-blind、source-only hammer。不得运行 VCS/simv、查询 license、启动任何 EDA、创建正式 attempt/result/quarantine，也不得自行制作 release。

核心攻击面：外层 `LANG=C LC_ALL=C` 下，未包装 Python 3.6 必须在真实绝对中文 `docs/359...` 路径失败；`PYTHONUTF8=1` 在本机仍必须被证明无效；R21 runner-local `C.UTF-8` 必须通过。逐个检查 12 个 guard 调用和 1 个 inline writer，无任何直接 Python 执行遗漏；同时证明 locale 未全局 export，M826 的 `license_gate` 与 `compile_and_run` 逐字未改。

双版本重跑原 atomic 12/12、final auth 8/8、四 receipt `false,false,true,true`、Unicode 5/5、closure 与 outer-C dry-run。确认 M826 release 已失效但 attempt 未消费，M826/M833 正式身份均不存在，M803 与五档 cycles 冻结。

只有 100/100 且 P0/P1/P2=0 才可授权一个新的 R21 true-release author；本 request 本身绝不授权 live VCS。
