# M838 request：M837/C2 R22 identity-compat source hammer

请由 fresh independent reviewer 对 R22 做 receipt-blind、source-only 打铁。本请求不授权 VCS、simv、license、EDA、formal identity 或 release。

核心攻击面是旧 M826 launch-chain identity 泄漏：reviewer 必须实证 M834 R21 的 PASS100 status 与四键 target 精确通过，同时拒绝旧 M826 status、错 M833 status、三键、缺键和额外键；M832 已判 spent 的 M826 release 不得以任何形式复用。

双版本必须重跑 atomic 12/12、final-auth 8/8、Unicode 5/5、R22 identity 11/11、closure、synthetic positive chain，以及 actual runner outer-C wrong-SHA rc3 / positive rc86 零副作用。还须核实 M803、五档周期、四 receipt、15 键授权、12+1 局部 C.UTF-8、VCS/simv outer-C 均未改。

只有 PASS100 且 P0/P1/P2=`0/0/0` 才能授权另一作者制作一次 R22 true release；reviewer 本轮不得直接 release 或 launch。
