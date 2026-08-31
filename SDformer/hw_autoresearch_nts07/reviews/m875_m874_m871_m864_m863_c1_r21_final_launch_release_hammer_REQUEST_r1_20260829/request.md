# M875：M863/C1 R21 fresh final-launch hammer request

请由不同于 M874 release author 的 fresh reviewer 对 exact R21 conditional release 做最终 source-only 打铁。reviewer 禁止运行 live runner、VCS、simv、license query、任何 EDA 或 workload，也不得创建 R21 result/attempt/quarantine。

PASS 输出必须写入固定路径 `reviews/m863_m533_r21_unit_delay_vcs_final_launch_release_hammer_r1_20260829/`，使用 runner 固定的 schema，达到 100/100、P0/P1/P2=0/0/0，精确绑定 release `2ee62a30...`、runner `456a07a0...`、candidate `b8b33c58...`，并重跑 typed authorization、source/event、closure 三负例、timeout fake suite 与 rc86 零副作用 stub。

即便 M875 PASS，caller 仍须独立复核 final-hammer 双封、计算并 pin 该包实际 outer-seal-file SHA 后，才可在 clean env 中执行 exactly one no-argument R21 runner。request 和 reviewer 均不获得 live launch 权限。

未来 functional PASS 不得提升 cycle、speedup、timing、PPA、energy 或论文 claim；M528 1.746753× 在此之前仍仅为 CPU same-ledger。
