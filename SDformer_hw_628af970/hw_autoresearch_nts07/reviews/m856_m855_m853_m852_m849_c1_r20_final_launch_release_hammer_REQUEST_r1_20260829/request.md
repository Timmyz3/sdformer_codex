# M856：M849/C1 R20 fresh final-release hammer request

请由不同于 M855 release author 的 fresh reviewer 对 exact M855 conditional release 做最终静态打铁。本请求不授权 live runner、VCS、simv、license、result/attempt 或任何 EDA。

必须固定 M852 source hammer 的 `e1cf9591...` / `6857efec...` / `ff7a616a...` 与 M853 candidate hammer 的 `85c42423...` / `5c9b13c...` / `3c0e85e6...` 双封身份。最终 review 必须写入：

`reviews/m849_m533_r20_unit_delay_vcs_final_launch_release_hammer_r1_20260829/`

只有 PASS100、P0/P1/P2=0/0/0，且 fresh 重跑 TB epoch-triplet、closure 三负例、timeout fake suite 与 rc86 零副作用 stub 后，才可使 M855 conditional release 生效一次 no-argument foundry-UNIT_DELAY functional VCS+simv attempt。

生产 simv 命令必须仍为 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`。即使未来 functional PASS，也不得提升 cycle、speedup、PPA、energy、timing 或论文 claim；M528 1.746753× 在此之前仍仅为 CPU same-ledger。
