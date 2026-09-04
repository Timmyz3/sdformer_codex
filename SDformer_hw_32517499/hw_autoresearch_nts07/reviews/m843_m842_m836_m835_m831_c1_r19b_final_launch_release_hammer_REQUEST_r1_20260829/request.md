# M843：C1 R19b fresh final-release hammer request

请由不同于 M842 release integrator 的 fresh reviewer 对 exact M842 conditional release 做最终静态打铁。请求本身不授权 live runner、VCS、simv、license、result/attempt 或任何 EDA。

必须明确 provenance：真正独立的 source hammer 是 M836/`318d913a...`；runner 固定路径中的 source/candidate review 是 M842 release integrator 写入的 compatibility authority，不得描述为新的独立 hammer。

最终 review 必须写入冻结路径：

`reviews/m831_m533_r19_unit_delay_vcs_final_launch_release_hammer_r1_20260829/`

只有 PASS100、P0/P1/P2=0/0/0，且 fresh 重跑 exact-edge、TB static、closure 三负例、timeout fake suite 与 rc86 零副作用 stub 后，才可使 M842 conditional release 生效一次 no-argument foundry-UNIT_DELAY functional VCS+simv attempt。

生产 simv 命令必须仍为 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`。即使未来 functional PASS，也不得提升 cycle、speedup、PPA、energy、timing 或论文 claim。
