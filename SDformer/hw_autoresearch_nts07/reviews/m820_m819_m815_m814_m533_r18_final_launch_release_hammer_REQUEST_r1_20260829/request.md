# M820 final-launch hammer request

请由新的独立 reviewer 对 M820 true release 做最终静态打铁。请求本身不授权运行 live runner、VCS、simv、license query 或任何 EDA，也不允许创建 result/attempt。

最终 review 必须写入 runner 固定路径：

`reviews/m814_m533_r18_unit_delay_vcs_final_launch_release_hammer_r1_20260829/`

只有 PASS 100/100、P0/P1/P2=0/0/0，且重新跑过 Python 3.6 source-static、closure 三负例、外部命令白名单和 rc86 零副作用 stub，才能授权后续一次 exact no-argument functional VCS+simv attempt。

R17 始终是 consumed failure。即使 R18 功能 VCS 后续 PASS，也不得据此声称 cycle、speedup、PPA、energy、timing 或论文 headline。
