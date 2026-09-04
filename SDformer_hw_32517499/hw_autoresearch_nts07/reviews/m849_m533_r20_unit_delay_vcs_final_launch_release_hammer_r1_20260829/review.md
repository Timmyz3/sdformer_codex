# M858：M849/C1 R20 最终发射打铁

结论：**PASS 100/100，P0/P1/P2 = 0/0/0**。

本次独立审计固定 M855 release `2086c964...`（manifest `4980d974...`，outer `d90a8f51...`）、runner `a02899d3...`、M852 与 M853 的双封 PASS100，并复跑所有 source-only 发射前测试。TB r9 相对 r8 仅有 P2 的 `build_reference/load_task/wait_done` 三个 epoch 字面值由 3 改为 14；normal frontier=13，P2 三元组为 14/14/14，13 个 normal cover、P2 minima 1/2、held-final 与 6 个协议攻击均未放宽。

函数闭包为 35 个定义、281 个调用、21 个固定 SHA 外部命令；删除定义、重命名定义和注入 stale 短名三种攻击全部被拒。伪 timeout 覆盖 fast/TERM/KILL/tee，返回码分别为 `(0,0)`、`(124,0)`、`(137,0)`、`(0,7)`，TERM/KILL 后均无孤儿。clean-env pre-mkdir stub 按固定事件序列运行，以 rc86 在 live VCS/license 边界前停止，VCS identity、license、compile、simv、result mkdir 均为 0。

因此，仅授权 root 以 SHA 为 `a02899d3...` 的 exact runner、绑定 SHA 为 `2086c964...` 的 release，执行恰好一次 functional VCS compile 和一次 simv。runner 不接受参数；运行时 identity/license/collision/resource/terminal gate 仍必须全部通过。任何 attempt 一旦被消费都不得重跑、续跑或改名。

本 hammer 不证明 RTL、coverage、timing、cycle、speedup、PPA、energy、full-network 或 paper claim。M528 `1.746753x` 仍只是 CPU same-ledger 数字，必须等正式 VCS 结果再改变边界。
