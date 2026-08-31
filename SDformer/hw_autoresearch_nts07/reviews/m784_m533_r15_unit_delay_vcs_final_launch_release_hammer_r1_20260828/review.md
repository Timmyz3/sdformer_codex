# M792：M784/M533 R15 最终发射打铁

结论：**PASS 100/100，P0/P1/P2 = 0/0/0**。仅授权固定 R15 身份的一次 VCS compile 与一次 simv；其余执行均未授权。

本评审没有调用 runner、VCS、simv、`vcs -ID`、`lmutil`、许可证服务器或任何 HDL/EDA 工具。它只做了固定证据图、真实 runner heredoc 和共享主机状态的只读核验。

关键闭环：

- release `6c3d4a1f...`、request `f140bae3...`、runner `0bff3424...`、source `d426deaf...`、candidate `114412ba...` 与两级前置 hammer 全部双封通过。
- R15 的 67 条 `require_regular_sha` 全部解析为普通非符号链接文件且 0 mismatch；移除三条 M782 边后，原 R14 64 条边的顺序与值逐项完全相同。
- 从真实 runner 抽取 M770 Python heredoc：sealed 正例通过；删除 `decision.r14_launch_authorized_now` 或换回错误旧键均以 `M770 launch boundary` fail closed。
- M782 明确将 R14 release 永久撤销；M779 不可再用于发射。R14/R15 result 与 R15 preflight 临时目录均不存在。
- `UNIT_DELAY` 编译定义恰好一次；`+notimingcheck` 和 `+no_notifier` 均不存在。R7 覆盖、P2 read-response、6 类协议攻击、task/global watchdog、资源 final-ack 与成功/失败双封门完整。
- 三次共享主机采样均高于阈值；同 UID EDA/VCS/simv 冲突为 0；session/user cgroup failcnt 均保持 0，under_oom/oom_kill 均为 0。
- `docs/359` SHA 保持 `dedde7ce...`。

唯一授权命令：

```bash
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 \
  VCS_ARCH_OVERRIDE=linux \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
  LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m784_m533_m528_dead_write_only_1rw_unit_delay_r15_exact_sha.sh
```

本 PASS 只准入一次功能 VCS 尝试；在 runner 产生双封 `RUN_COMPLETE` 前，C1 RTL、周期、PPA、能量和论文 claim 仍全部为 false。
