# M366 H67 ep35 ATLIF remaining-budget S10 preflight

M366 已把 M360 放行的 G12/ATLIF 流式捕获实现到可启动状态，但没有抢占 A800 上正在运行的 MVSEC valid。当前结论是 `READY_BUT_BLOCKED_BY_REAL_GPU_WORK`，不是 S10 性能结果，也不是 RTL 放行。

冻结论文身份是 H67 ep35、`no_running` BN。PAFT ep4 只作为诊断证据；M193 已拒绝其 hardware-accuracy promotion，M366 不混用 PAFT 权重或结果。

## 已完成

- exact-SHA 合同绑定 14 项身份输入、冻结 checkpoint/config、M248 S10 hook、DP-TME 权重/bias/threshold 流、M360 和 M193。
- 流式 hook 覆盖 81 个 live ATLIF site（45 个 T10、36 个 T2），不会落盘全分辨率 tensor。
- 重算 signed-Q8/Acc24 `resolved_at_k`、term skip、32-lane compaction issue cycles，并审计 range、nonfinite、bound、overflow、integer early-decision mismatch 和 float bridge mismatch。
- 本地与远端 dry-run 均通过；禁用 CUDA 的远端深层 CPU smoke 对 bound、integer decision 和 float event 对照均为 0 mismatch。
- smoke 首次暴露 scalar threshold reshape 错误；修复后才冻结当前脚本 SHA。
- runner 先做 exact-SHA dry-run，再要求四次连续 10 秒的真正空闲快照；任一 GPU context 或 ML train/eval/valid/profile 进程都会拒绝启动。

## 当前阻塞

2026-08-25T14:55:03Z，A800 有一个 1636 MiB compute context，MVSEC `eval_MV_flow_SNN.py --mode valid` 与 wrapper 都在运行。`outdoor_day1` 为 2425/2755；wrapper 默认还会串行跑 `indoor_flying1/2/3`，对应数据均存在。因此不能把瞬时 0% GPU utilization 当成空闲。

## 晋级边界

S10 必须同时满足：0 integer early mismatch、0 bound/range/overflow violation、至少 35% term skip、至少 25% 可执行 issue-cycle reduction、fixed-context 至少 1.03x。即使这些指标全过，也只能进入独立能量审计；suffix metadata/config、比较与 compaction 的净能量未证明为正前，M366 固定为 `NO_GO_RTL`。

本里程碑不承认 S10 opportunity、RTL、VCS、Synopsys PPA、能量、系统加速或摘要头条。
