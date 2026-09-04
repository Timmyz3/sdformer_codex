# M566｜M518 r4 per-point setup/area DC minimal source-only handoff

日期：2026-08-28  
状态：**SOURCE ONLY；fresh independent static hammer required；launch_now=false**

本包只修复 M563 的两个 P1，未调用 DC、VCS、runner、远端或大 CPU 任务，
也没有生成 admission、result、attempt 或 paired comparison。

## 两项且仅两项修复

1. runtime_final 采样现在更新或清零 <48 GiB 连续计数，并在最终门同时
   执行第三次判断；<40 GiB immediate 与 Mem/Swap/cgroup/collision 门保持。
2. runner 在任何 point preflight 目录、result workspace 或 attempt marker
   创建前，先将 M555 review、r2 quarantine、r2 attempt 的 live outer-seal
   文件 SHA 与 contract 冻结值逐项比较，并递归验证每个 SHA256SUMS 及
   SHA256SUMS.seal.sha256。任一失败 pre-attempt 退出。

其余 r3 边界全部冻结：per-point 隔离、paired schema-only、50/1175 双口径、
结构化 postcheck、一次 compile_ultra、零 hold fix、actual-exec PID tuple、
失败双封、64/128/32 preflight 与原 Tcl/RTL/SDC/filelist/DB 身份均不变。

## 冻结身份

- r4 runner: 5240712aeaf5dd3b50d68fb29389b1be5d27ba0611c7c50b9d744185c63a00c8
- frozen r3 Tcl: 8f189fc861722f4d5e9005c9301cd01d4fd3c515f5942287df0c58e8e00119e6
- r4 contract: fab51d46ddabff5254943cd1646be107f3fa173447f26cfd3f863b3657e65b5f
- contract outer-seal file: 6ca5402001209273d6d39e0926b8fec828f68182f8ae7c3905aef06e720cbd12
- docs/359: dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

只有 fresh reviewer 给出 P0=0、P1=0 后，root 才可另行逐点生成 one-shot
admission。本 handoff 不授权任何运行。
