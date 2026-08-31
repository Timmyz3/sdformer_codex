# M572｜M518 r4 两点 DC launch-admission candidate author handoff

日期：2026-08-28  
状态：**两个 launch_now=false 候选已双封；fresh candidate hammer required；NO RUN**

本包只创建 Fixed 与 rank3 两份彼此独立的点级候选。两者均为
`max_attempts=0`、`run_dc=false`，未调用 runner、DC、VCS、CPU 大任务或远端，
也未创建 result、attempt、最终 true release 或 paired comparison admission。

## 两点冻结身份

| point | top | candidate SHA256 | candidate outer-seal-file SHA256 |
|---|---|---|---|
| fixed | `m518_matched_fixed_t10_atlif` | `e83e2a47319a5fca165fb918adfb64659d1d968022aa946c52e8788bd5aa82a4` | `ef2624c318cc4c5f3d34181bc6c6ee29f94b214adcb35e6f0dbafe4f88514fc9` |
| rank3 | `m273_integrated_rank3_atlif` | `7c6fb69062707f542e310b9bcf2ab227ec0ee9397ada3d891e8dd8aea82f2958` | `9e41da07f22bb25998e680fe9508a865fa20af09f95f56a1f96bea4e9f8410ad` |

两点共同绑定 r4 runner `5240712...`、冻结 r3 Tcl `8f189f...`、contract
`fab51d...`、SDC、filelist、双 RTL、DC wrapper/actual executable、slow/fast DB，
并绑定 M568 PASS100 的 manifest 与 outer seal。公平口径为每点 50 个 source
declaration tuple 与 1175 个 DC bit port；两者不可混为同一计数。

每个候选分别冻结 canonical result/attempt 缺失、64/128/32 GiB preflight、
48 GiB 连续三样本/40 GiB immediate runtime 门、Mem/Swap/cgroup/同 UID EDA
碰撞 fail-closed、final ACK 门。任何候选或候选 hammer 都不能授权运行。

paired comparison 目前明确不存在，也不获授权。必须先让两点各自产生双封结果，
再分别通过 fresh independent point receipt hammer（P0=0、P1=0），之后才能另行
生成 paired comparison admission；point runner 不得生成比较结论。

`docs/359` 未改，SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
