# M605｜M579 technically non-runnable true-v4 template + launch candidate author handoff

日期：2026-08-28  
状态：**AUTHOR CANDIDATE ONLY；launch_now=false；fresh M606 hammer required；NO CPU RUN。**

## 交付

- 非生产 execution-template candidate：
  `contracts/m605_m579_true_v4_execution_contract_template_candidate_r1_20260828.json`
  - SHA256 `cdd2fdd07f5b5adfdec32d66ec1cd52fed7e8ed61f2a0eba3c155a3a9ea75a65`
  - schema 故意不是 production v4；M601 runner/analyzer 实际调用返回 `execution contract schema drift`，
    attempt 前拒绝。
- launch-admission candidate：
  `contracts/m605_m579_true_v4_launch_admission_candidate_r1_20260828.json`
  - SHA256 `55b3c951df3714a964836e13b3d5bc07f043b7deb74fde95826e44a0fba09c5e`
  - 当前权限为 `launch_now=false/run_cpu=false/max_attempts=0/execution_release=false`。

两项均有 member sidecar 与 outer sidecar。仓库内 production 路径
`contracts/m601_m579_paft_control_single_port_product_capture_execution_contract_r4_20260828.json` 仍不存在，
result/attempt/consumed/quarantine/PID staging 也都不存在。

## 冻结链

template 逐字节复制 M601 candidate 的精确 15-input mapping；future validator 必须重哈 15 inputs 与 80 packed
payload，正式 record 仍为零后才可消耗 attempt。M601 analyzer/runner、source contract/candidate 与 M603 PASS100
manifest/outer seal 全部 exact-SHA 锁定。chunk-major、M43/M504/M505、DMA=160、tail=2、commit=96,000、
8 blocks、九行 213,376 B 容量、M255 三 accuracy scope 与 64 帧 PAFT 退化 1.0189020311889285% 均不变。

## 资源与执行门

future release 只允许一次 80-record CPU replay、最多 3 workers、spawn。启动前必须重新做三次 live snapshot：
commit headroom >=48 GiB、MemAvailable >=128 GiB、SwapFree >=32 GiB、cgroup clean、零同 UID M579/CPU-DSE/
Synopsys/VCS/simv 冲突；随后 exact runner `--preflight-only` 必须再次通过。

冻结 M601 runner **不实现**上述 memory/collision gate；candidate 明确记录
`runner_enforces_memory_or_collision_policy=false`。因此该门在最终 release 中只能表述为 root live precheck，
不得冒充 runner 内建保护。author snapshot 只读通过，但不是未来 launch admission。

## 后续唯一合法顺序

1. fresh M606 candidate hammer 得到 score>=95、P0=P1=0；
2. root 同步生成 production true-v4 contract 与 true release（M607 identity），并双封；
3. fresh independent true-release hammer 再审；
4. root live resource/collision precheck + exact runner preflight 后，最多调用一次；
5. raw result 仍需 fresh independent result hammer。

本 author 没有运行 formal 80-record CPU、GPU、EDA 或 remote，没有创建正式 result/attempt。`docs/359` 未改，
SHA 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

