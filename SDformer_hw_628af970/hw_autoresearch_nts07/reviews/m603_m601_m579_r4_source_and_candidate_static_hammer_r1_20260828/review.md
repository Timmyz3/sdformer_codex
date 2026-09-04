# M603｜M601/M579 r4 source + launch-now-false candidate fresh static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_STATIC_HAMMER__NO_FORMAL_CPU_GPU_EDA_REMOTE`  
裁决：**PASS_SOURCE_AND_CANDIDATE_STATIC，100/100，P0/P1/P2 = 0/0/0。**

## 一、裁决

M601 r4 仅在 exact-SHA M594 r3 外增加 canonical-path hardening；没有改变冻结支持算术、chunk-major
任务顺序、M43/M504/M505 recurrence、15-input/80-payload 身份、M255 accuracy disclosure、九行容量或 claim
boundary。M598-P2-01 已关闭：runner 对 result/attempt/consumed/PID staging 使用 `-e OR -L`，对 quarantine
坐标拒绝任何既有目录项；analyzer 使用 `os.path.lexists` 并显式拒绝 symlink；result/quarantine/attempt 的最终
发布仍为 `renameat2(RENAME_NOREPLACE)`。

独立实际运行 immutable runner `--preflight-only` 通过：冻结 Python/NumPy、spawn child、M43/M504/M505、
八行 recurrence（ideal issue=6、liveness=8）、chunk-major anchor `[0,47,94,141]` 与 15-key 集均正确，正式
record/result/attempt 为零。独立临时 true-v4 contract 仅运行 `--validate-contract-only`，15/15 inputs 与
80/80 payload 重哈通过，正式 record 为零。

`launch_now=false` candidate 的 input mapping 与 source contract 精确相同；其 schema 故意不被 production
analyzer 接受，`launch_now/run_cpu/max_attempts/execution_release=false/false/0/false`。对 candidate 调用
runner `--execute` 在 attempt 前因 schema drift 拒绝，没有产生正式工件。

本 PASS 只授权下一步另建 exact-SHA true-v4 execution contract，并再做独立 true-launch admission/release；
**不授权**本 review/candidate 直接运行 80-record CPU。

## 二、M598-P2-01 closure

- dangling/live symlink fault：contract、canonical result、attempt、PID staging 均在 attempt 前拒绝。
- regular-file/directory confusion：result/attempt/consumed/quarantine 坐标均 fail-closed；terminal result 对
  dangling symlink 和 directory-as-file 均显式拒绝，CSV 走同一个 `require_regular_nosymlink`。
- cleanup 静态闭合：attempt 与 staging 搬入同一 unique quarantine staging，failure receipt 对 final result
  分列 `lexists/is_symlink/is_directory`，对 contract/runner/analyzer 保留 start/current SHA 与 lexists/symlink；
  tree 在 member/outer seal 前拒绝 symlink，quarantine final 为 NOREPLACE。
- success 静态闭合：terminal 复跑相同 15-input/80-payload validator；result member/outer seal、pre-publish
  identity、result NOREPLACE、attempt completion seal 与 consumed NOREPLACE 均保留。内部 terminal-receipt
  temporary rename 是 M594 r3 已有的 private staging 操作，r4 未新增 canonical 覆盖式 `mv`。

## 三、独立 fault matrix

以下均为临时 validator/fault test，未运行 formal record：

- 少 key、多 key、历史 path/SHA 漂移、schema、launch/run_cpu/max_attempts、top-level/runtime analyzer/runner
  identity、output coordinate、错误 expected contract SHA：全部非零拒绝。
- candidate `--execute`：schema gate 拒绝，attempt 前失败。
- result dangling symlink、attempt live symlink、consumed regular file、result directory、attempt regular file、
  quarantine staging dangling symlink、quarantine final directory、PID staging dangling symlink：全部拒绝。
- terminal result dangling symlink 与 directory-as-file：全部拒绝。
- fault 后 canonical result/attempt/consumed 均不存在。

## 四、冻结证据与边界

- r4 只委托 frozen r3 SHA `c684ac4d...83136fd2`；r2/r1 effective `__file__` 均正确指向 r4 顶层，冻结
  r2/r1/M43/M504/M505 SHA 全部重验通过。
- task order 为 `[sample, operator, row-chunk, partition]`；20,304 tasks/operator；末 chunk 56 rows；
  DMA=160、tail=2、commit=96,000/sample、8 blocks 未变。
- M255 同列披露 valid825 单 seed +0.5730215096601543%、十帧 5 win/5 loss、64 帧 PAFT 退化
  1.0189020311889285%；`accuracy_performance_pareto=false`。
- M528 九行容量为 213,376 B / 245,760 B，余量 32,384 B；macro integration/PPA/energy 仍 open。
- arithmetic-work/local-cycle/PAFT-control activity increment 不相乘；system/RTL/VCS/PPA/energy/headline 全 false；
  r4 不是新性能结果。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

