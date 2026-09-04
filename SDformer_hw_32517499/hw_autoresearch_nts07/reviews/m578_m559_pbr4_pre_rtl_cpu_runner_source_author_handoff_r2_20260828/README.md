# M578｜M559 PBR4 r5 repaired immutable CPU source author handoff

日期：2026-08-28  
状态：`R5_REPAIRED_SOURCE_ONLY__RUN_AUTHORIZED_FALSE__FRESH_STATIC_HAMMER_REQUIRED`

本交接响应 M574 对 r4 analyzer 的 `24/100, P0/P1/P2=3/2/1` 失败裁决。r4 冻结字节未覆盖；新增
r5 analyzer、runner、source contract 与 future schema。没有运行正式 CPU/runner，没有读取真实 M511 或
decoder weight payload，没有创建 result/attempt、launch admission/authorization/wrapper，也没有运行
RTL、VCS、DC、PT、PTPX、Formality、训练、GPU 或远端。

## 修复内容

- `CycleLedger.step()` 每个 edge 只收一个 primary class；ready xorshift 在所有 charged cycle 上推进，并用
  GF(2) jump 保持长 run 的精确 recurrence。四架构共享 source/frontier、M218 六 slice、weight refill、
  psum/directory/backing、dense output 和 r4 terminal FSM。
- A1-OSG/PBR4 使用显式 `phase,index` 的 4x4 context，A1-OSG 每个 bundle 都收 retire；SC8/ISO8
  constructor、PBR4 bundle epoch 与 OSG close/block drain 均为不可选择路径。27 个 primary class 均有明确
  success/stall/fault 收费路径。
- frozen decoder weight package 必须是四层 `COUT_CIN_KY_KX` signed INT8；binary source 恒为 typed
  `+1/source_sign=0`。每 contributor 依冻结顺序做 Acc24 modulo add，同时建立独立 reference accumulator，
  commit sequence 与 padded 384-B output data 都生成 SHA；mismatch 不再写常量零。
- result gate 同时合取 exact mismatch、ratio-of-sums、每 sample、weight active/refill、无 hidden state与
  PBR4!=OSG group/RMW/commit；support-only 单独要求每 sample>=1.0 且 psum traffic>=30%。
- preflight 重验 execution/source contract triples、future schema、M562/source/candidate/final review、六个
  goldens、canonical wrapper path+SHA+parent PID/starttime/cmdline、输入目录与语义 receipt。A1-only receipt
  在 PBR4 前后各 rehash 一次。
- exact 696.24M raw bit、926.88M block replay、11.04M dense destination、1600 row 均进入执行断言。
  当前本机缺 canonical M511/verification/r2 weight package时，未来只允许 preflight 在 attempt 前硬失败；
  不存在 synthetic data fallback。
- 全局 `try` 从 attempt 创建前开始；任一 post-attempt 异常都重封 consumed-attempt failure，并把 staging
  双封到唯一 quarantine；canonical result 保持 absent。

## 本 author 的轻量检查

仅执行 `/usr/bin/python3` 3.6.8 AST、`bash -n`、future schema `jq` 和 synthetic
xorshift/Acc24/M523/terminal self-test。自验 PASS 行为
`PASS M578 M559 repaired immutable analyzer static self-test`，不是正式 analyzer 结果。

下一门必须是 fresh independent、read-only source static hammer。P0/P1=0/0 才可允许后续
launch-candidate review authoring；即使 PASS 也不授权执行。

