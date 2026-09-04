# M545｜M542/M534 PBR4 pre-RTL CPU contract r2 修复交接

日期：2026-08-27  
状态：`R2_SOURCE_ONLY_REPAIR_COMPLETE__RUN_AUTHORIZED_FALSE__FRESH_STATIC_HAMMER_REQUIRED`

本交接只新增 r2 contract、future-runner schema、handoff、fresh hammer request 与双封文件。没有运行 CPU
analyzer，没有建立 runner/result/attempt marker，没有修改 RTL，没有运行 VCS/iverilog/Verilator/DC/PT/PTPX/
Formality/训练/GPU/远端任务，也没有修改 r1、M534 r2/r3/r4、`docs/524` 或 `docs/359`。

## 1. 被审对象

- contract：`contracts/m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r2_20260827.json`
- contract SHA256：`afdadd302cffdffedb34e0679c177424aa8fc9d7023e4fd9eecf7c4f5ff9bc63`
- contract member-sidecar SHA256：`57b5ecced98b433f1b0b2bc17da6faa81882ff3cc218833cc359a5ccb39951bf`
- contract outer sidecar：`contracts/m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r2_20260827.json.sha256.seal.sha256`

## 2. r1 三个 P1 的关闭方式

1. **T10 成为唯一执行分母。** canonical order 现在是
   `sample -> layer -> time 0..9 -> output_block -> source ordinal -> legal kernel`。每个架构都必须处理全部
   `S10 x 4 layers x T10`。冻结整数为：raw M511 `696,240,000 bit / 87,030,000 B`；block replay
   `92,688,000 bit/sample`、`926,880,000 bit/S10`；逐层/per-time/per-sample T10 logical/padded byte 与 dense
   destination 也写入合同。ready transcript 对每个 sample/layer/time 相同复位；time t 地址只有在全部三 beat、
   directory clear 与 epoch 状态退休后才能在 t+1 重用。
2. **numeric activity 与 source-sign 分类型。** numeric source `1` 是合法隐式 `+1`；独立 metadata
   `source_sign_bit=0` 才合法，`1` malformed；product sign 只来自 signed-INT8 weight。activity reference、
   descriptor hash 与 product miter 禁止复用一个字段解释两类对象。
3. **四架构语义不可由 runner 选择。** r2 规范性递归 import M534 r4→r3→r2 的 README+JSON 六个 exact SHA；
   precedence 是本合同 > r4 > r3 > r2。冲突、缺文件、SHA mismatch 或歧义都在 attempt 前 fail。合同还逐点
   冻结 SC8/ISO8/OSG/PBR4 的调度、prefetch、join、close、stall 与 cycle-increment 摘要；runner/launch/result
   identity 必须 pin 六个成员，不能本地替换算法。

## 3. 两个 P2 的关闭方式

- `out_valid && !sink_ready` 时 command/address/beat/data、内部 output/psum read、新 FINAL_OUTPUT request、
  beat accept/retire、transfer retire、directory-clear read/write/retire、cursor 与 owner 的 delta 全为 `0`；唯一
  状态变化是 `final_output_sink_stall_cycles += 1`，且进入严格 cycle conservation。
- 在任何 r2 文件写入前观测并冻结 `docs/524` SHA256
  `4f3ffe1c70027ef07efb3908c88876ec48cabc37ad8778150a6922f16c7abe0d`，交接时复核相同。该证明只覆盖
  本 author task 的起止字节一致性，不宣称此前 repository history。

## 4. 当前授权边界

- `run_authorized=false`；当前所有 CPU/analyzer/RTL/EDA/训练/GPU/远端/result count 为整数 `0`。
- future local CPU count 仍只是 `1`，所有 future non-CPU count 为 `0`；当前没有 runner 或 launch release。
- fresh static hammer 只有 P0/P1=`0/0` 才可 source-only PASS；即使 PASS，下一步也只能另行授权 runner source，
  不能直接执行 CPU 或开始 RTL。
- 当前没有 cycle、traffic、energy、PPA、system speedup 或论文 headline 结果。
