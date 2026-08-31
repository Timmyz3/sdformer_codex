# M542｜M534/PBR4 pre-RTL CPU 合同 fresh static hammer 请求

只做 fresh independent、只读静态审计。禁止运行 CPU analyzer、RTL、VCS、iverilog、Verilator、DC、PT、
PTPX、Formality、训练、GPU 或远端任务；禁止创建 canonical result 或 attempt marker。

重点审查四组问题：

1. 三种 A1 是否在 PBR4 不可见时先完成并封存，随后只选一个完整 S10 sum-cycle 最小的固定分母；
2. `FINAL_OUTPUT` 是否明确编码 beat 0/1/2、逐拍 ACK/反压/hold/退休/重复语义，且不存在免费 sink ACK；
3. final-output 是否有唯一 layer/output-block/y/x/channel/beat 到 `0x20000000` aperture 的地址和 byte 映射；
4. block-outer 重扫是否把 source bits/reads/logical bytes/padded bytes/cycles/stalls/symbolic energy 对候选和
   全部 baseline 同收费。

同时核实 `222,736 B` 已 superseded，`239,636 B` 仅是 modeled logical、foundry/CACTI/PPA false；M511
payload、signed-INT8 decoder weight package 和 runner source 均尚未准入，所以静态 PASS 仍不得执行。

只有 P0/P1=`0/0` 才能给 source-only contract PASS。输出必须双封；即使 PASS，下一步也只能另行申请
runner source authoring，不能直接启动 CPU 或写 RTL。
