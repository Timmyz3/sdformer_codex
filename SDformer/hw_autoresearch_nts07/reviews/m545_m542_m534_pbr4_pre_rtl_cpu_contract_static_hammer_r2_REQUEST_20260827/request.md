# M545｜M542/M534 PBR4 pre-RTL CPU contract r2 fresh static hammer 请求

只做 fresh independent、只读静态审计。禁止运行 CPU analyzer、candidate、RTL、VCS、iverilog、Verilator、
DC、PT、PTPX、Formality、训练、GPU 或远端任务；禁止创建 runner、canonical result 或 attempt marker。

被审对象：

- `contracts/m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r2_20260827.json`
- contract SHA256：`afdadd302cffdffedb34e0679c177424aa8fc9d7023e4fd9eecf7c4f5ff9bc63`
- handoff：`reviews/m545_m542_m534_pbr4_pre_rtl_cpu_contract_author_handoff_r2_20260827`

重点审查 r1 的五项 finding 是否关闭：

1. canonical loop 必须是 `sample -> layer -> time 0..9 -> output_block -> ordinal -> kernel`；原始 M511 总账
   必须精确为 `696,240,000 bit / 87,030,000 B`，block-replay 必须为 `92,688,000 bit/sample`、
   `926,880,000 bit/S10`，四个架构一律完整 T10，逐层 per-time/per-sample 聚合整数一致；
2. numeric source activity `1` 与独立 source-sign metadata 必须分类型：binary source sign bit 只能为 `0`，
   `1` malformed，product sign 只能来自 signed-INT8 weight；
3. M534 r4/r3/r2 README+JSON 六个 exact SHA 是否为规范性递归 import，precedence 是否唯一，runner 是否被禁止
   自选 SC8/ISO8/OSG/PBR4 的 scheduler、prefetch、join、close、stall 或 cycle increment；
4. `out_valid && !sink_ready` 时内部 read/request/accept/retire/directory-clear/cursor/owner/state delta 是否逐项
   为 `0`，唯一 increment 是否为 `final_output_sink_stall_cycles`，且其进入 cycle conservation；
5. `docs/524` pre-author SHA `4f3ffe1c...` 是否在 handoff 仍一致，并且证明范围没有扩大为历史 authorship。

同时复核 r1 已通过的三 A1 先封/固定 strongest、同资源 `239,636 B modeled logical`、M218
`6x16/L4/O8/FIFO4/Acc24`、三 beat output/address/payload、source bits/reads/logical/padded bytes/common symbolic
energy、严格 conservation、`>=1.30x` ratio-of-sums + every-sample `>=1.10x`、weight traffic 不增、OSG 等价
KILL 与 psum `>=30%` support gate。`run_authorized` 必须仍为 false。

只有 P0/P1=`0/0` 才可给 source-only contract PASS。即使 PASS，下一步也只能另行申请 runner source authoring，
不能启动 CPU 或写 RTL。输出 review 必须双封。
