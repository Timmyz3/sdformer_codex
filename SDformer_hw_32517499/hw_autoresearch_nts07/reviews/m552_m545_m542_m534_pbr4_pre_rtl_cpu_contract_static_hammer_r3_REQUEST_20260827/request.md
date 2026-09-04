# M552｜PBR4 pre-RTL CPU contract r3 fresh static hammer request

请由未参与 r3 authoring 的 reviewer 对以下双封 source-only 对象做 fresh read-only static hammer：

- `contracts/m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r3_20260827.json`
- `reviews/m552_m545_m542_m534_pbr4_pre_rtl_cpu_contract_author_handoff_r3_20260827/`
- M549 final review、r2 contract/handoff 与 M534 r4/r3/r2 的 README+JSON 六个规范输入。

审查必须独立复算/验证：

1. 四架构是否对每个普通合法状态都有唯一 ordered guard/action/state-delta/primary-class；
2. `A1-OSG` serviceable、`PBR4` ingress-vs-drain、same-edge retire/replace、SC8/ISO8 partial/tail flush；
3. group tile/bank-round、六 slice issue/L4 retire、每 destination 六读六写与四份 no-newline golden SHA；
4. exact T10 `92,688,000 bit/sample`、`926,880,000 bit/S10` 与 raw `696,240,000 bit/S10`；
5. numeric `+1`、独立 sign-bit0、sink-stall 零副作用、`239,636 B` modeled-logical-only；
6. strongest-A1/four-point identity不能由 runner 选择；
7. future closed authorization 同时绑定 result/attempt absence，以及 contract/runner/final-release 三套实际
   review member SHA、manifest SHA、outer-seal-file SHA，并要求 runner 重新 hash，而非信任 score 字段。

边界：零 candidate CPU/analyzer、runner、RTL、VCS/iverilog/Verilator、DC/PT/PTPX/Formality、训练、GPU、
远端与 result。`run_authorized=false`。只有 P0/P1=`0/0` 才能 source-only PASS；PASS 也不授权运行，只能进入
独立 runner-source admission。

