# M559｜PBR4 pre-RTL CPU contract r4 fresh static hammer request

请由未参与 M559 authoring 的 reviewer 对 r4 contract、author handoff、terminal goldens、future runner/
authorization/launcher schema 与双封身份做 fresh read-only static hammer，并完整读取 M556 final review 和 r3
contract/handoff/goldens。

必须独立验证两项 P1 是否真正关闭：

1. r3 common row8 已被替换而非合并；nonlast retire、last retire、clear start、1024 个按 0--1023 顺序的
   charged zero write、clear end、time retire、next owner 及 time9/layer/sample/cohort 分支，在所有 prior-state
   上都有唯一互斥 guard/action/state-delta/charge/class，且四个架构共用同一 FSM。独立复算 2-cycle 与
   1029-cycle terminal golden 的 exact no-newline SHA。
2. 未来身份是可拓扑排序的 DAG：immutable runner 不嵌后生 auth SHA；canonical auth 在 static/candidate/
   final reviews 后生成并独立双封；post-auth wrapper 冻结 auth triple 后独立 review；wrapper 不嵌后生 review
   SHA，运行时用 canonical review 绑定的 wrapper self-SHA 检查；wrapper review 后不存在更晚 author permit。

同时验证 exact T10、typed `+1`/sign-bit0、FINAL_OUTPUT stall 零副作用、`239,636 B` logical-only、固定
A1-STRONG 与 unchanged service/resource/GO gate。

边界：零 candidate CPU/analyzer、runner/authorization/launcher、RTL、EDA、训练/GPU/远端、result 与 attempt。
`run_authorized=false`。只有 P0/P1=`0/0` 才可 source-only PASS；PASS 不授权运行，只能进入独立 immutable
runner-source admission。
