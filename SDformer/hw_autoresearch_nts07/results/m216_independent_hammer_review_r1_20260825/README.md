# M216 independent hammer review

结论：**91/100，P0=0，M216 standalone sparse-FC2 frontend GO；完整
FC2/FFN、物理与系统 headline NO-GO；M218 service-island premodel GO。**

独立审查在 VCS 结果目录内部执行 `sha256sum -c`，全部通过。K1 定向
reference、K8 对 M214 cycle miter、K1 的 256 例 RTL/model recurrence、K8
对 M214 的 256 例 model identity 均为 0 mismatch。dense bank96 用例有 3,072
个 source event、8 个 output block，恰好产生 24,576 个 K1 group，守恒通过。

我没有使用 replay 已填好的汇总值来证明 payload extent，而是从冻结 manifest
重新筛出 120 条 `.mlp.fc2` Linear record，并按 shape/name 重算出 5,580,000
token、143,894,510 event；各 stage record 数为 20/20/60/20。K1/K8 周期分别
为 429,716,335 / 90,196,785，因此精确 frontend speedup 是
**4.764209001x**。独立重算的 `event × output_blocks` 是 412,900,394：它已经
删除所有扫描、调度和控制开销，任何仍受 `SOURCE_CAP=1` 约束的 K1 都不能低于
此数。故 active-depth 或 source order 不能把速度比压到 **4.577772855x** 以下，
own-best K1 cycle P0 在声明的 frontend 合同内关闭。

DC 的 r1 父 run 仍然是 `FAILED_OR_INCOMPLETE_DO_NOT_CITE, exit 40`，不得引用。
r2 recovery 的两个 exact-SHA 子 run 与 evidence manifest 均通过。直接解析两份
mapped netlist 得到 K1/K8 共有 2,770 个同名 sequential instance；K8 只多
`group_source_count_q[1:3]` 三个位，是 K1 常量折叠，不是 queue/window/storage
偷减。3 ns、ideal-clock、ZeroWireload、0-macro 下，K1/K8 cell area 是
20,436.696076 / 20,587.392080 um2，K8 logic overhead 为 0.737379%；两者
setup/hold 均 MET。条件 frontend throughput/logic-area 为 **4.729335849x**。

这个结果的边界必须写死：性能 replay 是 always-ready，M216 只输出 source
index，没有 weight SRAM、response latency/tag/backpressure、Acc24、accumulate 或
commit。若 K8 直接在一拍内消费 8 个 source 的 96-lane INT8 权重，隐含 supply
可达 6,144 bit/group；其 SRAM、bank、routing 与能耗完全不在当前 DC。因此这些
缺项不阻断严格的 standalone frontend 结论，但一旦把 4.764x 外推为完整 FC2、
FFN、物理或系统倍速，就会成为 P0 误述。

M218 可以进入预模型，但应把有限 lane-sliced weight response、bank conflict、
多 outstanding、epoch/flush、Acc24 context 与真实 backpressure 纳入周期；M216
只保留为 frontend upper-bound。另有 P1：补 native-cropped K1 area sensitivity，
补两个未命中的 terminal control coverpoint，并扩展跨 sequence/event-density replay。

`docs/359_DATE终局冻结_20260813.md` 的 SHA256 已独立确认仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
