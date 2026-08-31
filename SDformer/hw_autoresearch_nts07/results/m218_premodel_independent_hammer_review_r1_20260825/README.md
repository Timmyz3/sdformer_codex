# M218 premodel independent hammer review

结论：**91/100，P0=0，M218 RTL 开发 GO；fixed-latency/in-order premodel
带边界引用 GO；VCS/物理/完整 FC2、FFN 或系统倍速 NO-GO。**

本评审只在本目录新增证据，没有修改 M218 合同、分析器、原结果、RTL 或
`docs/359_DATE终局冻结_20260813.md`。原结果目录内执行 `sha256sum -c
SHA256SUMS` 全部通过；docs/359 SHA256 仍是
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 独立重算结论

审计器没有 import M168、M216 或 M218 分析模块。它从冻结 manifest 重新筛出
120 条 H67 ep35 `.mlp.fc2` Linear record，逐文件检查 SHA、大小和 popcount，
并重建 5,580,000 个 token 的 nonzero-96 descriptor、compact window 和
modulo-eight bank population。

| 项目 | 独立重算 |
|---|---:|
| event | 143,894,510 |
| K1 group command | 412,900,394 |
| M216 ordered K8 group command | 73,380,812 |
| independent-bank-queue oracle | 70,657,362 |
| ordered 开销 | 2,723,450 / 3.854446193% |
| active 128-bit bank-slice read（K1/K8 均相同） | 2,477,402,364 |
| weight byte（K1/K8 均相同） | 39,638,437,824 |
| result beat（包含 zero token） | 54,720,000 |
| zero token | 1,863,944 |

相邻窗口公式与 M216 `SOURCE_CAP=8` 严格一致：stage0 每个 compact D2
window 独立排空；stage1--3 只允许 `(0,1),(2,3),...` 相邻配对，每个 bank
先排空 older window 再使用 successor，因此一对 window 的 group 数恰为
`max_b(old[b]+successor[b])`。不允许跨 pair 重排。ordering overhead 只来自
该可执行顺序相对 whole-token independent-bank oracle 的约束，不应把
70,657,362 冒充 M216 可执行 group 数。

四个 stage 的 records 是 20/20/60/20，event 是
46,809,056 / 33,053,865 / 53,067,276 / 10,964,313；K1 group 是
46,809,056 / 66,107,730 / 212,269,104 / 87,714,504；ordered K8 group 是
12,494,340 / 12,042,708 / 34,593,020 / 14,250,744。stage0--3 的 primary
service speedup 分别为 2.834530x / 4.723876x / 5.686513x / 5.972984x。

## Service recurrence

独立 edge simulator 使用“同一 edge 先 retire、后 issue”的语义，与

`t[i] = max(t[i-1]+II, t[i-O]+L, t[i-(6*output_blocks)]+L)`

在 12,480 个小规模组合上 0 mismatch；其中命中 47,256 次 outstanding slot
同 edge 复用和 21,348 次 context 同 edge 复用。`P=6*output_blocks` 来自固定
`source-group -> output-block -> slice0..5` 请求顺序；同一 context 只有前一
response 在该 edge 被接受后才能复用。

完整 5×4×2 共 40 个 L/O/II 点、stage0/1/2/3 和 aggregate 都与 M218 原结果
逐项一致。主点 `L4/O8/II1` 为：

| 指标 | 数值 |
|---|---:|
| K1 service cycles | 2,552,566,588 |
| K8 service cycles | 515,449,096 |
| service speedup | **4.952121573x** |
| L1/O8/II1 K8 oracle cycles | 504,300,928 |
| primary throughput retention | **0.978371932** |
| K1 frontend/service interval | [2,552,566,588, 2,982,282,923] |
| K8 frontend/service interval | [515,449,096, 605,645,881] |
| conservative composed lower bound | **4.214618919x** |

所有 token 都计入 `6 × output_blocks` result beats 和最后一个 done cycle；
zero token 单独贡献 20,102,204 个 result/done tail cycles，没有被跳过。这里的
interval 仍是数学包络，不是 frontend 与 service 的可执行时序组合。

## 打铁判定

没有阻断“开始写 RTL”的 P0。以下七项是 P1，也是 M218 从 premodel 升级为
RTL/performance admission 前必须关闭的 exit gates：

1. FIFO4 尚未与 M216 backpressure、service、response/result drain 做时序组合。
2. 当前只证明 fixed-latency、in-order；variable-latency/OOO 未证明。
3. logical distinct-bank 可行不等于真实 SRAM port、RDW、latency、II1 和 routing 可行。
4. Acc24 signed INT8 数值、forwarding、overflow、slice/result 语义没有 miter。
5. soft flush 的 epoch/sequence ownership、truthful flush-ack 和 stale response 隔离缺失。
6. M218 还没有 scope-matched K1/K8 PPA；cropped K1 只能作为单列 sensitivity。
7. 相同 active reads/weight bytes 是 work conservation，不是 SRAM energy 结论。

因此可以立即开 M218 RTL，但首个 admission 必须 fail closed：固定 8 banks、
16 lanes/slice、6 slices、O8、FIFO4；K1/K8 使用同一 bank/FIFO/tag/Acc24 资源且
只改 `SOURCE_CAP`；每 bank 每拍最多一次 128-bit read；context 只能在 response
accept 同拍复用；所有请求携带 token tag、epoch16、request_seq32、output block、
slice 和 bank-valid ownership；wrong/duplicate/stale response 必须拒绝；soft flush
增加 epoch 并在真实 memory flush acknowledgement 前禁止新 header；zero token 也须
交付完整结果并在最后 beat accept 后 done。若要支持 OOO，必须新增有限 scoreboard/
reorder 证明，不能沿用 in-order recurrence 直接宣称通过。

## 允许与禁止措辞

允许：冻结 H67 FC2 payload 上，fixed-latency/in-order、pre-RTL standalone
slice-service 模型估算 K1/K8 为 2,552,566,588 / 515,449,096 cycles，主点
service speedup 4.952121573x，frontend/service 区间的保守下界 4.214618919x。

禁止把上述数字写成“M218 RTL achieved”、完整 FC2/FFN、physical、system 或
paper headline；禁止称 4.2146x 为 measured end-to-end；禁止用相同 byte 数推导
相同/更低能耗；禁止声称 1024-bit/cycle 已综合或 macro-qualified；禁止声称真实
memory conflict-free、OOO 已支持，或用 physically cropped K1 作为主基线。

可复现细节、40 点矩阵、每 stage 账本、攻击处置和完整 fail-closed 条款见
`m218_independent_recompute.json` 与
`m218_premodel_independent_hammer_review_r1.json`。
