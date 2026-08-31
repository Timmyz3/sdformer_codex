# M519 registered-release 独立静态打铁 r1

日期：2026-08-27  
结论：`STATIC_NO_GO__VCS_AUTHORIZATION_DEADLOCK__RTL_CANDIDATE_RETAIN`  
评分：**91/100**  
P0：**1**  
P1：**4**

本审阅是 receipt-blind 源码静态检查。未运行 VCS、DC、Formality、PT、PTPX、
Verilator、iverilog 或其他 RTL/EDA 工具；未修改 M519 RTL、SVA、TB、filelist、runner、
contract、旧 M219/M496 或 `docs/359`。

## 1. 裁决

M519 的 RTL 恢复方向正确，六个 RTL 的唯一功能语义变化可追溯为：

1. `free_slot_found` 只观察 registered `!sb_valid_q[slot]`；
2. `head_context_open` 只观察 registered `!ctx_busy_q[block][slice]`。

旧 M219 的 response-edge slot/context bypass 已删除；accepted response 仍在时钟沿清
scoreboard/context，资源从下一拍起可重新分配。独立 normalized diff 没发现 payload、
Acc24 算术、FIFO、epoch/generation/flush、SRAM bank mapping 或外部端口变化。其余五个
RTL 是冻结 M342/M349/M491/M495/M499 的新模块身份克隆，只把 K1/K1x8 service 指向
M519；K8 仍使用冻结 M218。旧 M219、M496 r3 contract 和 `docs/359` 的 SHA 均保持冻结。

两套 TB/SVA/filelist 与 exact runner 对 transaction、bit-exact numeric、weight、result/done、
request/response/result/raw stall、四个 protocol attack、两个 mid-flight reset、同拍不同资源
request/response、零同拍 released slot/context reuse、正的次拍 reuse 做了双向静态闭合。
VCS runner 的 25 个输入 SHA 全部匹配；两个 filelist 分别形成 K1-vs-K1x8 与 K8-vs-K1x8，
receipt 解析器还要求两次 K1x8 cycle 完全一致，禁止复制旧 M492/M497/M216 ratio。

DC Tcl 在 `ungroup` 与任何 compile 命令前运行 `check_timing`，读取 report 并对 TIM-209/
OPT-150 非零立即 `error`。三点均使用同一个 M519 matched top、同一个 filelist/SDC、3.000 ns、
slow/fast DB、ideal clock、ZeroWireload、flatten、两轮 ultra 与 hold-only。DC runner 另有
exact runner SHA、post-VCS launch admission、canonical path、原子 attempt marker 和 point
级资源/冲突门。

但是恢复合同存在授权死锁，因此当前不得启动第一次 VCS：

- recovery contract 的 `authorization.run_vcs=false`；
- 同一节 `next_authority` 要求 independent static hammer 后，另一个 sealed launch-admission
  在 **any tool execution** 前绑定 identity；
- 唯一 launch-admission 是 `m519_fc2_registered_release_dc_launch_admission...json`，其状态是
  `BLOCKED_PENDING_VCS...`，转成 authorized 的必要条件又包括已经存在并独立审阅的 VCS
  receipt/outer seal。

因此没有任何合法顺序能得到第一份 VCS receipt。按 fail-closed 合同优先级，本 r1 必须
`STATIC_NO_GO`，不能用审阅者推断覆盖合同原文。RTL/SVA/TB/runner 候选应保留；只需新身份
修正文义后做快速 r2 静态复核。

## 2. 静态通过项

### 2.1 身份与零漂移

| 项 | 独立观察 |
|---|---|
| 6 RTL + 2 SVA + 2 TB + 3 filelist | contract SHA 13/13 匹配 |
| VCS runner 显式输入 | 25/25 当前匹配 |
| shell / JSON | 两个 runner `bash -n`、两个 contract JSON parse 通过 |
| M219 RTL | `75c4690e...e1d47`，与 upstream ruling 一致 |
| M496 r3 contract | `e529aa8a...04f35`，与 upstream ruling 一致 |
| M496 failure review | inner manifest 与 outer seal 均通过 |
| `docs/359` | `dedde7ce...dfc4`，未变 |

### 2.2 六 RTL 的边界

| RTL | 静态角色与裁决 |
|---|---|
| registered-release service | 相对 M219 仅删除两条 same-edge bypass；**语义符合恢复要求** |
| standalone | 相对 M342 仅新 module 名与 SOURCE_CAP=1 service 替换 |
| K1 8-bank wrapper | 相对 M499 仅新 module 名与 standalone 替换；M499 no-reuse adapter 保留 |
| K8 8-bank wrapper | 相对 M491 仅新 module 名与 standalone 替换；M218/M490 保留 |
| K1x8 | 相对 M349 仅新 module 名及八个 service 替换 |
| matched top | 相对 M495 仅新 module/macro/三个 implementation 名；同一 public ports |

在 M519 service 内，`mem_req_valid` 不再通过 free-slot 或 context-open 依赖当前拍
`legal_response_accept`；历史 M496 的 `M219 rsp -> req -> M499 fault -> rsp` 路径因而在源码
层被切断。该判断只准入“源码依赖已移除”，最终 acyclic 仍只能由 DC precompile
`TIM-209=0` 证明。

### 2.3 验证闭合

- transaction：TB 独立 request/response tuple multiset、bank request/response/read 数量、
  group/request/response/context/result/done conservation；
- numeric：host integer `weight_value`/`reference_accum` 与两架构结果逐 lane bit-exact 比较；
- stalls：request、response injection、result、raw stall 均为 runner 必须非零项；
- protocol/reset：每套 TB 两架构各做 illegal header、spurious response 和 mid-flight POR；
- registered release：service SVA 断言同拍不能复用 released slot/context；TB scoreboard 独立计数
  violation 必须为零，同时 runner 要求 distinct same-edge 与 next-cycle slot/context cover 非零；
- architecture identity：第一套测 K1/K1x8，第二套测 K8/K1x8；Python receipt builder 拒绝
  repeated K1x8 cycle 不一致。

动态索引审计未发现默认身份的越界：`mem_rsp_slot` 为 3 bit，所有 M519 service instance 的
OUTSTANDING=8；bank/output-block 索引范围也是 0..7。bind 中的 unpacked-array dynamic select
是合法 SystemVerilog，但 VCS 工具接受性仍未被静态检查证明。

### 2.4 runner、negative path 与 DC 门

VCS runner 首条可执行 gate 要求 caller pin 当前 runner SHA；错误/空 SHA 在创建 canonical
result directory 前 exit 3，因此存在无副作用的 wrong-runner-SHA negative path。然后才创建
目录并校验所有输入，任一漂移 exit 10 且不能产生 positive receipt。

DC runner 当前仍被 blocked launch-admission 锁住。即使未来 admission 被正确填充，它还会
交叉锁 recovery contract、VCS receipt/outer seal、static review outer seal、Tcl、runner、
工具/库/SDC/RTL/docs359，并在第一次 `dc_shell` spawn 前原子消费 whole-line attempt marker。
K1/K8/K1x8 任一点失败均不会生成完整三轴准入 receipt。

## 3. P0 blocker

### P0-1｜第一次 VCS 的授权顺序不可满足

合同一面写 `run_vcs=false` 与“launch-admission before any tool execution”，另一面又要求该
DC launch-admission 必须先绑定 VCS receipt/outer seal。静态 review P0=0 本身没有被明确写成
一次 exact-SHA VCS 的授权主体。

最小修复：生成新的 recovery contract/runner identity，明确写成：

```text
P0=0 independent static hammer authorizes exactly one pinned exact-SHA VCS run.
DC remains forbidden until that VCS receipt is independently reviewed and a
separate DC launch-admission binds all final identities.
```

同时把 `authorization.run_vcs` 改成 `conditional_after_static_review_p0_zero`，保留 `run_dc=false`。
不能直接编辑 r1 contract 后继续使用旧 SHA；修复后 recovery contract 与两个 runner 的 pin 都
必须刷新，并请求 r2 静态 review。无需修改 RTL/SVA/TB。

## 4. P1 findings

1. service SVA 用 `mem_rsp_accept && !protocol_error` 近似 legal response。非法 response 在
   protocol_error sticky 更新前与 request 同拍时，属性可能把未释放资源当 release。建议 bind
   内部 `legal_response_accept`，no-reuse assertion 与 release cover 都以它为 antecedent。
2. wrong-runner-SHA negative path 静态存在，但 runner 不自动产出并 cross-link 独立 negative
   receipt。正式 VCS milestone 应先由主线程显式执行无副作用 negative preflight，并在 positive
   receipt/hammer 中记录 exit=3、无 result directory。
3. 历史 M496 中 OPT-150 只在 compile timing update 出现，而 precompile `check_timing` 明确出现
   TIM-209。当前 Tcl 会扫描 precompile report 的两种字符串，但可严格宣称的前置证明是
   `TIM-209=0`；`OPT-150=0` 还需结合最终 dc.log。不要把 report 中机械的 OPT-150=0 写成比工具
   实际诊断阶段更强的独立证明。
4. 两套 TB 在发现 `result_slice>=6` 后仍继续以该值索引 `[0:5]` reference/result store。合法 DUT
   路径不会触发，但若 DUT 正好输出非法 slice，testbench 可能先出现越界/X 噪声。建议 error 后
   guard 数值循环，使 malformed output 稳定 fail closed。

## 5. r2 复核门

1. 只修 P0-1 的 contract 授权顺序并刷新受影响 SHA pin；
2. 证明其他生产文件 SHA 未变；
3. recovery contract 明确区分“一次 VCS authorization”与“post-VCS one-attempt DC admission”；
4. VCS runner 仍在任何目录创建前验证 caller-pinned runner SHA；
5. 新 r2 静态 hammer P0=0 后，才允许 wrong-SHA negative preflight 与一次 positive VCS。

本 r1 不准入 SV compile、VCS behavior/cycles、组合图无环、DC、PPA、power、energy、完整 FC2、
FFN、系统倍速或 DATE headline。

