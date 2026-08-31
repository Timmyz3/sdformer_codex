# M518 production candidate 独立静态打铁 r1

日期：2026-08-27  
结论：`STATIC_NO_GO__DO_NOT_RUN_VCS__CAMPAIGN_DOES_NOT_IMPLEMENT_SEALED_V01_V20`  
评分：**72/100**  
P0：**6**  
P1：**4**

本审阅是 receipt-blind 的源码静态审查。未运行 VCS、DC、Formality、PTPX、
Verilator、iverilog 或任何其他 RTL/EDA 工具；未修改 RTL、SVA、TB、filelist、runner、
contract 或 `docs/359`。

## 1. 结论

M518 RTL 的主体结构值得保留：默认参数下 public ports 与 M273r2 逐项一致；独立枚举
确认冻结 schedule 是 `96*16+64=1600` 个互异且完备的 `(row,lane,time)` product；
五拍 config、五拍 raw、双 raw bank、17-cycle dense、cycles12--16 直接五次 FIFO push、
25-bit accumulator/26-bit update/Q24 final saturation 的源码结构与 sealed spec 基本一致。

但是当前生产候选不能运行并封成“V01--V20 PASS”。问题不在主 schedule，而在验证合同
把 sealed spec 的强测试缩窄了：精确饱和边界、release 全状态、reset 全状态、oldest/ownership
SVA 和 launcher negative-control 都没有闭合；contract 还重新编号并改写了 V01--V20 的
含义。若现在运行，runner 可以产出一份文字上声称 V01--V20 的 positive receipt，但那份
receipt 不等价于 sealed spec 的 V01--V20。因此 fail closed，判 `STATIC_NO_GO`。

## 2. 已静态通过的部分

### 2.1 身份与结构

- sealed spec 的 `SHA256SUMS` 与 outer seal 均通过；
- M518 RTL / SVA / TB / filelist / runner / draft contract 当前 SHA 已记录在 JSON；
- `bash -n` 通过，draft contract 可由标准 JSON parser 解析；
- filelist 只有且按顺序包含 RTL、SVA、TB 三项；
- 独立抽取端口得到 M273r2 与 M518 各 50 个 public ports，方向、名字和宽度表达式完全一致；
- `TAG_W=48`、`FIFO_DEPTH=16` 的非默认参数 elaboration fatal 存在。

### 2.2 数据通路

- config layout 是 weights `[799:0]`、bias `[1039:800]`、threshold
  `[1063:1040]`、padding `[1279:1064]`；final candidate 包含当前 beat 后才检查；
- raw beat 映射等价于 `X[2*beat+word/16][word%16]`；offending raw/config beat
  由 `&&!frame_error` 阻止提交；
- cycles0--11 每个 output scalar 累加三个 product，`sub==0` 从 bias 覆盖 stale acc；
- cycles12--15 每拍同时关闭 rows0--7 的一个 beat，并给 rows8/9 预计算两个 taps；
- cycle16 在一个 26-bit expression 中加入 taps8/9 后比较，没有读 pre-edge incomplete sum；
- 25-bit accumulator 只在 `wide_sum[25]==wide_sum[24]` 的动态断言条件下截回 25 bit；
- final saturation 后再做 signed threshold compare，threshold equality fires；
- 两 raw bank oldest-order mux、registered 16-beat FIFO、full pop/push、sticky registered
  fault 与 reset-only quarantine 的核心编码均与规格方向一致。

### 2.3 独立 schedule 枚举

静态审阅者没有复用 runner 的结论，另行枚举 17 cycles：active slots 为
`[96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,64]`；总 product 1600，
unique product 1600，并与 `{row 0..9} x {lane 0..15} x {time 0..9}` 完全相等。

这只准入“schedule 规格和 RTL 映射按源码看一致”，不准入 SV 语法、动态行为、cycle、
speedup、PPA 或 energy。

## 3. P0 blockers

### P0-1｜V02/V03 numerical campaign 不完整

sealed V02 要求 zero、random、alternating-sign、all-extreme；当前 TB 只有四个固定
profile，没有随机算例。sealed V03 要求 Q24 两个 rail 的 just-below / at / just-above。
当前 extreme profile 的 unsaturated totals 只有：

```text
-8,551,150, -8,226,025, 8,226,030, 8,551,155
```

它们能证明上下溢出方向，但没有覆盖 `8,388,606/607/608` 与
`-8,388,609/608/607` 六个精确边界。profile3 只证明 `7>=7` equality。

修复：增加独立、直接构造的六个 rail cases；增加固定 seed 的 random config/raw 集合；
oracle 继续使用 full-precision host integer，不能复制 DUT reduction/schedule。

### P0-2｜V06 oldest-ready 只做 combinational force，不是 sequential ownership proof

`oldest_selection_unit_attack` 在 negedge 强制 `raw_ready_q/raw_owned_q/order`，`#1` 只看
combinational `dense_selected_raw_bank`，随后立即 release force + reset。它没有经过接受
edge，没有证明 selected bank 被原子移出 ready、保持 owned 到 cycle16、tag/payload 对应，
也没有证明 bank1 真正 first issue/first result。

修复：构造合法时序让 bank1 比重新填入的 bank0 更老，或使用受控 white-box state 后至少
跨一个 issue edge，并核对 ready/owned/tag/result；同时新增 oldest selection SVA 和 dual-ready
cover。

### P0-3｜V16 release 状态矩阵没有执行

`finish_context` 只在 `send_tiles` 已发送完全部 raw beats并撤掉 `raw_valid` 后才拉高
`release_valid`。pressure context 可能让 release 等待 compute/drain，但没有覆盖 partial raw，
也没有分别证明 dense cycle0、12、16 和 FIFO drain 的 held-release 行为。它不能满足 sealed
V16 的文字合同。

修复：五个定向 case 分别在 partial raw、cycle0、cycle12、cycle16、FIFO-only drain 前拉高并
保持 release；每例证明零 early accept/retire，且仅在完整 drain 后一次 accept/retire。

### P0-4｜V18 reset 矩阵被缩成三例

当前 `reset_state_attacks` 仅覆盖 partial config、partial raw、dense cycle5，共 3 次。
sealed V18 还要求每个 dense phase、FIFO close stall、fault quarantine，以及每次 reset 后的
clean next context。当前最后一次 reset 后甚至没有再跑一个合法 context。

修复：至少覆盖 cycles0/11/12/15/16、FIFO-full close stall、registered quarantine；每次 reset
后检查所有 valid/owned/count/counters 的 deterministic empty state，再跑一组 N=1 exact
29-cycle + oracle context。

### P0-5｜required SVA/cover 不完整，close-stall assertion 可旁路

sealed spec 要求 ownership conservation、oldest selection、完整 close-stall atomic hold、
departure/push/tile conservation，以及 dual-ready、phase12 stall、phase16 stall、all five beats、
zero-tile fault、reset recovery covers。当前 SVA 缺这些专门属性/cover。

更严重的是 `ap_close_stall_holds` 的 consequent 是：

```text
protocol_error || fifo_credit_internal || (stable(cycle) && !fifo_push)
```

若 stall 后一拍 credit 恢复，`fifo_credit_internal` 直接让属性通过，完全没有检查 stall edge
上的 accumulators、raw ownership、dense tag/bank、FIFO write pointer 和 debug issue/push
counters 是否保持。当前 SVA 端口也没有这些 bind targets。通用 `cp_close_stall` 不能保证
phase12 和 phase16 都命中；`cp_beat4` 不能替代 all-five-beat cover；generic `cp_fault` 不能
替代 zero-tile fault cover；没有 reset recovery cover。

修复：扩展 bind targets，按 `$past` 写完整 state bundle hold；加 raw ownership/count equality、
oldest compare、`pushes-departures==fifo_count`、17 issue/5 push/tile、tile_done tag；新增并由
runner 要求所有 sealed covers 非零。accepted payload/control 同时加 `$isunknown` fail gate。

### P0-6｜V20 与 contract/runner identity 不闭合

runner 确实会检查六类输入 SHA，并提供 `M518_NEGATIVE_PREFLIGHT_TEST=1` 把 RTL expected SHA
改错的入口；但 positive runner 不执行、不验证也不 cross-link 一份 negative receipt，却会
直接输出“V01--V20”。runner 自身也没有 `M518_EXPECTED_RUNNER_SHA256` gate，所以修改 runner
即可改变校验/receipt 逻辑而不触发其输入 preflight。

此外 draft contract 重新定义了 sealed V 编号，例如 sealed V09 是 FIFO head stall，而 draft
V09 是“无 intermediate”；sealed V13 是 sustained-valid half-cycle probe，而 draft V19 才是
half-cycle monitor；sealed V16 是 release held across live states，而 draft V16 是 prologue/close
credit。marker-string 检查只确认 TB 文本中出现 `V01`...`V20`，不能证明语义一致。

修复：runner 首先用外部传入的 exact runner SHA 自校验；用独立 run dir 自动执行 wrong-SHA
negative control并要求 exit10、无 compile/simv/positive receipt；positive receipt cross-link
negative manifest；contract 的 V01--V20 逐字回归 sealed spec，runner 静态门解析明确 test ID，
不能只搜索 marker 字符串。

## 4. P1 findings

1. runner 没有机械比较 M273/M518 public-port signature；当前端口是 exact，但未来宽度漂移
   只能依赖 compile 的隐式告警/错误。应把 50-port canonical signature 哈希纳入 preflight。
2. runner 假设 `${VCS_HOME}` 路径就是 V-2023.12-SP1，没有记录并校验实际 `vcs -ID`；receipt
   中的 tool 字段目前是常量文本。
3. SVA 没有对 `busy`、`tile_done_tag`、tile_done exactly-once 与
   `debug_context_cycles/context_retire_cycles` 建立强一致属性；TB 只做部分 end-state计数。
4. run manifest 使用 run-dir 路径生成且没有 final `SHA256SUMS` 自身 outer seal。现有
   `RUN_MANIFEST + outer seal` 可保护核心 receipt，但 publication 结构应统一成可迁移相对路径
   的 member manifest + outer seal。

## 5. 修复后静态准入门

只有以下全部完成且新的 production identities 重新封存，才可请求 r2 静态打铁：

1. 补齐 P0-1 至 P0-5 的 TB/SVA，并使每个 sealed V01--V20 有唯一可审计 test/cover/counter；
2. contract 不再重编号或缩窄 sealed matrix；
3. runner 自身 exact-SHA，自动 negative-control，positive receipt cross-link negative evidence；
4. runner 需要的 covers 包含 dual-ready、phase12/16 stall、beats0--4、zero-tile fault、reset
   recovery，且每项 match > 0；
5. 重新执行纯静态 review，只有 P0=0 才允许主线程运行 Synopsys VCS。

## 6. 准入边界

当前准入：M518 是一个结构合理、schedule 静态完备的 **production candidate**。

当前不准入：SV compile、VCS behavior、V01--V20、29/80 RTL cycles、numeric equivalence、
RTL speedup、matched area、Formality、STA、power、energy、system speedup、PPA、headline。

`docs/359` 未修改；审阅结束时 SHA 为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
