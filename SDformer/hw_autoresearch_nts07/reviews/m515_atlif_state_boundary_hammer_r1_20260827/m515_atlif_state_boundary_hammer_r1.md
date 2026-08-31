# M515 ATLIF state-boundary 独立打铁 r1

日期：2026-08-27  
结论：`NO_GO_CURRENT_UNCONDITIONAL_PASS__CONDITIONAL_STATE_BOUNDARY_GO`  
评分：**71/100**  
P0：**2**  
P1：**4**  
EDA / GPU / simulator 实际执行：**否**

## 裁决

底层硬件结论基本成立，但当前 M515 `RUN_COMPLETE` 和 paper-safe sentence 不能原样
准入。独立人工审计支持下面这个**有条件结论**：在 calibration hook 为空、训练侧
threshold update 停止、参数/缓冲冻结的 inference 模式下，当前 ATLIF forward 对
每个完整 temporal tile 独立计算；M273 T10 RTL 只在所有 live ownership/valid、
pipeline 和 result FIFO 状态排空后接受 release。因此该 standalone boundary 不需要
跨 tile、跨 frame、按空间索引的 membrane-state SRAM。

当前 artifact 的两个 P0 是证明与数字口径，不是已经发现了隐藏 membrane：

1. AST 检查不是传递 def-use 证明，也没有冻结动态 calibration observer；
2. `8,515 bit excluding tags/order/control` 不是完整或自洽的寄存器账。

修复这两个 P0、重新生成纯静态结果后，不要求新 ATLIF arithmetic RTL；M289 已经
证明 exact RTL 的 retained sequential state 位于标准单元中，但该证据必须被 M515
合同直接绑定。

## 身份、重放与篡改检查

- contract SHA：`5a0b87e80141e5a63d5a9f5429eba20805977cd38821be2f0f0892426d3b6aa9`。
- analyzer SHA：`0b33d283ceb06275f3dbcfd2c9ec14ad13d03613483ed570929e88eec1e16443`。
- 六个 frozen inputs 全部为 regular non-symlink，实际 SHA 与 analyzer/contract 一致。
- result member seal 通过；`SHA256SUMS` 内容 SHA 为用户给出的
  `8035858cb74dc75868cf171df684db80fee6d996b2f9dc23a52bbf8f74fef84e`；
  outer-seal 文件自身 SHA 为
  `8d997065c5c15986593abc6ce908307d7069387f18248ee0ba9e6ef3854d2071`。
- exact analyzer 在临时输出上重放得到 JSON SHA
  `effcb0b84fe48dd6de399745a0f7909e60492b77e75886281ac7f6fedfc32f5d`，
  与 sealed result byte-identical。
- 临时副本单字节 drift 被 analyzer 以 SHA error 拒绝且不产生 output；把 algorithm
  换为 symlink 同样被拒绝。sealed result JSON 的临时篡改使 member verification
  rc=1。
- `docs/359` 未修改，仍为 `dedde7ce...`。

所以结果可复现、输入防漂移与 member tamper detection 是通过的；P0 位于被重复
生成的分析逻辑本身。

## P0-1｜AST dataflow 不足以证明 mutable observer 不在输出路径

analyzer 只对硬编码变量集合
`flattened/latent/h_seq/negative_scale/spike/out` 的**直接赋值语句**收集
`self.*` load，再与 forward 内的 `self.*` stores 求交。它没有建立传递 def-use。
本轮用合成 AST 反例静态复现：

```python
cache = self.update_value
spike = cache
self.update_value += 1
out = spike
return out
```

analyzer 得到 `written={update_value}`、`output_path_attrs` 不含 `update_value`、交集
为空，尽管输出显然读取 mutable state。因此这个算法只能描述当前若干直接语句，
不能作为一般 output-dependency proof。

冻结源码本身还有一个真实未冻结入口。forward 353--355 行执行：

```python
observer = getattr(self, "_h9_calibration_observer", None)
observer(h_seq.detach(), self.thresh.detach())
```

这个 dynamic attribute 不会被 `self_attributes` 发现。callback 在 `self.act(...)`
之前执行，而且 `detach()` 共享 tensor storage；未受约束的 callback 可以原地修改
threshold，影响同一次 forward 输出。仓库里的已知 `BudgetObserver` 当前只读，但
它的实现没有被 M515 冻结，analyzer 也没有证明实际 inference instance 上 hook
不存在。

另外，`update_value`、`quantile_value`、rate 与 importance observers 虽不在当前
forward 的内建输出路径上，训练侧 `threshold_update()` 会在 forward 之间读取它们
并更新 `thresh`/negative scale。只有明确停止该调用并冻结参数，才能把它们排除出
inference state。

**修复门：**绑定一个 inference-instance/config manifest，强制 calibration hook
不存在、threshold updater disabled、参数/缓冲 frozen；analyzer 对 `out` 建立递归
def-use（含 branch predicates、call targets、dynamic `getattr`）或对任何未证明纯的
callback fail closed。之后结果只能写“under frozen inference”。

## 算法底层判断｜条件成立时无跨 tile/frame recurrent state

对 exact source 人工展开后，内建输出路径是：

```text
current x_seq -> flatten -> dense W or factor R/L -> bias/center
              -> frozen threshold activation -> reshape(out)
```

没有 membrane tensor、spatial index、previous-frame input 或跨 forward recurrence。
`T=10`/`T=2` 都把完整 temporal dimension 放在当前 `x_seq` 中；可变 counters/EMA
仅作训练或观测。因而在上述 frozen-inference 条件下，算法层面的 no-persistent-
membrane 结论成立。它不是“所有运行模式均 stateless”。

## RTL release/drain 独立展开

当前 M273 port list 没有 previous membrane/frame/state 输入输出。`work_empty` 同时
要求：

- 无 partial raw fill、两个 raw bank 均无 ownership；
- stage1 inactive；两个 intermediate bank 均未 reserved；
- stage2 inactive、product invalid；
- FIFO count 为零。

`release_ready` 还要求 config loaded、无 protocol error、至少处理过一个 tile，并且
`raw_valid=0`。数据使用链也 fail closed：raw bank 在完整五 beat 后才 ready；
stage1 phase0 不读取旧 accumulator；intermediate 在 valid 前完整写入；product/FIFO
只在 valid/count 非零时读。因此 release 后的旧 payload bits 不会成为下一 context
的 live input。

这足以支持“无 live cross-context membrane state”，但**不等于寄存器物理清零**。
release 只清 `config_loaded_q`；raw/inter/product/FIFO payload、tags、orders、config
bits 会保留 stale don't-care 值。JSON 的 `state_survives_release=false` 与 README 的
“retains no tile state”必须改成“no live/valid tile state survives release”。

analyzer 当前只搜索少量字符串和四个 port 名，方法本身不能证明上述语义；本轮是
基于 exact pinned RTL 的独立人工展开。后续应把 live-state invariant 做成结构化
表或 SVA，但这不需要发明新的 datapath。

## P0-2｜8,515-bit 账不是完整且标签自相矛盾

`8,515` 的加法本身可重复：

```text
1536 + 2*1280 + 48*24 + 2*384 + 147 + 16*147 = 8515
```

但它不能称为 `payload_total_bits_excluding_tags_order_and_control`：

- product 的 147 bit 实际为 tag 48 + beat 3 + valid-bits 48 + data 48；
- 每个 FIFO entry 的 147 bit 同样含 tag 48 与 beat 3；
- 所以 8,515 内已经含 **867 bit tag/beat metadata**；
- 同时它漏掉独立保持的 decoded live config **1,109 bit**：right factor 240、
  requant 5、valid 120、negative 120、shift 360、bias 240、threshold 24；
- 还漏 raw/inter tags、orders、ownership、pointers、phase/control 和 debug/context
  counters。

按 exact RTL declarations 独立枚举，pre-optimization sequential state 上界是
**10,470 bit**。M289 exact-source gate netlist 实际有 **9,639 个一位 sequential
cells**（9,638 个 DFKCNQD1 + 1 个 DFKCNQD2）；常量/不可观察位优化解释了它低于
RTL declaration count。M515 的 8,515 只是 selected mixed payload/metadata subset，
不是总 working register count，也不能直接等同物理 FF 数。

**修复门：**删除当前 8,515 paper sentence，或明确标成 selected subset 并列出
included/excluded fields。论文主表建议使用 `10,470 RTL-declared bits (upper bound)`
与 `9,639 synthesized sequential cells` 两列，不把 cell count误写成 SRAM bit。

## M289 是否确实包含 working registers

事实结论是 **是**，但 M515 的绑定链不充分：

- M289 `input_sha256.txt` 的 RTL SHA 正是 `11d5c6c4...`；
- sealed mapped netlist 中独立数得 9,639 个 DFF instances；
- sealed area report为 102,852.287739 um2 total cell area、19,432.224313 um2
  non-combinational area、9,639 sequential cells、0 macro。

因此 retained working registers 确实已经包含在 M289 logic-only standard-cell area
里。不过 M515 contract 没有冻结 M289 contract、input identity、area report、
netlist或 evidence seal，只冻结了一个 macro mapping JSON；当前 result 不应单凭它
发布 `already_charged_in_m289_logic_area=true`。修订版应直接绑定 M289 evidence与
独立 M302 review，并保留 ideal-clock/ZeroWireload/0-macro、非 paper-PPA 的标签。

## Trace 重算

独立 CSV 重算与 result 一致：1,840 records，其中 ATLIF 930、attention 120、
operator 790；ATLIF 为 T2 480、T10 450；sample 0--9 各 93 条。该统计只证明冻结
S10 workload population，不证明 state liveness 或 memory capacity。

## System config/weight memory 边界

边界方向正确但还需写透：

- **inside M273/M289：**单 context 的 config capture/decoded factor、bias、threshold
  registers，以及 tile-local raw/intermediate/product/FIFO state；
- **outside：**为全网络/所有层保存并搬运 weights/config 的 backing store，输入输出
  activation buffers、NoC/DRAM traffic、105 个 ATLIF 调用的 provisioning；
- **scope mismatch：**M273 是 T10 rank3 tile engine，T2 population 只在 algorithm/
  trace 层被统计，没有被该 RTL 的 state/accounting proof覆盖。

所以不得从 M515 写“ATLIF memory closed”或“full-system state memory closed”；只能写
“M273 T10 standalone live membrane-state boundary closed under frozen inference”。

## P1

1. RTL proof 是 substring/forbidden-name scan，不是 port parser、liveness或 use-before-
   overwrite proof；exact source 的人工审计通过，但 analyzer 应结构化该证明。
2. stale physical bits 在 release 后仍存在；只清 validity/ownership。修订所有
   `state_survives` 措辞为 semantic/live state。
3. M289 exact-source area事实可核，但 M515 没有直接冻结 M289/M302 evidence chain。
4. T2 RTL 与 network-wide weight/config/activation backing store 未覆盖；scope 必须
   收窄到 M273 T10 standalone boundary。

## 当前唯一合法论文句

> With calibration hooks and training-time threshold updates disabled, the
> frozen ATLIF forward maps each complete temporal tile independently. The
> exact M273 T10 RTL accepts context release only after all live ownership,
> pipeline, and result-FIFO state has drained; stale payload bits remain
> unobservable. Thus this standalone boundary requires no spatially indexed or
> cross-frame membrane-state SRAM. The exact-source M289 logic-only netlist
> contains 9,639 sequential standard-cell cells and zero macros; network-wide
> weight/config and activation storage, Fixed comparison, trained rank-3
> accuracy, power, and system performance remain outside this claim.

在 P0 修复前，不得使用原句中的“8,515-bit payload working set (plus tags/control)”，
也不得引用当前 `RUN_COMPLETE` 作为无条件 PASS。
