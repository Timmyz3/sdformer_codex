# M515r2 ATLIF state-boundary 独立打铁

日期：2026-08-27  
结论：`GO_CONDITIONAL_STATIC_BOUNDARY_ONLY__NOT_RUNTIME_COMPLIANCE`  
评分：**94/100**  
P0：**0**  
P1：**3**  
EDA / GPU / simulator 实际执行：**否**

## Literal verdict

M515r2 已关闭 r1 的两个 P0，可以准入**条件式静态边界句**。它不能升级为“实际
部署实例已经满足 frozen inference”，不能升级为系统 memory/PPA/performance，也
不覆盖 T2 RTL。

本轮独立核验的身份为：

```text
contract   319e07f7e1896da97c59a450cb61dad3363f9fb4db76dd8de34a1419fc75db37
analyzer   6c4d079c098ada7a639c0e2e15831b95a6c504d29a83bc2168ecc2427c6b54dc
deployment 7b5bd132567821a2e1690b1544e8d5e9b6303f54dec9ca9a9e1fad617c3691f1
result     ad763198cd6428ede76724cca9fac109cfceeea225b2886e13a5f4cf9aab7c66
```

当前 analyzer 的 recursive def-use 不是通用 Python 程序证明，RTL liveness 也仍
主要是 exact-source fragment binding；但在 pinned source 上，本轮人工展开没有
找到 output-affecting recurrent state 或 live state 穿过 release。这些限制列为
P1，不再阻断准确收窄后的论文句。

## 重放、seal 与篡改

- exact analyzer CPU 重放产生 JSON SHA `ad763198...`，与 sealed result
  byte-identical。
- result `SHA256SUMS` 内容 SHA 为 `1f38ce44248225bc7529d9a493d8b3b24c5ddcb090f0915ceb0ebea80ca5e804`；
  outer-seal 文件自身 SHA 为
  `3482f43298c9fb61e6bca44a3af035af8b06c9c2c55f94e7a129713bb74bb223`。
- deployment manifest 临时单字节 drift 被 SHA gate 拒绝，output 未创建；algorithm
  临时 symlink 被 regular/non-symlink gate 拒绝，output 未创建。
- sealed result JSON 临时篡改使 member `sha256sum -c` rc=1。
- M289 evidence manifest outer seal 与 M302 member/outer seals 独立复验通过。
- `docs/359` 未修改，仍为 `dedde7ce...`。

## r1 P0-1 closure｜条件部署与 output dependency

r2 不再把部署条件冒充实测：contract、deployment manifest、result 与 README 均
明确写出 `NOT_RUNTIME_MEASUREMENT` / `runtime_instance_compliance_measured=false`。
必须满足的条件完整列出：calibration hook absent、threshold/optimizer update disabled、
autograd disabled、parameters/buffers frozen、每次 forward 输入完整 temporal tile。

动态 callback 也从“被 AST 漏掉”改为 fail-closed identity：pinned forward 中必须
恰有 `_h9_calibration_observer` 这一 dynamic `getattr`，但部署合同要求它不存在。
因此它不会在条件式推理路径上接触 `thresh.detach()`。

新的局部 def-use 已修复 r1 的直接别名漏洞。本轮合成 probe：

```text
cache=self.update_value -> spike=cache -> out
```

现在得到 `self:update_value` dependency，并与 forward write 相交，能够拒绝。
对 exact forward，回溯集合含 `input:x_seq`、threshold、weight/factors、bias、center、
mode/rank/activation call target；forward 内写入的 rate/quantile/update/importance fields
与返回依赖交集为空。

人工继续展开 `self.act` 的四个可选 surrogate：其 forward 只使用调用参数和 autograd
ctx；在 contract 的 autograd-disabled 条件下，没有模块 recurrent state。算法核心
仍是 current complete `x_seq` 的 temporal matrix transform，没有 membrane tensor、
previous-frame 输入或空间索引状态。

### P1-1｜recursive 仍是 local、非 interprocedural

该分析器能沿普通 local-name 链递归，但仍不是通用 def-use：

- `owner=self; spike=owner.update_value` 被记录成 `local:self`，会漏掉 attribute alias；
- `spike=self.helper()` 只记录 `self:helper`，不分析 helper body；
- 当前 output 本身调用 `self.act`，其纯性由本轮人工源码审计补上，并非 analyzer证明。

此外 `act`、mode/rank、`sp` 等普通 Python configuration attributes 不是 Parameter 或
buffer。建议 deployment manifest 再加
`output_path_configuration_attributes_immutable=true`，并让 analyzer 对 `self` alias
和 allowlisted callees做 interprocedural 回溯。当前 exact SHA 无上述恶意 alias，
所以这是 P1，不是 P0。

## r1 P0-2 closure｜10,470 与 9,639 分列

20 项 RTL-declared sequential-state breakdown 独立重加为 **10,470 bit**：

```text
config capture/control/decoded  = 1536 + 5 + 1109
raw payload/metadata/control    = 2560 + 96 + 64 + 4 + 53
stage1 control/accumulators     = 6 + 1152
inter payload/metadata/control  = 768 + 96 + 64 + 4
stage2 + product                = 5 + 148
FIFO entries/pointers/count     = 2352 + 13
debug + context/retire/tile     = 320 + 115
total                           = 10470
```

该表明确包含 tag/beat/control，不再使用 r1 的 8,515 “excluding tags”错误句。
result也明确把 10,470 pre-optimization declared bits 与 synthesized cell population
当作不同 metrics。

M289 关系独立复算通过：

- `input_sha256.txt` 的 RTL line为 exact `11d5c6c4...`；
- mapped netlist有 9,638 个 `DFKCNQD1` + 1 个 `DFKCNQD2`，合计 **9,639 个一位
  sequential cells**；
- area report为 total cell 102,852.287739 um2、non-combinational
  19,432.224313 um2；M302 复算也是 9,639 sequential、0 macro；
- result保留 ideal-clock、ZeroWireload、logic-only 与 paper-PPA=false 边界。

所以 “working registers are charged in the exact-source M289 logic-only area” 合法；
不得把 10,470 与 9,639 相减解释成 SRAM compression，也不得把 9,639 cells 写成
9,639-byte/bit macro capacity。

## release｜stale 与 live 已正确分开

r2 wording 已修正：physical stale bits不会在 release 清零，但 no live/valid tile
state survives。exact RTL 的 `work_empty` 覆盖 partial raw fill、raw ownership、
stage1、intermediate reservation、stage2、product 与 FIFO count；release还要求
`raw_valid=0`。下一 context 中：decoded config 在 config complete后才使能 raw；
raw 五 beat complete后才 ready；stage1 phase0以零作为 accumulator base；inter、
product、FIFO payload 都在 valid/count 可见前覆盖。人工审计支持 stale bits
unobservable and overwritten-before-valid-reuse。

### P1-2｜RTL liveness 仍未被 analyzer 完整机械证明

analyzer 绑定了关键 fragment与 frozen RTL SHA，但没有结构化解析所有 ports、所有
valid/use-before-overwrite关系，也只用四个名字搜索 external state port。对当前 exact
source人工审计足够；为长期防 drift，建议把完整 live-state invariant/overwrite表
纳入 analyzer，或后续以 release→next-context SVA封存。该 P1 不要求新 datapath。

## scope 与 conditional contract

scope现在清楚且未偷换：

- inside：M273 T10 standalone 的 local config capture/decoded registers、raw/inter/
  product/FIFO 与控制状态；
- outside：T2 RTL、network-wide weight/config backing、activation buffers、NoC/DRAM、
  Fixed、trained rank3 accuracy、SAIF/PTPX、cycles/energy/system speedup、paper PPA；
- deployment manifest是**要求未来/实际部署遵守的合同**，不是本轮测得 hook absent
  或 model instance frozen。论文只能作条件陈述，不能写 “we measured all deployed
  instances stateless”。

### P1-3｜result pack 未自绑定 analyzer/audit contract，部分跨证据关系靠已知 SHA

result JSON identity已包含 deployment、RTL、M289/M302与 docs，但没有把 r2 analyzer
自身或 r2 audit contract写入 identity；README也没有列这两个 SHA。analyzer pin了
M289 input/manifest/seal，但没有程序化解析 input file中的 RTL relationship或执行
seal check，当前关系由 exact known hashes和本轮独立复验补足。

建议下一版将 audit contract/analyzer identity写入 result，并显式检查 M289 input
RTL line、manifest outer seal、M302 member/outer seal。当前封存 triple由本 hammer
直接绑定，故非 P0。

## 唯一合法论文句

> Under an explicit deployment contract—not a runtime-compliance measurement—
> requiring calibration hooks absent, autograd and training-time threshold and
> optimizer updates disabled, and parameters, buffers, and output-path
> configuration frozen, the pinned ATLIF forward maps each complete temporal
> tile independently. The exact M273 T10 RTL accepts release only after all
> live ownership, pipeline, and result-FIFO state drains; stale physical bits
> remain unobservable and are overwritten before valid reuse. Thus this
> standalone boundary requires no spatially indexed or cross-frame membrane-
> state SRAM. The exact-source M289 logic-only netlist contains 9,639 one-bit
> sequential standard-cell cells and zero macros; T2 RTL, network-wide weight/
> config and activation storage, Fixed comparison, trained rank-3 accuracy,
> power, energy, and system performance remain outside this claim.

该句是 boundary/completeness 句，不是硬件性能贡献或 DATE headline。
