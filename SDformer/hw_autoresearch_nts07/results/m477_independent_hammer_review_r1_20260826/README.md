# M477 失败 DC 独立 hammer（2026-08-26）

## 裁定

**94/100，P0=0、P1=1；runner 拒绝准入完全合理。**

正式 verdict：`PASS_FAIL_CLOSED_NO_PPA_GO_BOUNDED_TWO_VARIANT_DSE`。

M477 是一份有价值的失败诊断，不是通过的 DC/PPA 点。42,370.649130 µm² 等数字只能帮助选择下一步结构，不能进入论文 PPA、性能或能效表。

## Receipt-blind 报告复核

独立脚本直接读取 sealed DC reports，而非用 failure receipt 作为数字来源：

- Cell area：42,370.649130 µm²；
- Cells：41,849，其中 sequential 5,508、combinational 36,340；
- Macro：0，ideal clock，ZeroWireload；
- Setup worst slack：0.0000 ns，MET；
- Hold timing worst slack：+0.0101 ns，MET；
- Max capacitance：2 个违反，total slack -0.0363；
- Max transition：1 个违反组，total slack -0.2161 ns；
- Max fanout：3 个违反，fanout 为 80/32、61/32、57/32，total slack -102。

必须特别纠正口径：**-0.2161 不是 hold slack，而是 max-transition slack。**当前 sealed failure receipt 已按这个分类记录，且没有遗漏 max-capacitance。

## 为什么 fail-close 是正确的

`dc_shell` 正常完成并返回 0，setup/hold path summary 也为正；但 runner 要求 max-delay、min-delay、max-capacitance、max-transition 和 max-fanout 五组约束全部无 violation。实际只有前两组 clean，因此 runner 以 33 退出并写入：

```text
status=FAILED_OR_INCOMPLETE_DO_NOT_CITE
runner_exit_code=33
```

这避免了“setup summary 为 0 就算 PPA 通过”的错误。Failure receipt 中所有 admission 字段均为 false；manifest 覆盖 26 个非 seal 文件，manifest 和 outer seal 均校验通过。docs/359 未修改。

## P1

评审过程中发现并已在最终 sealed failure receipt 中修正的分类/算术为：

   ```text
   42370.649130 / 37316.285232 = 1.135446598
   delta = 13.5446598%
   ```

最终 receipt 现已正确写成上述 `1.135446598 / 13.5446598%`，并正确区分 max-transition、max-capacitance 与 hold timing；最终 reconciliation mismatch 为 0。因此不再记作 open P1。

唯一 remaining P1：`+13.54%` 是 whole-design diagnostic delta，不是因果 ablation。Sequential cells 增加 999、combinational cells 增加 5,516，说明方向值得优化，但不能把全部差异直接归因于第二个 1152-bit slot；coherence/control 和 DRC buffering 也混在其中。

## Bounded next gate

只允许两个变体，不继续无界扩展：

1. 首选：一个 full-width 1152-bit slot，加窄的 valid/ID/source skid 或 replay。发生 macro response + RAW forward 双入队时，对年轻请求显式 backpressure/replay，不再用第二个 full-width 标准单元数据槽。必须重跑 M478 九条 base cover、old=5/new=1 targeted attack，以及重复 collision/no-loss/no-reorder 压测。
2. 只有拿到真实 SRAM `.db` 和行为模型时才做：把 registered/holdable SRAM output 当第一 response stage，只保留必要 valid/ID 或 bounded skid。Macro output register 不能免费切掉；面积、时序和功耗必须包含在 macro 合同内。

共同决策门：

- 独立 VCS hammer 先通过；
- 同一 3.000 ns SDC 与 slow/fast library 重跑；setup/hold、capacitance、transition、fanout 全 clean，否则立即 NO-GO；
- clean final cell area 至少比 M477 诊断点低 5%；
- 同一冻结 transaction trace 的额外 cycle 不超过 dual-slot 的 2%；
- 两个变体都失败则停止 queue DSE，本轮不再开第三种结构。

选中唯一变体后再做 Formality。M473 performance、system speedup 和 DATE headline 在整个过程中仍为 false。

## 复核

```bash
python3 results/m477_independent_hammer_review_r1_20260826/audit_m477_independent.py \
  --root .
```
