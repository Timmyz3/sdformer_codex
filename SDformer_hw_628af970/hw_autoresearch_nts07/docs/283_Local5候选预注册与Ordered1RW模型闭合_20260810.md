# Local5 候选预注册与 Ordered 1RW 模型闭合

> 日期：2026-08-10  
> 前序：`docs/282_Local5同窗全Head硬件统计预注册_20260810.md`  
> 本轮唯一问题：在正式 profile 揭晓前，冻结一套不可 post-hoc 换参的
> Local5 候选、强基线和合法 1RW 决策模型。**本 v1 已在正式 profile 前被第二轮独立评审否决，不得用于候选裁决。**

## 1. 结论

首轮 DATE 评审的两个表面 P0 曾按下列方式处理：

1. `[rtl]` 首次触达直写与复访 RMW 的接收间隔已用真实
   `qfit_direct_1rw_acc_bank` 锁定为 `1/2` 拍；
2. `[模型]` 新 reference 按 source、lane、gate、destination 的 RTL 顺序重建 term，
   显式模拟 first-touch、RMW、五 bank 原子组播、跨 head valid preserve、
   flush、vector readout 和共同 scalar serializer；
3. `[prereg]` `C0--C3` 候选、参数、stage 并行度、bootstrap seed、
   `1.20x` 门槛及源码 SHA 已写入预注册合同；
4. `[test]` 10 个 Python 测试与 1 个 RTL timing lock 通过；
5. `[test]` 13800 head group、621 万 source descriptor 的四候选端到端合成
   规模干跑通过，耗时约 `2m38s`、峰值内存约 `262 MiB`。

这些结果只闭合“模型与决策合同能否运行”。第二轮评审进一步证明模型
边界不公平，因此真实 Local5 结果即使生成，也不得交给本 v1 裁决。原结论仍为
`[待验证]`，不得引用合成 fixture 的候选比率作为算法或架构收益。

v1 合同状态：`INVALIDATED_BEFORE_PROFILE`。替代版本见
`docs/284_Local5公平Ordered前端候选V2预注册_20260810.md`。

## 2. 冻结候选

| 候选 | Relation | Projection Acc | 定位 |
|---|---|---|---|
| C0 Direct-Recompute | 每 output tile 精确重算 | 直接合法 1RW TCFM5 | 强基线 |
| C1 SRAC2-Recompute | 每 output tile 精确重算 | 每 bank 两槽 source-resident Acc context | 短生命期候选 |
| C2 Direct-ERM7 | 7 KiB critical-only exact memo | 直接合法 1RW TCFM5 | 跨 tile 候选 |
| C3 SRAC2-ERM7 | 7 KiB critical-only exact memo | 每 bank 两槽 source context | 双生命期主候选 |

预注册文件：

```text
contracts/local5_joint_candidate_prereg_20260810.json
SHA256=11dbcb3eff33e23617782b97f6886ccecfd3e59be938eb195071561e0f764e5f
```

参数冻结为：

- SRAC2：每 bank 两个 context，descriptor latency=`3`，每 head 边界 flush；
- ERM7：`7 KiB`、`512x112-bit`、critical-only、head-order admission、容量 miss
  精确 fallback；
- stage head/output tile：`3/6/12/24`；
- 四候选共用 B2v 跨 head preserve 边界、450 个最终向量读与
  `450x32` scalar serializer。

B2v 只是公平实现边界，已被 `docs/281...` 否决为独立贡献。

## 3. Ordered 1RW 参考模型

源码：

```text
scripts/local5_joint_candidate_reference.py
SHA256=493300ec98b8ecd735fc61d73ab81ae7184065cdecbb06c5d0e88a9f91c78e6d
```

### 3.1 Term 顺序

与 `qfit_source_multicast_term_builder` 一致：

```text
source id ascending
  -> active K lane ascending
    -> unique nonzero gate in first-role-occurrence order
      -> five-color destination multicast
```

同一 source 在一个颜色 bank 中最多对应一个 destination address。同 term
的五个角色在颜色 bank 上互异，因此各 bank 并行，但 term 作为原子
组播单元，只要任一目标 bank 复访就等待该 RMW 完成。

### 3.2 Direct 1RW

```text
first touch: 1 write command, initiation interval = 1
revisit:     1 read + 1 write, initiation interval = 2
mixed term:  any revisited destination makes the atomic term interval = 2
```

head0 以 `run_accumulate=0` 清 valid metadata，后续 head 以
`run_accumulate=1` 保留同一 Acc 空间。最后 head 后才执行一次 vector readout。

### 3.3 SRAC2

SRAC2 不试图缓存通用 token，而是利用 Local5 五色几何下“一个 source
在每 bank 仅有一个目标地址”的性质，在寄存器 context 内合并该 source
的所有 lane/gate 整数增量。模型显式计入：

- dirty victim writeback；
- 已 materialized 地址的 refill read；
- 五 bank 操作并行的最大延迟；
- 下一 source descriptor latency 未被当前 source term 隐藏的停顿；
- 每 head 边界的 dirty context flush。

## 4. 配对决策统计

评估器：

```text
scripts/evaluate_local5_joint_candidates.py
SHA256=08a982236b934824a0ae5f8dccb34fd61e5aa7965b98dd4df957c7d1b92b22bd
```

每个 sampled joint-window 在四候选下使用同一批 head descriptor、同一 output
tile 数和同一最终 serializer。每个 sample 用 `analysis_weight=stage_windows`
汇总为 HT 整帧周期估计。

对 C1/C2/C3 分别对 C0 做：

1. 100-sample paired bootstrap；
2. 18-sequence cluster paired bootstrap；
3. 总体与每 stage 的 inverse-probability-weighted window p95。

唯一晋级门槛：

```text
min(sample one-sided 95% LB, sequence one-sided 95% LB) >= 1.20x
AND overall weighted window p95 non-regression
AND every-stage weighted window p95 non-regression
```

过门槛只表示 `PROMOTE_TO_MINIMAL_RTL`，不表示已经成为 DATE 贡献。

## 5. RTL Timing Lock

入口：

```bash
bash sim_qfit/run_local5_joint_candidate_reference_checks.sh
```

`[rtl]` 接收周期为：

```text
accepts=2,3,5,7,10,12
```

前四项验证 first-touch 后可隔 `1` 拍继续发射，而复访 RMW 使后继事务
间隔为 `2` 拍。后两项验证 `run_accumulate=1` 后已有地址仍走 RMW。

## 6. 正式规模干跑

合成 fixture 形状与正式合同一致：

| 项目 | 值 |
|---|---:|
| sample | 100 |
| block | 12 |
| joint window | 1200 |
| head group | 13800 |
| source descriptor | 6210000 |
| evaluator wall time | 约 2m38s |
| evaluator max RSS | 约 262 MiB |

fixture 特意设为每 source 只有一个短 term：SRAC2 无法用当前 term 隐藏
descriptor latency，ERM7 也因 direct service 不小于 relation build 而不 admission。
因此四候选裁决全为 `REJECT_MODEL_PROMOTION`，证明决策器不会无条件偏向
新候选。这些 fixture 比率不得进入论文结果。

## 7. 证据边界

| 声明 | 证据 |
|---|---|
| Direct 1RW 首触/复访间隔 | `[rtl]` 极小 timing lock |
| source/lane/gate 顺序、bank 地址、SRAC2/ERM7 周期 | `[模型]` |
| 预注册候选和 SHA | `[prereg]` profile 揭晓前冻结 |
| 13800-group 可执行性 | `[test]` 合成数据 |
| 真实 C0--C3 周期与晋级裁决 | `[待验证]` |
| SRAC2/ERM7 整合 Acc32 bit-exact | `[待验证]` 只在模型过门槛后执行 |
| OpenROAD/DC/STA/SAIF/PTPX | `[待验证]` |

## 8. 第二轮独立 DATE 评审

第二轮独立审稿结论为 `Reject / Major Revision`：证据包 `2.8/5`、
Ordered 1RW 模型可信度 `2.1/5`、架构筛选严谨性 `2.2/5`、DATE 就绪度
`2.0/5`。正式候选 watcher 被禁止启动。

四个 P0 为：

1. Direct 漏计公共 active descriptor capture，SRAC2 却单独加入
   `descriptor_latency=3`，比较不公平；
2. 模型让 SRAC2 跨 head preserve，但仓库 GASR2C `run_start` 会清
   backing-valid 和两槽；
3. ERM admission 随 Direct/SRAC2 后端变化，且 replay/fallback 控制偏理想；
4. v1 文件均未跟踪，缺少 profile 输入绑定的不可变预注册锚点。

P1 还包括共同 readout/serializer 未做端到端 timing miter、sequence 顺序绑定
不足、多候选未控制 family-wise error，以及 runner 未运行 evaluator 单测。

这些问题已在 v2 中进行模型级修复和预注册；GASR2C-P preserve RTL 与共同
readout timing 仍明确为 `[待验证]`。v2 必须再次独立评审，不能因代码已改就
自动视为 P0 闭合。

原计划检查项为：

1. Direct 1RW 首触/RMW/跨 head preserve 是否真正对齐 RTL；
2. SRAC2 的两槽、预取、驱逐、flush 模型是否有隐含理想化；
3. ERM7 的 admission、容量与 frontend/backend 重叠是否与既有 RTL 合同一致；
4. 共同 B2v 边界和 scalar serializer 是否公平；
5. 预注册 SHA 是否足以阻止 profile 后换参。
