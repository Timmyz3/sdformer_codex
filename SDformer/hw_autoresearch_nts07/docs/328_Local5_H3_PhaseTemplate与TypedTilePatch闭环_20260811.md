# Local5 H3 Phase-Template 与 Typed Tile-Patch 闭环

## 1. 本轮裁决

本轮完成的是 formal phase/event archive 的单窗可扩展表示 canary，不是新的数值算法
或硬件性能优化。

```text
sealed v9 H3 deferred-valid          REJECT，保留为负结果
sealed v10 H3 held-valid runtime      PASS [rtl]+[rtl-build-provenance]
phase-template + typed tile-patch   PASS [rtl-trace-derived]
independent row-by-row expansion    PASS 862,507/862,507
Acc32                               PASS 43,200/43,200，零失配
template/patch tamper regression    PASS 9/9 被拒绝
formal G0                           DENY
```

有效结果：

- `results/local5_h3_phase_template_patch_canary_v2_20260811`
- `results/local5_erep_numeric_rtl_release_v10_phasepatch_20260811`
- `results/local5_h3_phase_template_tamper_regression_v4_20260811`

首次 v8 generator dry-run 因 binding SHA 长 64 字节而 `origin_dictionary=S24` 失败，
保留在 `results/local5_h3_phase_template_patch_v8_generator_dryrun_20260811`，不计
PASS。修正为 `S64` 后才生成 v2 dry-run 和正式 v9 archive。

v9 的 3.5/5 独立复审进一步发现：当时 `weight_rsp_valid` 在 hold 期间被拉低，只是
把 valid 向 DUT 延后两周期，不是真正 held-valid。v9 保留为负结果，不得宣称
backpressure 覆盖。v10 把 `weight_rsp_valid/ready` 明确定义为服务适配器上游接口，
DUT 侧另用 `dut_weight_rsp_valid/ready`；hold 期间真实出现 `valid=1,ready=0`，并逐周期
写入 protocol telemetry。

## 2. 为什么必须保存 tile patch

同一结构类型的 head/tile 执行具有相同 `event,origin` 序列，但 identity-derived
service 的 delay 和绝对 cycle 依赖 tile/head/source/lane/out 身份。静态复制一个
模板不能重建真实执行。

本轮将 archive 拆成：

1. **结构模板**：每类只保存一次 `event_code,origin_code`；
2. **实例表**：保存 template class、tile、head 和 patch offset；
3. **typed tile patch**：逐行保存 cycle、tile/head/source/lane/out、delay、flat index
   与 payload dictionary code；
4. **payload dictionary**：去重 available/accept 和跨 tile 复用的 payload。

patch 不是“异常差分”，而是精确语义的一部分。容量结果包含 patch，不再沿用早期
只算 base-template 的乐观数字。

## 3. 七类结构模板

| 模板类 | 实例 | 模板行 | 展开行 |
|---|---:|---:|---:|
| prefix | 1 | 7 | 7 |
| head_seed | 3 | 52,630 | 157,890 |
| inter_head_gap | 6 | 2 | 12 |
| head_accumulate | 6 | 81,430 | 488,580 |
| tile_tail | 3 | 72,003 | 216,009 |
| tile_transition | 2 | 3 | 6 |
| suffix | 1 | 3 | 3 |
| 合计 | 22 | 206,078 | 862,507 |

base event 复用因子为 `862507 / 206078 = 4.185x`。该数字只描述 `event/origin`
骨架复用，
不包含 patch 成本，不能写成完整存储缩减。

## 4. 完整 archive 容量

| 项 | 字节 |
|---|---:|
| 原始候选 CSV trace | 52,519,869 |
| 完整 NPZ archive（含 typed patch） | 24,734,032 |
| 文件大小缩减 | 2.123x |
| payload dictionary entry | 56,467 |

NPZ 的 typed array `nbytes` 合计为 24,728,046，与文件大小接近，说明结果不是靠
ZIP 对重复文本做不可控
压缩。2.123x 只用于验证 archive 存储，不是片上 SRAM、周期、
功耗或 ASIC PPA 收益。

## 5. 独立展开与 v8 等价

独立 verifier 不导入 generator，自己检查 NPZ 精确 member set、dtype、offset、字典
边界和 instance/template 长度，然后逐行展开 11 列 CSV。

| 检查 | 结果 |
|---|---|
| 展开行数 | 862,507 |
| 展开 trace SHA | `1ea50c7b56dc3f57c750af732d99a12f73f68c13d67851dcc5f40349f749cfa6` |
| 与源 candidate trace | 逐字段、逐行、byte-stream SHA 完全一致 |
| handshake cycle-free ledger | 126,198 条，与 v8 一致 |
| boundary cycle-free ledger | 26 条，与 v8 一致 |
| state cycle-free ledger | 717,849 条，与 v8 一致 |
| core-all cycle-free 全序 ledger | 844,073 条，与 v8 一致 |
| Acc32 | 43,200，candidate/v8/software expected 三方一致 |

cycle-free ledger 去掉的是合法服务等待造成的绝对周期差，仍保留 event、身份、delay、
index、origin 和 payload。它不能替代 cycle-exact 性能比较。

## 6. 非空洞 weight backpressure 与 payload 解码

v10 release 在 TB 服务适配器上游边界冻结
`FORCE_WEIGHT_RESPONSE_HOLD_CYCLES=2`。服务侧 `weight_rsp_valid` 在 hold 期间保持为
1，`weight_rsp_ready` 为 0；DUT 侧使用独立的 `dut_weight_rsp_valid/ready`，在 adapter
允许接受后才完成 transfer。数值 DUT 未修改。

| 检查 | 数量 | 结果 |
|---|---:|---|
| relation payload 逐字段解码 | 4,050 | 与真实 Q/K/mask、tag/identity 一致 |
| weight payload 逐字段解码 | 9,216 | 与真实 INT8 权重、tag/identity 一致 |
| final payload 逐字段解码 | 43,200 | 与 actual Acc32、坐标、last 一致 |
| weight available→accept delta | 9,216 | 全部恰为 2 cycle |
| `valid=1,ready=0` telemetry | 18,432 | 每个 response 连续 2 cycle |

因此 v8 复审的“weight 没有 held-valid 覆盖”和“payload 只比相等、不解码内容”两项
P2 已在本轮变为实测证据。

## 7. 负测试

密封独立 expander 对以下九类 archive 篡改均 fail-closed：

| 篡改 | 结果 |
|---|---|
| 修改一个 template event code | PASS_REJECTED |
| 修改一个 tile patch cycle | PASS_REJECTED |
| 修改一个 payload dictionary code | PASS_REJECTED |
| 修改 instance class code | PASS_REJECTED |
| 修改 patch offset | PASS_REJECTED |
| 修改 instance tile identity | PASS_REJECTED |
| 修改 instance head identity | PASS_REJECTED |
| 制造 dictionary code 越界 | PASS_REJECTED |
| 修改 patch identity | PASS_REJECTED |

篡改用 manifest 已重算 archive SHA，故失败来自逐行展开不等价，而不是只靠文件 SHA
挡住。

## 8. 证据边界

- `[rtl]`：sample2/stage0/block0/window249/H3 单窗；
- `[rtl-build-provenance]`：v9 同时密封构建 H3/H6/H12/H24，仅 H3 已运行；
- `[rtl-trace-derived]`：模板和 patch 来自真实 v9 trace；
- 699,951 cycle 是带定向服务 hold 的验证环境时延，不能与 v8 681,519 cycle 作架构
  性能比较；
- 当前尚未把 template archive 独立展开成旧 `WindowCommandWork`；
- 尚未运行 H6/H12/H24 template canary，也没有 100-sample/1,200-window archive；
- formal G0 保持 `DENY`。

## 9. 下一步

独立 DATE 复审通过后，按一个缺口一轮推进：

1. 把单窗 template archive 独立适配成旧 `WindowCommandWork`，对 phase/resource/
   identity ledger 做同构 miter；
2. 再扩 H6/H12/H24，检查模板类是否需要按 head-role 或 stage 分裂；
3. 根据多规模 patch 密度重算 1,200-window 容量与 wall-time；
4. 以上完成后才恢复 formal archive 生成和 G0 admission。

## 10. 两轮独立 DATE 复审

首轮对 v9 的评分为 `3.5/5 Reject`。P1 是所谓 weight held-valid 实际只延后 valid，
并未出现接口 `valid=1,ready=0`。该意见成立，v9 已降为负结果。

修复后的 v10/v2 复审为 `4.5/5 Accept`，但只接受 H3 单窗 canary。审稿代理独立
重算确认：

- 9,216 个 weight response 各有连续 2 个服务侧 held-valid 周期；
- 18,432 条 stall telemetry 的 identity/payload/cycle 均正确；
- core-all cycle-free 全序 ledger 在 v10/v8 两侧均为 844,073 条，SHA 同为
  `0b0c501d5fe61ec0fc6983cc5dd1cf75c4703960217d252e0d070165ccd13f3b`；
- 43,200 Acc32 零失配；
- 9 类 template/patch 篡改全部 fail-closed；
- complete 直接绑定与 v10 release 复验通过。

复审保留三项 P2：当前覆盖的是服务侧 producer+hold-adapter，不是 DUT 自身 deassert
ready；尚缺单独 `instance_head_flip`；sample/stage/block/window、PASS 事务数与 baseline
仍是 H3 固定口径。因此允许进入“参数化后再跑 H6”，不允许直接宣称 H6/H12/H24
通过。
