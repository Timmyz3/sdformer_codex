# M409：M408 H67 q32 全量静态 codec VCS 独立打铁

结论：**PASS，允许进入真实 q32 配置/模式 miter 与 selected standalone DC。**
评分 95/100，P0/P1/P2=`0/0/5`。本里程碑没有新速度。

## 独立结果

我先校验了 M408 stimulus 和原 VCS run 的 inner manifest + outer seal，再从
M401 开始逐项复核输入 SHA 与身份链：H67 ep35 checkpoint、`no_running`、M40
冻结的 `zurich_city_09_a` 十样本、M41 checkpoint-bound INT8 weights，以及
M338 到 M73 disjoint train-only S32 catalog 的链接均未漂移；PAFT checkpoint 未被
用于这条 population。`docs/359` SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

独立 Python 审计没有调用 M408 exporter，而是从四份 M41 权重和 M338 q32 center
重新计算全部 442,368 条物理记录。逐条检查 1281-bit 布局、321 hex digits、
`operator/partition/center_id/global_output_block` 顺序、narrow sign extension、wide
`zero-extended low8 + signed high4<<8`、128-bit padding，结果全部 0 mismatch：

- blocks/source lanes：`442,368 / 42,467,328`；
- narrow/wide：`112,167 / 330,201`；
- contributions：`772,569`；
- checked contribution lanes：`74,166,624`；
- 数值范围：`[-1089, 1059]`。

这里必须固定命名空间：42,467,328 是独立 source lanes；74,166,624 是 wide
记录拆成 low 与 signed-high 两个贡献后被检查的 contribution lanes，不能把后者
包装成 source lanes。

随后从 exact-SHA RTL/SVA/TB 重新编译 Synopsys VCS V-2023.12-SP1 并全量复跑，
compile/sim rc 均为 0，复现 blocks/source lanes/narrow/wide/contributions=
`442368/42467328/112167/330201/772569`，metadata、arithmetic、narrow semantic、
padding、protocol、assertion 均为 0 mismatch/failure。

## `cp_wide_pair=330200` 的判定

原 SVA 报 330,200 次 wide pair，确实比 wide block 330,201 少一条；最后一条
stimulus 恰好是 wide。为避免凭经验解释，我又在不改主线 RTL/SVA/TB 的条件下
绑定了一个独立 procedural probe，并用 Synopsys VCS 再跑一次全 population：

`narrow/wide_low/wide_high/wide_pair/pending/order_error =
112167/330201/330201/330201/0/0`。

因此功能上不存在少发或乱序。主 TB 在最后一个 wide-high 被接受并 drain 后立即
`$finish`，VCS 对两拍 SVA cover 的终拍收口少记一条。这是 P2 coverage accounting
问题；功能 checker 和独立 probe 都覆盖了最后一对。

M408 全量测试固定 `contribution_ready=1`，所以 `cp_output_stall=0`、`cp_fault=0`。
这不升级为 P1，因为 exact-SHA M405R3 directed VCS 已覆盖 147 个 output stalls、
4 个 adapter attacks、2 个 shell global-fault attacks，且 atomic leak=0；但不能声称
已经覆盖“全 population × backpressure/attack”的交叉场景。

## 边界和下一步

M40 是从 M401 继承的冻结 runtime context；M408 静态 stimulus 实际枚举的是 M41
权重与 M73/M338 catalog center，不消费 M40 runtime rows。因此它证明 codec 全域，
不证明真实 q32 matcher 的 occurrence/order。

允许下一步：

1. 跑真实 q32 configuration/pattern miter，覆盖真实 M40 row、两 pass、early stop、
   tie、phase release 与 M384 双 replay 边界；
2. standalone leaf DC 可并行启动；integrated selected-slice DC 的证据应在真实 miter
   之后收口；
3. 后续仍需 Formality、PrimeTime，以及真实 SAIF 后的 PTPX。

本里程碑只准入 full static codec VCS。它没有新 speedup，也不把 1.156371x 升级成
RTL-measured/system speedup；没有能量、物理 SRAM、paper PPA 或 DATE headline。

独立 reviewer 首次 wrapper 在 compile 前因把 `RUN_MANIFEST.sha256` 第一行内容哈希
误当成 manifest 文件自身哈希而 fail-closed；该目录保留。更正后 exact-SHA 全量复跑
与 terminal-pair 第二次 VCS 均通过，没有修改任何主线 RTL/SVA/TB/contract/docs359。
