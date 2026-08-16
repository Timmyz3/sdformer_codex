# Local5 ep44 十二 Block 有界生产 Job 回放

日期：2026-08-15。

## 结论

Local5 ep44 已新增一项严格受限的 `[rtl]` 生命周期证据：从密封的 100-group population 中，为 12 个 Local5 block 各选择一个真实 `OUT_DIM=2` group，经现有生产 tagged-job 路径执行：

```text
raw Q/K
-> Query-Silent / score / Shiftmax5
-> inverse-stencil relation frontier
-> source-owned term
-> TCFM5
-> Acc32 readback
```

Icarus 与 Verilator `--assert` 的逐 block 账本完全一致。12/12 jobs、5,400 个 token response、768 个真实 checkpoint INT8 weight response 和 10,800 个 Acc32 result 均通过，Acc32 零失配；token、weight 和 result 三类随机反压均实际命中。Yosys `hierarchy -check` 与 `check -assert` 报告 0 个结构问题。

该结果只补充 **12 个 block 的独立 tagged-job 生命周期覆盖**。它不是同一 sample/window 的 138-head cohort，不含 cross-head reduction，也不是 1,320-window encoder 调度或性能评估。

## 身份

- checkpoint：Local5 ep44；
- checkpoint SHA256：`19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57`；
- 输入 population：ep44 score-projection 100-group sealed vectors；
- 数据：真实 raw Q/K、hardware-order Q7/Q1.7、真实 checkpoint INT8 权重；
- 输出：每个 job 900 个 pre-bias/pre-BN/pre-requant/pre-residual Acc32 值；
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未改变。

主证据：

- `results/local5_ep44_12block_job_replay_v3_20260815/report.json`；
- `results/local5_ep44_12block_job_replay_v3_20260815/complete.json`；
- report SHA256：`fabeea3d7cd6ac8c2f3be67a8fa159aaa5abf997a6efc32cb143ed5d01382871`；
- plan SHA256：`5ca509725377f7b84cc994066f53ba145732f9689da3e534875209843a3c0c38`。

## 选择合同

选择规则是：每个 block 取首个 nonempty group；若该 block 在 100-group population 内没有 nonempty group，则取首个 group。因此 12 个 job 中 10 个 nonempty、2 个 empty。

这是为了覆盖两类控制路径的 **coverage-seeking correctness selection**，不是 outcome-independent 性能抽样。82,063 个仿真周期及各 block 周期只能用于双仿真账本一致性检查，禁止作为论文性能、加速比或 encoder 外推。

12 个 job 来自不同 sample、window、head 和 output pair。把它们顺序运行只验证 tag、权重上下文、结果退休和跨 job 清理，不验证真实网络中的 block-to-block 数据依赖。

## 负向证据与修复

首次 plan 将 output pair 错误映射为 `output_channel/2`，与生产接口按 32-channel output tile 编号的合同不一致。旧 plan 目录被保留，不作为证据；v2 改为 `output_channel/32`，同时记录 tile 内 channel offset。

首次仿真还暴露了 TB 的同拍竞态：任务线程在 `job_done` 握手上升沿切换 `current_job/job_tag`，监视线程可能用新 tag 校验旧完成事件。最终 v3 修复为任务线程事件式等待 `completed_jobs` 记分板，并在后续下降沿读取稳定账本后再启动下一 job。生产 RTL 未修改，失败目录未冒充通过结果。

旧 plan 已显式改为 `REJECTED_SUPERSEDED`，首次失败结果目录增加了 `REJECTED.json`，v2 增加了 `SUPERSEDED.json`，避免后续脚本误把旧 `PASS` 字段、不完整日志或存在诊断调度歧义的旧包当成最终证据。

## Claim 边界

可以写：

> Local5 ep44 的现有生产 tagged-job 链在覆盖全部 12 个 block 的独立真实 group 上，通过双仿真、SVA、随机反压和真实 INT8 Acc32 bit-exact 生命周期回放。

不能写：

- 完整 12-block encoder 已实现或数值闭合；
- 同一 window 的全部 head 已在 RTL 中归约；
- 82,063 cycles 是网络性能；
- 全输出、bias、BN、requant、residual 或 decoder 已闭合；
- 本结果提高 DATE 架构创新分；
- 本结果替换 `docs/359` 的冻结性能列；
- H81 已有 RTL。

本 TB 会核对 DUT 请求的 job tag、head、output tile、lane 和 out，并从当前选中 group 的密封权重对中返回数值；它不独立证明全局 output tile 到权重存储地址的变换。该地址合同需由完整 weight-loader/encoder shell 另行验证。

## 仍缺的严格系统证据

若要把当前结果升级为同窗全 head 的生产数值证据，需重新导出一个预注册 sample/window 的完整 138 个 head-row，并执行 12 tiles、138 head jobs 和 cross-head reduction；现有 ep44 100-group population 对任一 `(sample, stage, block, window)` 最多只有两个 head，不能从当前向量拼出该证据。

DC、Formality、PTSTA、SAIF/PTPX 仍在另一台有 Synopsys 和目标库的服务器执行。本结果不要求重打现有 `194436Z` 交接包，因为该包的论文活动合同仍是旧冻结身份，本次结果也不改变 frozen PPA top。
