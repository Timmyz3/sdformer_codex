# M1617｜M1613 C2 registered-fault directed source 不同作者 hammer

日期：2026-09-01

状态：`PASS_M1617_M1613_C2_REGISTERED_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT`

评分：99/100；P0=0，P1=0，P2=1。

## 裁决

M1613 的 exact-SHA source-only 包通过不同作者静态审阅。M1617 只准许在另行生成、双封且绑定 runner、source contract 与本 `review.json` SHA 的 M1618 release 存在后，消耗一次未来 VCS compile 和一次 simv；本轮没有运行 VCS、simv、DC、Formality、PT 或 PTPX，也没有创建 release。

## 语义闭合

- filelist 恰好两行，只选择 M1609 successor 和 M1613 TB；同名冻结 M214 predecessor 不会同时 elaboration。
- 合法 terminal raw/descriptor 在同周期接受，采样沿后 `raw_valid` 故意 linger，确认组合 `illegal_request=1`，同时要求公开 `protocol_error=0` 且内部 `fault_q=0`。
- 非法 header/raw 在呈现周期均为 `ready=0`、`accept=0`、`protocol_error=0`；上升沿后 `#1ps` 要求 `fault_q=protocol_error=1`。
- sticky SVA、两类 sticky directed check、三次 reset clear 都在 exact TB 中；共 12 个 `#1ps` settled sampling 点。
- PASS token、receipt 和 contract 全部固定 `performance=false`；该运行不产生 cycle、speedup、area、power 或 paper claim。

## 故障注入

独立审计器实际拒绝了以下变异：同时注入旧 M214 定义；删除 settled sampling；弱化合法 terminal 假错检查；删除非法 header/raw 锁存检查；破坏 sticky/reset；把 VCS compile 或 simv 扩成两次；绕过 M1617/M1618；让 other-UID simv 误阻或漏掉 same-UID 冲突；改变 result namespace；把 attempt 移到工具之后；注入 performance/speedup claim。

Python 3.6 与 Python 3.12 均为 14/14 PASS，作者静态测试两环境均为 9/9 PASS，runner `bash -n` PASS。M1611 和作者交接树的内外 seal 均通过。

## 一次性执行边界

M1618 release 尚不存在，因此当前仍不可执行。未来 release 必须精确绑定：

- runner SHA `f2b3888879cb5a6af4396eb8b4971510453a47622299e17dd6702925587c0b29`；
- source contract SHA `248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d`；
- 本 review SHA；
- 权限恰为一次 compile、一次 simv、seed 1613、零自动重试、零其他 EDA。

runner 在所有 SHA、fresh review/release、fresh namespace、same-UID collision 和环境门之后，才原子创建 attempt；失败不会回收 attempt。

## P2 与红线

本 TB 只覆盖 compactor-local fault boundary。它没有覆盖 M1611 要求的 M216/service outer error OR-chain。因此未来 directed PASS 也只能证明本地 registered-fault 时序，不能宣称集成错误传播、性能或论文结果。

docs/359 未修改，SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
