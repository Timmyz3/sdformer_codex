# M1568：M1502 C2 mapped 首拍失败只读法证

结论：**M1502 已消费且失败，不能重跑；现有日志不能把三位 DUT fault 解聚，但第一嫌疑是 `protocol_error` 锥中的 mapped 四态/X 污染，而不是 endpoint、数值溢出、陈旧响应或 M1493 的 Python `SOURCE_CHAIN` 问题。** 置信边界必须保留：M1334 在 28.5 ns 只断言了 `!(protocol_error || numeric_overflow || stale_response_seen)`，没有分别打印三位值，因此“确切是哪一位、是 1 还是 X”尚未由封存证据直接观测。

## 封存边界与首故障

- `failure.json` 固定在 `SIM_k8_0`：VCS compile 1、simv 1、SAIF 0、PTPX 0；canonical result 不存在，automatic retry=false。
- 28.5 ns 的 aggregate SVA 首先失败。到该边沿，`cp_source` 只有 1 次命中，而 endpoint request、commit、done 均为 0。
- `endpoint_fault` 有一条独立 SVA，封存报告没有 endpoint failure；同时没有已接受 memory request。故 endpoint 类在这个采样边沿没有证据，不能用它解释首故障。
- `stale_response_seen` 需要 request/response 身份路径；首故障前 endpoint accept 为 0，response/commit cover 也为 0，所以它很不可能是原发故障。
- `numeric_overflow` 位于 K8 service 的注册累加锥；首故障发生在首个 source accept 后、首个 memory request 前，还没有权重响应或累加提交，数值溢出作为原发故障在因果上最弱。
- 剩下的 `protocol_error` 汇聚 frontend、service、core-adapter、memory-adapter 和 consistency fault，且 mapped netlist 的该锥很大。它也允许 X 通过 aggregate SVA 产生同样的 failure 文本。因此排序为：`protocol_error-or-X` 最高，stale 很低，numeric 极低，endpoint 在已观测边沿未见。

## 更像 mapped 四态问题，而不是 RTL/source-chain 功能 bug

M1502 所谓 `SOURCE_CHAIN` 修复只删除 Python runner 中不存在的方法调用；本次已经成功编译并进入 mapped case0，所以不是旧 Python 异常复发。

冻结 RTL 的 M859 equal-bandwidth VCS 已让 K8/K1x8 共 10 个 clean case 通过，且五档 exact cycle 通过；这不是 mapped 等价证明，但显著降低了“同一合法 raw case 在 RTL 中必然触发协议错误”的可能性。相反，M1050 已在同一 C2 mapped 家族观察到：首个 raw accept 后、首个 memory request 前的 25–28 ns，fault/control 锥被 X 污染；编译期二进制初始化后同 workload 通过。M1502 的 compile prefix 没有任何 initreg/reset-isolation 修复。因此，M1502 在 28.5 ns 的同相位 aggregate failure 与既有 mapped-X 指纹吻合。

这仍不是“已证明 K8 exact root cause”。原因是 M1502 没有保留三位 fault 的逐位值，也没有保留 internal fault taps。静态锥只能给出因果排序，不能把 X 精确定位到 frontend/service/adapter 中的某个 flop。

## 最小 successor：一次双 DUT、逐位首故障诊断

不要先改算法或 memory endpoint，也不要把 `+initreg` 直接当硅修复。允许的新工作应是新 namespace 下的**一次 VCS compile + 一次 K8 case0 sim**，无 UCLI、无 SAIF、无 PTPX：

1. 同一个 TB、同一份 case0 stimulus 并排实例化 frozen RTL K8 top 与 mapped `ARCH_MODE1` top，各自使用独立但同构的 reset-safe memory model。
2. 用 `case equality` 分开记录 `protocol_error`、`numeric_overflow`、`stale_response_seen`、8 位 endpoint fault；任何 X 单独标成 `FAULT_IS_X`，禁止继续用三位 OR。
3. 对 mapped DUT 只读 tap 已保名的 `frontend_compactor_fault_q`、`frontend_paired_sink_fault_q`、`core_adapter_fault_q`、`service_fault_q`、`memory_adapter_fault_q/stale_q`；每个 posedge 写一行固定字段，记录 header/raw/request/response handshake 和 fault vector，首个异常立即停止。
4. 这一次 root diagnostic 不加 `+vcs+initreg`，否则会掩盖要定位的 X；也不复用 M1502 `simv`。编译前必须由独立 source hammer 证明只有 additive TB/checker/filelist 和新 namespace。

一次运行的判决门：

| RTL K8 | mapped K8 | 判决与唯一后继 |
|---|---|---|
| clean | fault/X | mapped reset/valid-isolation 或综合等价问题；先修 RTL 可观测隔离并重综合，禁止只靠 TB 放宽 |
| 同位 binary protocol fault | 同位 binary protocol fault | 才检查 raw tail/valid linger 合同；最小 RTL 修复是只在采样请求边沿锁 fault，不能让一次已接受的 held-valid 组合尾巴制造第二请求 |
| clean | binary protocol fault | 对第一个分叉 internal tap 做 focused Formality/RTL↔netlist cone miter |
| clean | clean | M1334/UCLI/assertion sampling 绑定问题；修 checker，不改 datapath |

若结果确认为 mapped X，DATE 可执行的强修复是给参与 fault/ready 判定的 valid/fault state 完整 reset，或用已 reset 的 valid bit 对未初始化 payload 做可证明隔离，再重新综合。`+vcs+initreg+random` 只可作为多 seed 诊断/活动建模工具；没有 reset/隔离证明时不能包装成硬件功能修复。

## 论文与执行边界

本法证不生成新性能、功耗、SAIF、PPA 或系统倍率。M1502、它的 private build 和任何 partial axis 均不可引用。M803 RTL VCS 结果仍可按原有限边界引用；mapped activity/PTPX 仍为空。
