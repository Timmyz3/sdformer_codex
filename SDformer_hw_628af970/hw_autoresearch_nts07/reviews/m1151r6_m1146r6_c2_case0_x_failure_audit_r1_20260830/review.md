# M1151R6：M1146R6 C2 frozen-netlist case0 X 独立失败审阅

## 裁决

**M1146R6 已消费唯一 attempt 并正确 fail closed，不得 retry，也不能升格 mapped functionality。值得做一次新的 additive、VCS-only 定位；不值得重跑 DC 或继续给 observation shadow 打补丁。**

本审阅只读固定 attempt、case0 log、TB、RTL、标准单元模型和已冻结 mapped netlist。没有调用 VCS、simv、DC、launcher，没有修改旧 attempt/work、RTL、网表或 `docs/359`。

## 第一根 X 与级联

128-cycle 原子 bitmap 把因果顺序钉得很清楚：

| 周期 | 新增 X | 解释 |
|---:|---|---|
| 3 | bit 3 `obs_protocol_error`; bit 6 `obs_fault` | `obs_fault` 是 `protocol_error | numeric_overflow | stale_response_seen` 的直接别名；本周期 numeric/stale 都已知，因此只有一个独立根：functional `protocol_error` |
| 4 | bit 7 `obs_bank_request_accept` | request handshake 开始被 X 污染 |
| 5 | bits 10/12/16/19 | request-accept popcount 进入 outstanding/request/active-read/bank-request shadow counters |
| 9 | bits 9/15/17/21 | result-accept 进入 fifo/result/live-slot/bundle-response shadows |
| 12 | bit 2 `obs_busy` | 功能 live-state 最后被污染 |

终局 union 为 `2b96cc`，共 12/22 个 observation；另外 10 个保持已知。这个形状排除了“337-bit async shadow bank 自己首先坏掉”：shadow 只是在后续边沿忠实采到了 X accept/result 事件。

## 复位与单元结论

- TB 保持 `rst_core=1` 跨五个正沿，在负沿释放，距首个业务正沿 1.5 ns。没有同沿释放竞态。
- mapped netlist 中恰有 337 个 observation-shadow `DFCNQD1BWP35P140`，均为带 active-low `CDN` 的异步清零单元；active-high `rst_core` 经反相后极性正确。
- case0 编译没有 `UNIT_DELAY`、SDF 或 initreg。标准单元 `DFCNQD1` 的 `negedge CDN` 明确把 Q 清零。
- 因为 cycle 0--2 的全部 22 个观察量已知，且 cycle 3 首先坏的是 functional protocol control，不支持把 TB reset 时序、shadow reset chain、异步复位极性或库模型列为第一根因。

## 根因边界

目前能严格定位到 **functional handshake/protocol cone，在 observation shadow 之前**；仅靠 22 个顶层口，不能诚实地唯一指定某个内部寄存器。

两个最小候选都真实存在，必须由下一次内部观测区分：

1. M1058 顶层把成对的 core/adapter request/response accept 用四态 `!=` 比较并组合进 `consistency_fault_now`。只要成对 accept 同时为 X，`X != X` 仍为 X，会把一个 flow-control X 直接放大为 `protocol_error=X`。
2. TB scalar-memory 的 `mem_req_ready` 在 `mem_req_valid=0` 时仍无条件索引 `pending_q[mem_req_slot]`。如果 invalid 周期的 slot 是 don't-care/X，ready 会变 X 并反向进入请求 ready/accept 锥。另一个可能是 functional synchronous-reset/data D 锥的四态 reconvergence；现有 log 不能在二者之间裁决。

因此，不能把本次失败写成“mapped reset 缺失”，也不能把 memory model 宣判为唯一根因。

## 唯一建议的 additive successor

新 namespace，只做一次 frozen-netlist VCS diagnostic，禁止 DC：

1. 一个 TB 同时记录 retained internal fault Q（compactor、paired sink、service、core-adapter、memory-adapter）、顶层 request valid/ready/slot/accept 和 22-bit bitmap；仍在 128 cycle 末 fail closed，不得延迟或掩盖 X。
2. 同一 TB 的第二个 DUT 使用 valid-qualified scalar-memory ready：invalid request 不索引 slot，valid request若 payload 非已知则 ready=0并单独报 protocol error。两个 DUT输入完全相同。
3. 若原 memory 复现 X、valid-qualified endpoint 为 0-X 且数值/握手通过，才允许把问题定为 TB endpoint contract，并以新 namespace 做正式 mapped cases。
4. 若两者都 X，则停止 TB 修补；在新 RTL namespace 显式导出/寄存 `core_protocol_error`、`adapter_protocol_error`、`consistency_fault_now/q` 与四个 accept，修 functional valid/reset hygiene，先 RTL VCS 再唯一一次 DC/mapped VCS。
5. 禁止把 `protocol_error` 强制为 0、用 initreg、`set_case_analysis`、延后 checker 或观察-only shadow 来制造 PASS。

这一步值得继续，因为 C2 已有独立的等带宽面积效率价值；但只值得一个定位门。若 valid-qualified endpoint 仍不能把第一根 X 收敛到单一内部锥，就停止 mapped-observation 扩张，保留 M903 的 logic-only component 口径。

## Claim boundary

现在合法的结论只有：compile/simv 生成成功；case0 在 cycle 3 首次出现 functional protocol X，随后污染 request/result shadow；M1146R6 失败且不可重试。当前仍没有 mapped functionality、SAIF/PTPX、系统倍速或 Table-A 准入。
