# ATLIF DP-TME 协议采样与 Fail-Closed 签核

## 本轮结论

本轮关闭了 `hitflow_dptme_array` 的协议验证证据缺口，未修改其数值 datapath 或握手状态机。最终证据表明：

1. H67 epoch30、全分辨率 `480x640` checkpoint 生成的 81 条 ATLIF 命令，在 Icarus 与 Verilator 中均完成 `25920` 个 hidden 和 `25920` 个 event 比较，零失配 `[rtl]`；
2. 合法 checkpoint 流在时钟采样沿的 `protocol_error` 次数为 0 `[rtl]`；
3. 非法 tag、T10 提前 last、first/last 同拍单步命令各被拒绝 1 次，且控制状态、累加结果和输出状态均未推进 `[rtl]`；
4. checkpoint 与 directed Verilator 均实际启用 `--assert`，日志包含运行时 marker，报告器对缺失 marker、错误比较数、错误模拟器身份和不完整 directed coverage 全部 fail-closed `[rtl]`；
5. 独立 reviewer 经三轮审查后给出本轮范围内 `5/5，可签核`。该评分只针对协议闭环，不代表整体 DATE 论文评分。

主报告为：

- `results/h67_ep30_checkpoint_atlif_dptme_protocol_20260809/report.json`

## 拍间组合脉冲的根因

RTL 定义为：

```text
protocol_error = step_valid && !protocol_ok
step_ready     = output_space && protocol_ok
step_fire      = step_valid && step_ready
```

真实 checkpoint TB 在握手上升沿后等待 `0.1 ns` 才撤销 `step_valid`。首拍在采样沿合法接收后，`busy_q` 已进入 busy，但 `step_first=1` 和 `step_valid=1` 在这段很短的拍间时间仍保留，因此组合 `protocol_error` 暂时为 1。它表示“若当前输入在下一个采样沿仍保持，将成为非法首拍”，不是已经接收了非法事务。

因此本轮没有把组合拒绝信号改成寄存器，也没有改变接口延迟。正确的验证口径是：

- 拍间组合电平用于即时拉低 `step_ready`；
- 协议错误计数在时钟采样沿进行；
- 只有 `step_fire` 才能推进状态。

## Fail-Closed 验证链

### 合法真实流

`tb_checkpoint_atlif_dptme.sv` 新增沿采样错误计数。报告器硬锁以下常量，而不是只搜索 PASS 字符串：

| 指标 | 必须值 | 实测 |
|---|---:|---:|
| commands | 81 | 81 |
| hidden comparisons | 25920 | 25920 |
| event comparisons | 25920 | 25920 |
| hidden mismatch | 0 | 0 |
| event mismatch | 0 | 0 |
| sampled protocol error | 0 | 0 |

Icarus 和 Verilator 的完整结果必须逐字段相同。Verilator 还必须在日志中给出 `ASSERTIONS=enabled`，否则报告生成失败。

### 非法定向流

`tb_hitflow_dptme_array.sv` 在每种非法事务之前快照：

```text
busy / mode / steps_seen / group_valid / tag
out_valid / out_events / out_hidden / out_slot_valid / out_tag
```

非法事务跨过一个真实采样沿后再比较快照，并动态累计结果：

| 非法场景 | reject | 状态推进错误 |
|---|---:|---:|
| 进行中 tag 改变 | 1 | 0 |
| T10 提前 last | 1 | 0 |
| first/last 同拍 | 1 | 0 |
| 合计 sampled protocol error | 3 | 0 |

该 directed 流同时由 Icarus 和 Verilator 执行。两份日志必须分别声明正确模拟器身份；Verilator 必须声明 SVA 已启用。不能用同一份日志冒充两个模拟器结果。

### 产物绑定

最终 JSON 记录了相关 RTL、TB、SVA、runner、向量生成器、报告器和 manifest 的源码 SHA256，并记录以下六份证据日志 SHA256：

1. checkpoint Icarus；
2. checkpoint Verilator；
3. directed Icarus；
4. directed Verilator；
5. Verilator lint；
6. Yosys check/stat。

报告器单测覆盖：非零 sampled error、旧格式缺字段、错误比较数量、缺 SVA marker、directed 覆盖不完整、directed 模拟器身份错误。连同 TESC profile 身份单测，最终相关 Python 单测 `14/14 PASS`。

## 三轮独立评审

| 轮次 | 评分 | 结论 | 主要问题 | 整改 |
|---|---:|---|---|---|
| 第 1 轮 | 3/5 | 不可签核 | checkpoint 漏 `--assert`；未硬锁比较数；directed 未绑定 | 启用 SVA；硬锁 `81/25920/25920`；directed 纳入同一 runner/report |
| 第 2 轮 | 4/5 | 不可签核 | directed 身份未锁；状态不推进为打印常量 | 显式模拟器身份；运行时状态快照和动态计数 |
| 第 3 轮 | 5/5 | 可签核 | 无阻断或高风险 finding | 六日志与源码哈希复算一致 |

评审结论不能被解释为整体 accelerator 或 DATE 投稿已签核。它只证明这个接口子系统的协议证据链已达到当前开放工具流程下的 fail-closed 标准。

## 数值与部署边界

整数 RTL 相对同一整数金参考为零失配 `[rtl]`，但 checkpoint 局部固定点 ATLIF 与原浮点事件仍有：

- `1175 / 25920 = 4.533179%` event flip `[prof]`；
- `deployment_accuracy_signoff=false`。

这不是 RTL 错误，而是部署数值桥尚未通过端到端 valid825。静态 site scale、下游 event-times-threshold folding、BN/requant、残差、skip 和 full encoder 均不在本轮签核范围。论文中只能写“ATLIF 组件整数 RTL bit-exact”，不能写“完整网络定点无损”。

## 对双线工作的影响

- Motion：ATLIF DP-TME 的接口可信度已补齐，可继续使用同一物理阵列支持 T10 与 T2；它不是新的架构贡献，也不改变 TESC/RQTB 的晋级门槛。
- Local5：若复用同一 ATLIF DP-TME 单元，可继承协议验证方法，但必须使用 Local5 自身 checkpoint 数值报告，不能直接继承 H67 的 `1175/25920` 数值。
- 下一步仍是读取 Motion fullres T450 的 all-12-block projection 与 RQTB 结果，再决定是否推进相同同步 K-SRAM、相同反压的真实 RTL 对照。
