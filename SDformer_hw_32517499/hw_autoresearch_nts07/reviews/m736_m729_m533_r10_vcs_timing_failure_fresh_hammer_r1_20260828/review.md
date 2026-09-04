# M736：M729/M533/M528 r10 VCS 封存失败独立打铁

## 裁决

**PASS（failure classification），98/100。** M729 结果包自身双重封存完整，VCS 编译成功，runner 对“simv 返回 0 但没有功能 PASS token”的处理是正确的 fail-closed。该结果仍是 `FAILED_DO_NOT_CITE`，不能推出功能正确、时序正确、性能或 PPA。

本次失败**没有证明 C1 算术/协议 RTL 有功能缺陷**。直接失效链是：RTL 零延迟寄存器/组合输出在 SRAM 采样沿后同一仿真时刻改变宏输入，slow foundry model 的 hold notifier 随即污染 Q/存储体，随后 cleanroom oracle 在 420 ns 发现 slot0 数据不一致。物理 hold 是否满足仍需 `.db` 下的 STA；本次 VCS 既不能证明满足，也不能证明硅上违例。

仅授权**一个新的、功能用途的修复候选身份**，不授权立即启动 VCS。候选必须使用 foundry 模型自己明示的 `UNIT_DELAY` 功能模式，保持相同 RTL、TB cleanroom oracle、SVA、攻击与覆盖门；不得使用 `+notimingcheck`、不得编辑 foundry 文件、不得引入 behavioral SRAM fallback、不得给宏时钟加隐蔽 skew。候选 PASS 只能写 `functional_vcs_only=true`；slow-corner setup/hold 必须由后续 DC/PT/STA 单独闭合。

## 证据完整性

- `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 均重新校验通过。
- `RUN_FAILED_OR_INCOMPLETE.json`：`status=FAILED_DO_NOT_CITE`，`phase=functional_and_coverage_gate`，`runner_exit_rc=1`，`child_rc="0"`，`paper_citable=false`。
- `ARTIFACT_INVENTORY.json` 将内部 symlink 的原始 target、解析后路径、字节数和 SHA 一并绑定；本次复核未发现外逃或特殊文件伪装。
- `compile.log` 明确显示 slow foundry `.v`、macro adapter、top r2、SVA r2、TB r4 全部编译和链接完成；没有 compile error。
- `sim.log` 没有 PASS token；在 420 ns 由 TB r4:1125 报 `slot0 foundry response identity/data mismatch`。

## 根因分解

### 1. 宏模型按其公开接口规则正常工作，不是接口语义错误

foundry 模型头部明确写明：control/data/address 不支持在 positive clock edge 同时切换，必须和时钟留出时序间隔。slow-corner specify 的 hold 要求为 CEB 38.6 ps、WEB 42.6 ps、A 45.6 ps、D 127.5 ps；notifier 行为会污染 Q，CEB/WEB notifier 还会污染整个存储体，D notifier 会污染相应 word。因此后续 mismatch 是模型预期的 fail-loud 行为。

### 2. 直接前因是宏引脚同沿零延迟切换

`sim.log` 共 2,223 条 timing violation，且全部是 hold：

| 时间 (ps) | CEB | WEB | A | D | 合计 |
|---:|---:|---:|---:|---:|---:|
| 406500 | 9 | 0 | 0 | 0 | 9 |
| 409500 | 0 | 0 | 36 | 502 | 538 |
| 412500 | 0 | 9 | 36 | 452 | 497 |
| 415500 | 0 | 9 | 9 | 0 | 18 |
| 418500 | 0 | 9 | 0 | 1152 | 1161 |

这些时刻都恰好是 3 ns 时钟的 positive edge。九个 slice 同时受影响，和一个逻辑 1152-bit row 的 bit-slice 绑定吻合。

### 3. 不能把根因简化成“TB 把所有 stimulus 放在 posedge”

- 正常任务的 `prep_*` 在 negedge 驱动；failure 发生在第一轮正常执行，六个攻击 task 尚未运行。
- TB 的 `psum_write_ready/row_complete_ready` 确实在 posedge 通过 `always_ff` 更新，是一个次级同沿输入来源。
- 但宏的 CEB/WEB/A/D 来自 top 内部 `scratch_*_w` 与 `row_final_packed_w`；top 的状态寄存器也在同一 posedge 零延迟更新。因此只把 ready 改到 negedge 不能根治全部 2,223 条违例。
- 这属于“RTL 功能仿真零 c2q + 带真实 specify/notifier 的宏”边界不匹配。真实实现是否满足 hold 取决于标准单元 c2q、布线和 STA 修复，不能由零延迟 RTL VCS替代。

### 4. slot0 mismatch 由 notifier 污染充分解释，尚无独立 RTL 功能反例

首个 CEB hold violation 在 406.5 ns；409.5--418.5 ns 又发生大量 A/WEB/D hold violation。模型的 notifier 会将 Q 或存储内容置 X。TB 在每个 negedge 对 slot identity/data 做 exact cleanroom 比较，420 ns 的 mismatch 出现在这条污染链之后。日志没有在首个 timing violation 之前给出 cleanroom mismatch、`$error` 或 assertion failure，因此不能把该 mismatch 独立归因为 C1 数据通路逻辑。

### 5. child rc=0 不能越过内容门

VCS 在本次 `$fatal` 后打印 `$finish` 并以 0 返回。runner 随后要求唯一 PASS token、coverage token、覆盖下限和零 failure signature；PASS token 不存在，故在 `functional_and_coverage_gate` 返回 1 并双封失败收据。这正是所需的 fail-closed 行为，不能改成“simv rc=0 即 PASS”。

## 唯一合法修复候选

候选应只改变验证模式身份和 runner/contract，不改 top r2、macro adapter、TB r4、SVA r2：

1. 对同一个 checksum-verified foundry `.v` 使用其文档化的 `+define+UNIT_DELAY` 模式；这是功能模型身份，不是 slow timing identity。
2. 收据必须显式写 `macro_model_mode=foundry_UNIT_DELAY_functional`、`functional_vcs_only=true`、`timing_verified=false`、`paper_citable_timing=false`。
3. 保留唯一 PASS/coverage token、11 个 normal cover、P2 consecutive-read identity、六种 attack、fatal/error/assertion 零容忍。
4. runner 必须额外拒绝 `Timing violation`、`$fatal`、`$error`、assertion failure；即使 simv rc=0 也不得跳过内容门。
5. 新身份在静态 hammer、candidate、final release 完成前不得启动；授权上限仍为一次 VCS compile + 一次 simv。
6. 功能 PASS 后另用 slow `.db` 做 macro-inclusive DC/PT setup/hold；在此之前 C1 仍不可标为 physical/paper PPA ready。

不授权的“修复”：`+notimingcheck`、`+no_notifier`、编辑 foundry `.v`、用寄存器数组替换宏、仿真专用宏时钟延迟、降低 oracle/coverage、把 r10 failure 改写为 PASS。

## 评分与发现

- P0：0
- P1：2
  - M736-P1-01：M729 没有给出功能结论；slow-model notifier 先污染，物理 hold 仍开放。
  - M736-P1-02：后续必须把 functional UNIT_DELAY 和 slow-corner STA 分成两个身份，禁止交叉借 claim。
- P2：0
- 打铁质量：98/100
- 设计功能状态：`NO_CONCLUSION`
- r10 论文状态：`FAILED_DO_NOT_CITE`
- 新功能候选：`AUTHORIZE_ONE_CANDIDATE_ONLY__NO_LAUNCH_YET`

## 冻结检查

`docs/359_DATE终局冻结_20260813.md` 复核 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
