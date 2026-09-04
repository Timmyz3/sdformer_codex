# PPDI Bank Executor 独立审阅（2026-07-22）

## 1. 审阅结论

**总判定：CONDITIONAL。**

- P0：0 项。
- P1：4 项，分别涉及 stale response 与 Acc ready/valid 稳定性、engine-originated error 清除、partial commit 后外部 Acc 失效耦合、有限 epoch 回绕。
- P2：2 项，均为 TB/SVA 覆盖与因果性检查不足。
- PASS：在合法命令源持续保持 payload、没有 flush/stale 干扰的前提下，两位 `destination_done_q` 能使两个 Acc 端口分拍且每个有效端口每条 command 最多握手一次；`cmd_ready` 只在所有有效目的已提交或本拍提交时出现，`term_done` 与 term-last command retire 同拍，product 只在 term-last retire 时释放。

本结论只覆盖 **PPDI 单 bank executor 叶模块**。不等价于完整 PPDI，不等价于 H67 四 stage bit-exact，不形成周期加速、面积、功耗、EDP 或 PPA 结论。下述 Verilator assertion 运行均为动态仿真，不是 formal。

## 2. Findings（按严重度）

### P1-1：stale response drain 可撤回已拉高但未握手的 Acc valid

**状态：CONDITIONAL。**

**证据：**

- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:185-190` 把 `!stale_weight_response_fire` 放入 `acc_update_enable`，因此任何旧 epoch response 被 drain 的拍都会把两路 `acc_update_valid` 强制为 0。
- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:209-217` 允许 stale response 在 product engine 当前并不等待 response 时仍被 ready/drop。
- `verif_hitflow/gatestack_ppdi_dctf32_bank_executor_assertions.sv:70-83` 正确要求 Acc valid 在 stall 下保持，但现有 TB 的 stale response 只发生在替代 term 尚未形成 product 时，未覆盖“Acc valid 已拉高且 ready=0”的并发窗口。
- 临时定向动态实验在 partial-pair 的 odd 端口 stall 后注入不同 epoch response，Verilator 在 assertion 文件第 147 行，即 `p_odd_update_stable`，于 745 ns 失败。该实验不是 formal。

**影响：** exactly-once 计数本身不会因此立即重复或漏写，因为 done mask 未被清除；但输出 ready/valid 合同被破坏，sink 已观察到的 pending valid 会无 flush 地撤回。该行为也给组合仲裁和系统级协议审计引入不必要的例外。

**最小关闭条件：** stale drain 不得屏蔽已经 resident 的 Acc update。实现可选择让 stale response 与 Acc commit 独立并行，或在不破坏 Acc valid 保持的前提下延后 stale drain。新增 even-stall、odd-stall、partial-done 三种窗口的定向回归，要求现有稳定性 assertion 通过、stale 计数准确、每个有效目的恰好一次、command/done 不提前。

### P1-2：单拍 `clear_error` 无法可靠清除 engine-originated sticky error

**状态：CONDITIONAL。**

**证据：**

- wrapper 在 `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:291-323` 先执行 clear，再把电平型 `engine_protocol_error` 重新置入自身 sticky error，表现为 new-error-wins。
- product engine 在 `rtl_hitflow/gatestack_decoupled_product_engine.sv:139-180` 对自身错误采用 `!clear_error` 门控，表现为 clear-wins。
- 当 child error 已为 1 时，单拍 clear 的同一上升沿上，child 清零，但 wrapper 仍采样 child 的旧值 1 并重新置位；clear 撤回后 wrapper 保持 1。
- 临时定向动态实验以同 epoch、错 tag response 触发 child error，再给一个周期 `clear_error`，实际观察到 wrapper `protocol_error` 仍为 1。该实验不是 formal。

**影响：** docs/147 第 69 行声明“clear 与新错误同拍采用 new-error-wins”，但一个已经存在的 child sticky level 不应被无条件解释为“本拍新错误”。软件或 context controller 若按单拍 clear 合同工作，会得到不可清除的表象。另需明确：同 epoch 错身份 response 当前不 ready/drop；若 producer 遵守 valid-until-ready，它只能依赖 context flush/abort 退出，单纯 clear 不解决通道阻塞。

**最小关闭条件：**统一 parent/child 的 clear 优先级与事件语义，并明确 malformed same-epoch response 的恢复动作是 clear 还是 flush/abort。回归至少包含：child 旧错误加单拍 clear 后 parent/child 均清零；clear 与真正的新 command error 同拍仍满足选定优先级；错身份 response 持续 valid 时系统按合同进入可恢复路径。

### P1-3：partial commit 的正确性依赖同一 flush 原子失效外部 Acc，本叶模块未关闭该依赖

**状态：CONDITIONAL，文档边界描述基本正确。**

**证据：**

- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:311-320` 允许一个 parity 端口先完成外部握手；`rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:282-287` 的 flush 只清 executor 状态和 done mask，不能回滚已经发生的外部写。
- docs/147 第 73-80 行明确承认叶模块没有回滚外部 SRAM，并要求完整 projection 同拍清 Acc group valid bitmap；这个边界界定是正确的。
- `rtl_hitflow/hitflow_banked_accumulator.sv:187-196` 确有 flush 清 valid bitmap/busy 状态的机制，但当前 PPDI executor 尚未集成该 Acc。
- TB 第 439-459 行只检查 executor 输出在 flush 时被屏蔽；scoreboard 仍把 flush 前的 even fire 计入累计 commit，没有实例化 Acc，也没有证明该写在新 group 中不可见。

**影响：** 若 PPDI executor flush 与 Acc flush 不是同一个无漏拍、同 reset domain 的 context-abort 事件，flush 前的单端口部分写可能在后续 bias/final 路径中继续有效。这不是叶模块内部 done mask 能解决的问题。

**最小关闭条件：**在首个 PPDI projection 集成顶层把同一 context flush 同拍送达所有 PPDI executors、对应 Acc banks 及 group-valid 生命周期状态；建立集成 TB：先提交一个 parity、flush、用相同 tag/token 重启，逐元素确认新 group 从逻辑零开始，旧 partial write 和旧 final 均不可见。该测试通过前，不得把 partial-flush 标为端到端 PASS。

### P1-4：默认 4-bit epoch 只能隔离有界迟到响应，回绕后存在 ABA 别名

**状态：CONDITIONAL。**

**证据：**

- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:209-217` 只比较 `weight_rsp_epoch != epoch_q`；默认 `EPOCH_W=4`，连续 16 次 flush 后回绕。
- docs/135 第 73-86 行已经正确记录该限制，但 docs/147 的 PPDI 叶模块边界没有重申或显式继承该系统约束。
- 当前 TB 只验证一次 flush 后的旧 epoch response，不覆盖回绕。

**影响：**若 response 的最大生存期允许跨越一个完整 epoch 周期，旧 response 可再次等于当前 epoch，并进入 child identity 检查；在 tag/channel/tile 也复用时会形成 ABA 污染风险。PPDI 沿用了标量 executor 的既有边界，并未新增该风险，但不能因此把 stale isolation 写成无界保证。

**最小关闭条件：**在 PPDI 集成合同中选择并落实至少一项：证明最大 response 寿命内 flush 次数小于 `2^EPOCH_W`、复用前排空旧请求、扩大 epoch，或使用未决请求表/更强 generation。增加参数缩小后的 wrap 定向测试，并在 PPDI 文档与接口规格中引用该有界条件。

### P2-1：command 源稳定性 assertion 漏掉 first command 内部预取的首个周期

**状态：CONDITIONAL。**

**证据：**

- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:176-178` 可在外部 `cmd_ready=0` 时先让内部 product engine 接受 first command；外部 command 直到目的提交完毕才 retire。
- `verif_hitflow/gatestack_ppdi_dctf32_bank_executor_assertions.sv:57-65` 仅在 `term_active_q && cmd_valid && !cmd_ready` 时检查下一拍稳定。first command 被内部接受的当拍，旧 `term_active_q=0`，因此 source 若在下一拍撤回/改变 payload，此 property 不会捕获第一次违规。
- TB 的 `set_command`/`wait_command_accept` 始终保持 payload，未做故障注入。

**影响：**RTL 在合法 ready/valid source 下可工作，但验证没有完整证明 docs/147 第 67 行的“活动命令未 retire 前保持所有字段稳定”，尤其缺少内部 prefetch 与外部 retire 分离的边界。

**最小关闭条件：**将 source-side property 覆盖所有 `cmd_valid && !cmd_ready` 周期，或显式增加 first-capture 后直到 retire 的 held-command property；分别注入 first payload change、first valid withdrawal、continuation sequence/tag/token change，确认 assertion/`protocol_error` 的预期行为。

### P2-2：TB/SVA 对 product 生命周期和 exactly-once 因果链覆盖不足

**状态：CONDITIONAL。**

现有定向覆盖有效但较窄：split pair、same-cycle pair、only-even、only-odd、两 command 复用一次 product、两类非法 start、partial flush、一次 stale epoch 均有实际检查。bind 使用 `(.*)` 且当前 elaboration 通过，不是空 bind；SVA 也包含 ready/valid、parity、done mask、flush 和 stale 计数检查，不能称为空洞。

仍缺少以下关键场景或直接 property：

1. odd 先提交、even 长 stall，以及两路交替多周期 backpressure；
2. paired 且 non-last 的 command，随后 continuation 再 paired/单目的，用于证明 non-last retire 不释放 product、last retire 才释放且只产生一次 term_done；
3. back-to-back term、continuation 间 `cmd_valid` bubble、last commit 与 flush 同拍的优先级；
4. `acc_port_fire -> 下一拍 done bit`、同 command done bit 单调、未 fire 不得凭空置位、每个有效 bit 在 retire 前恰好一次的因果性 assertion；
5. sequence jump、重复 first、错误 identity、非法 head-last、gate zero、token 越界和 malformed continuation；
6. stale response 在 weight request、weight wait、product output、单端口 partial-done 各阶段到达；
7. clear/error 同拍矩阵和 epoch wrap。

**最小关闭条件：**补齐上述高风险定向场景；增加独立参考计分板，以 `{term identity, command sequence, parity port}` 为 key 统计 expected/actual commit，而不是只依赖当前 TB 的 token 一次性数组；增加 done-mask 因果 property。动态 assertion 通过只能表述为所运行轨迹通过，不能写成状态空间证明。

## 3. 七项重点判定

| 审阅项 | 判定 | 说明 |
|---|---|---|
| 两端口分拍 exactly-once commit | PASS/CONDITIONAL | done mask 结构可防同 command 已完成端口重发；现有 split 测试通过。结论仅限合法 held command，且需关闭 stale/valid 问题和扩充因果性检查。它不保证 term-wide token 去重，后者属于 whole-term adapter；也不提供下游 eventual-ready 活性保证，ready 永久为 0 时 command 会安全阻塞而不是 retire。 |
| `cmd_ready` / `term_done` / product 释放 | PASS（静态）/CONDITIONAL（验证） | `cmd_ready` 以 all-destination-complete 为门槛，`term_done` 仅在 last retire，`product_ready` 仅在 last retire；未见重复释放或漏释放的直接 RTL 路径，但 paired non-last、back-to-back term 未被专门验证。 |
| first/continuation/ready-valid 稳定性 | CONDITIONAL | identity/sequence 检查存在；first prefetch 的 source stability property 有首拍空洞，stale drain 会实际撤回 pending Acc valid。 |
| partial commit 后 flush | CONDITIONAL | executor 本地屏蔽和清状态成立；外部 Acc invalidation 依赖已在 docs/147 正确声明，但未由 PPDI 集成仿真关闭。 |
| stale epoch 与 clear/error 优先级 | CONDITIONAL | 一次 flush stale drop 通过；有限 epoch 回绕和 child/parent clear 语义未关闭。 |
| TB/SVA 完整性 | CONDITIONAL | 不是空 TB/SVA，但关键交叉场景和 exactly-once 因果 property 不足，现有 71 个非 reset 周期不能支撑广义签核。 |
| 架构价值与 DATE 声明 | CONDITIONAL PASS | 叶模块证明了“共享一次 term product + 奇偶物理端口 + command 整体 retire”的可实现性；完整论文价值仍依赖 adapter/fabric 的 command-work 降低转化为端到端周期/能效收益。 |

## 4. 相比普通双端口写的架构价值边界

单看本叶 RTL，新增内容主要是两路目的字段、两路 valid/ready 和 2-bit commit mask；这本身不能作为 DATE 架构贡献。可辩护价值来自组合关系：whole-term 验证后的奇偶配对减少 command 数，bank-local product 在 term 内只生成一次，三个 bank 同时把同一 product multicast 到既有偶/奇 Acc 物理端口，并由有序 fabric/整体 retire 保持工作守恒。

因此 docs/144 第 54-63 行对价值来源的界定是合理的，30.270% 只能称真实 trace 的 command-work 降低，不能直接称周期加速。最小 DATE 晋级证据仍是：

1. PPDI adapter、双目的 fabric、三个 executor 与 Acc flush 生命周期完整连接；
2. H67 S0-S3 全量 `acc32` 逐元素零失配，term/weight/bias/final 工作量守恒；
3. 与标量 DCTF/DCTF-2C 在相同开放映射、相同 memory 位分账和相同约束下比较真实周期；
4. 目标 SRAM、STA、活动率功耗和完整 projection 面积归一吞吐证据齐全后，才讨论 PPA/EDP；本轮没有这些证据。

## 5. 实际执行与证据一致性

实际阅读：PPDI executor RTL、对应 TB/SVA/bind/runner、docs/144、docs/147、结果目录全部文件；为确认跨模块边界，额外核对了 product engine、标量 executor、Banked Accumulator flush RTL、docs/135 与 docs/138。

实际复跑：

~~~text
PASS PPDI DCTF32 BANK EXECUTOR cycles=71 commands=5 weight_req=5 acc={4,4} done=4 stale=1
Icarus: PASS
Verilator 动态 assertion: PASS
Yosys hierarchy/check/stat: PASS
Erie RTL/TB: 0 error, 0 warning
~~~

结果包输入 SHA-256 与复跑时指定的 engine/RTL/TB/SVA/bind 一致。Yosys 运行只有无约束 hierarchy/check/stat，不是目标库综合、STA 或 PPA。官方 runner PASS 证明现有定向轨迹通过，不抵消本审阅的两个可复现负向场景。结果 JSON 只记录 build 目录日志的绝对路径，日志本体未复制到结果包；若该目录要作为可搬移审阅证据，最小补强是将 build/仿真/lint 日志一并固化并记录哈希。

## 6. 最小关闭清单

1. 关闭 P1-1：stale drain 不再撤回 pending Acc valid，并通过三类 stall 交叉回归。
2. 关闭 P1-2：统一 child/parent clear 语义，单拍 clear 和 malformed-response 恢复测试通过。
3. 关闭 P1-3：PPDI executor 与 Acc 使用同一 context flush，partial-write quarantine 集成测试逐元素通过。
4. 关闭 P1-4：固化 epoch 最大在途寿命/回绕合同并验证。
5. 补齐 P2-1/P2-2 的 source stability、paired continuation、product release、done-mask 因果和错误优先级覆盖。

以上关闭前，建议状态保持 **`[rtl-leaf][CONDITIONAL]`**；不得升级为完整 PPDI、完整 projection、DATE 性能结论或 PPA 结论。

## 7. 第二次独立复审

### 7.1 复审结论

**总判定：CONDITIONAL。** 本节以 2026-07-22 第二轮当前 RTL/TB/SVA/runner/results 为准，更新并覆盖首轮 finding 的关闭状态，但不删除首轮审计轨迹。

- P0：0 项。
- P1：1 项 OPEN，pending generation 目前只凭 epoch 释放，尚未对错误 identity 的 stale response fail-closed。
- P2：2 项 OPEN，分别是 clear property 未排除同拍新错误、结果日志哈希与 build 日志的可搬移证据仍不完整。
- PASS：首轮 stale drain 撤回 resident Acc valid、旧 child sticky 单拍 clear、叶级 partial commit 共同 flush、first-prefetch 稳定性空洞、done-mask 因果与 paired product 生命周期等问题均已有对应 RTL/SVA/TB 关闭证据。

本轮仍只签 **PPDI 单 bank executor 叶模块及其与一个真实双 bank Acc 的局部 flush 合同**。不是完整 PPDI adapter/fabric/三 bank projection，不是 H67 四 stage bit-exact，不形成周期加速或 PPA/EDP 结论。所有 assertion 结果均来自动态仿真，不是 formal。

### 7.2 首轮 Findings 逐项状态

| 首轮项 | 第二轮状态 | 独立复审依据 |
|---|---|---|
| P1-1 stale drain 撤回 pending Acc valid | **CLOSED** | `acc_update_enable` 已不再包含 `stale_weight_response_fire`；TB 在 even 已提交、odd valid stall 时 drain 一个真实占用的 pending generation，odd valid、token、tag、values 和 done/ready 边界保持，现有动态稳定性 assertion 通过。 |
| P1-2 单拍 clear 无法清旧 child sticky | **CLOSED（功能问题）** | wrapper 只在 `!clear_error && engine_protocol_error` 时重锁 child 电平；同 epoch 错 identity 先产生 parent/child sticky，撤回错误 response 后单拍 clear 实测两者同时清零。错误 response 本身仍不 ready，恢复合同是 context flush/abort，不是 clear。clear property 仍有同拍新错误的 P2 问题，见 7.4。 |
| P1-3 partial commit 依赖外部 Acc invalidation | **CLOSED（叶级集成）** | 新 TB 实例化真实 `hitflow_banked_accumulator`，executor 与 Acc 共用同一 `flush`。旧 command 实际 commit 偶/奇=`1/0`；同 tag/token 重启后 token2 final 逐 lane 仅为新 product `2*(3+lane)`，旧 `7*(10+lane)` 不可见。完整三 bank projection 仍是上层准入条件。 |
| P1-4 4-bit epoch 回绕 ABA | **OPEN（P1）** | 空闲 generation 选择、8 项满表阻塞和 drain 后恢复均已实现并实测；但 pending bit 的释放只验证 epoch，不验证被取消请求 identity，尚不能无条件称 fail-closed，见 7.3。 |
| P2-1 first command 内部 prefetch 首拍稳定性 | **CLOSED** | 新 `p_first_prefetch_holds_command` 从 `start_fire && !cmd_ready` 开始检查下一拍全部 command 字段；随后由原 active-command property 延续到 retire。 |
| P2-2 product 生命周期与 exactly-once 因果覆盖 | **CLOSED（本轮指定子项）** | 已增加 fire 后 done、done 必有 cause、已完成端口不重发；paired non-last 不产生 done/不新增 weight，paired last continuation 奇端口先提交、偶端口后提交并只释放一次 product。更广状态空间仍不能由 175 个非 reset 周期替代。 |

### 7.3 P1 OPEN：pending generation 可被错 identity stale response 提前释放

**状态：OPEN / CONDITIONAL。**

**证据：**

- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:222-235` 以 `stale_epoch_pending_q[weight_rsp_epoch]` 判定 pending stale；没有保存或比较该 generation 原请求的 group tag、input channel、physical tile。
- `rtl_hitflow/gatestack_ppdi_dctf32_bank_executor.sv:378-386` 在任何 pending epoch response fire 时直接清 bit；若此前满表，还立即把该 epoch 作为当前 generation 恢复使用。
- TB 第 367-406 行确实先制造了一个被 flush 取消的请求，但注入 stale 时只改 `weight_rsp_epoch`。其余 response 字段仍是随后当前 term 的 tag/channel/tile，而不是被取消请求的 `12'h440/7'd22/6'd10`。因此该轨迹证明的是“pending epoch 可 drain 且不扰动 Acc valid”，不是“被取消请求的完整 identity response 返回”。
- 满表恢复用例第 582-594 行同样只改 epoch 后清 bit。当前两次 stale 计数都没有验证 canceled request identity。

**影响：**在“每个请求恰好返回一次且 epoch 不会被错误 response 冒用”的可信 response-source 合同下，bitmap 能禁止正常 ABA 复用；但若一个错 tag/channel/tile 的 response 携带某个 pending epoch，当前硬件会把 generation 释放，而真正旧 response 仍可能在途，之后复用该 epoch 会重新引入 ABA。当前对 active generation 的错 identity 采用 error 加 flush，却没有对 pending generation 采用同等 fail-closed 语义。

**最小关闭条件：**二选一并固化为接口合同：

1. 每个 pending generation 保存足够的请求 identity，只在 epoch 与 identity 都匹配时清 bit；错 identity 置错且不得释放 generation，随后由 flush/abort 恢复；或
2. 明确 response source 保证 epoch 是不可伪造的唯一 request identity、每请求恰好一个 response，tag/channel/tile 对 stale drain 不参与安全判断，并把“fail-closed”声明限定在该假设内。

无论选择哪项，TB 都应使用被取消请求的真实 tag/channel/tile/epoch 重放一次，并另加 pending epoch 加错 identity 用例，检查选定合同下 bit 是否允许清除。

### 7.4 P2 OPEN：clear property 与 new-error-wins 优先级不一致

**状态：OPEN。**

`verif_hitflow/gatestack_ppdi_dctf32_bank_executor_assertions.sv:158-162` 的 `p_clear_old_child_error` 只要看到 `clear_error && engine_protocol_error`，下一拍就要求 parent/child 均为 0；它没有排除同拍 `command_protocol_bad`、product identity error 或 `unknown_stale_response_fire`。RTL 第 373-377 行和 docs/147 第 69 行则明确采用新错误胜出。因此“旧 child error + clear + 同拍新 parent error”是 RTL 应保持 parent error 的合法优先级场景，却会被该 property 判失败。

**最小关闭条件：**把 old-child-clear property 的 antecedent 限定为本拍无新错误，并为 new-error-wins 单独增加 property 与定向优先级矩阵。旧 child 单拍 clear 的现有功能用例保持 PASS，但不能用当前 property 覆盖所有 clear/error 并发语义。

### 7.5 P2 OPEN：结果包已有日志，但可搬移证据仍未完全闭环

**状态：OPEN。**

runner 第 93-101 行已把叶仿真、Acc 集成仿真、Yosys 和 Erie 日志复制到结果包并生成 SHA-256，首轮“日志只在 build 目录”的主要问题已修复。当前输入与八份已复制日志在原路径执行 `sha256sum -c` 全部通过。

剩余问题：

1. `log_sha256.txt` 记录绝对路径；复制结果目录后执行校验会访问原目录，而不是被复制的日志。原目录仍存在时甚至可能得到 PASS，但没有认证副本内容。
2. `iverilog_build.log`、`verilator_build.log`、`acc_flush_iverilog_build.log`、`acc_flush_verilator_build.log` 未复制到结果包，也未进入日志哈希；runner 虽会因 warning/error 退出，但可搬移证据不能独立复核编译/展开日志。

**最小关闭条件：**在结果目录内以相对文件名生成 hash manifest，并复制/hash 两组 Icarus 与 Verilator build 日志。复制到临时目录、修改其中一份日志后，`sha256sum -c` 必须针对副本失败，才算可搬移证据闭环。

### 7.6 六项整改验收表

| 整改 | 判定 | 说明 |
|---|---|---|
| 1. stale drain 与 resident Acc valid 解耦 | **CLOSED / PASS** | RTL 依赖已移除，真实 pending generation 与 odd stall 同拍 drain 的动态轨迹通过。 |
| 2. parent/child clear 统一、错 identity 需 flush | **CLOSED / PASS（功能）** | 旧 child sticky 单拍 clear 已通过；same-epoch 错 identity 不 ready，必须 flush/abort。P2 仅针对 assertion 未排除同拍新错误。 |
| 3. 真实 Banked Acc 共同 flush | **CLOSED / PASS（叶级）** | 旧 partial=`1/0`，同 tag/token 恢复 final 逐 lane 只含新 product；三 bank 完整 projection 仍未实现。 |
| 4. pending-generation bitmap 防 ABA | **OPEN / CONDITIONAL** | 空闲分配、8 项满表 fail-closed、drain 恢复通过；pending bit 仍可被错 identity response 仅凭 epoch 清除。 |
| 5. first/done/paired/odd-first 覆盖 | **CLOSED / PASS** | 指定 property 与定向轨迹均实际存在并在本轮动态运行通过。 |
| 6. 结果包复制日志并哈希 | **OPEN / CONDITIONAL** | 运行日志已复制并哈希；绝对路径 manifest 与缺失 build logs 尚未满足可搬移独立复核。 |

### 7.7 本轮实际执行

官方短 runner 第二次独立复跑结果：

~~~text
Leaf Icarus:       PASS cycles=175 commands=7 weight_req=16 acc={6,6} done=5 stale=2
Leaf Verilator:    PASS（动态 assertion）
Acc flush Icarus:  PASS old_partial=1/0 replacement_final_token2=1 bias=4 updates=6 writes=6
Acc flush Verilator: PASS（动态 assertion，同上计数）
Yosys hierarchy/check/stat: PASS
Erie executor RTL / leaf TB / Acc integration TB: 0 error, 0 warning
~~~

结果包九个输入文件 SHA-256 与当前文件一致，八份已复制日志 SHA-256 在当前原路径校验通过。Yosys 仍是无约束 hierarchy/check/stat，不是目标库综合、STA 或 PPA。

### 7.8 第二轮最终边界

关闭 7.3 的 pending-response identity 合同前，状态保持 **`[rtl-leaf][CONDITIONAL]`**。7.4 与 7.5 是 P2 验证质量/证据可搬移问题，不推翻已通过的叶模块定向功能轨迹，但必须在扩大声明前关闭。即使全部关闭，也只能写“PPDI executor 叶模块和局部 Acc flush 合同通过”，不得写完整 PPDI、完整 projection、H67 加速或 PPA/EDP 已证明。

## 8. 第三次独立复审

### 8.1 Findings 优先结论

**总判定：PASS（仅限第二次剩余 P1/P2 及 PPDI 单 bank executor 叶级范围）。**

- P0：0 项 OPEN。
- P1：0 项 OPEN；第二次 pending-generation identity 问题已 **CLOSED**。
- P2：0 项 OPEN；clear property 与结果包可搬移证据两项均已 **CLOSED**。

本轮不重新扩展首轮已关闭功能项的状态空间，只核查第二次留下的一项 P1 和两项 P2。“PASS”不表示完整 PPDI、完整 projection、H67 四 stage 或 PPA/EDP 已验证。

### 8.2 P1：pending generation 完整 identity 释放

**状态：CLOSED / PASS。**

**实现证据：**

- RTL 第 93-100 行为每个 pending epoch 保存 `group_tag/input_channel/output_tile`，并为当前 outstanding request 保存同样的 identity。
- RTL 第 229-245 行仅在 pending bit 存在且 `epoch/tag/channel/tile` 全部匹配时生成 `stale_response_identity_matches`；错身份 stale response 仍 ready/drop，但生成 `unknown_stale_response_fire`。
- RTL 第 342-347 行在 flush 将已发出未返回请求的 identity 写入该 epoch 表项；第 397-411 行对错身份响应置 `protocol_error`，并只在完整 identity 匹配时清 pending bit/恢复满表 generation。因此错身份不会提前释放 epoch，不再留下第二次指出的 ABA 窗口。

**定向动态证据：**

- TB 第 367-376 行以 `tag=12'h440/channel=22/tile=10` 发出真实请求后 flush，得到真实 pending generation。
- TB 第 379-412 行先提交偶端口、保持奇端口 `valid=2'b10` stalled，然后注入同 epoch 但错 tag `12'h441` 的响应；实测响应被 drop 并报错，pending bit 保持，Acc stalled valid、`cmd_ready` 和 `term_done` 边界不受扰动。
- TB 第 413-435 行清旧 error 后注入原请求完整 identity，此时 pending bit 才清除，奇端口 valid 在 stale drain 前后持续稳定，后续仅提交一次。

**最小关闭条件验收：CLOSED。** 已满足“保存 pending request 足够 identity、错 identity 只 drop+报错且不清位、完整 identity 才释放”，且不依赖 resident Acc valid 撤回。

### 8.3 P2：old-child clear property 与 new-error-wins

**状态：CLOSED / PASS。**

`p_clear_old_child_error` 在 SVA 第 162-167 行已将 antecedent 限定为：`clear_error && engine_protocol_error`，且本拍无 `command_protocol_bad`、无 product identity error、无 `unknown_stale_response_fire`。这与 RTL “旧 child sticky 可清，同拍新错误胜出”的优先级一致，不再将合法新错误保留误判为 clear 失败。官方 runner 中该动态 assertion 通过。

**最小关闭条件验收：CLOSED。** 第二次 finding 所列三类同拍新错误已全部从 old-error-clear 前提中排除。本结论只说明该动态 property 不再与 RTL 优先级矛盾，不把动态 SVA 称为 formal 或全状态空间证明。

### 8.4 P2：结果包可搬移性与完整日志闭环

**状态：CLOSED / PASS。**

- runner 第 90-92 行在 repo 根目录下以 repo 相对路径生成 `input_sha256.txt`；本轮从 repo 根执行 `sha256sum -c results/gatestack_ppdi_dctf32_bank_executor_20260722/input_sha256.txt`，9 份输入全部 `OK`。
- runner 第 93-106 行已复制叶级与 Acc flush 的 Icarus/Verilator build log，结果目录共有 12 份 `.log`；第 107 行在结果目录内以 `./*.log` 生成相对路径 manifest。
- 本轮在结果目录内执行 `sha256sum -c log_sha256.txt`，12/12 全部 `OK`。再将整个结果目录复制到临时目录，副本仍 12/12 通过；篡改副本中 `leaf_iverilog.log` 后校验失败，证明 manifest 验证的是搬移后副本内容。

**最小关闭条件验收：CLOSED。** 相对路径 manifest、四份 build log 纳入、结果日志自校验、repo-root 输入校验和副本篡改检出均已实测满足。

### 8.5 第三轮实际执行

官方短 runner `sim_hitflow/run_gatestack_ppdi_dctf32_bank_executor_checks.sh` 本轮独立复跑退出码为 0：

~~~text
Leaf Icarus:          PASS cycles=177 commands=7 weight_req=16 acc={6,6} done=5 stale=3
Leaf Verilator SVA:   PASS（动态 assertion，同上计数）
Acc flush Icarus:     PASS old_partial=1/0 replacement_final_token2=1 bias=4 updates=6 writes=6
Acc flush Verilator:  PASS（动态 assertion，同上计数）
Yosys hierarchy/check/stat: PASS
Erie executor RTL / leaf TB / Acc integration TB: 0 error, 0 warning
Input SHA-256:        9/9 OK（从 repo 根执行）
Result log SHA-256:   12/12 OK（结果目录与搬移副本）
Tamper check:         PASS（副本一份日志篡改后校验失败）
~~~

真实 `hitflow_banked_accumulator` 共同 flush 集成仍为 PASS：旧 partial 偶/奇写入数为 `1/0`，同 tag/token 恢复后 final `token2=1`，计数 `bias=4, updates=6, writes=6, mismatch=0`。

### 8.6 第三轮最终边界

第二次剩余的 P1/P2 最小关闭条件已全部满足，因此本文对 **PPDI 单 bank executor 叶模块与真实 Banked Acc 局部共同 flush 合同** 给出 `PASS`。该结论仍不包含 PPDI adapter/fabric、三 bank projection 系统级组合、H67 四 stage bit-exact/周期收益，也不包含目标工艺 STA 或 PPA/EDP。本轮 SVA 只在动态仿真中执行，不是 formal。
