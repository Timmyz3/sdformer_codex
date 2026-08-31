# M144r2 independent hammer review r1

结论：**88/100，P0=0，P1=2，P2=4**。

M144r2 可以接纳为“正常非回绕、端点与控制器原子复位”前提下的独立控制器里程碑：原生产 VCS exact-SHA 测试被全新编译/仿真复现，独立攻击覆盖了精确行序、32 位 completion identity、完成顺序、零工作端点、同沿 barrier、新行 lookahead、fence 和 commit；3 ns exact-SHA DC 也逐项复现。

它还不能接纳为长期不复位的通用接口，也不是性能 headline：复位 epoch 和 32 位 sequence wrap 各有一个可复现的静默错误；engine arithmetic、descriptor/result SRAM、matched-frequency、physical/power/energy 和 system speedup 均不存在。

## 精确证据复现

- 封存 VCS 的 9 个输入、4 个输出以及封存 DC 的 20 个 evidence manifest 条目全部通过 SHA256 校验。`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。
- Synopsys VCS V-2023.12-SP1 fresh replay 精确复现生产 PASS：4 banks、5 jobs、5 rows、8 descriptors、5 次 PWP、5 次 correction、5 次 completion、1 barrier、1 commit、3 attacks。五项 cover 分别为 12/13/27/1/5，零 assertion failure。
- 单独编写的 attack TB 在原生产 RTL 上通过：`positive_rows=404 row_attacks=5 sequence_attacks=1 completion_attacks=1 endpoint_attacks=1 commit_attacks=2 fence_boundaries=1`。生产 RTL、合同和 sealed runs 均未改动。
- Synopsys DC V-2023.12-SP3 使用 exact RTL/filelist/SDC/Tcl 和同一 TSMC28 HPC+ 库重新综合，复现 3902.472014 um2、4998 cells、847 sequential cells、742 ports、0 macros、setup +0.0019 ns MET、hold +0.0000 ns MET、77 logic levels。映射 Verilog/SDC 与 seal 仅生成时间不同，面积/QoR 数值完全一致。
- Fresh DC 与重新打开的 mapped DDC 均无 TIM-209、OPT-150、ELAB-312、Error 或 Fatal；`check_design=1`、`check_timing=1`，五类 constraint 均无 violation。

## 独立协议攻击结果

以下行为均通过：

- 最大合法窗口严格接收 384 行，row id 恰为 0..383，并只允许第 383 行 close；
- malformed first row、early close、skipped row、late close 和中途 tag drift 全部 fail closed；
- 同 bank、同 16-bit tag，但错误 32-bit sequence 的 stale completion fail closed；
- correction completion 乱序 fail closed；
- zero-work 的同周期 launch/completion 被拒绝，而 accepted launch 后恰好一个周期的 completion 合法；
- barrier 与新 row 同一时钟沿接受时，新 bank 被标为 post-fence；四 bank lookahead 可继续填充，但 post-fence PWP 在 matching commit 前不能越界；
- wrong commit tag 和 duplicate commit 均 fail closed。

## P1-1：reset epoch alias

攻击先接受 bank 0/tag `0x6000`/sequence 0 的 PWP，随后复位 wrapper，再分配出同一 bank/tag/sequence 身份。此时注入复位前遗留 completion，`protocol_error` 仍为 0，新 bank 被推进到 `WAIT_CORRECTION`。因此，模块本身无法区分 reset 前后的同值身份。

若系统合同能证明 endpoint engine 与 wrapper 原子复位/flush、复位后恢复服务前已销毁全部旧响应，则此攻击不可达；否则必须把 epoch 带入 endpoint identity，或增加 flush request/ack 并在 ack 前阻止重新分配。

## P1-2：32-bit wrap 会错序并提前 commit

独立 TB 将可达的 pre-wrap 状态加速到 `0xfffffffe`，之后只用正常握手分配三个工作，sequence 分别为 `fffffffe`、`ffffffff`、`00000000`。M142 的 unsigned `<` oldest selector 会先选新 sequence 0，而不是旧 sequence `fffffffe`。同时 fence=0 时，`next_completion=fffffffe > 0` 让 `outer_commit_valid` 在三个工作均未完成前提前出现，且不触发 protocol error。

真实 heldout workload 远小于 2^32；所以最小修复可以不是复杂 modular arithmetic，而是冻结并检查“每次 reset 少于 2^32 个 allocation”，在将分配 `0xffffffff` 前 fail-close/saturate。若要支持长期运行，则 oldest selection 和 barrier drainage 都必须使用一致的 epoch/modular half-range 规则。

## DC 性能边界

面积数字可复现，但 timing 余量极薄：critical path 是 `lower/bank_sequence_q_reg[1][0] -> correction_accept`，2.5481 ns、77 级逻辑，setup 仅 +1.9 ps，即 3 ns 周期的约 0.063%；hold 报告为四位小数下的 +0.0000 ns。当前仍是 ideal clock、ZeroWireload、无 routing、无 macro 的 flattened controller cut。

这说明 M144r2 更像 correctness/control closure，而不是性能优势。若它继续留在性能候选路径，应优先切断 sequence/oldest/correction-accept 长组合锥；否则只把它作为 cycle simulator 的控制规则，不把 3 ns 作为 routed Fmax。

## Findings

- **P0 (0)：** admitted standalone non-wrap scope 内没有阻断问题。
- **P1 (2)：** reset epoch stale completion 可静默 alias；32-bit wrap 下 oldest 顺序和 barrier drained 判断同时失效。
- **P2 (4)：**生产 seal 未直接纳入大部分新边界攻击；没有 RTL-to-netlist Formality seal；3 ns DC 只有 1.9 ps setup margin 且是 prephysical；742-port controller cut 没有 engine/SRAM/power。

## 建议下一步

1. 最便宜的 correctness closure：冻结最大 allocations/reset，sequence 到边界前 fail-close；冻结 endpoint 与 wrapper 的原子 reset/flush 合同。若集成环境做不到，再增加 epoch/flush RTL。
2. 将本 review 中除 backdoor-wrap 加速外的正常协议测试并入 production VCS/SVA；wrap 用小参数同构实例或 formal bounded proof 覆盖。
3. 跑 exact-SHA Formality，确认 final mapped netlist 与 RTL 等价。
4. 若目标是性能，不再扩大 wrapper 功能；先 pipeline/partition 77-level correction-accept cone，再做 matched-frequency PT。SRAM、engine arithmetic 和其延迟/能量未接入前，不得把该控制器 DC 结果外推成加速比。

机器可读结论见 `m144r2_independent_hammer_review_r1.json`；本目录核心证据由 `manifest.sha256` 固定。
