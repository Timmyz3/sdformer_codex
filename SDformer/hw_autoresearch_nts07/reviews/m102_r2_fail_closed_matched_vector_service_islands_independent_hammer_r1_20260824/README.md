# M102 r2 fail-closed matched vector-service islands：独立打铁评审

日期：2026-08-24

结论：**81/100，当前 SHA 对 common-period production-only DC 为 NO-GO。** r2 已经关闭 r1 评审所描述的“fault 已登记以后，再释放 `output_ready`”泄漏路径，但没有关闭非法请求到达当拍的接受窗口，也没有实现合同声明的 reset-only sticky fault。

## 证据完整性

- sealed VCS input manifest 15/15、output manifest 7/7、runner SHA 全部复核通过。
- baseline/candidate compile 与 simulation RC 均为 0；PASS 行完全一致；未发现 assertion、fatal、error 或 compile warning 签名。
- candidate cover 与合同一致：PWP 7、正 correction 2、负 correction 1、fallback 2、stall 4、protocol fault 32、buffer quarantine 3、metadata error 1、PWP→correction seam 2。
- cycle-ledger manifest 2/2、r1 hammer manifest 3/3、M102 preflight manifest 6/6 通过。
- 重新运行 SHA-pinned analyzer，输出与已封 JSON byte-for-byte 相同。
- `docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## P0：同拍非法请求仍可接受旧输出

RTL 只把已经登记的 `request_fault_q` 放进 `protocol_error`：非法请求在时钟沿之前令 `request_semantically_valid=0`，但 `protocol_error` 仍为 0。若此前的 M82 结果被 stall，而 `output_ready` 与非法请求在同一周期拉高，则：

1. 旧 `m82_output_valid` 仍使顶层 `output_valid=1`；
2. 顶层和 M82 的 output ready 尚未被 `protocol_error` 隔离；
3. 旧结果在该沿被接受并从 M82 清除；
4. 同一沿之后 `request_fault_q` 才变为 1。

独立 VCS witness 在 r2 精确 RTL SHA 上确认：

```text
WITNESS M102_R2_SAME_EDGE_INVALID_ACCEPT preedge output_accept=1 protocol_error=0 semantic_valid=0 m82_valid=1
WITNESS_CONFIRMED M102_R2 old output retired on invalid-request edge
```

现有 TB 先保持 `output_ready=0` 注入非法请求，等待 fault 登记后才拉高 ready，因此错过这个顺序。现有 SVA 也只断言 `protocol_error |-> !output_valid && !output_accept`，没有断言当前拍的 `service_valid && !request_semantically_valid` 必须立即禁止旧输出接受。

## P0：故障不是 reset-only sticky

`phase_load_ready` 没有被 `protocol_error` 门控，而合法 phase load 会直接把 `request_fault_q` 和 `phase_poison_q` 清零。SVA 的 sticky 属性还显式排除了 phase-load handshake。独立 witness 在上述同拍泄漏清空 M82 后，不施加 reset，成功通过 phase load 清除了 fault：

```text
WITNESS_CONFIRMED M102_R2 request fault cleared by phase load without reset
```

因此合同、RUN_COMPLETE 和 ledger 中的 `sticky_until_reset=true` 以及“quarantined until reset”不是当前 SHA 的普遍性质。

## 周期账本复算

- baseline：`371,461,096 × 3 = 1,114,383,288` cycles。
- PWP：`11,164,284×3 + 32,360,036×4 + 13,936,011×4 + 1,509,043×5 = 226,222,255` cycles。
- correction/fallback：`188,148,490 × 3 = 564,445,470` cycles。
- candidate service：`790,667,725` cycles。
- same-clock service-slot work ratio：`1.4094204844392757×`。
- 8,640 phases 的单 context parser：`8,640×128 = 1,105,920` edges。
- 再计每 phase 一个 load edge：candidate 为 `791,782,285` cycles，ratio 为 `1.407436500047485×`。

算术正确，但两者都只是 SHA-pinned analytical ledger。VCS candidate 仅跑 8 个合法向量，不是该 workload population 的 actual-record RTL replay；`1.4074365×` 也不是 physical、frequency-normalized、system 或 headline speedup。

## 分级问题

### P0

1. `M102-R2-H-P0-01`：非法新请求到达当拍存在旧 stalled 输出接受窗口。
2. `M102-R2-H-P0-02`：request/metadata fault 可被 phase load 清除，不是 reset-only sticky。

### P1

1. SVA/TB 没有覆盖“invalid request 与 ready release 同拍”的次序，也没有 reset-only sticky assertion。
2. analyzer 把 `sticky_until_reset` 和 `fault_quarantine` 写成常量 true；sealed cover 不能证明遗漏的次序或恢复路径。
3. 当前只有含 SVA/TB 的 directed-VCS filelist，没有 baseline/candidate production-only DC filelist，无法对当前身份做 exact-SHA matched DC。
4. memory response mux、SRAM、decoder/ECC 和 bank enable 仍是 port cut；`bank_select_pwp` 也不受 fault 门控。logic-only DC 不能证明物理共享端口或 fail-silent memory side effect。
5. candidate directed legal population 只有 8 个向量、一个合法 metadata 形状；metadata base/terminal/overflow 边界和不同 seam 组合仍较薄。

### P2

1. `cp_fault_quarantines_buffered_output=3` 是同一个攻击持续三拍的 occupancy matches，不是三个独立反例。
2. runner SHA 使用独立文件记录，不在 primary input/output manifest 内；当前 SHA 虽然匹配，后续 DC 应统一 provenance。
3. analyzer 固定并校验 contract 数字，没有从 M88 payload 重新推导 population；本次独立复算正确，但 sealer 本身不是第二实现。
4. `1.409420×` 与 M88 bounded `1.409375695×` 口径不同，不能混用。
5. same bandwidth 仍不等于 equal area；candidate 存储和外部 mux 成本不能由本轮 VCS/ledger 推断。

## GO / NO-GO

- exact-SHA sealed VCS/SVA：**GO**。
- baseline directed functional：**GO**。
- candidate 已覆盖的合法 service/seam：**GO（directed only）**。
- r2 测试顺序中的 post-fault buffered-output quarantine：**GO（bounded ordering only）**。
- full fail-closed、same-edge quarantine、reset-only sticky：**NO-GO**。
- `1.409420×`：**GO（analytical service-slot only）**。
- `1.4074365×`：**GO（analytical parser/load-inclusive boundary only）**。
- 当前 SHA common-period production-only DC：**NO-GO**。
- physical/equal-area/system/headline：**NO-GO**。

## 下一步

1. 定义组合 `fault_now = protocol_error_registered || (service_valid && !request_semantically_valid)`，用它同时门控顶层 output、M82 output ready、service ready，以及未来真实 memory enable；避免只在下一拍隔离。
2. request/metadata fault 只允许 `rst_core` 清除；如果产品确实需要 phase reload 恢复，则修改合同为显式 recovery 协议，并验证旧输出如何 flush，不能继续写 `until reset`。
3. 新增 same-edge ready-release witness、request/metadata phase-reload recovery attack 和相应 SVA，重新 seal VCS；sealer 应由行为检查导出 sticky admission，而不是常量 true。
4. 修复并 reseal 后，再冻结两份 production-only filelist、同一顶层边界、同一 TSMC28 库/period/ideal-clock/ZeroWireload recipe 和 precompile resource audit，才准入 logic-only common-period DC。

本评审未修改生产 RTL/SVA/TB/contracts/results，未使用或引用开源工具结果，未修改 `docs/359`。
