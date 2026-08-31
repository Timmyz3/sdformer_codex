# M230 independent hammer review

**Score: 88/100. P0: 1. P1: 7. P2: 4.**

M230 的冻结 trace recurrence 数学成立。独立脚本没有导入 M224/M225/M230 production analyzer，而是重新读取 100 个 bit-pack payload、重建 raw/spatial/temporal parent residual、形成 K8 group/source/context occupancy，并用四个物理 slot 的逐边沿模型复核 M229。

封存链、100/100 payload SHA、M225 group/read/service 账本和 M230 aggregate 均为零 mismatch。40,016 个 directed/random L1/L2 用例中，`nonempty period = service + response_latency + 4` 没有 off-by-one。

单 descriptor 的边沿定义如下：

| latency | header | descriptor | request | response | first/last replay | done accept | next header | period |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | `service + 5` |
| 2 | 0 | 1 | 2 | 4 | 5 | 6 | 7 | `service + 6` |

四 credit、full queue 时 `tail==head` 的同拍 pop/push、response 注册后一拍才能 replay、last pop 后 done fence、done accept 后下一拍才能接新 header 都已显式建模。因此 recurrence 本身 **GO**；没有触发 recurrence P0。

## Independent numbers

| item | F1 | F2 | F4 |
|---|---:|---:|---:|
| raw K8 service cycles | 1,010,523,752 | 627,381,048 | 453,854,808 |
| raw latency-2 trace cycles | 1,077,720,212 | 694,577,508 | 521,051,268 |
| speedup vs raw K8/F1 | 1.000000x | 1.551620x | 2.068357x |
| DC logic area | 18,219.222 | 24,013.206 | 35,715.078 um2 |
| logic area ratio | 1.000000x | 1.318015x | 1.960297x |
| trace throughput / logic area | 1.000000x | 1.177240x | 1.055125x |

Spatial-parent F4/L2 对 raw K1/F1 的 composed recurrence 是 `2.155535x`。这些数字可以叫“冻结 H67 binary-FC1、固定 latency、无 stall 的 trace recurrence”和“island logic-only area diagnostic”，不能叫物理 SRAM、完整 FC1/FFN 或系统结果。

## Layered verdict

| scope | verdict |
|---|---|
| M51/M225/M226/M229/M230 SHA 与 100 payload | **GO** |
| raw/spatial/temporal K8 独立重建 | **GO** |
| M229 L1/L2 recurrence | **GO** |
| `1.551620x / 2.068357x` | **GO，限 fixed-latency no-stall binary-FC1 slice** |
| `1.177240x / 1.055125x` | **GO，限 island logic-area diagnostic** |
| active-32 chunk 作为下一主攻 | **NO-GO / 降级** |
| 可执行 FC1 group lifecycle | **NO-GO** |
| physical throughput/area/energy | **NO-GO** |
| complete FC1/FFN/system/headline | **NO-GO** |

## P0

1. **M230-P0-01 — recurrence 还不是可执行硬件闭环。** M229 没有 empty-group bypass、mask/sign descriptor producer、Acc19 地址/初始化/parent seed/final commit，也没有物理 weight/Acc SRAM wrapper。M230 的 1-cycle empty 和 fixed-response always-ready 是外部模型动作。它不推翻当前 bounded trace 数学，但阻止用 F2/F4 做物理 Pareto 选择、paper PPA 或完整 FC1 声明。

## P1

1. Raw trace 中 148,932 个 empty stream 无法由 M229 自己完成：它拒绝 zero context mask，也必须收到 `descriptor_last`。必须补 summary/bypass RTL。
2. 公式要求 descriptor 连续可用、`weight_req_ready`、response、`acc_update_ready`、`done_ready` 全部无 stall；contract 没有逐项冻结所有 ready 假设，尤其 done。
3. scan 仅按 aggregate fixed cycles 收费；尚无把 K8 pixels 变成 source/context/sign descriptor 的 bounded-buffer 或 streaming schedule。全量物化会漏存储，在线生成会引入 stall。
4. 14,592-bit Acc19、F-way bank plumbing、zero/init、spatial seed、96-lane commit、group/output-block 地址和 commit backpressure 全在 port cut 外。
5. DC 是 0 macro logic-only。18,219/24,013/35,715 um2 不能覆盖物理 SRAM port、布线和能耗；当前 logic TPA 只能用于诊断。
6. active-32 skip 应降级。Raw/spatial 只有 `1.066573x/1.067059x`；即便每个 skipped raw chunk 免费省一整拍，也只省 1,797,628 cycles，不足当前 raw F4 trace 的约 `0.36%`。
7. 目前只有 10 samples、10 个 binary FC1；两层 stage-3 FC1 fallback，且没有汇总 per-record speedup min/mean/max，也没有 FC2/ATLIF 联动。

## P2

1. 独立模型已验证 full-credit `tail==head` 同 slot pop/push，但 production VCS 应增加一个明确的 L2 cover。
2. 分析器写 in-order fixed response，而 RTL 有 slot/tag/epoch/source 身份并支持合法乱序；应只冻结 memory policy，不要误写成 RTL 限制。
3. M230 seal 通过 manifest/result 间接绑定 100 payload，没有在顶层 SHA256SUMS 逐文件列出。本 review 已重哈希全部 100，属于打包卫生问题。
4. `2.155535x` spatial composition 的 choice/seed 周期虽已收费，但 parent-output 可用性和 seed/Acc 端口冲突仍未执行，只能作为 secondary premodel。

## Minimum next milestone

不要继续把 active-chunk 当主线。最小而高杠杆的下一步是 **FC1 -> ATLIF -> FC2 two-tile elastic fusion bridge**：

1. M229 完成一个 `8 contexts x 96 channels` FC1 block 后，用明确 tag/epoch/group/output-block 身份交给 bridge。
2. bridge 直接执行冻结量化 ATLIF state/threshold，产生八个 96-bit spike mask，不落完整 FC1 activation tensor。
3. 将 spike mask 转置成 FC2 source/context descriptor，直接喂 held-weight service；用两 tile buffer 重叠 FC1 finalize 与 FC2 replay。
4. 加 empty-tile bypass、真实 backpressure、Acc/mask/weight SRAM 合同，并以“store ATLIF reload”为 matched baseline。
5. 在同一 100-record trace 上收费 FC1、ATLIF、FC2，再做 directed/trace VCS 和 matched Synopsys DC。

这个桥可直接攻击 FC1 的 34.56M final-commit 周期、activation materialization 和 FC2 rescan；比最多约 0.36% 的 chunk-skip 上限更值得投入。

本 review 只新增本目录，没有修改 production RTL/脚本/合同、论文或 `docs/359`；后者 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
