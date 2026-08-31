# M335 独立接口与周期锤审

结论：**63/100，P0/P1/P2 = 2/7/3；当前 headline NO-GO，但两条小模块路线都值得立即闭合。**

最重要的口径纠正是：`4.764209x` 来自 **M216 H67 FC2 的 K1→K8 always-ready standalone frontend**，分子/分母分别是 `429,716,335 / 90,196,785` cycles。它不是 M307 Conv 的数字，也不能接到 M321 matcher 上。M307 的选定 `tau=[1,0,1,1]` 只有：

| Conv 端口 | vs bit-sparse | vs tau0 exact |
|---|---:|---:|
| WIDE144 | 1.644554x | 1.067506x |
| SHARED96 | 1.292379x | 1.048245x |

## A. FC2：M216→M218→M304

M216 的 4.764209x 在其限定范围内是可信的：120 条 H67 ep35 FC2 record、5.58M token、143.895M event，K1/K8 使用同脚本和 scope-matched frontend。把 K1 控制开销完全删掉，仍有 `412,900,394 / 90,196,785 = 4.577773x` 的 frontend oracle 比值。

M218 已经有独立 service RTL、VCS 和 3 ns logic-only DC；但它和 M216 尚未连接跑 frozen trace。L4/O8/II1 premodel 给出 service ratio `4.952122x`，frontend/service 极端重叠区间为 `4.214619x–5.785795x`。这只是固定延迟、顺序响应模型的筛选区间，不是 SRAM/commit 后的物理上下界。

M304 不是 FC2 链的闭合，而是把 M216/M218 协议壳迁移到 FC1/Conv G4 的另一个 predesign。它对 cropped K1 可形成约 3x 的 task-only 数字，但单 G4 比现有 M218 K8 慢 2.52–2.71x；G4x4 相对强 K8 的上限只有 FC1 `1.4787x`、Conv `1.5878x`，且四读/每 bank 尚未解决。不能继承 4.764x。

最小可执行方案不是全系统 scheduler，而是一个连接模块：

1. M216 K1/K8 frontend；
2. FIFO4、O8 和 M218 tagged slice service；
3. 八个有限端口 weight bank，显式 L/II、仲裁、response tag 和 backpressure；
4. Acc24 context、result drain、BN2/residual 与 final commit；
5. 在同一 120-record trace 上逐拍 miter group/request/response/context/result/done。

这条线最有希望先得到可信的 **4x 级模块结果**。在连接执行器出数前，只能写 “4.764x standalone FC2 frontend”，不能写 complete FC2/FFN、physical 或 system speedup。

## B. Conv：M251/M280/M307→M321/M329

M307 并非把 matcher、packer 和 PWP DMA 直接当成零：它按 phase 计算 `matcher_rows+16`、`ceil(assignments/8)+4`，并以 32 B/cycle 加载 12,288 B weight + 18,432 B PWP，即 960 cycles/phase。问题是 17,280 个 phase 全部由 compute 绑定，其他成本都被 `max()` 和完美双缓冲隐藏；center metadata、真实 bank conflict、有限 FIFO、neuron/commit 和 candidate-only area 没有闭合。

存在一个更严重的接口 P0：M307 在 `distance>tau` 但 `1+distance<population` 时仍需使用最近 center 的 PWP，再执行 residual correction；M321 在 `out_snapped=false` 时却把 `out_selected_pattern` 设回 original pattern，而且没有 center index。按冻结账本，至少 **5,161,708** 个 selected assignment row 需要这条缺失路径，占 assignment 至少 **45.91%**。因此当前 M321/M329 不能实现 M307。

最小修复是让 matcher 无条件输出 `best_center + center_index + distance`，另行输出 `snapped`，保留 original 并生成 `residual_mask=original XOR best_center`。随后做一个 Phi 风格的局部 K-first 执行器：raw-row/prefilter → latency2/II1 matcher → finite match FIFO → pattern-index PWP prefetch → 8-unit cross-row pack → PWP/correction compute → neuron/commit。所有 stage 必须逐拍 ready/valid，只有独立端口和非满队列时才允许 overlap。

M321 的 latency2/II1 映射到每 partition：若接受 N 行，沿用 M251 的记账方式应写 `N+2`；精确首发到末退边沿是 `N+1`，二者必须在 N=1/2/burst directed trace 中冻结。M307 phase 全局串行时，一个 q16 matcher 可以共享；若并行多个 partition，则 matcher 数量和 256-bit center 供数按并发数线性增加。

M329 只提供一个 q16 matcher 的 bounded logic 代价：`1,997.982 um2`、3 ns、PT setup `+1.1141 ns`、hold `+0.0180 ns`、0 macro、Formality 通过。M307 的 “same-resource” 没有把这块逻辑和 center/PWP/metadata SRAM 加进候选，也没有给 baseline 等面积资源，所以不能直接写 throughput/mm2。

简单敏感性不是 admission，但能看出风险：

| 模型 | WIDE144 | SHARED96 |
|---|---:|---:|
| M307 完美 overlap 上限 | 1.6446x | 1.2924x |
| matcher/packer/center 串行，已有 eligible compaction | 约 1.5674x | 1.2443x |
| 无免费 compaction，51.84M raw row 全部入 matcher | 约 1.4156x | 1.1467x |

若再假设每个最终 Acc24 输出经同一个 32 B/cycle 端口 commit 一次，WIDE raw-scan screen 约为 `1.4065x`。这些不是保证下界；真实 bank/FIFO/commit 可能更差，所以当前尚无正的 executable-cycle lower bound。

## q 扩展建议

q 扩展必须另立 DSE，不能写进当前 q16 M307 实测。旧 M70 heldout k16 的 q8/16/32/64/128 vector-op screen 为 `1.115/1.189/1.450/1.710/1.948x`；相应 signed19 全 catalog PWP 约 `24/48/96/192/384.75 MiB`。q128 使用了 `220,683/221,184` entries，不能靠全局 pruning 解决容量；signed12 也约 243 MiB。

单个 M321 做 q128 至少八 pass/row；八路复制约 `15,983.856 um2` logic-only，尚不含 2,048-bit center 接口、全局 minimum reduction 和 SRAM。另一个方案是真正的 q-entry 1-D systolic matcher，以 q-cycle fill 后 1 row/cycle运行。两者都必须先采集 **tile-local working-set、pattern-index PWP prefetch 和 bank-conflict trace** 才能判断是否可行。

优先级：先修 M321 nearest-center/index P0；并行做 M216→M218 connected FC2 executor。然后做 q16 Conv 局部执行器，最后才跑同 cohort 的 q 扩展和 macro/energy DSE。`docs/359` 保持 `dedde7ce...` 未动。
