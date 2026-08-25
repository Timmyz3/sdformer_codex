# M117 W384 prefetch transpose 独立打铁评审 r1

日期：2026-08-24  
评分：**86/100**  
严重度：**P0=1，P1=7，P2=5**

结论：M117 的首 key、下一 key 预取 identity 调度和 254/254 条件式无 group bubble 经独立商业 VCS 打铁通过；真实 768-bit 权重 payload、lane SRAM、shared arbiter、numeric mapper、accumulator 与物理性能仍未实现，因此 2.535462× 仍只能称为 M109 软件投影。

本评审只写本目录，未修改 production RTL、SVA、TB、contract 或 sealed evidence。`docs/359_DATE终局冻结_20260813.md` 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 独立 VCS 结论

独立 TB 使用 Synopsys VCS V-2023.12-SP1，构造了两组 seeded sparse full-key 窗口。事件以 reverse key/row 顺序进入，调度器必须按 ascending key/row 输出；service 端承受重复随机 stall，prefetch 端在 254/254 主实验中保持 ready。

| 指标 | 结果 |
|---|---:|
| full-key sparse windows | 2 |
| 主实验 ingress/service events | 768 / 768 |
| weight prefetch accepts | 256 |
| load tokens | 768 |
| zero-bubble scoreboard | 254 / 254 |
| simultaneous-accept SVA subset | 246 |
| early-prefetch final-stall cover | 9 |
| service stall cycles | 402 |
| max repeated stall | 14 |
| stall releases | 63 |
| fill/drain ping-pong overlap | 388 |

另有独立定向攻击：初始 key prefetch stall 7 周期；final event 上下一 key identity stall 6 周期；预取已接受后继续 hold final event 4 周期。最终只有两次预取接受，没有重复请求，next key 的 load0 紧随 final-event retire 出现。

## 首 key 与下一 key 是否重复或漏读

首 key 路径在 READY descriptor dispatch 时发出 `source/block/context`，接受后下一可见 service token 为同 identity 的 beat0。

下一 key 有三种时序：

1. final event 与 prefetch 同拍接受，下一周期直接显示 next-key beat0；
2. final event 被 stall，而 prefetch 提前接受，`next_key_prefetched_q` 记住完成并停止重复请求，event 以后 retire 时直接进入 beat0；
3. final event 先 retire、prefetch 尚未接受，`drain_prefetch_wait_q` 暂停 service，并持续保持同 identity，接受后再进入 beat0。

独立 scoreboard 对两组各 127 条 key transition 全数检查，得到 **254/254**。SVA 的直接 simultaneous-accept 子集命中 246 次；early-accept cover 命中 9 次，其中包含专门攻击 descriptor。该结果是 directed evidence，不是形式化穷举。

## Ready/valid、close grace 与 descriptor_done

service 和 prefetch 在 stall 中所有 identity/sideband 均稳定。标准 streaming 下，上一项接受后下一周期可直接换成新 event/close payload；若保持完全相同的已接受 identity，则 exact grace 阻止重复接受。本次覆盖两次 last ingress event exact grace 以及紧随其后的 close exact grace。

两组 empty 与三组 nonempty descriptor 的 `descriptor_done_empty/base/context` 顺序均正确。两个 empty descriptor 连续提交时，`descriptor_done` 会连续两个周期保持高、identity 每周期变化。这个接口可按“每个高周期都是一次 completion”消费，但不能由下游只检测上升沿；同时它没有 `done_ready`。该约束必须在集成合同里冻结或改成 valid/ready。

双 bank 的 fill/drain ping-pong 在 388 个周期并行活动。需要注意 ingress 并非通用 backpressure：若没有 fill bank，仍呈现 `event_valid` 会触发 sticky protocol fault，所以上游必须先有 credit。

## 最大缺口：identity 不是 payload

`weight_prefetch_*` 只有：

- source：4 bit；
- block：3 bit；
- context：16 bit；
- 合计：23 bit identity。

接口没有 96-lane INT8，也就是 768-bit 权重向量；三个 counted load beat 也只有 beat identity，没有 256-bit read request/data/response、SRAM response-valid、shared arbiter、bank conflict 或 residency 信号。因此 `weight_prefetch_accept` 只证明“请求 identity 被接受”，不能证明下一周期 load0 所需的数据已经到位。

这里还有一个会直接决定 2.5× 是否成立的同步 SRAM 边界：

```text
若 256-bit sync read 从 load0 开始：
load0: issue beat0
load1: response0 + issue1
load2: response1 + issue2
event0: response2 + consume full 768b
```

这种实现必须有 `load2 response -> first event` 的 tail-bypass；如果 event0 只能读取上一拍已注册完成的 768-bit vector，就必须再插一个 bubble。当前 RTL 既没有 payload response，也没有 tail-bypass。另一方面，如果设计意图是 `weight_prefetch_accept` 仅在整条 768-bit 已驻留时才为真，也必须把这种“full-residency completion”语义写进接口，而不能继续把 request acceptance 与 payload ready 混用。

同样尚缺：

- 同步 256-bit 三 beat 路径和 load2 tail-bypass；
- 196,608-bit dual bitmap 的 foundry macro/真实 latency；
- PWP/correction numeric mapper；
- signed20 accumulator 接入；
- exact heldout descriptor/payload replay；
- macro-inclusive STA/PTPX 与 matched baseline。

所以 M117 是 scheduler transition kernel，不是完整加速 datapath。

## 2.535462× 与 bubble 风险

M109 数字为：

```text
baseline = 1,114,863,448 cycles
candidate = 439,708,199 cycles
projection = 2.5354620417×
```

距离 2.5× 只有 6,237,180.2 cycles，即每个 active group 最多平均容纳 **0.754075** 个额外 cycle。若真实 payload 路径每组增加一个 bubble：

```text
candidate = 439,708,199 + 8,271,296
          = 447,979,495 cycles
ratio     = 1,114,863,448 / 447,979,495
          = 2.4886483878×
```

这会直接跌破 2.5×。因此 254/254 identity-port 结果值得保留，但不能把 always-ready identity 口等价成真实权重 SRAM。

## 打铁分级

P0 是 payload delivery 缺失：M117 用于守住 2.5× 的关键机制仍停留在 identity handshake，尚未在数据通路实现。

P1 包括：one-bubble 风险、lane SRAM/shared arbiter 缺失、numeric mapper/accumulator 缺失、bitmap 仍是 behavioral array、全 trace 未执行、ingress 依赖外部 credit、back-to-back done 无 handshake。

P2 包括：SVA 只覆盖 simultaneous 子集、directed 而非 formal、descriptor_done 不等于 commit、base+row 未限制溢出、precompaction 成本在模块外。

## GO / NO-GO

| 项目 | 判定 |
|---|---|
| production exact-SHA sealed VCS | **GO** |
| independent commercial VCS | **GO** |
| dispatch/next-key identity prefetch | **GO directed** |
| stall identity、no duplicate/no skip | **GO directed** |
| 254/254 zero transition bubble | **GO，限 always-ready identity port** |
| empty/nonempty done 与 ping-pong | **GO directed，done 必须逐周期采样** |
| 768-bit payload/lane SRAM/shared arbiter | **NO-GO** |
| numeric mapper/accumulator | **NO-GO** |
| actual heldout scheduled replay | **NO-GO** |
| M109 2.535462× 软件投影 | **GO，qualifier mandatory** |
| scheduled/physical/system/headline 2.535462× | **NO-GO** |

复跑：

```bash
cd hw_autoresearch_nts07/reviews/m117_w384_prefetch_transpose_independent_hammer_r1_20260824
./run_vcs_m117_independent_hammer.sh
```

runner 会拒绝覆盖现有 `vcs_sealed/`；新复跑应使用新的 sealed 收据目录。
