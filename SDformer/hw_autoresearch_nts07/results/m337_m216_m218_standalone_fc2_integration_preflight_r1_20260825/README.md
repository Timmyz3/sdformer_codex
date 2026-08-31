# M337：M216 K8 frontend + M218 tagged slice service standalone FC2 集成预检

结论：可以组成一个不依赖全系统调度的 `raw4 event -> signed Acc24` standalone FC2 模块，进入实现为 `GO`；但不能把现有两块 RTL 直接广播 header 后简单拼接，也不能把当前边界称为 checkpoint-equivalent 的完整 FC2+BN2+SN2 层。

最小集成只有三处控制逻辑：

- header 必须对 M216/M218 做原子 fork/join，保证 top、frontend、service 三个 accept 同拍；
- M216 的 `token_done` 是 frontend exhaustion，必须桥接成 M218 的 `frontend_done`，最终 top done 只能来自 M218；
- K8 group 的 tag、output block、source count、bank mask 和八路 source channel 已逐位兼容，可以直连。

每个 group 在 M218 内展开成 6 个 16-lane request。精确映射为：

`destination = output_block * 96 + slice * 16 + lane`

`source_channel = raw_beat_index * 96 + bitmap_row * 8 + bank`

外部 weight store 必须是八个等深、每 bank 每拍至多一读的 128-bit word bank。一次 K8 request 对全部 active bank 原子接收，响应必须原样回显 epoch/slot/generation/tag/bank mask，并在 backpressure 下保持完整 1024-bit 最大 payload 稳定。主性能点沿用 `L4/O8/II1`，但还需 sweep latency 和 outstanding。

唯一公平 K1 基线不是 M219 单块，而是：

`M216(SOURCE_CAP=1) + onehot-to-scalar bridge + M219 + 同一八 bank memory + 同一 Acc24 commit`

候选则为：

`M216(SOURCE_CAP=8) + M218 + 同一八 bank memory + 同一 Acc24 commit`

两边必须从同一 raw header/beat population 开始，独立运行，cycle 都从 header accept 数到 final token done accept。不能把 M216 的 `4.764x` 和 M218 的 `4.952x` 相加或相乘。

网络源码中的 FC2 是 `bias=False`，因此 Linear bias 精确为零；M218 的 raw Acc24 后仍缺 weight scale/dequant、BN2、SN2、residual/写回。第一轮应明确只做 standalone INT8 accumulation core。完整模型层语言必须等这些后处理拥有 checkpoint-bound 数值合同和 Python miter。

当前实现起步无 P0；有 6 个 P1 集成缺口。任何 connected speedup 需要先补 exact top、同边界 K1、可执行 weight response model 和 VCS cycle miter；任何 complete-FC2 表述还必须补 weight-set identity、scale/round 和 BN2/SN2/commit。

建议下一步 M338 只实现约 400–650 行 production wrapper/bridge，加 1,400–2,100 行 memory model、TB、SVA 和 runner。首轮只跑固定 L4/O8/II1 的 B=1/2/4/8 bit-exact/conservation，随后再加 stalls、OOO response、reset、fault 和 frozen H67 replay。

评分 `86/100`，`P0=0, P1=6, P2=3`。M337 未写 RTL、未运行 VCS/DC/EDA、未修改合同或 docs/359。
