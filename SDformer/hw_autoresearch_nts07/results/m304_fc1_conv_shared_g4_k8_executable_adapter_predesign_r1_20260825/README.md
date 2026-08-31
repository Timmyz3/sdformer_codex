# M304 FC1 + 3x3 Conv 共享 G4/K8 可执行适配器 pre-design 审阅

结论是“分层 GO”：先做单 G4/K8 可执行适配器，用它封住 beta=0 exact、mask/scan/commit 开销和真实 cycle；在存储端口 DSE 解决之前，不直接做 G4x4 RTL；当前不允许宣称相对 M218 K8 有 3–5x 增量加速。综合评分 74/100，P0=0、P1=7。

## 最小可复用路径

M216 可直接复用 ready/valid 协议外壳、双 buffer、8 个固定 source bank、每 bank 最老 source 优先选择、K8 group hold 和 fail-closed done/tag/count fence。但必须把 FC2 的“两个 window 合并 + output-block 重放”改成“当前/下一 destination group 的剩余 source bitmap ping-pong”。Conv beta=0 最大 864 个 `channel*9+tap` source key，即每个 modulo-8 bank 108 个；M216 现有 8x96-bit/96-per-bank 容量不够，需扩到至少 9x96-bit/108-per-bank，或在 cycle model 中显式分 tile。

M218 可直接复用 FIFO4、O8 epoch/slot/generation/tag scoreboard、同拍 slot reuse、OOO response 验证、response skid、soft flush/stale quarantine/timeout 和 8 路 signed INT8 reduction tree。需把固定 FC2 `8 block x 6 slice x 16 lane` Acc24 上下文改成可寻址的 `context_slot x dest_group x 4 lane`，并加入 op/module/context/dest-group 身份。M219 作 K1 golden/fallback，M220 的 recurrence、stall/OOO 攻击和 identity miter 框架可扩展到 FC1/Conv。

M275r2 没有 RTL，只能复用 ping-pong bank lifecycle、valid fence、role switch、no-use-before-valid 和读写冲突分开记账的方法。其 materializer 仅改变 2,530/352,335,120 cycles，不是本适配器的加速来源。

## 为什么必须 destination-major

不能先打包 8 个 source，再对它们的 destination mask 做 OR。beta48 下 FC1/Conv 保留 task 比例仍有 0.6763/0.6298，8 个 source 对同一 group 同时被跳过的机会很小，会变成“节省 active-bank 能量，cycle 近乎不降”。

可执行算法应为：对每个 destination group 构造 `remaining = active_source_bitmap & keep_mask_column`，8 个 bank 各选一个 source，握手成功后只清除本 group 的已选 bit，清空后才前进。mask 应按 destination-group column 存储，并预取 current/next column。固定 beta48 的 FC1 + Conv source/group mask 合计 1,090,944 bit，即 136,368 byte。

## beta=0 exact 和 beta=48 附加门

beta=0 必须强制 keep-all，不读 lossy mask；包括所有 active source x destination-group task，包括数值为零的权重；与 beta48 共用同一 K8 scheduler/accumulator，不能暗中切 K1。在 per-row scale/bias/dequant 身份未接入前，只能声称 dense INT8 reference bit-exact，不能声称原 float forward exact。Conv 还需独立验证 padding/stride/tap 与 pixel context。

beta48 只能在 idle/token fence 处开启，并绑定 checkpoint/module/mask/quant/no_running-BN/S10 sample 的 SHA 和身份。先跑同一 S10 的 beta0，再跑 beta48，AEE 绝对增加不得超过 0.02。丢失或不匹配 mask 必须 fail closed；运行时 `selected + skipped == beta0 tasks`。FC1/Conv 的现有 raw INT8 bound 分别为 9,936/21,504，不是网络输出误差证明。

## 性能的诚实上限

M218 每个理想 request 是 `8 source x 16 dest = 128` contributions；单 G4/K8 只有 `8 x 4 = 32`。因此在 beta48 task 全部转化成 cycle 的过度乐观假设下：

- 单 G4 对 M219 K1 是 FC1 2.96x、Conv 3.18x；但对 M218 K8 反而分别多 2.71x/2.52x cycles。所以它是 calibration RTL，不是最终性能点。
- 要保持 128 contributions/cycle，需要 G4x4/K8。其 beta48 task-only 上限相对 M218 K8 只有 FC1 1.48x、Conv 1.59x；相对 M221 96-lane source baseline 是 1.97x/2.12x，仍未扣除任何开销。
- G4x4 每拍可能要求每个 source bank 读 4 个不同地址。M218 每 bank 只有 1 读，必须在 32 个物理 bank、4 倍复制/多端口或串行化中选择；串行化会把收益吃掉。这是当前阻止“相对最强 K8 再 3–5x”的主要真实瓶颈。

## 建议的实现顺序

1. 先做单 G4/K8 destination-major frontend、256-bit response service 和可寻址 G4 Acc24，只跑 beta0。
2. 用 M219/M220 框架做 FC1 K1-vs-K8 miter，另建 Conv padding/stride/tap software reference，攻击 stall、OOO、flush 和坏身份。
3. 加入 column-oriented mask SRAM current/next prefetch 和 beta48 gate，但只在 paired S10 过关后打开。
4. 并行 DSE 32-bank、4 倍复制和两拍 G4x2，选宏端口/冲突/能耗帕累托前沿。
5. 最终在同一 VCS cycle model 中报 K1、single-G4、G4x2、G4x4，再进 Synopsys DC/PT/能耗，不只报对弱 baseline 的一个倍数。

完整机器可读细节、输入 SHA、接口、状态、门禁和评分见 `m304_fc1_conv_shared_g4_k8_executable_adapter_predesign_r1.json`。本次未修改任何既有里程碑、RTL、contract 或 docs/359。
