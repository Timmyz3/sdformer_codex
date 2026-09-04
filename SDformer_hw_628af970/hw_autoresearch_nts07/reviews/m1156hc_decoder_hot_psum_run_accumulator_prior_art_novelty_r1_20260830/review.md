# M1156HC｜Hot-psum destination-run accumulator prior-art / novelty 边界

## 裁决

**可以继续，但必须并入 C2，不能作为第四个贡献，也不能声称发明了 accumulator/cache/output-stationary。**

当前 M1155HC 单 D0 独立重放给出的 `1.979143×` local CPU cycle-model speedup 与 `98.924981%` backing-operation reduction是很强的机制筛选结果，但仍只有 H67_ep35 的一个 D0 call；在 D0–D3、最终 checkpoint、RTL/VCS、DC/PPA 和 memory-inclusive accounting 完成前，不得成为摘要或系统表数字。

## 哪些部分是经典机制

1. **当前 destination 留在寄存器中连续累加**是 output-stationary/local-psum accumulation 的基本形式。Eyeriss 已系统讨论在本地存储和 PE 间累积 psum、减少 psum 数据移动。
2. **valid/dirty/tag、fill、hit、evict、flush**是普通 write-back cache/run accumulator 协议；entry 数为 1 不构成新颖性。
3. **按输出坐标聚合稀疏贡献**属于 Gustavson sparse accumulation / sparse merge 的传统边界。GAMMA 用 Gustavson dataflow 与 cache/buffer 混合存储；SpArch 在线合并 partial matrices。
4. **K3/S2 transposed convolution 的零模式、相位拆分与重排**已有 GANAX 等直接前作。这里要特别纠正：GANAX 是 **ISCA 2018**，不是 MICRO 2018。
5. **事件/脉冲的稀疏 descriptor 解码和流式处理**已有 FireFly-T、Prosperity、Phi 与 ELSA。后者还明确采用 bundled AER 和 spiking Gustavson-product。

因此禁止把 `one-entry hot-psum cache`、`destination-run accumulator`、`exact fill/evict/flush` 或 `output stationary` 单独称为 novel。

## H67 真正可以 claim 的对象差和协议差

对象差不是“又一个 cache”，而是下面五项必须同时成立的约束映射：

- producer 是 **binary ATLIF 的 typed K8 source descriptor**，乘积为 signed local-INT8，而非稠密 deconv 输入或通用 SpMSpM fiber；
- H67 的 **K3/S2 polyphase source order** 对精确 key `(timestep, destination, output_block)` 形成自然连续 run；冻结 D0 中 4,417,036 个非冷引用全部是 reuse-distance 0；
- datapath 必须保留完整 **96×Acc24**，即 2,304 bit/288 B，不能偷换成 288 bit；另加 16 bit metadata，总计 290 B；
- 固定 240 KiB 已占 243,200 B，只余 2,560 B；该 direct one-entry stage 放入后剩 2,270 B，因此不需要 CAM 或新 SRAM 宏；
- exact protocol 必须完整收费：cold fill、hit dependency、key-change dirty eviction、timestep-terminal flush、dense commit，且不能把 source-term 数当 psum-update 数。

这组对象/协议差可以写成“针对 H67 decoder descriptor stream 的 resource-constrained specialization”，不能写成“新的 sparse accumulator 原理”。

## 如何并入 C2

不要增加 C4。把现有 C2 改写为一个完整的 typed sparse execution path：

- **C2-a：source side**——typed signed K8 descriptor、共享 96-lane Acc24、已有等带宽面积效率；
- **C2-b：destination side**——利用 mapper 保证的 contiguous destination run，在 Acc24 出口增加一项 290-B exact writeback coalescer，再接既有六-bank 1RW psum SRAM。

论文图中把两者画在同一条 descriptor→issue→Acc24→run accumulator→backing SRAM 路径。实验中只做 C2 accumulator off/on 消融，并保持 descriptor、weight、compute、dense commit、96 lane、六-bank 1RW 和 240 KiB 完全一致。这样 novelty 落在“typed descriptor protocol 跨 source issue 与 destination lifetime 的联合映射”，而不是冒充第四个算法。

## DATE claim 模板

### 当前可用的保守模板

> We specialize the C2 typed-sparse execution fabric for H67's K3/S2 decoder. The polyphase descriptor order exposes contiguous runs of the exact `(timestep, destination, output-block)` key, allowing a single 96×Acc24 write-back stage to coalesce backing-psum accesses through exact fill, dirty-eviction, and terminal-flush semantics. The 290-B stage fits within the residual capacity of the fixed 240-KiB memory budget and introduces neither an associative matcher nor a new sparsity definition.

### 完成 D0–D3、最终 checkpoint 与 RTL/PPA 后才可追加

> Under identical descriptors, weights, compute width, dense commits, six-bank 1RW backing SRAM, and total 240-KiB capacity, the destination-run stage reduces backing-psum operations by `[X%]` and decoder cycles by `[Y%]`, with zero exactness mismatches and `[area/power overhead]`.

### 禁用措辞

- `the first/novel partial-sum accumulator/cache`
- `the first output-stationary SNN/deconvolution engine`
- `a novel Gustavson accumulator`
- `the first exploitation of transposed-convolution/polyphase locality`
- `the first event-driven sparse decoder`
- `a new sparsity paradigm`
- 用当前单 D0 的 `1.979×` 宣称 decoder/full-network speedup

## 必引 primary sources

完整链接和引用边界见 `references.md`。最低必引组合是 Gustavson、Eyeriss、GANAX、GAMMA 或 SpArch、Prosperity、Phi、FireFly-T、ELSA，以及事件 SNN 光流 decoder 的 workload 来源。
