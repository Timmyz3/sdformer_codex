# M534 r2 下一条 RTL 候选筛选独立打铁 r1

日期：2026-08-27  
被审对象：`reviews/m534_next_rtl_candidate_screen_r2_20260827`  
模式：只读静态审计；零 CPU/VCS/DC/PT/PTPX/Formality/训练/远端运行；零 RTL 修改  
裁决：`FAIL_CLOSED__AUTHOR_R3_SCREEN_REPAIR_ONLY__NO_PRE_RTL_CPU_CONTRACT__NO_RUN__NO_RTL`  
评分：**90/100**；P0/P1/P2 = **0/2/2**

## 1. 结论

r2 已经实质修复 r1 的结构性 P0。一个合法 M523 8-lane bundle 现在先进入收费的
`8x128-bit` atomic ingress；即使 cross-event packing 令八条 lane 全落入一个 phase，也可先整体
accept，再用该 phase 的四个 context 分两波完成 exact partial-RMW。下一 bundle 在 ingress 和 16 个
context 排空前整体 backpressure。context drain 不等于 semantic close，final commit 仍依赖 canonical
frontier/fence。因此，在有限且公平的 memory/sink response 假设下，本审未发现正常 bundle 必然
deadlock、early-close、drop/duplicate 或偷第十七个 context 的结构性路径。

同资源方向也明显收紧：`A1-SC8/A1-ISO8/A1-OSG/PBR4` 共享 M218 的 6x16、L4/O8/FIFO4、八个
weight bank、四个 psum bank和同一 external link；strongest baseline 是在整个 S10 上固定选择的一个
architecture point，不允许 per-sample oracle。ELSA 的 bundled AER、Gustavson、single-RMW 与 dependency
completion 已被承认为直接前作；只有过 A1-OSG 后才保留 ConvTranspose parity-frontier 的窄对象差。
这些修改关闭了 r1 的 baseline、prior-art、S10/S100、CPU/DC 混门和候选 B 1% 容差问题。

但是 r2 还不能直接生成可执行 CPU contract。其 exact 数值输入和 external psum backing 的存在性状态
仍有两组会改变周期/traffic/reference 的歧义。当前唯一授权下一步是 author r3 screen repair；r3 经新
reviewer 得到 P0/P1=0 后，才可 author 仍为 `run_authorized=false` 的 pre-RTL CPU contract。

## 2. 已通过的核心复算

### 2.1 atomic ingress 与 progress

- M523 interior 首个 full-8 的 `4 odd/odd + 2 odd/even + 2 even/odd` 已由四 context/phase 覆盖。
- 任意 cross-event 单 phase 八 destination 不要求八个常驻 context：八条 lane 已由 atomic ingress
  同拍保存；前四个 destination 做完收费的 partial-RMW 并释放后，后四个再搬入。
- 一个 active bundle epoch 内最多八条 lane；单 context 的八 contributor slot 不会因该 epoch 自身溢出。
- psum/weight/join/backpressure 均只能阻止下一状态，不能把 partial-RMW 解释为 output completion。

这一结论仅认可 r2 的有限状态结构，不是 exhaustive proof，也不生成任何性能数字。未来 reference
仍须覆盖 r2 列出的 single-phase-eight、同 bank multi-round、dirty eviction/restore、RAW、frontier、fault
drain 和小尺寸穷举。

### 2.2 222,736 B 算术

独立加总如下：

| 项 | byte |
|---|---:|
| 8x128x128b weight | 16,384 |
| 4x1024x384b psum | 196,608 |
| 4x4x256b join | 512 |
| 4x128x128b tag/frontier | 8,192 |
| four 384b RAW bypass | 192 |
| 8x128b atomic ingress | 128 |
| 18x128b descriptor FIFO | 288 |
| O8/FIFO4/two skids | 432 |
| **合计** | **222,736** |
| `240 KiB - 合计` | **23,024** |

weight tile 的 `16 Cin x 9 K x 96 Cout = 13,824 B` 与 `108` 个 128-B aggregate refill beat 一致；
每个 weight bank 的 18 个 flattened key x 6 slice = 108 row，也与 `128x128b` 几何一致。每 phase
128 destination x 6 slice = 768 row，装入 1024-row psum bank。上述算术通过，但“已列项目之和正确”
不等于完整 backing/control state 已收费，见 P1-02。

## 3. P1 findings

### P1-01｜binary source 的 exact 数值与 identity 没有冻结

r2 把 ingress lane 写成“至少携带”tag/time/source-channel/kernel-index/destination/phase/fence，并把
context contributor 写成 `{flat_source_key16,sign1}`，但没有定义：

1. M511 decoder descriptor 的 source value 是否严格为隐式 `+1`；
2. `sign1` 是 source sign、weight sign 还是 product sign；
3. M523 原接口的 `source_y/source_x/kernel_y/kernel_x` 是原样存储，还是由 destination/kernel 无损重建；
4. cross-event bundle 中 event identity、canonical source frontier 和 duplicate detection 使用哪组字段。

M523 当前是 binary descriptor transport，不携带 signed activation payload；M218 则是 signed INT8 weight
reduction。若 source 确为 binary `1`，r3 应明确冻结，并把 source coordinate 的保留/可逆重建公式、
event ordinal/fence ownership和 128-bit ingress、256-bit context 的逐字段 bit allocation写全。若 source
不是隐式 `+1`，现有 slot 没有数值 payload，Acc24 reference 不能定义。这个问题必须在 CPU contract
之前关闭，不能让 analyzer 自行选择解释。

### P1-02｜backing 存在性和 mandatory state 尚未形成 closed ledger

r2 同时规定：dirty victim 写回 external psum backing；之后 miss 可 restore；“从未存在”的 destination
只做六次 zero-fill write。问题是 resident tag 被替换后，当前 8,192-B tag array 不再保存该 destination
是否曾写回。对稀疏 source stream，仅由当前 contributor 也不能一般性区分：

- 该 destination 过去没有 nonzero contributor；或
- 过去已有 partial sum、但其 tag/data 已被逐出 external backing。

二者分别收费 `6 cycle` 与 `41 cycle`，会改变 candidate/baseline speedup。r3 必须冻结且四点共享以下
一种 exact 方案：

- external backing 对全部 destination 预初始化为零，所有非 resident miss 都统一 restore，并收费 frame
  clear/初始化或其既有系统身份；或
- 增加 persistent backing-valid/address identity，按完整四层 output-block 空间计容量、端口、清零和
  访问；或
- 仅对静态可证的 canonical first-contributor 使用 zero-fill，其余一律 restore，并给可执行判定式。

不得用 resident tag、未收费 Python set 或“23,024 B 余量足够”代替这份状态。这里与完整 state ledger
是同一个阻塞：总和正确，但 r2 只列出数据阵列、O8/FIFO4 和两个 skid。一个可执行模型还需要至少明确归属：weight
resident tile identity/valid、refill beat/counter、shared-link arbiter、pending psum write、victim/restore
command、global canonical frontier/fault epoch，以及 P1-02 选择所需的 backing state。部分可以是
standard-cell control、部分可能是 SRAM bitmap；两者都不能隐身。

r3 不必把 standard-cell bits伪装成 SRAM byte，也不必现在估 mapped area，但应给：

- 每个 state 的 exact bit count、bank/port/owner、四点是否共享；
- SRAM/bitmap 项进入 `<=240 KiB` 加总并重新给 headroom；
- standard-cell 项进入 future CPU state/comparator ledger并在 paired DC 收费；
- 128-bit ingress与256-bit context逐字段 packing，证明没有 overflow/hidden side state。

只在 contract 结果阶段“entry 不足则 fail”太晚；否则 contract author 可以在 23,024-B 余量里隐式增加
candidate-only scheduler/backing state。

## 4. P2

1. `A1-OSG` 很可能在同资源下吃掉 PBR4 的大部分或全部差值；r2 正确把等价设为 KILL。未来 CPU
   合同应输出 group/RMW/commit sequence hash，不能仅比较 total cycles 后口头判断等价。
2. S10 的 `1.30x`/逐样本 `1.10x` 只是单序列 fast-kill。即使通过，也不能写 multi-sequence robustness、
   full-network或 system speedup；system 行继续等待同 cohort denominator。

## 5. 候选 B

B 的修复通过静态筛选：cycle、accept、request/response、active-bank read、data byte、Acc24 update、
result beat 和 done-cycle 全部为 exact integer zero delta；mapped-gate PTPX 继承 net/leaf 100% annotation、
active-cone nonzero-toggle、同 contiguous window/macro boundary，dynamic=`internal+switching`。它仍必须等待
独立双封 M519 K1/K8/K1x8 canonical，不产生 cycle/system speedup，也不获本 hammer 的 RTL/EDA 授权。

## 6. 身份、seal 与授权

- 被审 `README.md` SHA256：`fb2eeb6346e4a61b26d6a4f062e1b062fcb1e9a5f7f5e09b3c197c1b4dd64257`；
- 被审 JSON SHA256：`e8d3a0050343d105b87c6b809cd7d743f7b6e133201072e9115cb6cf1b7d8b0b`；
- 被审 member seal 与 outer seal 均在其目录内通过；13 个 frozen input SHA 全部与当前文件一致；
- `docs/359` SHA256 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改；
- `git diff --check` 通过；本 hammer 未运行 CPU、Synopsys、训练或远端任务，未实现/修改 RTL。

授权矩阵：只允许 `M534 r3 screen repair authoring=true`。pre-RTL CPU contract、CPU run、RTL、VCS、DC、
PT/PTPX、Formality、训练、远端、performance/energy/PPA/system/headline 全部为 false。
