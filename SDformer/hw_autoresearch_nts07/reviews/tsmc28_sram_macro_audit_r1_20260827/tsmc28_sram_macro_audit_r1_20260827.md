# TSMC 28 nm SRAM 宏映射独立审计 r1

日期：2026-08-27  
状态：`PARTIAL_FAIL_CLOSED`  
范围：C1 selected Conv/PWP、M473/M498 parent scratch/dual-slot/psum、C2/FC2 K1/K8/K1x8、C3 ATLIF、A1 attention buffer。  
审计性质：只读；未执行 memory compiler，未启动 license server，未运行 VCS/DC/PT/SAIF/PTPX，也未修改 `docs/359` 或任何生产文件。

## 1. 技术结论

本机存在完整的 TSMC N28 HPC+ SP/DPSRAM compiler 文档与 QRT，但 `/opt/tech/tsmc28/Memory` 内没有任何已生成的 `.db/.lib/.lef/.gds/.v/.spi`。另一个既有私有交接目录存在一个校验完整的 HVT `128x128b 1RW SP` 宏，可立即作为 C1 PWP SRAM 的真实 foundry-view 证据；目前没有任何已生成 DPSRAM 宏。

因此，当前证据只允许写入两类数字：

1. `TS1N28HPCPHVTB128X128M4S` 的生成宏面积、周期、访问时间和 datasheet 电流，以及由其组成的 C1 容量/宏数小计；
2. 明确标注为 `[foundry-QRT model]` 的 SP/DPSRAM 映射敏感性。

当前仍不能声称 `paper_ppa_ready`、post-layout、macro-inclusive module power 或 energy/frame。特别是 M498 的 64x1152b 同步 1R1W parent scratch 没有生成 DP 宏；C2 的 288 KiB 权重存储还未被计入逻辑-only DC。两者都可能显著改变面积/能效结论。

## 2. 关键发现

| 模块/存储 | 逻辑需求与端口 | 优先映射 | 宏数 | 面积模型 | 时序模型 | 证据与准入 |
|---|---|---:|---:|---:|---:|---|
| C1 Q32 center PWP | low `128x768b` + high `128x512b`，两个独立只读执行 bank | `6+4` 个已生成 `128x128b 1RW SP` | 10 | 87,583.606 µm² | slow `tcyc=0.616 ns`, `tacc=0.4679 ns` | `GENERATED_VIEW`；只准宏小计，不准 integrated PPA |
| C1 Q32 correction | `128x768b`，phase-separated 1RW | 6 个已生成 `128x128b SP` | 6 | 52,550.164 µm² | 同上 | `GENERATED_VIEW`；若 load/compute overlap 要 ping-pong，则需复制 |
| C1 descriptor | `32x48b`, phase-separated 1RW | `SP 32x48m4s` | 1 | 2,937.195 µm² | slow `0.479/0.351 ns` | `FOUNDRY_QRT`；无生成 views |
| M473/M498 parent scratch | `64x1152b`，每周期 1R+1W | 首选 `16x DP 64x72m4f` | 16 | OPEN | OPEN | compiler-legal，但不在抽取 QRT、无生成 views |
| parent scratch QRT exact fallback | 同上 | `32x DP 64x36m8f` | 32 | 473,034.720 µm² | slow `0.853/0.605 ns` | 仅 QRT 模型；面积已经大于当前逻辑切片量级 |
| dual response slots | `2x1152b`，同拍弹性队列 | 保持寄存器 | — | — | — | depth 小于宏最小值，宏延迟会改变协议 |
| one resident psum bank | `64x1824b` | 条件式 `19x SP 64x96m4s` | 19 | 113,087.468 µm² | slow `0.500/0.379 ns` | 仅当 read/write 互斥；端口协议未冻结 |
| full 8-block row-tile psum | `8x64x1824b=114 KiB` | `152x SP 64x96m4s` | 152 | 904,699.744 µm² | 同上 | QRT bank sum，不是 floorplanned area |
| C2 FC2 weights | `8x2304x128b=288 KiB`，每 bank 1R | 每 bank `2048x128 + 256x128 SP` | 16 | 558,507.032 µm² | worst slow `0.800/0.701 ns` | exact-capacity QRT 组织；K1/K8/K1x8 容量相同，仅激活数不同 |
| C2 context K1/K8 | `48x384b`，same-edge update/result | 候选 `6x DP 48x64m4f` | 6 | OPEN | OPEN | compiler-legal；adapter、延迟 tag 和 exact port proof 缺失 |
| C2 context K1x8 | 8 份 `48x384b` | 当前寄存器 | — | — | — | 上述问题乘 8，暂不宏化 |
| C3 M273 ATLIF working state | 多个 depth 1/2/16 存储及双向 FIFO | 当前寄存器 | — | — | — | 小于宏最小深度或端口/延迟不兼容；全网膜电位容量未冻结 |
| A1 row + descriptor stores | `4x225x32b + 2x225x20b`，六个 phase-separated 1RW bank | `6x SP 256x32m4s` | 6 | 21,188.772 µm² | slow `0.477/0.346 ns` | QRT proxy；无 TSMC 生成 views、非 integrated A1 PPA |
| A1 slot FIFO + score directory | `32x16b` 2-enq/2-deq；`163x10b` + scan | 当前寄存器 | — | — | — | multi-access/组合选择语义不适合直接换同步 SRAM |

不得把上表面积直接相加成芯片面积：其中包含相互替代的 parent scratch 组织、one-bank 与 full-eight psum 情形，也包含不同模块的未集成模型。

## 3. 对当前 DATE 硬件结论的影响

### 3.1 C1 可以建立第一条真实 foundry-macro 证据链

已生成 `128x128b SP` 宏的 13 项文件校验全部通过，包含 slow/fast `.db/.lib/.v/.ds`、`.lef` 和 `.gds`。C1 center 加 correction 共 16 个宏，生成宏面积小计为 **140,133.770 µm²**。在 slow `ssg0p9v125c`、0.9 V 下，根据 datasheet `uA/MHz` 电流计算：

- center 全宽并行读：`10 x 11.6754 x 0.9 = 105.079 pJ/active read cycle`；
- correction 全宽并行读：`6 x 11.6754 x 0.9 = 63.047 pJ/active read cycle`。

这只是 macro-pin 内部活动能量。datasheet 明确不包含 pin power，且这里也未包含 bank select、地址/数据互连、外围寄存器、时钟树与漏电积分。只有获得真实 access trace 和集成 PTPX 后，才能换算为 frame energy。

### 3.2 M498 的 parent scratch 是高风险物理税

逻辑要求是同步 1R1W，而现有生成宏只有 1RW SP，不能同周期读写。最紧凑的 compiler-legal 组织是 `16x DP 64x72m4f`，但当前没有对应 QRT 行或生成 view，因此面积和能耗必须保持 OPEN。

可审计的 exact-capacity QRT fallback `32x DP 64x36m8f` 面积为 **473,034.720 µm²**；另一个过配置 proxy `16x DP 128x72m4f` 为 **285,350.640 µm²**。这些模型说明：dual-parent 的“一拍修复”可能由 DP 宏面积主导，不能只凭 logic-only DC 决定是否升格。

### 3.3 C2 的逻辑-only DC 尚未计入约 0.559 mm² 权重 SRAM

FC2 的 8 个权重 bank 总容量为 288 KiB。按 QRT exact-capacity 拆成每 bank `2048x128 + 256x128`，总面积为 **558,507.032 µm²**。最慢宏在 slow corner 的 `tcyc/tacc=0.800/0.701 ns`，单看宏时序可满足 3 ns，但地址译码、bank adapter、返程寄存、logic path 和布局互连尚未 STA。

K1、K8 与 K1x8 必须使用同一容量基线；性能/能耗差异只能来自每周期激活 bank 数和调度，不得把 K1 少激活误写成少部署容量。tt1v85c nominal 模型下，deep segment 每个 bank request 约 22.213 pJ；K8/K1x8 全 8-bank 激活上界约 177.704 pJ/cycle，K1 为 22.213 pJ/cycle。实际能量必须按 deep/tail 访问计数加权。

### 3.4 C3 和 A1 不应为了“宏化”破坏协议

M273 的主要工作状态深度为 1、2、16，且包含同拍 push/pop FIFO；直接使用同步 SRAM 会引入容量浪费和额外拍。当前保留寄存器是正确选择。完整 ATLIF membrane state 的 network-level depth/width/端口尚未冻结，不能把 working-state 审计外推到全网状态 SRAM。

A1 的六个 row/descriptor bank 可用 `6x256x32 SP` 做 QRT proxy，面积约 **0.0212 mm²**，相对 C2/parent scratch 较小。slot FIFO 和 score directory 具有多访问/组合扫描语义，仍应保留寄存器，除非后续设计 banked adapter 并重新验证周期。

## 4. Foundry 资产与证据等级

### 4.1 官方 compiler 支持范围

SP `tsn28hpcpd127spsram_20120200_180a` 是单端口同步读写 SRAM，单宏最大 1 Mbit：

- mux4：depth `32, 48, ... 1024, 1056, 1072, ... 8192`，width `8..144`；
- mux8：depth `64, 96, ... 2048, 2112, 2144, ... 16384`，width `4..72`；
- mux16：depth `4096, 4224, 4288, ... 32768`，width `2..39`；
- 不支持 NWORD/NMUX 值：260、772、1284、1796。

DPSRAM `tsn28hpcpdpsram_20120200_130a` 是真双端口同步读写 SRAM，A/B 口各自独立时钟，单宏最大 72 Kbit：

- F mux4：depth `32, 48, ... 1024`，width `4..72`；
- F mux8：depth `64, 96, ... 2048`，width `4..36`；
- F mux16：depth `128, 192, ... 4096`，width `4..18`；
- M mux4：depth `32, 48, ... 2048`，width `4..72`；
- M mux8：depth `64, 96, ... 4096`，width `4..36`；
- M mux16：depth `128, 192, ... 8192`，width `4..18`；
- F 不支持 WL=W/CM：68、132、196；M 不支持：132、260、388。

DPSRAM 不同端口对同一地址同时 read/write 或 write/write 在时序窗口冲突时结果不确定，wrapper 必须保留 address-collision assertion、forwarding 或禁止合同。read/read 可以并发。

### 4.2 PVT corner

与现有逻辑 setup/hold 使用方式一致的 1.0 V family 包括：

- nominal：`tt1v25c`, `tt1v85c`；
- slow：`ssg0p9v-40c/0c/125c`；
- fast：`ffg1p05v-40c/0c/125c`。

0.9 V family 则是 `tt0p9v25c/85c`、slow `ssg0p81v...` 和 fast `ffg0p99v...`。现有 logic DC 的 setup/hold 是 `ssg0p9v125c` / `ffg1p05vm40c`，与 1.0 V family 宏一致；但既有 M448 PTPX 使用 `tt0p9v25c` stdcell。若将 1.0 V SRAM QRT 电流与 0.9 V stdcell power 混在一张表，会形成 PVT 身份错误，所以 macro-inclusive power corner 仍为 OPEN。

### 4.3 QRT 指标定义

- QRT `readc/writec` 单位是 `uA/MHz`，不含 leakage 和 pin power；
- 在明确电压下，每个 active access/cycle 的内部动态能量为 `I(uA/MHz) x V(V) = pJ`；
- area 是 macro 本体面积和，不含 halo、routing channel、power grid 和 wrapper；
- cycle/access time 来自 macro 模型，不是集成模块时序；
- DPSRAM QRT 的内部 type 列出现 `spsram`，但包名和 databook 身份明确为 DPSRAM。本审计按包身份解释，并保留此文档不一致性。

## 5. 数据来源与可复核性

Foundry 文件：

- SP databook：`/opt/tech/tsmc28/Memory/tsn28hpcpd127spsram_20120200_180a/AN61001_20180416/TSMCHOME/sram/Documentation/documents/tsn28hpcpd127spsram_20120200_180a/DB_TSN28HPCPD127SPSRAM_20120200_180A.pdf`，SHA256 `50a5d5badfb08b10fa06b93bbda0ba032e40e582d60b052e54098b9b589eaa17`；
- SP QRT：同目录 `tsn28hpcpd127spsram_20120200_180a_Quick_Reference_Table.xls`，SHA256 `0174110a52d12521e30cb13071792c2401010c77927ae9f417de9ca6e8ea7d31`；
- SP compiler tar：SHA256 `062e2c0b9b39bea923fd8a7b2af5ed6f25c8940bcfd14a11885dc972bbe1430a`；
- DP databook：`/opt/tech/tsmc28/Memory/tsn28hpcpdpsram_20120200_130a/AN61001_20180125/TSMCHOME/sram/Documentation/documents/tsn28hpcpdpsram_20120200_130a/DB_TSN28HPCPDPSRAM_20120200_130A.pdf`，SHA256 `539282bfd56deacb7b10a93e58b532100d41a6355707ed238987290ca46ca185`；
- DP QRT：同目录 `tsn28hpcpdpsram_20120200_130a_Quick_Reference_Table.pdf`，SHA256 `319f8ebb979bb00cafafa468bfd43b1b4a14511a3bbb56cffbd4962336ac3ebf`；
- DP compiler tar：SHA256 `08f6171a4eced076f894f9821cd169f3b392cc069d9a42d159a6c8a34e09f32d`。

已生成宏资产：

- `/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821`；
- manifest SHA256 `c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f`；
- 2026-08-27 复核 `sha256sum -c`：13/13 `OK`；
- slow DS：area 8,758.3606 µm²，cycle/access `0.6160/0.4679 ns`，read/write `11.6754/11.1923 uA/MHz`，leakage `66.6783 uA`；
- fast DS：cycle/access `0.3811/0.2887 ns`，read/write `13.3596/12.5927 uA/MHz`，leakage `3.0361 uA`。

逻辑需求来自冻结或现有 contracts/RTL，包括 M426、M433、M468、M477/M498、M219/M349、M273 和 A1 physical wrapper。机器工作树当前存在并行工作的未提交改动；本审计按文件 SHA/只读观察封存，没有修改任何输入。

本目录的结构化附件：

- `foundry_qrt_selected_rows_r1.csv`：40 条选中 QRT/generated-datasheet 行；
- `tsmc28_sram_mapping_r1.json`：逐模块映射、面积/时序/能量字段和 fail-closed 标签；
- `report.html`：便携式主要阅读界面；
- `artifact.json`：HTML 报告的可复现源。

HTML 报告只制作一张横向面积量级图，用来直观暴露隐藏存储税；图中明确混合了不同模块范围和替代组织，因此不能相加。所有端口条件、OPEN 状态和精确映射仍以离散表格为准。

## 6. 方法

1. 校验官方 databook/QRT/compiler tar 的身份与 hash，只读取文档和 tar 内容；
2. 从 databook冻结合法 depth/width/mux/segment 范围、端口语义和 PVT corner；
3. 从 QRT 抽取本项目相关配置的 area、cycle、access、read/write current 和 leakage；
4. 从既有 contracts/RTL 冻结逻辑 depth x width、端口并发、banking 和当前周期语义；
5. 优先使用 exact-capacity、exact-port 的合法配置；若只有 partition/overprovision 或端口未证，则标为 `QRT_PROXY`/`OPEN`；
6. 只对校验完整的 generated views 标记 `GENERATED_VIEW`；QRT 与 CACTI 不得冒充生成宏；
7. 电流只转换为 per-active-access 内部能量，不推导 frame energy；
8. 检查 3 ns 可行性只限 macro 自身 cycle/access；未做 integrated STA。

## 7. 局限性与稳健性

- `/opt/tech/tsmc28/Memory` 没有生成 views；唯一生成宏位于外部私有交接树，迁移到当前正式 handoff 前必须保留原 manifest；
- 未运行 compiler，因此 compiler-legal 配置的 PPA 仍是 OPEN；
- 未取得 DP `64x72m4f` 生成宏，M498 不得升格 physical-ready；
- 未冻结 M498 psum 同周期读写、C2 context adapter 或全网 ATLIF membrane-state 端口；
- QRT area 不包含 floorplan halo/routing，且宏数较多时 bank/address/power-grid 开销可能显著；
- QRT 电流不含 pin power；没有 access trace、standby 时间、clock/interconnect 活动，不能给能量/帧；
- 当前没有经过审计并绑定到这些配置的 CACTI 结果。旧 32 nm CACTI proxy 不得与 TSMC28 QRT/generated-view 数字混排；
- QRT 的 nominal 1.0 V 和已有某些 stdcell PTPX 0.9 V 口径不同，须重新统一 PVT；
- 所有 `tcyc<1 ns` 只说明 macro local timing 对 3 ns 有余量，不等于 RTL↔macro 集成 STA 收敛。

## 8. 后续执行门与命令前置条件

本审计没有执行下列动作。正式生成必须满足以下前置条件后再由主流程启动：

1. 在隔离 staging 目录重新校验 compiler/doc tar SHA；确认 `/opt/tech/tsmc28/Memory/MC2_2012.02.00.d` 的平台兼容性；
2. 确认 TSMC 授权 license 已由管理员提供并运行。不得由自动审计脚本启动或改写 license server；
3. 优先生成并封存以下 exact 配置及 SVT/HVT 选择：
   - SP：`32x48m4s`, `64x96m4s`, `128x128m4s`, `256x32m4s`, `256x128m4s`, `2048x128m4s`；
   - DP：`48x64m4f`, `64x72m4f`, `64x36m8f`；
   - `2304x128m4s` 按 databook 是合法直接配置，但在当前 QRT 模型中仍用 `2048+256` 分区，以保持数值可审计；
4. 每个宏必须收集 NLDM `.lib`、转换/提供 `.db`、functional `.v`、`.lef`、`.gds`、`.ds`，建立双层 SHA seal；
5. setup/hold 使用与 stdcell 对齐的 slow/fast PVT，power 则选择同电压 nominal corner，不能跨 0.9/1.0 V 拼表；
6. 为 DP wrapper 增加 same-address collision assertion/forwarding；为 C2/context 和 A1 adapter 冻结同步返回 latency/tag；
7. 通过 exact-SHA VCS 后，才能执行 macro-aware DC/PT/SAIF/PTPX；宏位置、halo、bank decoder 与互连必须进入约束；
8. 只有从 cycle simulator 导出逐宏 read/write/access/idle trace，并计入 leakage、pin、clock、interconnect 后，才可报告 mJ/frame。

Fail-closed 准入门：

- C1：16 个 SP 宏可先进入 macro-area sensitivity；若需要 ping-pong，必须用复制后的面积/能量重算；
- M498：`64x72m4f` DP views + wrapper VCS + integrated STA/DC/PTPX 缺一不可；
- C2：所有方案使用同一 288 KiB 容量；K1/K8/K1x8 按真实 bank-access trace 比较；
- C3：除非 network-level membrane state 规格冻结，否则保持 stdcell；
- A1：生成 `256x32` views 并完成同步 adapter 前，只能使用 QRT model；
- 全系统：禁止把 generated、QRT、CACTI 三种证据等级混成单一 “measured silicon/implementation” 数字。

## 9. 待回答问题

1. M498 parent scratch 在 admitted schedule 中是否真的要求每周期同地址或异地址 1R+1W？能否通过 phase split 降为 SP，而不损失 M473 的 fused 假设？
2. C1 correction phase 是否需要与下一 tile load 重叠？如果需要，16 宏小计必须增加 ping-pong 副本。
3. C2 每个 bank 的 deep/tail 访问比例是多少？K8 是否存在显著 idle bank，可据此 clock/power gate？
4. one resident psum bank 是严格读写互斥还是 concurrent update？这决定 SP 与 DP 选择。
5. full-network ATLIF membrane state 的精确 depth、width、端口和跨 timestep lifetime 是什么？
6. A1 row/descriptor 能否用已生成 `128x128` 宏重新打包而不增加读放大和 adapter latency？
7. 统一论文表最终采用哪个 nominal power corner，如何与 stdcell、SRAM、DRAM 模型保持同一电压/温度身份？

在这些问题闭合前，最稳妥的论文表述是：C1 有真实 foundry-generated SP macro 小计；C2/M498/A1 是明确标注的 foundry-QRT 宏敏感性；C3 小状态保持寄存器；全模块 macro-inclusive PPA 和 energy/frame 仍待正式集成。
