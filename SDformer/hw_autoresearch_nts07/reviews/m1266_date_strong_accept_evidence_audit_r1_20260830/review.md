# M1266｜DATE Strong-Accept 独立证据审计（2026-08-30）

## 结论先行

当前证据更接近 **3.4/5，Borderline / Weak-Accept 边缘**，还不是 Strong Accept。项目已经不再是“只有 simulator 数字”：C1 有 28-nm 宏感知 3 ns setup/area，C2 有等带宽三轴 DC，C3 有 Fixed-T10 DC；但最重要的投稿闭环仍缺失——**最终 checkpoint 未绑定、decoder 未完成、Table-A 全系统生产行为 0 行、功耗/能量与 hold/PT 未闭合**。

Strong Accept 的短板不是再缺一个新 idea，而是现有三个 component row 还没有被一条最终 checkpoint、decoder-complete、memory-inclusive 的同资源系统行串起来。

本审计只读封存证据，不运行 EDA、GPU 或 remote，不修改任何生产结果，也不修改 `docs/359_DATE终局冻结_20260813.md`。审计时该文件 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 当前可引用证据及边界

| 线 | 已闭合的可引用数字 | 合法论文口径 | 仍禁止的升级 |
|---|---:|---|---|
| C1 exact 1RW product-capture | 10 sample / 812,160 task raw-CPU same-ledger：434,242,823 vs 763,908,050 cycle，**1.7591725402×**；容量账 214,912 B < 240 KiB | “冻结四层 bottleneck Conv 上的 raw-CPU same-ledger component opportunity” | 不得写 RTL/mapped/system speedup；不得与 C2/C3 或 Prosperity 倍率相乘 |
| C1 macro-aware physical slice | 28 nm、3.0 ns、9 个 128×128 1RW 宏、147,246.39209 µm²、setup WNS **+0.001795 ns** | “macro-aware component setup/area candidate” | hold 有 9,992 条 diagnostic violation、worst −0.09 ns；full 214,912 B storage、power、energy、RTL cycle 均未闭合 |
| C1 R12 verification | R11 编译/链接通过但仿真因 TB object-boundary violation fail-closed；M1256 判定不是 M935/M1162 RTL bug。R12 改为 child-seam boundary-only，TB/RTL/SVA 冻结 | 只能说明验证重构方向；R11 失败取证可作为纪律证据 | M1264 对最终 checker 找到 4 类 P1 reachability 漏洞；**无 release、无 R12 VCS PASS、无 integrated-random claim** |
| C2 typed signed K8 | K8 1,913 vs 等带宽 K1×8 1,945 cycle，**1.01672765×**；**4.541078× throughput/mm²**；logic area **−77.6104%** | 五个冻结 directed workload、logic-only pre-macro、等带宽 K8-vs-K1×8 | 不得把 K8-vs-single-K1 的旧 4.76×写成稀疏 headline；不得写 full-network/energy/PPA |
| C3 Fixed-T10 | 62,433.503388 µm²、3 ns setup WNS **+0.0003 ns** | Fixed-T10 exact component 的 logic-only setup/area | hold worst −0.02 ns、9,741 violations；无 throughput/speedup/power/system；rank-3 不准复活 |
| Decoder | 审计快照已写出 **67/120** 个 address-timed diagnostic call；运行仍在继续 | 仅 diagnostic progress；每行仍标 `speedup_admitted=false`、`final_checkpoint_rebind_required=true` | 不得把 partial decoder、D0/D2/D3 官方 Prosperity 机会或 D1 diagnostic 当 ours/system |
| Checkpoint | ep30、ep32 checkpoint 已存在；四候选 binder/release source 已独立 hammer，严格要求 legacy-ep29 + resume-ep30/32/34 四个 strict-valid825 | 可写“最终选择和重绑流程已准备” | ep34 与四份 strict-valid825 尚未全部封存；M1259 明确 `checkpoint_selected_now=false`、`hardware_rebind_authorized=false` |
| Full system | M1118 component annex 有 C1/C2/C3 三行 | 三个分栏 component rows，限定语必须逐行保留 | **Table-A full-system production rows = 0**；无 system speedup、energy/frame、FPS、paper-PPA-ready |

## 第一性原理评价

### Novelty — 3.5/5

C1 的可辩护对象差是“240 KiB、1RW parent/PWP capture、signed ATLIF event-flow 下的 exact product reuse”，不是重新命名 Prosperity。C2 的价值是 typed signed K8 共享 Acc24/端口的面积效率；C3 是部署完整性与 Fixed-T10 特化。三者能组成 coherent architecture，但 C2 的公平周期只有 1.017×、C3 尚无性能结果，因此目前只有 C1 像主性能贡献。

### Soundness — 4.4/5

这是当前最强维度。raw-CPU same-ledger 结果有独立重算与 15 类攻击；C1/C2/C3 数字均有 fail-closed claim boundary；M1250/R11 失败没有被包装成 PASS，M1256 正确区分 TB 对象边界与 RTL bug，M1264 再次阻止 checker lexical pass 冒充 executable reachability。缺点是这种严格性尚未转化成完整系统证据。

### Significance — 3.3/5

C1 的 1.759× component opportunity 和 C2 的 4.541× throughput/mm² 都像 DATE 数字，但一个仍是 raw CPU cycle，另一个周期只提高 1.017×。没有 decoder-complete 全网分母时，无法证明这些结果能变成 ≥1.10× 的系统收益；因此 significance 还不能升到 Strong Accept。

### Implementation — 3.6/5

C1 macro-aware setup、C2 等带宽三轴、C3 Fixed-T10 setup 已是真正 Synopsys 证据，明显超过仅 RTL/simulator 阶段。但 C1 R12 尚无 VCS PASS，三个点均未有完整 hold/PT/PTPX，C1 full storage 未集成，C2/C3 为 zero-macro logic-only。实现完成度足够支撑 Weak Accept 讨论，不足以支撑强物理结论。

### Evaluation — 2.8/5

主要失分来自 Table-A 0 行、decoder 仅 67/120 diagnostic、最终 checkpoint 未选、无最终多序列 trace-weighted cycle/traffic/energy、无同资源系统 baseline。官方 Prosperity decoder 结果只能放 external opportunity table，不能补这个洞。

### Presentation — 2.5/5

证据边界组织很清楚，但当前 C1/C2/C3/decoder/final-checkpoint 的最新叙事尚未收成可投稿的六页正文和主表。大量 M 编号与合同对内部审计有用，论文中必须压缩为三条贡献、三张主表/图和一张证据等级脚注。

### 综合

六维均分约 **3.35/5**，裁决为 **Borderline，Weak-Accept 边缘；Strong Accept = false**。若现在投稿，系统性能/能量行缺失会让严格的硬件审稿人给 Weak Reject；如果下面 P0 闭合，不新增机制，也可升到约 3.7–3.9 的 Weak/Accept 区间。要到 4.0+ Strong Accept，需要最终同资源全系统结果本身有说服力，而不是增加更多 component ratio。

## 从 Weak Accept 到 Strong Accept 的真正 P0

1. **最终 checkpoint 四选一并重绑**：完成 ep34 与 ep29/30/32/34 四份 strict-valid825；只执行一次 M1257/M1259 允许的 selector；封 checkpoint/config SHA、AEE/AAE、missing=0/unexpected=0、最终整数参数与活动身份。此前的 ep35/ep29 checkpoint-dependent 周期和能量不得转移。
2. **C1 功能与物理链闭合**：修正 M1264 指出的四个 checker reachability P1 后，只做一次新 hammer 和一次 R12 boundary-only VCS。论文必须同时保留真实 M935 integrated directed evidence；boundary-only random 不得冒充 integrated random。随后将 raw-CPU 1.759×与可执行 RTL/mapped service schedule 建立对照，完成 Formality/PT hold，并明确 full 214,912 B storage 的宏/端口代价。
3. **Decoder-complete replay**：先让当前 120-call diagnostic 自然完成并封存；最终 checkpoint 选出后重抓并重放 D0–D3，D1 做最终数值桥/bit-exact miter。输出四层同资源 cycle、traffic、capacity，而非 partial/official-artifact 倍率。
4. **Table-A 至少一条 production system row**：同一最终 checkpoint、同一资源、同一完整网络分母下，给出 baseline 与 proposed 的 cycle/FPS、SRAM+DRAM traffic、area、energy/inference；必须 decoder-complete、memory-inclusive、三序列分层。目标至少 **≥1.10× system speedup**，最好 ≥1.15×，同时能量有明确优势。
5. **系统功耗与物理边界**：至少对 headline 路径完成 SAIF/PTPX + SRAM/DRAM energy model；hold/PT 未闭合的 component 只能留 annex，不能写 paper-PPA-ready。

## P1（不阻塞第一条系统行，但决定审稿上限）

1. C2 补 fixed-latency SRAM、PT hold、Formality 和 component SAIF/PTPX；保留 1.017×周期与 4.541×吞吐/mm²同句呈现。
2. C3 若不能得到可信 throughput/energy，就降为 C1/C2 的 Fixed-T10 service island，不单列第四贡献；补 coefficient/state memory 与 PT hold。
3. 多序列至少三条 DSEC sequence，分别报 event density、decoder density、cycle、traffic、energy，避免只报 weighted mean。
4. 对标表分三层：任务/网络质量；iso-workload accelerator mapping；ours physical/component/system。Phi、Prosperity、FireFly-T 和官方 artifact 均加对象与分母脚注。
5. 将 negative results 压缩成一张消融：空 tile、temporal XOR、payload residency、lazy-PWP 等，证明为何最终选择 C1，而不是继续开新 matcher。
6. 立即建立最新 LaTeX 骨架和 Table A/B/C；不等待所有数字才写。否则 Presentation 会单独拖到 reject 区。

## 最小收口序列

1. **停止新增硬件 idea**；保护 docs/359，完成 M1264 的 bounded checker repair → fresh hammer → 唯一一次 R12 VCS。
2. **并行等待训练**：ep34 → 四份 strict-valid825 → one-shot selector；等待期间让 decoder diagnostic 120/120 完成，并准备最终 rebind 命令，不启动第二条昂贵探索。
3. **一次最终 capture**：选择出的 checkpoint 上生成统一多序列 ordered trace、D0–D3 payload、权重/ATLIF/BN/attention identity manifest。
4. **一次最终 replay**：C1/C2/C3/decoder 共用 address-timed SRAM/DRAM 账本，产出 baseline/proposed 的完整 cycle 与 traffic。
5. **物理/能量收口**：只对进主表的配置做 PT/SAIF/PTPX 与 memory-energy；支撑模块不再扩结构。
6. **Table-A/B/C 独立 hammer**：Table A = 完整系统；Table B = ours component；Table C = external normalized opportunity。任何缺列 fail-closed。
7. **写作**：三条贡献固定为 C1 product-capture、C2 typed signed shared fabric、C3 fixed-neuron/system integration；decoder 是完整性，不是第四 novelty。

按这条最小序列，真正决定 Strong Accept 的单一关口是：**最终 checkpoint 绑定的 decoder-complete、memory-inclusive 全系统行是否达到 ≥1.10×（最好 ≥1.15×）且能量同步下降**。若没有，该项目仍可凭 C1 + C2 面积效率争取 Weak Accept，但不应声称 Strong Accept ready。
