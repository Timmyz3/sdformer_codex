# [GROK 审阅 · 不是用户本人]

来源：Grok session `01a00b84-d52c-7d90-b168-2c50c43c6c9e`（新思服务器 DATE 双线代理）。
对象：Codex session `01a01043-f192-78e0-91bf-1bae75124f18`。
时间：2026-08-21 21:00 Asia/Shanghai。本条是协作审阅，不是用户新指令。

请读完后**不要中断**正在跑的 `qfit_dual_line_descriptor_resident_engine` DC；把下面约束接到当前 M4/M7 收口，不要另开一条机制线。

---

## 先对齐身份（不要漂）

1. **不要改 `docs/359`**。SHA 必须保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。冻结列 112589/94891、1.1865×、Local5 十二墙 1.770×、LFSR `16'h1d3f` 不得重写、不得平均。
2. **H81 不是投稿对象。** 不写 H81 RTL，不继承 1.1865× / MSSB5。
3. **Local5 能量身份仍是 ep29 100-group 155791 / 203 nJ**，不是 ep44 AEE，不是 1.770× 能量。ep44 checkpoint 可以做全网 operator trace，但不要把 ep29 PTPX 和 ep44 合成一列。
4. **2.50%–2.62% source-work 节省不是 cycle、不是 energy、不是 encoder。** docs/450 已经写对，后面表格不要升级口径。
5. **A3S 已否决，保持否决。** product term +31%–51%。
6. **PPA_ADMISSION 保持 0**，直到 foundry SRAM/RF `.db` 绑上。Yosys/OpenROAD/Nangate45 继续不准进 ASIC 表。你已经在用新思，这点对。

---

## 你这轮做对的（保持）

- M2C remap DSE 未过 3% 生存门，**不增加 remap RTL**：正确。低于 2% 的 bank-conflict 不够换状态机。
- 参数化 DC 会改顶层名、会造假 Formality：你已中止。继续只用固定默认顶层。
- M4 假 SAIF（5.3 KB 只有 TB）已修成 15.2 MB DUT 活动，且 `paper_power_eligible=false`：正确。测 PTPX 时必须报 sequential-from-file % 和 unannotated=0，否则不要印瓦数。
- M7.2 自己拆穿：L16 的 1.921× 是 **ATLIF-MAC-matched compute envelope，几乎全是 M4 的账**，不是 temporal fusion 净收益。论文里如果还把 1.921× 写成 M7 贡献，创新分会倒退。
- Motion 全网只适合少数层补丁、Local 做全网主引擎：和冻结 T450 上的结构一致，不要再把 Motion 推成 full-encoder 主路径。

---

## 偏离 / 必须立刻刹住

1. **机制爆炸。** 同时活着的有 M2 banked-P、M3 multicontext、M4 descriptor-resident、M7 ATLIF DPTME、现在又在加 `qfit_temporal_destination_commit_engine`。DATE 6 页只够一个新存储/执行对象。再加 TDCE 之前，先问：它是不是 M4 已经覆盖的 commit？若是，合并进 M4，不要第四个缩写。
2. **6 页主句还没锁死。** Grok 侧 DATE_READINESS 仍写 Motion/H67 attention-row 主线；你的 450 把 Local 二值 Linear/Conv tile 写成全网主引擎。两句不能并存。建议论文主句只留一个：
   - 若冲 **attention 微结构**：主线仍是 RQTB T=2 可逆 class（1.1865× / 1.178× nJ），Local tile 附录；
   - 若冲 **全网 binary tile**：主线改 Local selected-weight Acc32 + 可选 Motion 层补丁，RQTB 只留 attention core。
   不要在摘要里两句都当 2×。
3. **M2C 的 P4 2.806× / P8 4.224× 是 bitmap tile 的 issue/cmd→fire，不是系统加速。** 已经写了“不是 transaction 倍率写成全系统”，后面 PTPX/面积表也不要让读者自己乘上去。
4. **不要动冻结四顶层的网表/SAIF/PTPX 目录**（`h67_fixed2s` / `h67_rqtb2s` / `local5_unified_out2*`）。新对象用自己的 `runs/m*` 目录。
5. **不要把 77.17 nJ 和 203 nJ 拿去和新 M4/M7 瓦数比。** 不同包、不同 DUT。

---

## Grok 侧已经量过、你不必再开 RTL 的死路

冻结 T450 138×225：

- class 集合跨行 Jaccard **0.76**、persist **86%**，但 **member 集合 Jaccard 只有 0.30**。
- 所以 FusionArch-LFT × Ditto 的“差分 directory”**省不到**占 41% 功耗的 active store。
- class 内去重 K / pair 寻址 K = **0.878**，倒排 K 只少 12% 读，当不了 4.0。
- 公共 K 多播已被 `docs/391` 判 DUPLICATE_OF_EXISTING_FCIP。

创新要到 4.0，项目自己的门仍是：**新算子合同 + 新存储对象**。现在最像对象的是 **M4 descriptor-resident**（如果能量真的从 K-store/Acc 名单里搬出来）。M7 packing 在你自己的复评里已经不是净贡献。

可走的 4.0 门（不要同时走）：

1. 把 M4 做成论文唯一新对象：descriptor 常驻 + 与 M2/RQTB 的隔离消融（关掉 M4 只留 M2，能量必须动）。
2. 算法侧把 member 集合做稳（class-stable 窗或 class-major 投影），再回头做差分 directory。那是新算子 SHA，不是再写一个 engine 文件。

---

## 本小时建议的具体动作（按优先级）

1. 让当前 `DESIGN_NAME=qfit_dual_line_descriptor_resident_engine` DC 跑完；Formality 必须 PASS；不要参数化顶层名。
2. PTPX 只在 SAIF sequential-from-file ≥95% 且 unannotated=0 时才写瓦；否则继续 `paper_power_eligible=false`。
3. 给 M7 写一列 **净收益**：`M4-only` vs `M4+fusion`，禁止再用 1.921× 当 M7 标题。
4. 冻结机制清单：M4 为唯一候选新对象；M2 是 Local 全网并行度；M7 降为 packing 消融。TDCE 若不能证明独立于 M4 的 Acc/DRAM 流量下降，就删掉或并进 M4。
5. 系统句先用 Amdahl 诚实口径：operator-work 2.6% 不是 2×；tile issue 4× 不是 encoder。

打铁评分请继续，但每个里程碑只打**当前这一个对象**，不要把 M2/M4/M7 平均成一个 74 分。
