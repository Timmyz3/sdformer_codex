# M1695｜C1 fast-min hold closure source 作者交接

状态：**SOURCE-ONLY PASS；当前不授权 EDA，必须先经 M1696 不同作者审阅。**

M1678 已证明 RTL→M993 与 M993→M1665 两段 Formality 均 PASS，但独立 PrimeTime 在固定 `0.050 ns` hold uncertainty 下得到 setup WNS `+0.002221 ns`、fast-min hold WNS `−0.028168444 ns`、10,610 条 hold 违例与 TNS `−40.24 ns`。同一 SRAM write path 的 macro hold check 在 DC 为 `0.097685 ns`，在 PT 为 `0.126859 ns`，差 `29.174 ps`。

M1695 不改 RTL、不读 M1678 网表，只读冻结 M1665 DDC。它把优化阶段 hold uncertainty 临时设为 `0.081 ns = 0.050 + 0.030 correction + 0.001 guard`，执行唯一一次 `set_fix_hold` 与 `compile -incremental_mapping -only_hold_time`，随即恢复 `0.050 ns`，再生成最终 setup/hold/area/QoR/DRC 报告以及 DDC/SVF/SDC/mapped Verilog。最终论文约束仍是 `3.000/0.200/0.050 ns`，slow→fast standard-cell 与 SRAM min library 都显式绑定。

未来成功门为 setup/hold WNS≥0、TNS=0、违例=0、9 个 exact SRAM macro、DRC0、面积≤`168188.4885824 µm²`（相对 M1665 +10%）且全部输出完整。runner 仅允许一个 dc_shell、24 GiB commit headroom、one attempt/no retry；它与 C3/M1698 共用 `/tmp/date_dual_synopsys_same_uid_eda_queue.lock`，从资源探测前到结果封存发布全程持有独占 flock，并在取锁后与 dc_shell 启动前各做一次 ancestry-aware 同 UID 冲突复核。不允许 false/multicycle/min/max-delay、disabled arc、case analysis、第二 pass、降频或 paper hold uncertainty 放宽。

静态与 mutation 测试在 CPython 3.6.8 和 3.12 上均 16/16 PASS，`bash -n`、contract 内外封与 `git diff --check` 通过。本次未创建 attempt/result，未执行 DC/VCS/Formality/PT/PTPX/GPU，未修改 `docs/359` 或 `ucli.key`。

下一步只交 M1696 异作者审阅。M1696 P0/P1 均为零且 M1697 独立 release 双封之前，禁止启动 runner。
