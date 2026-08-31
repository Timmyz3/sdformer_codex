# M221 统一共存与论文贡献收口独立打铁评审

结论：**86/100，P0=0，P1=8；M221 作为“逻辑图共存、资源记账和论文收口计划”可以封存，作为“已经实现的统一硬件、共享 SRAM 已证明或三项论文贡献已完成”则 NO-GO。**

M221 最重要的判断是正确的：不能把每个 M 号都写成一项贡献。M165/M166 是 M167 的演化证据，M219 是 K1 基线，M220 是验证 miter，M36 是 census；这些都不应再计为 production block 或独立贡献。M167、M216、M218 的 logic-only 面积小计也重算为 `138927.726185 um2`，M218/M219 面积、周期和条件性 throughput/logic-area 算术全部精确一致。M221 自身 `SHA256SUMS`、十项外部证据 SHA 和受保护 `docs/359` SHA 均通过。

功能域层面，RQTB attention core、FFN BN1/sn2 temporal-normalization、FC2 和 patch embed 位于不同算子/图阶段，**没有发现同一数学算子被两个贡献重复声称**。因此“可以通过薄 phase/epoch/tag 合同组合”是可信的架构方向。不过当前还没有任何岛真正实现 M221 的统一 typed interface；patch 岛甚至只有机会 census。因此最安全的表述是“这些机制在图上无冲突，存在一个可实现的顺序组合计划”，不能简写成“这些硬件已经统一共存”。

## 独立复核通过项

- M221 内部三文件 SHA 全部通过；所有 `evidence_bindings` 与磁盘 SHA 一致。
- 冻结 620302905-cycle activity-weighted envelope 中，patch/FC1/global-ATLIF/FC2 的账本分别为 199420620、118370114、128020500、41413997 cycles，对应 share 与 perfect-removal ceiling 算术正确。
- M167+M216-K8+M218-K8 的已知 logic-only 小计是 `29489.291809 + 20587.392080 + 88851.042296 = 138927.726185 um2`；M165+M166、M219 和 M220 均未重复叠加。
- M218/M219 的 `15.604369%` 面积增量、`4.952121573x` service-cycle 比和 `4.283680292x` 条件代数 throughput/logic-area 均重算一致，且 M221 没把它升级成完整 FC2 或系统指标。
- 所有统一面积、统一周期、能耗、共享端口、paper PPA、system speedup 和 headline flag 均保持 false；没有发现越界 headline。

## P1 问题

1. **“can coexist=true”只能理解为逻辑计划。** 现有 M167/M216/M218/RQTB 接口并未服从 M221 typed transaction；patch RTL 尚不存在。合同自身的 `physically_integrated_rtl=false` 和 `shared_sram_ports_proved=false` 已经诚实限定，因此 GO 只适用于图级可组合性。
2. **平面 phase list 不是完整 H67 执行图。** 实际网络需要 stage/block 循环，并在每个 block 中交替 attention 与 FFN；还存在 downsample、decoder、prediction、sn1 和未优化算子的 baseline service。当前 list 没冻结循环、旁路或模块间依赖，不能直接作为顺序 cycle simulator 的最终 schedule。
3. **共享 SRAM 合同尚不可执行。** typed fields 没有显式 memory-space/port/address/burst/precision/payload-kind/credit；也没有 buffer depth、ready/backpressure 和 phase-drain liveness recurrence。`one phase owner` 是必要条件，不足以证明端口无冲突或不会死锁。
4. **production 资源替换关系不完整。** M159 明确原模型让 fc1/fc2 time-share 一个 Linear engine、sn1/sn2 time-share ATLIF engine。M221 计入专用 M167 和专用 M216+M218，却未说明原共享 Linear/ATLIF engine 哪些功能被替换、哪些仍需保留，也未把 FC1 producer、sn1 ATLIF、rank-factor preload 等列入 subtotal exclusions。小计仍可作“可见子块小计”，不能作完整 resource-dedup ledger。
5. **M167 的热点归因必须收窄。** `global_atlif=128.0205M` 包含全网 ATLIF，不等于 M167 可优化范围；M167 当前只面向 FC1 后的 BN1/sn2 temporal path，冻结 FFN-local sn2 ATLIF 是 36.48M cycles。FC1 的 118.370114M 也由独立 producer 承担。论文不能用 20.64% global-ATLIF 或 19.08% FC1 share 直接为 M167 的已达覆盖背书。
6. **C1 尚不满足 M221 自己的 contribution eligibility。** RQTB 有 exact RTL 周期和 OpenROAD 代理，但现有冻结说明明确没有 DC/PTPX/目标 SRAM；因此可保留为主要机制候选，不能称为符合当前“VCS critical RTL + DC timing/area”门槛的完成贡献。
7. **C2/C3 都还是候选而非完成贡献。** C2 的独立 M167 评审仍有 5 个 P0：numeric composition、barrier controller/storage、coefficient/rsqrt、checkpoint/accuracy 和 matched operator cycles。C3 的 M216→M218 连接未实现，M220 只证明 M218↔M219 的固定 L4 directed matrix，而且 O8/context recurrence 在该矩阵中不 binding。
8. **三项故事存在“机制拼盘”风险。** quotient attention、dynamic-BN temporal kernel 和 FC2 bank coissue 在图上不重叠，但目前也没有一个已验证的共享数据流或端到端指标把它们统一起来。论文可以先以 C1+C2+C3 为候选目录；最终应只保留通过 connected cycles、宏面积/能耗和 checkpoint 正确性的 2–3 项，而不是预先承诺三项都写成主贡献。

## GO / NO-GO

- **GO：** 将 M221 用作统一接口需求、资源去重规则、里程碑筛选和论文贡献分组合同。
- **GO（严格限定）：** 表述为“RQTB、BN1/sn2 temporal path、FC2 和 patch 位于不同图域，可由顺序 phase contract 组合；物理组合仍待证明”。
- **NO-GO：** “已经实现统一 accelerator/top”“共享 SRAM/phase/barrier 已证明”“138927.726185 um2 是加速器面积”“4.283680292x 是实现吞吐面积比”。
- **NO-GO：** 当前就把 C1/C2/C3 全部写成已完成论文贡献，或把 hotspot share/局部倍率相乘成系统结果。

最小收口顺序应是：先做一个 M221 phase-wrapper/adapter contract canary，覆盖真实 block iteration、drain、stale-tag 和有限 buffer；同时补全“旧共享 Linear/ATLIF engine → 新岛”的替换表。之后才分别用 connected M216→M218、M167 numeric/barrier、RQTB Synopsys/macro 结果决定最终保留哪两到三项贡献。M222 patch screen 可以继续，但在 RTL/cycle gate 前仍只是候选扩展。

`docs/359_DATE终局冻结_20260813.md` SHA256 复核仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
