# M221 Motion/H67 分层硬件统一收口合同

结论：**所选硬件岛在图和协议层可以共存，但不能把每个 M 号都写成一项论文贡献。**

统一收口采用一个很薄的 `phase + epoch + generation + tag` 合同；patch/conv、RQTB attention、动态 BN/rank-3 ATLIF 和稀疏 FC2 按网络顺序执行。共享 SRAM 每拍只能有一个 phase owner，切换前必须 drain。这样不需要复杂全系统 scheduler，也不要求当前就综合一个巨大的顶层。

## 论文贡献建议

| 论文角色 | 归并对象 | 当前判断 |
|---|---|---|
| C1：Motion quotient/descriptor attention | RQTB 及原 attention 支撑模块 | 保留，但局部倍率不能与其他岛相乘 |
| C2：barrier-aware phase-shared temporal normalization | M161 代数/流量、M163–M166 演化、M167 selected kernel | 有创新性；仍缺数值、phase wrapper、rank/coefficient SRAM、BN2 |
| C3：context-amortized bank-coissued sparse service | M216 frontend、M218 K8 service、M219 K1 基线、M220 miter | 当前最强的新硬件点；仍缺 M216→M218 连接、宏、能耗和完整 FC2 |

M36 census、M159/M156 拆分、M219 baseline、M220 miter、各代 superseded RTL 都是证据，不是独立贡献。M147–M154 只有在 patch/conv 同资源 cycle 闭合后才能合成一个机制，不能逐模块列贡献。

## 资源去重

- 选择 M167 后，不再把 M165+M166 面积叠加；M167 仍未包含 storage/controller/rsqrt。
- M216 与 M218 是 FC2 的 frontend 和 service，当前各计一次；M219 只作 K1 基线，不能计入 proposed area。
- 目前可见 TSMC28 logic-only 子块的条件小计是 138,927.726185 µm²，但缺 RQTB、patch、所有 SRAM macro、BN coefficient/BN2/residual 和顶层仲裁，**不是统一加速器面积**。

## 性能优先级

冻结 620.303M compute envelope 中，patch/FC1/ATLIF/FC2 占比分别为 32.15%/19.08%/20.64%/6.68%。M218 的 4.952× FC2 service 结果即使作为整个 FC2 的理想替换，包络敏感性也只有约 1.056×，因此下一步不再继续雕刻 FC2 控制细节。M219 独立 DC 评审为 90/100、P0=0；K8 service 相对 cropped K1 只增加 15.604% logic-only 面积，条件性 service throughput/logic-area 为 4.284×。该数仍不是 achieved RTL、宏感知或整加速器指标。

M222 优先审 patch 的 `K-bank multi-source + 3x3 line buffer + PED shared input/commit`。若同资源结果低于 1.5×，立即转 FC1 的 Acc19/expanded-destination K-bank premodel。所有局部模块的周期必须在顺序 cycle simulator 中相加，禁止倍率相乘。

机器可读边界见 `m221_motion_layer_islands_unified_coexistence_r1.json`。本文档没有修改论文正文，也没有修改 `docs/359`。
