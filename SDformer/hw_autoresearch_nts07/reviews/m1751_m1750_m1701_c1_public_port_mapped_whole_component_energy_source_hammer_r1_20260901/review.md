# M1751 independent source hammer

裁决：`PASS_M1751_M1750_C1_WHOLE_COMPONENT_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_CAMPAIGN`。P0/P1/P2 均为 0；本轮没有运行或查询 EDA/license，也没有创建 M1752、attempt 或 result。

M1745 的 P0 已被实质修复。M1750 的唯一 primary 是当前 whole mapped C1 design 的 `report_power` 四分量，包含 9 个已链接 SRAM Liberty；checker 强制 `switching + internal + leakage = total`。源码不存在 selected-macro power report、top-minus-macro 或 PTPX 与 datasheet SRAM 相加。外部 SRAM 读写/泄漏模型仅保留为独立 alternative sensitivity，改变其读写计数不会改变 PTPX primary。

corner 口径明确为 mixed-corner component estimate：standard cell 为 TT 0.9 V 25 C，SRAM macro Liberty 为 SSG 0.9 V 125 C；不得称为 single-corner signoff。工作负载仍只是 ep34-density-conditioned 的 64-row directed component activity，residual/psum 为 synthetic，不是 frame、完整 C1 或系统能量。

独立 hammer 在 CPython 3.6 与 3.10 下分别完整重算 51,840,000-row support ledger，并各拒绝 5 个 runtime、4 个 SAIF、4 个 power/accounting 变异。public-port TB、exact DUT-only SAIF window、100% net/leaf annotation gate、M1751/M1752 authority-before-attempt 与 one-shot namespace 顺序均通过静态检查。

下一步只能先创建独立双封且 exact-SHA 的 M1752 release；M1751 本身不授权绕过 release 直接启动。未来 candidate 仍须独立 result hammer 才能引用。
