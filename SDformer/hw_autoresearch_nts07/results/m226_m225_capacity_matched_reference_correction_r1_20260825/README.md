# M226 M225 等容量公平基线纠偏与 raw RTL 选型

M225 的全部精确计数保留，但自动 F2/F4 admission 被撤回。`1.802367×/2.248493×` 同时包含 K1→K8 grouping、spatial parent 和 F 路更新，不能称为纯 held-weight multicast。

## 公平拆账

- K1/F1 → raw K8/F1 grouping/descriptor：1.028613×；
- raw K8/F1 → spatial K8/F1 parent：1.189066×；
- 同 spatial parent、同 K8/同14,592-bit状态下，F2/F4 multicast：1.473619×/1.838372×，均没有达到原1.5×/2.0×机制门；
- 相对等容量 raw K8/F1，parent+multicast 组合仍为1.752230×/2.185947×，但暂时只能作联合敏感性。

第一版 RTL 改选更简单的 raw path：不需要parent选择、负方向生成和parent-output seed。相对等容量 raw K8/F1，raw K8/F2为1.568695×，raw K8/F4为2.112902×；十样本范围分别为1.559855–1.578961×、2.084994–2.147819×。

下一里程碑必须做同一参数化 K8/F1/F2/F4 RTL：三者均保留14,592-bit Acc19 context state、768-bit weight口和同一tag/epoch协议；同时计入C384时至少3,072-bit presence mask、3,072-bit sign mask、768-bit held-weight寄存器、256-bit scanner、source walker和replay控制。最终只依据3ns matched DC、throughput/area与SAIF/PTPX选择，不能依据2.1129×周期单独选F4。

本纠偏没有产生新cycle、RTL或PPA，也不构成完整FC1/FFN/系统加速。论文正文和`docs/359`未修改。
