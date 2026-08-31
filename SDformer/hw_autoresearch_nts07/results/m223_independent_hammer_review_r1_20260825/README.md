# M223 独立打铁评审

结论：**91/100，P0=0，GO**。M223 对 M222 的纠偏是正确且 fail-closed 的：exact-binary patch 的公平主基线应为 96-lane add-only K1/D96，而不是只给 K1 定价 96 个 INT8 MAC；96-MAC 只能保留为传统 dense 次级对照。

独立重算固定资源表：当前 `8×128-bit` 合同内，最好合法点 K4/D32 只有 `1.104921×`，M218-like K8/D16 为 `0.946285×`；K8/D32 的 `1.793944×` 需要 16 个 128-bit bank 等价宽度、256 lanes，不能作为同资源胜利。因此 patch 性能 RTL 转为 NO-GO、主线转 M224 FC1 是正确决定。

身份核对也通过：M222 与其独立评审绑定 SHA 完全匹配；M223 自身 4/4 校验通过；`docs/359` SHA 未变。独立检查 M51 manifest 后，M223 所称 10 个 FC1、100 份 bitpack 均存在且 100/100 文件 SHA 匹配，均为 Linear `fc1`，输出维度为输入的 4 倍。不过这些身份还没有写入 M223 的绑定区，必须由 M224 正式封存。

## 三个 P1

1. M221 仍把 M216/M218 写为最强 C3 性能贡献，与 M223 的收窄并存。最终主账本必须明确用 M223 取代该措辞，不能让两套贡献表同时有效。
2. K1/D96 的 add-only 逻辑公平，但 equal-capacity SRAM row placement、六选八路由、accumulator/commit 端口仍未闭合。M223 只能作为负 DSE admission，不能升级为物理性能对比。
3. FC1 转向只写了数量，没有在 M223 内绑定 M51 manifest、10 个 module index 和 100 个文件 SHA；M224 必须补齐。

另有两个 P2：Acc19 需绑定范围证明；stage-3 两个 conventional FC1 需写出精确身份和非二值证据。

论文口径上，M223 的贡献收窄是加分项：M216/M218 现在只能是 context/request-update amortization 的支持机制，patch 不能占第三贡献；FC1 是否能成为第三贡献，要等 M224 在强 K1 基线下过门。每个层模块、baseline、miter、negative screen 都不能各算一个贡献。
