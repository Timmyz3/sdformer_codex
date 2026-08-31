# M173 H67 FC2 scan-width exact-payload DSE r1

M173 在 M172 已验证的 group-held recurrence 上，对冻结 120 个 FC2 payload
重新计算 64/96/128/192/384-bit scan。每个点都包含 beat 内 bank-unique K4
fragmentation、one-beat prefetch、zero token、`Cout/96` replay 和 token-done
延迟。

| scan width | K1 wall | K4 wall | K1/K4 |
|---:|---:|---:|---:|
| 64 | 446,528,624 | 179,057,955 | 2.493766x |
| 96 | 437,234,151 | 157,504,597 | 2.776009x |
| **128** | **432,951,702** | **146,423,753** | **2.956841x** |
| 192 | 428,961,896 | 135,135,765 | 3.174303x |
| 384 | 425,370,073 | 123,793,034 | 3.436139x |

选择 128-bit，不选择更宽点。它是最小 power-of-two 且在四个 stage 全部超过
2x：2.151304x / 2.898767x / 3.160068x / 3.295855x。192/384 只保留为
带宽上界。

下一 RTL 必须同时修复 M171 r1 的物理失败：不能复制两套嵌套 priority loop，
而应复用一个 selector，先并行得到八个 bank-present/first-row，再用四级 8-bit
lowest-onehot 选 bank。预锁定目标是 3 ns 下显著少于 M171 的 103 logic levels。

2.956841x 仍只是 exact-payload analytic frontend boundary；尚无 128-bit bitmap
memory 供给、RTL/DC、weight SRAM、M169 composition 或 complete FC2/system speedup。
`docs/359` 未修改。
