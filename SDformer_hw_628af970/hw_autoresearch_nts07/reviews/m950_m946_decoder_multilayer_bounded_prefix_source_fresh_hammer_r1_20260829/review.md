# M950｜M946 D1/D2/D3 bounded-prefix source fresh hammer

## 裁决

**100/100，P0/P1/P2 = 0/0/0。** M946 新冻结身份通过独立 source hammer，可进入“下一份独立 bounded-prefix execution contract”的设计；本审阅本身不授权任何执行，更不授权 full-row、production、结果发布或 Table-A。

审阅中先发现默认 `/usr/bin/python3` 为 3.6.8、无法加载 M896 的解释器缺口。作者修复后，source 在加载 M896 前固定 M925 同源解释器 `/opt/anaconda3/envs/pytorch310/bin/python3.10`，版本 3.10.18，SHA-256 `9f78cd...15`。独立复测证明默认 Python 现会在 M896 import 前 fail closed；该 P1 已闭合，故最终冻结身份无遗留 finding。

## 冻结语义与 selector

M946 没有复制或改写 mapper、transaction generator、resource、address equation、transaction order、RUN-GTLS recurrence 或六类 cycle priority。它只调用冻结 `M785.iter_record_transactions`，再用冻结 `M890.truncate_transactions` 截成 `1K/10K/100K` expanded-request prefix。

独立检查了冻结 M686 的全部 30 个 D1/D2/D3 S10 row：layer、module、sample 与 manifest route 一一匹配；D0、负 sample、sample 10 和非枚举 prefix 均被拒。命令行同样拒绝 full-row、production 和 output。

数值边界保持正确：D1 的 manifest 是 `EXACT_SCALED_BINARY_BITPACK`，但输出只能标为 `COMMON_CHARGED_FULL_SHAPE_DIAGNOSTIC_NONHEADLINE`；D2/D3 才是 `EXACT_BINARY_SUPPORT`。因此合法表述仍是“D0/D2/D3 exact-binary support subset + separately charged D1 diagnostic”，不能称 four-layer exact acceleration 或 decoder complete。

## 独立运行结果

在冻结 Python 下：compile PASS、static checker PASS、6/6 unittest PASS；synthetic 10K 通过 M768/M861/M890/M896 exact miter。real sample0/A1/t0 的 D1/D2/D3 各 1K 也全部 exact：三行均为 1,000 expanded request、1 compressed transaction、2,024 diagnostic cycles，cycle-class 分解均为 active-service 1,000 + dependency-completion 1,023 + compute 1。三层 address/order hash 分离且与作者收据一致。

1K 均尚未到 commit，故 `e3b0...b855` 是空 commit sequence 的预期 SHA，不是 commit coverage。它们也不是 full-row latency。

作者 D1 100K receipt 的递归 seal 与投影算术经独立重算：`2×timeout=2,522 s < 21,600 s`，`2×projected memory=1,613,402,396 B` 同时小于当时 MemAvailable 与 commit headroom。即使这两门通过，`full_row_authorized` 仍固定为 false。D2/D3 的目标 request 仍只是 sizing proxy，本审阅没有运行它们的 10K/100K。

## 下一步边界

下一份 contract 只能授权明确枚举的 bounded prefix，并应让 D2/D3 从独立、逐层的受限 prefix 开始。100K gate 通过后仍需新的独立 release 才能 full-row；production、result publication、paper/Table-A admission 都不随本审阅自动获得。

