# M1747 source-author receipt

结论：**PASS source-only；可进入 M1748 不同作者审阅，但不授权分析。**

M1727 的唯一生产调用在 payload replay 前因旧 sample-order schema literal fail-closed；双封失败回执绑定了 exact traceback、M1727/M1729 身份、result/work 均不存在和 `automatic_retry=false`。M1747 只允许 exact canonical schema `m1544_ep34_m1458_sample_order_r1_v1`，并同时要求 exact `sample_order.json` SHA-256 `d4f1f6e...f773` 与 M1744 review/manifest/outer 三重身份。

TSBG B4/B8 ordinary same-capacity LRU comparator、资源未定价边界、S2 FC1 channel-multiplicity 修复、PATCH blocked、FC2 NO-GO 与所有 admission/claim boundary 都直接复用 exact M1727。Python 3.12/3.6 各 15/15 tests PASS，9 类负向突变全部拒绝；没有 capture verify、analysis、GPU、EDA、network 或 result write。
