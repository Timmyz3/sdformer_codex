# Fusion ROUND3 短杀门卡

日期：2026-09-13  
前情：ROUND2 **无 KEEP_PROBE**；cand1 公平背压后 23=23 → KILL_LAYOUT；普通旁路、BN/V 融合仅 **BASE_ONLY**。  
本轮四刀：**弹性直供 / 小整数基底+例外 / gate-summary×NRV / 驻留 R24**。  
纪律：不复活已杀布局；不碰 Codex；不是 Stage B / ASIC PPA / 标题 X。

通用否决：
- 放松 same-port / 状态 / 背压 / 顺序公平  
- 只赢 always-ready，公平背压即打平  
- BASE_ONLY 通用融合冒充候选 X  
- cells/局部拍数好看但 job 周期不降

---

## R3-A　弹性直供（elastic direct feed）

- **B**：source→preview 之间用小弹性缓冲/valid-ready skid 直供，允许短背压吸收；避免强制 store→reload。  
- **A**：ROUND1 普通旁路（BASE_ONLY）；1-port same-state；保序 ready/completion。  
- **X（待证，窄）**：不是旁路本身；待证是「**最小弹性容量**在公平背压下仍能保留有效读写/服务下降」。  
- **对照**：store-reload；零缓冲旁路；同容量 FIFO 不直供；相同背压序列。  
- **杀**：  
  1. 公平背压下 job 周期 ≥ store-reload / 同 FIFO  
  2. 只缩内部 held/读写旗标，服务不降  
  3. 需无限/明显更大 FIFO 才赢  
  4. 乱序/丢 token/ready 环  
  → **KILL_LAYOUT**；普通旁路仍仅 BASE_ONLY。

## R3-B　小整数基底 + 例外（small-integer basis + exceptions）

- **B**：将 source/lifting 系数表示为共享小整数基底（移位/加）+ 稀少例外系数；例外走旁路或微码。  
- **A**：常量编译/CSE、PoT/DeepShift、普通系数量化均为底座。  
- **X（待证）**：只有当「基底共享 + 例外稀疏」改变**有限端口/状态/服务**才有差分；系数量化本身 ≠ X。  
- **对照**：普通 Q12 常量；纯 PoT；独立小整数分解（无共享）；同误差预算 LUT。  
- **杀**：  
  1. 例外率高，微码/选择税吃掉基底复用  
  2. 同资源周期不胜常量 CSE / PoT  
  3. 只减系数位数（当前系数存储非瓶颈），不减整段服务  
  4. 未评数值函数/AEE 即报创新  
  → **KILL_LAYOUT**，保留量化底座。

## R3-C　gate-summary × NRV

- **B**：在 source/PSN 侧生成组级 gate-summary，向 NRV/供数侧发「组内仍需哪些行/列」摘要，尝试减少无用导航/读行。  
- **A**：Gustav NRV/源∩W；组完成/门界；摘要编码为公共底座。  
- **X（待证）**：门摘要与 NRV **双消费者并集义务**共设计，必须真删物理行/请求；summary 本身 ≠ X。  
- **对照**：无摘要 NRV；逐门 bitmap；整字/整组退休；oracle 摘要上界。  
- **杀**：  
  1. 双消费者并集仍满读（复现 ROUND1 cand5）  
  2. 摘要生成/传输/解码税 ≥ 减少导航  
  3. 只减少逻辑行代理，物理请求/job 周期不降  
  4. oracle 上界也无 ≥5% 余量  
  → **KILL_LAYOUT**；不杀 Gustav/NRV 家族。

## R3-D　驻留 R24（resident R24）

- **B**：在有限 RF/寄存预算内固定驻留 24 个高复用 source/intermediate 条目，减少重读与翻译；统一 ordinary/lifting 权限。  
- **A**：Gustav 驻留/F_live、LoopTree 占用语言、普通 cache/RF 替换策略。  
- **X（待证）**：R24 不是 X；只用于检验**当前工作负载是否真是驻留瓶颈**，若成立才反哺 F5/F7。  
- **对照**：R0；同容量 LRU/静态 top24；ordinary 与 lifting 均得 R24；oracle top24 上界。  
- **杀**：  
  1. job 周期/物理读不降，或只改善命中率  
  2. ordinary 同权限获益相同/更大 → **BASE_ONLY**，非 X  
  3. 状态预算/端口增加抵消收益  
  4. oracle 上界薄（<5%）  
  → **KILL_LAYOUT** 或 **BASE_ONLY**；不据此杀 F5/F7。

---

## ROUND3 筛序

1. **弹性直供**：先用 ROUND2 同一公平背压，防止又只赢 always-ready。  
2. **gate-summary×NRV**：先跑 oracle 并集上界；满读直接杀，不写 RTL 深化。  
3. **驻留 R24**：ordinary/lifting 双臂同权限；若通用受益则 BASE_ONLY。  
4. **小整数基底+例外**：先固定例外率/误差代理；例外税高直接杀。

记分牌 verdict 限定：`KEEP_PROBE` / `BASE_ONLY` / `KILL_LAYOUT`。  
**无 `ADVANCE_X`**：ROUND3 玩具 RTL 不足以证明标题 X。
