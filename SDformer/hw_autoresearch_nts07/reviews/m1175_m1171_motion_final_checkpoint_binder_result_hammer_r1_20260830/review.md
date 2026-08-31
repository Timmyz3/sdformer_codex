# M1175｜M1171 Motion 最终 checkpoint binder 独立结果打铁

结论：**PASS**。远端 M1171 结果的精确成员集、内外封存、终止 token、一次性 attempt、三代 binder 与 launcher 源码身份、配置、排名、五个 checkpoint 和五个 profile 均独立重核一致。14 类封存与语义变异全部 fail-closed。

## 最终身份

- 选择规则：五个标准 valid825 profile 中精确 AEE 最小，epoch 作平局裁决。
- 选择：epoch 29。
- checkpoint SHA256：`2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a`。
- config SHA256：`c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955`。
- profile SHA256：`e4fcb2f5d9e8153ce670cf4208a4095c283bb167accb3f06d28c00dff44b8f72`。
- valid825：AEE `1.209876834190253`，AAE `5.406798340046045`，AAE_Benchmark `5.148612399245754`。
- 活动：72,036,342,053 spikes，发放率 `0.05604385744236269`。
- `63130.11851729999 uJ` 仅是 spike-activity proxy，**不是硬件能量**。

## 解锁范围

E0 最终 checkpoint 与部署配置身份在本次打铁后准入；E1 的标准 valid825 绑定成立。E1 的 dyadic/quantized/hardware-order 数值和 E2–E6、E8 的工作现在可以按各自 gate 启动，但结果仍未准入。E7 必须等待 E2–E6 后才能做真实 trace SAIF/PTPX 与统一 Table A。

这不是硬件 replay 完成，也不产生任何硬件或系统加速、功耗或能量结论。

## 不可继承

旧 ep35/ep24 的活动、稀疏、C1 source-row/周期、decoder payload/地址周期、attention Q/K/RQTB、SAIF/PTPX、Table A、范围与压缩结果均不得平移到 ep29，必须按 E1–E8 重新绑定。拓扑本身不依赖 workload 的 logic-only 面积证据可保留其物理身份，但 workload 周期、活动与能量仍须 ep29 重放。

## 攻击

拒绝了额外成员、错误 token、未重封 payload 篡改、重封后的 selected epoch/AEE/checkpoint SHA/授权/E0–E8 策略篡改、CSV 漂移、profile typed-zero→bool、排名重排、源码 SHA、attempt retry 和 docs/359 SHA 漂移，共 14 类。

本审计只读；未启动 GPU、训练、验证、capture、replay、VCS、DC、PT 或 PTPX，未复制 checkpoint。`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
