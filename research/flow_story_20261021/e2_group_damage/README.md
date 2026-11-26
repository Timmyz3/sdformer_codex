# E2 有界动作：数值核心准备，不是已选路线

2026-10-21。已实现纯 inference 核心和 metadata 构造器；**没有选实验 family，没有创建实际动作合同，没有读取 GT，没有 GPU，没有标签采集器运行。** family 与实际 Q6-W 由 root 依据 E1 决策后固定。源码可用于准备，不构成 E2 已满足条件启动的结论。

## 实际执行

`group_precision_core.py` 使用与旧 R27 Q6 一致的 q8（CPU FP32、每输出 maxabs/127、ties-to-even、clamp[-127,127]）及同一个 scale：

- 真实尾区：bits0–7，即 Q8。
- 其余未选完整9×9：bits2–7，即 Q6。
- 选定完整9×9动作：bits4–7，即 Q4。

生产者按 plane 真正减少发射：bit0/1 只作用 tail，bit2/3 作用非 Q4 行，bit4–7 作用全部行。没有先算 Q6 再减去低位、没有清零伪装跳算。三域共享 q8，保留全部空间、T10、全输出通道，canonical 行位置重建后进入原 BN/PSN。实际执行只接受 B=1，避免把不同身份合批后改变 current-BN 数值合同。

每层记录实际 q8/scale SHA、三域行数与 SHA、tile坐标、逐 plane scope/rows/显式 CUDA padding、逻辑 row-plane 总数。逻辑量应为 `8*N_tail+6*N_Q6+4*N_Q4`，不能当作净周期。dense reference 三域分别对 `q8`、`q8>>2<<2`、`q8>>4<<4` 做整数点积；输出采用相同一次 scale。

纯 inference 不含 STE。`GroupAffine.forward` 必须处于 `torch.no_grad()`，`installed(...,ste=True)` 明确拒绝。若借用旧 `capture_full`，仅使用 `ste=False` 的 plane/dense 两路径。与旧 Q6 的全图比较、工程前向数需由后续 collector 明确单列；不能套用三路径训练核验计数。

## 候选动作构造

`group_contract.py` 从 SHA 固定的 R27 完整 inventory 提取全部28目标的 T/H/W/K/Cout，接受调用者选定的 `family_targets`，不自行选择家族。

`build(geometries,family_targets,source_state_sha256=...)` 为每个可参与 affine 的完整tile按 row-major 划分六段，每段含 `floor(N_tile/6)` 个tile。对应段跨目标组成六个互不重叠的动作，各组核内可取消工作严格相等：

`2*T*81*ceil(K/128)*ceil(Cout/16)` 每tile。

剩余完整tile保持 Q6，真尾始终 Q8。少于六个完整tile的目标（例如 stage3 的每affine仅两个tile）明确列为未参与，不把同一tile复制进多个组来凑六个。若选定家族无法形成六个非空组则拒绝。

返回状态为 `UNFROZEN_CALLER_REQUESTED_METADATA_CANDIDATE`；它仍要求 root 绑定实际 Q6-W、可用共同前缀、数据身份、成本和执行源版本之后再采集。source-state 字段在 CPU 结构测试中是明确的测试占位，不能直接用于真实实验。

`actions()` 返回 22 个唯一动作：背景、六 singleton、十五 disjoint pair。`materialize()` 返回全部28个目标的显式CPU mask和摘要，即使目标未被选，也有其零mask。与固定几何不符的真实输入会在 affine forward 被拒绝。它仅保证等核工作，不保证等供数/缓存/调度总费用。

## 之后 collector 必须保留的合同

- 同一个实际恢复 Q6-W；换 W 或量化图必须重采，禁止复用旧损伤标签。
- root 冻结的16个 E2 身份，每身份完整22动作，正式352前向；任何背景重放/独立整数核验均单列工程次数。
- 保存完整最终二维光流原数组（含全部有效GT对应图域），四head摘要、状态/mask/输入/BN/实际plane记录；不要只存被动区域，不能把真实联合 pair 输出替换为 singleton 向量相加。
- 不在 collector 内读取开发 GT 决定布局、动作或停止；质量由独立 reducer 计算，全部16区/序列/尾部保留。
- 所有动作标签与每身份背景相关，不能将352行当352个独立场景；pair比较在相同核预算内，静态选择与预测拟合需保持既定泄漏说明。

## CPU 验证范围

`test_cpu.py` 检查三域/完整tile拒绝、独立 int64 黄金值、逐plane真实发射量、旧R27 Q6背景逐字节、FC与conv halo整数结果、真实原 MLP/ResBlock 的 current-BN plane/dense一致性、实际inventory下六组等工作量及不重叠拒绝。

测试调用 patch 和 FC1 家族只为覆盖metadata构造，不是为真实实验选家族，没有保存候选布局供执行。四次原模块微型前向不计全网络实验。
