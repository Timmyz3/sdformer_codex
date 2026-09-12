# 固定 full-D 的 z＋phase 精确重写普查

状态：本文记录执行前的精确普查；后续固定双臂已在 [PHASE_EXECUTION.md](PHASE_EXECUTION.md) 完成，此处保留原普查的证据与结论范围。

结论：有一个可编码的替代接口，但尚无性能证据。全部 gword/q8 下 z 只需 signed9，phase 为 unsigned11；真实四窗相对 g=0 的 phase 差分有稀疏性。它可以把显式 signed19 D×Ug 转成 U16×z9 与 U16×phase 差分，但完整查表、精确编码、选择、额外 MAC 和存取可能吃掉余量。本轮仅运行整数普查，不修改主线 kernel、不扩新 card、不训练、不做 EDA。

B：既有 full-D 消费者必须承担 Dg+c 编码与连续 PED，D×Ug 不能因为门已就绪而免费。问题是同一数值函数能否改用较窄的连续操作数。A：整数商余分解、仿射量化的尺度/常数外提、条件查表与完整 CSE 均是已有手段；沿用[已核验原始来源](../literature_owned/PRIMARY_SOURCES.md)，本轮不声称模恒等式或条件量化本身新颖。候选 X 仅是当前有限机上的具体执行放置，必须经后续同函数实际端口调度胜出才成立。

父严格为旧 `stage_20260912/algorithm/hardware_exports/{ordinary,lifting_raw}` R24+onepass，使用原 `full_g_q8` 参数且全部 step=2048，U/V 均为原 Q16。四窗均为既存 corner/interior 的 16 个消费 anchor；捕获的真实 updated/gate 只供本次数值普查，没有将其声称为免费硬件入口或重跑真实 prefix。精度对应既有 full-D 函数，不借新 320 步学生的 825 结果，也未重跑 AEE。

## 精确函数与舍入限制

设 `s=2048`，`b=Dg+c=s*a+phi`，`a=floor(b/s)`，`0<=phi<s`。保留原 `q=clip[-128,127](RNE((x-b)/s))`，定义 `z=q+a`，则

```text
xhat = sat24(b+s*q) = sat24(s*z+phi)
```

该恒等式包括 q8 clipping；不能把 z 再截断为 signed8。全部 1024 gword×256 q8×10 时间行的直接普查确认，两父分别满足 z∈[-249,254] 与 [-245,255]，均为 signed9。重建范围分别为 [-509421,520237] 与 [-499732,523281]，普遍不触 sat24；仅对当前参数可因此在原 U RNE 前提取常数。`phase_results.json` 另给完整三角界，分解后的累加仍在 signed48 内。

若编码器直接读取 `(a,phi)` 而不构造 b，令 `x-phi=s*k+r`、`0<=r<s`，精确编码为

```text
up = (r > 1024) or (r == 1024 and ((k-a) & 1))
z  = clip[k+up, lower=a-128, upper=a+127]
```

上式中 `clip[value, lower, upper]` 表示按给定上下界截断。half-tie 时 z 的目标奇偶等于 a 的奇偶，因为原 q 的目标是偶数。直接 `RNE((x-phi)/s)` 会错：ordinary 的 gword=0、t=0、x=-262550 时，b=-1430、a=-1、phi=618，原 q=-128、z=-129；普通 ties-to-even 的 z 却为 -128。原 q clipping 对应 z 的区间 `[a-128,a+127]`，这两个界和 parity 都不能省掉。

两父共验证 5,242,880 个 gword/time/q8 重建组合，以及 1,352,920 个合法 I24 舍入/截断边界输入，全部精确。边界集包括 half-tie 两侧、q8 两端和远端截断，并非枚举全部 2²⁴ 输入；一般正确性由上述商余式证明。错误的普通 z ties-to-even 在同边界集中产生 51,390 处差异。

## 四窗统计

每窗为 15,360 个 T/C/P 标量、1,536 个 T10 gate word；H8 按原连续通道 0..7、8..15 等划分，不重排通道。`delta_phi=phi-(c mod2048)`。

| 父/窗口 | 实际 z 范围（均 signed8） | gword 类别 | gword=0 | delta_phi 非零 | T/H8 整组全零 |
|---|---|---:|---:|---:|---:|
| ordinary/corner | [-70,74] | 11 | 83.01% | 16.99% | 660/1920 = 34.38% |
| ordinary/interior | [-59,78] | 19 | 75.20% | 24.75% | 380/1920 = 19.79% |
| lifting_raw/corner | [-70,72] | 9 | 86.07% | 13.93% | 750/1920 = 39.06% |
| lifting_raw/interior | [-60,74] | 17 | 76.69% | 23.31% | 430/1920 = 22.40% |

四窗的 T10/H8 完全全零比例恰与上表相同。全部 phase 本身都非零，因此必须先提取 g=0 常数，才有这里的差分跳过机会。通用 1024 gword 中，两父各有 1024 个不同 phase 向量；delta_phi 的通用非零率超过 99.8%。所以 9–19 个实见类别只是这些窗口的统计，不能替代完整表或合法 fallback。实际 z 恰好落 signed8 也不能覆盖通用 signed9 要求。

四窗没有 q8 clipping 或重建 sat24。依次检查重建 61,440 值、原 U RNE 前 15,360 值、U 15,360 值和最终 PED 61,440 值均 0diff；V 的原 RNE、bias 和 sat 全部保留。完整直方图、按时间位宽、H8 内类别数和普通 fixed/affine 统计在 `phase_results.json`。

## 一个可直接编码的后续接口，尚未执行

固定 P1 放置：从真实 projection SRAM 读取 T10 gword；gword=0 走共同常数旁路，否则一次有费系数表响应取得十行 `(a,phi)`。编码实际 UPDATED 为通用 signed9 z，按上述 parity 和平移截断完成；消费者在原 U RNE 前执行

```text
U*xhat = 2048*(U*z) + U*delta_phi + (c mod2048)*sum_h(U[:,h])
```

随后接共同 P1 keep-Z 和原 V。这样无需显式 Ug 或 D19×Ug，但必须支付 U16×delta_phi12；11bit 正 phase 若直接使用则几乎没有零旁路。可以在同一个已取权重向量下连续处理 z 与 delta_phi，是否减少实际 CR、SR 或 RF 指令应由后续执行器判定。本轮没有计时或隐含一槽新操作。

| 必须收费的义务 | 固定范围或代价约束 |
|---|---|
| 表与 cold fill | phase-only 最小 14,080B，但精确编码还需 a；a9+phi11 共 25,600B 逻辑内容。每 gword 打包一条 32B CR 响应需 32,768B；每项展开32bit需40,960B，若每行对齐64B则65,536B。表在共享 coefficient SRAM，不能塞入既有512×128指令ROM冒充免费表 |
| 选择与解码 | 每窗 1,536 个真实 gword 选择；共同 g=0 旁路后非零选择为214–381个，尚未计跨词复用。32B 行内十个20bit字段仍需真实 selector、提取、符号扩展和 RF 写，不能把一个 CR 响应当十次免费输入 |
| 编码 | 每窗仍处理全部15,360个真实I24，付 x−phi、quotient/remainder、tie parity、平移clip和code写入；原 full-D 同样拥有 g=0/CSE 旁路 |
| 码与 phase spill | 全窗 z9 打包17,280B，z16展开30,720B；phase11另打包21,120B，phase16另30,720B。若保留 gate word 供后查需原3,072B及实际读/选择；避免 phase spill 需要明确 RF 寿命和重查次数 |
| phase MAC | 按当前标量零旁路，四窗需6,420–11,403个额外8-lane phase MAC义务；H8整组跳过不能用标量零率替代 |
| 完整控制 | 原 full-D 先计算 Ug 的未CSE AAC8 为666–1260项，稠密 D×Ug 为4800项；这些宽度与phase MAC不同，禁止直接加总比较槽数，且必须给原full-D完整CSE/合法动态旁路 |
| 常数、RF与 U/V | phase0 行和常数同普通 affine 获得相同预计算/缓存权限；跨回调原Conv2/F/projection会覆盖RF，需恢复。全部只用共同96RF、64B staging、CR256单响应、128KiB系数/状态池 |

最强分母仍是同 Q16 U/V 的普通 fixed/affine，给相同 P1、CSE、Uc、源驻 RF 和最佳合法码格式；它们原 q8 每窗15,360B，不能强迫普通路径展开16bit来衬托候选。第二分母是同函数 full-D＋完整 CSE，包括同一张表可同时构造 b 的公平展开控制。候选表32KiB在高系数地址可放下，但冷fill和竞争仍收费。

后续若选择执行，只实现上述一次 P1 有费表/phase 放置，先比较同函数 full-D 完整消费者，再比较最强普通 fixed/affine。若无实际服务收益，停止这次 phase 放置的性能句，保留精确位宽与parity结果；不据此否定全部条件表示，不自动启动下一轮模型或新 card。本轮结论停在“可编码且有有限稀疏余量”，没有新性能或精度收益。

复现：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 phase_probe.py`。只写 `phase_probe.py`、本文件与 `phase_results.json`。
