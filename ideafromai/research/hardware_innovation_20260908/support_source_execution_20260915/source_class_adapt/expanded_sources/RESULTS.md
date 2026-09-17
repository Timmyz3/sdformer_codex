# 32 帧固定输入扩展：独立结果聚合

新 class 在 31 个未参与 pair 选择的训练探针上，相对 exact-code 的源服务合计减少 **1.592% / 1.651%（ready / BP）**，两种服务条件均 31/31 帧为正。但相对普通 static64+预取，ready 减少 9.990%，**BP 增加 2.983%**，BP 只有 6 帧正、25 帧负。应保留类别选择的薄增量，同时保留普通强控的反证；不能把小表两帧的 2.5–2.7% 当扩大后的恒定收益。

本次只读 CSV/日志并用 Python 3.12 聚合，未新增 RTL 运行、重新选择 pair 或借用旧 AEE。真实输入来自原 32 帧训练缓存，各固定 sample0..31，共1,024包；frame0 是 pair 选择帧，frame1..31 是未用于此次选择的训练探针，**不是 validation，也不是对学生/字典训练的独立留出集**。

## 服务与实际字节

所有臂使用同一源 RTL、128B cache、普通 next-X/图预取权限和既定 D-only 熵序；图臂均 packed32。服务为 CSV `cycles`，已含 start、逐包参数取数与背压，不另加 go；每包从 reset/cache cold 开始。这是原始 X→源 MAC→判码模块，尚不含 FC1 消费流量或新 W″ 的网络质量。

| 输入范围 | 臂 | 服务 ready / BP | T10 通道任务 | X 字节 | 图字节 ready / BP |
|---|---|---:|---:|---:|---:|
| frame0（选择帧） | static64 + PF | 41,504 / 44,128 | 2,048 | 65,536 | 0 / 0 |
| frame0（选择帧） | exact-code32 + PF | 35,957 / 43,771 | 1,690 | 54,080 | 60,688 / 60,672 |
| frame0（选择帧） | 旧 class32 + PF | 35,919 / 43,525 | 1,688 | 54,016 | 60,000 / 60,000 |
| frame0（选择帧） | 新 class32 + PF | 35,026 / 42,614 | 1,642 | 52,544 | 62,336 / 62,336 |
| frames1..31（训练探针） | static64 + PF | 1,286,624 / 1,367,968 | 63,488 | 2,031,616 | 0 / 0 |
| frames1..31（训练探针） | exact-code32 + PF | 1,176,829 / 1,432,418 | 55,438 | 1,774,016 | 2,151,056 / 2,150,272 |
| frames1..31（训练探针） | 旧 class32 + PF | 1,173,106 / 1,422,911 | 55,244 | 1,767,808 | 2,125,536 / 2,125,088 |
| frames1..31（训练探针） | 新 class32 + PF | 1,158,089 / 1,408,772 | 54,474 | 1,743,168 | 2,168,832 / 2,168,224 |
| all32 | static64 + PF | 1,328,128 / 1,412,096 | 65,536 | 2,097,152 | 0 / 0 |
| all32 | exact-code32 + PF | 1,212,786 / 1,476,189 | 57,128 | 1,828,096 | 2,211,744 / 2,210,944 |
| all32 | 旧 class32 + PF | 1,209,025 / 1,466,436 | 56,932 | 1,821,824 | 2,185,536 / 2,185,088 |
| all32 | 新 class32 + PF | 1,193,115 / 1,451,386 | 56,116 | 1,795,712 | 2,231,168 / 2,230,560 |

字节是实际接受的128bit请求×16，包含 padding、重复读取和未选中分支预取。图字节不是驻留容量；prefetch 是 X/图读的子集，不能再加一次。每包参数读取 static30字、图臂21字；总读字节、参数字节、各帧名字与明细见 [summary.json](summary.json)。例如31帧新 class 相对 code 的图读取反而增加17,776/17,952字节，但 X 读取少30,848字节，合计实际读字节少13,072/12,896。相对旧 class，源少读仍未抵消图读增加，总读字节反而增加18,656/18,496；周期减少不能外推成统一减少带宽。

## 帧级正负数

正/平/负按逐帧周期减少/相等/增加定义，不由源任务数代替。范围百分比由总服务计算，不平均单帧百分比。

| 范围 | 对照 | ready 减少 | ready 正/平/负 | BP 减少 | BP 正/平/负 |
|---|---|---:|---:|---:|---:|
| frame0（选择帧） | 新 class / exact-code | 2.589% | 1/0/0 | 2.643% | 1/0/0 |
| frame0（选择帧） | 新 class / 旧 class | 2.486% | 1/0/0 | 2.093% | 1/0/0 |
| frame0（选择帧） | 新 class / static64 | 15.608% | 1/0/0 | 3.431% | 1/0/0 |
| frame0（选择帧） | exact-code / static64 | 13.365% | 1/0/0 | 0.809% | 1/0/0 |
| frames1..31（训练探针） | 新 class / exact-code | 1.592% | 31/0/0 | 1.651% | 31/0/0 |
| frames1..31（训练探针） | 新 class / 旧 class | 1.280% | 31/0/0 | 0.994% | 31/0/0 |
| frames1..31（训练探针） | 新 class / static64 | 9.990% | 31/0/0 | -2.983% | 6/0/25 |
| frames1..31（训练探针） | exact-code / static64 | 8.534% | 31/0/0 | -4.711% | 4/0/27 |
| all32 | 新 class / exact-code | 1.622% | 32/0/0 | 1.680% | 32/0/0 |
| all32 | 新 class / 旧 class | 1.316% | 32/0/0 | 1.026% | 32/0/0 |
| all32 | 新 class / static64 | 10.166% | 32/0/0 | -2.782% | 7/0/25 |
| all32 | exact-code / static64 | 8.685% | 32/0/0 | -4.539% | 5/0/27 |

31帧探针中，新 class 相对 code/旧class 两种服务条件均无帧级负例；图相对 static 的负例来自BP条件。具体如下：

- 新 class / static64，BP 正例 frame[1, 4, 15, 18, 20, 22]，负例 frame[2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31]；最差 frame7 `zurich_city_05_a_0158.npy`，44,128→47,830，增加 8.389%。
- exact-code / static64，BP 正例 frame[1, 15, 18, 22]，负例 frame[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31]；最差 frame7 `zurich_city_05_a_0158.npy`，44,128→48,501，增加 9.910%。
- 新 class / code 最小增量：ready frame13，39,148→38,819（减少 0.840%）；BP frame13，47,615→47,267（减少 0.731%）。
- 新 class / 旧class 最小增量：ready frame13，38,951→38,819（减少 0.339%）；BP frame3，45,206→45,118（减少 0.195%）。

## 实际 gold 与协议检查

新表1,027×6 = **6,162任务**，旧 class 表1,027×2 = **2,054任务**；合计 **8,216**，其中真实输入 8,192、诊断 24。两份日志各有1,027条病例 PASS 和最终 ALL PASS。逐任务固定6组×T10，合计 **492,960 个最终 code/class 标号**实际比较；性能聚合排除诊断。

TB 对实际发出的 **472,658 个通道包**逐一比对全部 T10，共 **4,726,580 个 signed48 U 值**及同数门位。没有将未生产的 CPU 全量 oracle 算作 RTL 检查。记录共执行 47,265,800 个标量 MAC，但单个乘积没有逐值 monitor。输出接受组数 49,296；原 TB 还检查请求/输出背压保持、唯一通道/组以及完成时计数。

独立逐任务复核7条费用/状态恒等式，共 **57,512** 条：`xwords=2×channels`、`scalar_mac=100×channels`、`words=boot+xwords+graph_words`、`cycles=Σstate−1`，以及 MAC/CLEAR/DECIDE 状态次数。另核 **16,432** 条 boot 请求/响应状态至少为参数字数的下界；BP 会延长这两段，不能误要求周期恒等于字数。原64包本次共 **512条重叠配置记录与旧 CSV 全部字段一致**，不只是结果值相同。见 [新表](source_cycles.csv)、[旧class表](old_class/source_cycles.csv)、[新日志](run.log)、[旧日志](old_class/run.log)。

## 消费路径与 T10 粒度边界

`train_exact_support_probe.py:70–82` 的 h=A×X+bias−center 为各目标 t 独立的行点积，nearest-code 逐 t/C16 执行。`Student.source()` 返回 emitted/raw gate/soft 供训练辅助；部署替换函数 `:226–236` 只取 emitted（forced_code 的投影门）×theta，raw gate/soft 不向网络返回，fc1 再消去 theta。源 U 仅用于当前门计算，没有部署输出。

`MS_Spiking_Mlp.forward:164–181` 的实际顺序是 sn1→drop1→fc1→bn1→sn2→drop2→fc2，`run_bn_probe.build_model()` 明确 `model.eval()`，dropout 为推理恒等。在此部署链未见 sn1 原 rawg/U 的其他消费者。外层 `:840–845` residual 保留的是进入 MLP 前的 x，不是这里的 rawg/U；训练 auxiliary 和调试捕获也不能算本接口的额外推理消费者。

**共同生产全部 T10 是当前 controller 粒度，不是算法硬约束。** `source_classifier.sv:189–203` 以 source_t 遍历10输入，同时求10目标 t；因此一通道收费100标量MAC。若某个 t 的 cofactor 前沿只需要一个门，数学上只需该 A 行的10项点积，其他 t 的门可不生产。非因果 PSN 要求所需源时间 X 可用，并不强迫所有目标门都被求值，也没有 gate-to-gate 递推要求共同退休。

有限 holding 的 partial-t 生产是尚未纳入此表的普通强对照。它必须给 exact-code/class 相同的逐 t 需求、A 行读取、输入暂存与预取权利，并实际收费排队、重读和持有状态。当前整 T10 对照仍是有效实现结果，不能当成不可削减的算法下界。新 frontier RTL 在另一分支测试，本报告不提前给它收益或合法性收据。

源码：[Student.source 与部署替换](../../../algorithm/train_exact_support_probe.py)、[build_model eval](../../../algorithm/run_bn_probe.py)、[当前TB](../../tb_source.cpp)、[当前源RTL](../../source_classifier.sv)、[MS_Spiking_Mlp与外层residual](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:164)。
