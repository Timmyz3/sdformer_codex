# r0：原生数据流、有限值分解与真实结构剪枝

2026-09-13，接续 [上一轮三套 RTL](../pro_fusion_trials_20260913/REPORT.md)。**本阶段已完成实际RTL筛选、六臂diverse10及独立审阅。** [完整阶段结果与下一步](REPORT.md)。

| 已完成路线 | 实测结果 | 去留 |
|---|---|---|
| 原生三数据流 | dense含每块新输入：强W-major 798,816→source-major 480,384周期 | 留作更强公共底座，非新X |
| 消费者枚举 | 同源优先进一步到427,320周期；同请求/同结果；三个mask少11.05%–11.52% | 控制器实测有效，位图迭代本身仍是先验 |
| 整数Winograd＋坐标剪枝 | 精确/坐标剪枝仍为同核direct的1.965/1.630×周期 | 本固定布局不晋级，保留其他表示接口 |
| 25%真实结构剪枝 | physical/magnitude/coordinate AEE 1.190684/1.185997/1.172116；fresh NB0 1.462941 | 均过十帧质量筛查；physical未胜幅值控制 |

原生核与Winograd核内部端口不同，不能横比绝对周期；所有RTL表只对应同一帧八块，未完成整层、整网bittrue或PPA。源码与结果摘要纳入Git，模型、数据、编译产物及浏览器登录状态不纳入。

实际代码：[原生执行器](native_sparse/native_sparse.sv)、[消费者枚举](consumer_enumeration/native_sparse.sv)、[完整Winograd](winograd/winograd_tile.sv)。[质量数据](data_and_quality/README.md) · [浏览器登录入口](browser/README.md) · [独立原生审阅](REVIEW_NATIVE.md) · [枚举审阅](data_and_quality/REVIEW_ENUMERATION.md) · [Winograd审阅](native_sparse/REVIEW_WINOGRAD.md)。

## 本轮问题与固定对照

昂贵对象为 matched-dense stage320 的 patch residual r0 `conv2.0`，完整 C96/N96/T10，原生连续 4×4 输入生成 2×2 输出。上一轮的64个离散 patch不能替代连续输入。r0名义算术占比约10.678%，不是已测硬件时间份额。

| 路线 | 借入 A | 本网剩余问题 B | 本轮要试的增量/控制 | 完成判据 |
|---|---|---|---|---|
| 原生数据流 | Gustavson 目的供数、ELSA局部状态、普通两源部分和 | 低共同活动配对少补算却多取W；旧叶未计原生gather与真实端口 | 固定单源口/8个W bank/8个psum bank；output-major、weight-major、source-major同硬件 | 完整3840输出、原生地址、真实读写/等待/末drain；比较最佳公共底座 |
| 特定分解 | 完整F(2×2,3×3) Winograd、有限域分析、WINS与零跳过 | 变换可把廉价binary AAC变成更多加法、膨胀W和状态 | 整数精确变换与直接AAC；支持掩码关闭实际词；变换/逆变换全计费 | 相同整数函数零差、同资源周期；不能只报2.25×乘法减量 |
| 有损剪枝 | 普通结构剪枝、同nnz幅值控制 | 独立权重置零不一定删掉完整广播词 | 固定N8×Cin4×完整3×3块、25%点；实际源活动参与评估 | 新前向diverse10 AEE与原NB0比较，再用新W/掩码进同RTL；局部误差不当AEE |

这里的循环重排、bank、Winograd与块剪枝均是借入底座，X尚待比较；不能因实现绿或比弱布局快就宣告新颖性成立。负结果只停所测布局，不否定所有表示与接口。

AT-LIF `{0,θ}`，静态θ可折权；非因果T10和连续消费者不被跳过。硬件本轮首先闭合线性叶，未接norm/PSN/残差时明确写出。诊断Q16不继承FP32 AEE，近似权重另测。当前质量政策以原SDformerFlow NB0为准，未重新引入+0.005门。

分工目录：`data_and_quality/` 真实捕获及算法质量；`native_sparse/` 原生数据流；`consumer_enumeration/` 唯一追加的控制器比较；`winograd/` 分解核；`browser/` 本机Pro登录辅助（浏览器状态和凭据只在用户私有目录，不进入仓库）。生产nts07、主稿、docs359和H81保持只读。
