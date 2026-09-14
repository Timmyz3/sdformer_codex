# 连续第二级的 Winograd 融合：先写洞和对照

2026-09-14，空间R16完整RTL之后的一处有界融合；不是宣布新的Winograd。

**B**：普通空间R16已经保留前级{0,θ}跳过，但连续1×3第二级仍占held/disjoint raw core的56.5%/55.2%。完整3×3输入复用不会消除这些连续乘积。旧“Winograd把二值源变多位”的反对只适用于前级，不能杀掉后级接口。

**A**：普通3×1→1×3空间分解、Gustav式源共享/局部psum、Winograd F(2,3)。[Lavin/Gray CVPR2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)及其[公开生成器](https://github.com/andravin/wincnn)是算法先验；不是复现其GPU硬件。低秩+Winograd本身已有[2023预印本](https://arxiv.org/abs/2301.11180)，分离卷积+Winograd加速器也已有[2025作者论文页](https://research.tue.nl/en/publications/algorithm-hardware-co-design-for-accelerating-depthwise-separable/)。两者只做有界最近邻核查，不宣称已读全器件实现。

**待验证增量**：仅在连续横向级切换求值表示，使前级仍是便宜的脉冲AAC；原地复用有限Z bank并在原始p32/I24边界前精确恢复。该适配可能有性能，但“分解+Winograd”不能单独撑新颖标题。

**数值与资源**：原signed13 Q2直接变换会扩大至signed15；禁止暗换乘法器。只试一次signed11 Q2量化，令3tap系数和严格适配原13bit乘法操作数；普通factor及direct展开共同获得相同11bit函数。z15加减为z16，适配现有19bit另一操作数。4个变换域输出累加器、输入变换holding、额外W/cache字和重建周期全部付费。不新加latent舍入，最终除2必须精确偶数。

**最强对照**：同函数ordinary factor RTL（源跳过、缓存、双P15、真实Z/p）、同函数expanded-W sparse AAC，及既有q13格式的质量/资源Pareto。不能只同未分解稠密MAC比。

**停止布局的门**：功能不等立刻修复；新q11网络AEE不优于同环境NB0则不晋级。付变换/配置/消费/BP后同ordinary factor没有净周期就停这一放置；保留连续级快卷积家族。灰区10–15%留一处有界接口，不扫bit/rank。全网乘积数与RTL周期分开，不借用q13的825质量。
