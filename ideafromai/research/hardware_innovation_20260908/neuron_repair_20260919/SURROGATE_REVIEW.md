# 二电平神经元 surrogate 审阅（2026-09-19）

范围：仅 H9 的 `official_atlif` 二电平 \(y=\theta H(h-\theta)\)，\(\theta>0\)；已读 [THETA_BN_REVIEW.md](../handoff_review_20260919/THETA_BN_REVIEW.md)、H9 `atlif_ternary_psn.py` 与 `t57_gradfix_train.py`。只做只读源码审阅和 CPU 标量数学核对，未修改 overlay、未用 GPU、未训练、不涉及 H12。**先修清真实训练回路，再比较明确定义的代理规则；硬阶跃没有唯一“正确的 surrogate 导数”。**

**两种候选的差别。** 记 \(g=\mathbf1[h\ge\theta]\)、\(u=h/\theta-1\)、\(p=(1-|u|)_+\)、上游梯度 \(a=\partial L/\partial y\)。用三角核积分 \(F'(u)=p\) 定义 STE：`B(u)=stopgrad(H(u)-F(u))+F(u)`。

| 规则 | 输入局部导数 | 阈值局部规则／共享标量梯度 |
|---|---:|---|
| 当前 H9 official | \(p\) | \(-\operatorname{mean}(a p^2)\)，没有 `.abs()` |
| T57 margin | \(p\) | 局部 \(g-p\)，实际为 \(\sum a(g-p)\) |
| normalized-coordinate：\(y=\theta B(h/\theta-1)\) | \(p\) | 局部 \(g-(h/\theta)p\)，即 \(\sum a[g-(h/\theta)p]\) |

T57 可解释为 margin 坐标 \((h-\theta)/\operatorname{stopgrad}(\theta)\) 的代理；normalized 坐标允许分母随 θ 求导。两者 forward 相同，θ backward 不同。例如 \(h/\theta=0.5,1.5\) 时，normalized 的 θ 局部导数是 −0.25、+0.25，T57 是 −0.5、+0.5。normalized 满足局部尺度恒等式 \(hD_h+\theta D_\theta=y\)，适合作为可由明确 STE 计算图核验的主候选；这不证明其训练精度优于 margin。两种代理核在 \(0<h<2\theta\) 外均为零，但 \(h\ge2\theta\) 时幅值项仍给 \(D_\theta=1\)。

**共享 θ 和 AMP 合同。** 当前 θ 是 0-d 参数，广播反传应对使用该 θ 的全部元素求 `sum`；`mean` 是额外的梯度归一化，不能当成同一个计算图的自动求导。loss 已取平均时再对 θ 取平均会额外除以神经元元素数；跨层元素数不同。不能笼统以“学习率乘 N”声称 Adam、ε、梯度裁剪后仍等价。T57 还将 saved h、θ 转为 `grad_output.dtype` 后重算 g，这与原 forward 的比较不保证相同：低精度 h 与 0-d fp32 θ 的类型提升，以及重新比较时的 cast，都可能改变边界。候选应保存**原 forward 的 bool gate**；保持原 forward 算术／输出 dtype，用 fp32 计算代理核和 θ 乘积归约（float64 测试保留 float64），最后按对应输入 dtype 返回梯度。记录 saved h、θ、output、grad_output、归约 dtype。任何改变比较精度的 forward 改造须另立实验臂。AMP 须在真正的 optimizer step 前 `unscale_ → clip → step`；梯度累积时只在累积结束 unscale，候选之间顺序一致。

**θ-fold 的成立条件。** θ 固定为推理常数且该边进入线性消费者时，\(W(\theta g)+b=(W\theta)g+b\)，bias 不缩放，神经元比较阈值仍保留；按输入通道／列折叠并覆盖所有消费者。等式发生在线性输出处，后接 BN 用 batch 或 running 统计都不妨碍此处实数恒等式。sign/support 消费者在 θ>0 时可消去幅值，不能删去支持位；Q 仍有前向作用。非线性门、skip、拼接、多消费者须逐边证明；K 的 gate 载荷仅在 gate 对幅值不敏感时才能搬出 θ。FP32 舍入、量化、溢出与后续阈值翻转须另验，不能由代数恒等式宣称全网逐位等价或零成本。此项只是推理重参数化，不保证折叠前后训练梯度相同。

**最小验收标准。**

1. **前向：** 相同 h、θ 下与原 official 输出逐位一致，含 h=θ、相邻浮点数、θ 非整数及 AMP 实际 dtype；原 gate 与 backward 使用的 g 完全一致。记录门差异数，不能只看最大输出误差。
2. **梯度：** 在 \(\theta\in\{0.2,1,3.2\}\)、\(h/\theta\in\{-0.5,0,0.5,0.9,1,1.1,1.5,2,2.5\}\) 上检查正／负 upstream 的 VJP、零 upstream、标量共享与广播归约。对采用广播 sum 的候选，`sum` loss 复制元素应让 θ 梯度倍增，`mean` loss 完整复制应不变；mean 对照臂按其额外 1/N 规则核验。normalized 还须满足上述尺度恒等式。用三角核积分的**局部光滑延拓**检查解析式，不对硬 forward 做 finite-difference gradcheck。
3. **数值：** 对已量化的相同 h／gate，比较 fp32 基准与 AMP backward；检查大元素数归约不溢出、scaled loss 在 unscale 后的 θ 梯度一致（浮点容差内）。有效更新另记 Adam m/v、ε、lr、clip 系数及 θ 写回前后值；“梯度非零”不等于“参数可见更新”。
4. **图连通性：** 只有上游 a 非零时，换 θ 代理才可能恢复该层梯度；当前 sn_q 的布尔 score 路径需独立诊断，不能以其 θ 未动判定代理失败或模块无用。

CPU 已核对 54 个局部延拓／规则组合，两种导数最大绝对误差分别为 1.01e−6、5.01e−7 以下（中心差分步长 1e−6θ），normalized 尺度恒等式误差 <1e−14。延拓在每个锚点固定 \(c=g-F(u)\)：normalized 为 \(\theta[c+F(h/\theta-1)]\)，margin 为 \(\theta[c+F((h-\theta)/\theta_0)]\)，其中 θ₀ 固定于锚点。这里只验证公式，不替代 PyTorch／AMP 实现测试；本地未安装 torch。

**推荐二值实验臂。** 先以官方 backward、η=0，完成旧／新 AMP 裁剪顺序的固定 batch 配对诊断。之后固定正确训练回路、同 checkpoint／RNG／optimizer 初态／BN／数据／预算，短探针依次比较：O=`−mean(a p²)`；K=`−mean(a p)`；Mμ=`mean[a(g−p)]`；MΣ=`sum[a(g−p)]`；NΣ=`sum[a(g−(h/θ)p)]`。相邻差异分别隔离核幂、幅值项、归约和 normalized 坐标；不把合并倍率全归因于“补幅值项”。仅在数值与更新合同通过后，再以 O、MΣ、NΣ 做同预算二值恢复与匹配验证集评测；η 的手动阈值增量随后单独恢复，不能以 θ 能动作为最终精度验收。
