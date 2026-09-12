# BN 三种浮点函数：真实网络成对比较

2026-09-12，三臂完成。新用户准入规则是优于同评价口径的 SDformerFlow baseline；主线程/phase_aee已核实本地 upstream 复现 NB0 fullres ep29 的同十帧/同valid_pixels基线为 **1.45460286107**。本页所有 `legacy_* +0.005` 字段只保留为旧门历史诊断，**不再据此淘汰候选**。本表六个学生/BN组合均低于该同十帧基线，满足当前“可继续考虑”的精度条件；没有继承 valid825 结论。该基线不是作者发布检查点，采用最终头；候选采用粗头，两者可比较任务质量，不能把全部差值归因于BN。详见[同帧基线与口径](../../../accuracy_baseline/README.md)。下面的原 CUDA BN 指两个现有学生的原 BN，不能冒称原 SDformerFlow 网络。

| 原学生 | 原 CUDA BN | 原四支路中心化 Engine | 两支路单遍 Engine | 单遍−中心化 |
|---|---:|---:|---:|---:|
| ordinary | 1.161187075541 | 1.172122721941 | 1.166841775747 | −0.005280946194 |
| lifting_raw | 1.186585517185 | 1.188048987398 | 1.187834853773 | −0.000214133625 |

数字均为固定 diverse10 的 AEE 帧均值；没有训练、valid825 或整网硬件性能声明。两学生原 CUDA 对照十帧逐帧复现历史值，单遍带 flow 观察的复跑也逐帧复现初轮。所有结果和逐帧预测变化在 `three_arms_summary.json`。两种 Engine 函数都改变原 CUDA 数值运算；单遍在此匹配比较中比中心化的 AEE 更低。因此不能把全部精度差归给单遍方差公式，更不能据此杀单遍家族。

只替换 `sttmultires_unet.encoders.swin3d.patch_embed.proj.norm_layer`。原学生权重、fixed integer 源与消费者、raw I24、连续 PED、其他 BN/PSN、原 coarse exit 和 AEE 公式保持一致。单遍是 BNFF/FlexAcc 公共强底座，不是新 X。

`numeric.cpp` 是部署算术伴随，分别实现原四支路中心化两遍、两支路 sum/sumsq 单遍；两者均保留各自 paired256 树序、原 seed+三次 Newton 与分开的 affine MUL/ADD。它不模拟端口，不能产生硬件服务数字。两函数、两学生的本地 288 个统计和18,432,000个输出，分别与对应完整 Engine 逐位相等；见 `local_exact_check.json`、`centered_exact_check.json`。A800 在网络评价前直接把完整 GPU 归一化输出与各自保存的 Engine 输出比较，全部0位差。**CUDA 方法的该匹配字段为 null**；其复评前顺带完成的独立 helper 检查已另名，不能误读成 CUDA BN 与 Engine 一致。

真实 flow 比较来自 GPU 的480×640输出，先于 GT valid-mask；全图 flow 距离分母和 AEE 有效像素分母不同。单遍相对原 CUDA 在 ordinary 的3帧、lifting的5帧产生预测变化，其余对应帧整幅相等；有变化帧全图平均flow距离约0.76–1.18，不能仅用 AEE均值接近宣称预测不变。限定 `zurich_city_07_a_0001` 的首翻门诊断已完成，见下段与 `first_flip/result.json`。

入口：`evaluate.py --root <BASE>`；`--original-cuda` 重评原 BN，`--centered` 用旧中心化部署函数，`--capture-flow` 将实际预测暂存在远端 `/tmp/onepass_aee_flow/` 以生成成对统计。所有原始三臂报告均保留。没有修改生产树、主稿或 EDA。

首个门差在 stage0 首块 `mlp.sn1`：前面的 `attn.proj_sn/sn_q/sn_k/attn_sn` 均逐位相同。该叶18,432,000门只翻1位（T8,y68,x152,c15，1→0）；实际膜从 **1.0** 变为 **0.9999999403953552**，θ=1，跨越一个向下 ULP。该T10输入非零，整叶仅一列拥有此输入，不是首差成片翻门。完整投影域133,147/192,000个空源向量全部对应raw=0，但其中PED全96通道同时为零的交集为0；BN默认值并未原样绕过连续PED。首差已在注意力残差之后，脚本没有臆造直接默认坐标映射，也未推断尚未测量的非零PED重复类。当前观察支持一般近阈值放大，不能把它包装成默认类专属问题。

诊断只在RAM保留打包门位和稀疏近阈值记录；实际膜来自ATLIF已有观察口。没有重构算法前向；修正了观察位置以遵循 `conv_res → sn → BN` 的真实顺序和 helper 在发门后释放连续状态的边界。原始/单遍两次AEE分别复现1.870729870522 / 1.919150395469。
