# 固定整数R8至真实消费者

固定上一轮Q1/Q2/output_scale，不训练、不改变格式、不重选rank。复用A800 Python3.12环境，仅把此前新链diverse10扩展为官方valid825；dense/block/Cin三臂825不重复。历史NB0门1.44535253468097。原mux已失效，根代理使用已授权认证重建 `/tmp/codex_r8_consumer_a800_20260914.sock`，已重新检查GPU空闲。

先读实际MS_ResBlock.forward与当前LiteralForward：r0 sn2→conv2→norm2→ADD原r0输入identity→r1.sn1，后者先write24(I24)再时域矩阵/阈值。运行时确认BN2的training/track_running/mean/var/eps及归一化轴，不凭名字编造固定BN。I24是signed24、fraction14、torch ties-even后饱和，不能移过任意线性消费者。

一次825运行的首帧固定zurich_city_09_a_0001，同8个旧固定空间块捕获source4×4、z/p/identity/BN2/r0/r1输入/I24，完整T10/C96或R8与2×2输出；存真实BN参数和全域统计。全部整数z/p用FP64精确模拟并检查范围，之后同原float32消费者。先发原数值合同，再与硬件端协调定点出口，不能把近似fold称FP32 bittrue。

所有新写入限此data目录及远端同名owned镜像。旧r0_stream_fusion_20260914与生产只读；无hash、新环境、额外训练或同三臂重跑。

## 按真实范围冻结的新消费者与公平质量分母

运行时BN2实际为固定eval统计。与硬件端共同冻结唯一新数值合同：`a=RNE(scale*gain*2^40)` signed32，`b=RNE(offset*2^20)` signed32，`J=RNE(identity*2^20)` signed32饱和；`wide=p*a+((J+b)<<20)` int64，最终一次ties-even右移26后饱和到I24。先新消费者diverse10，通过历史NB0后再独立825。禁止重选格式。整帧p/J/I24在此十帧首帧自然导出，无额外捕获前向。

新消费者十帧实测与原浮点消费者有可见AEE差异。根代理因此明确授权完整825之后，顺序复跑同A800/env312的原PSN/SDSA NB0 ep29官方825，作为同环境分母；历史1.44535253468097单独保留。NB0入口用accuracy_baseline保存的原配置、原checkpoint、remap v1、78处BN no_running及完整模型flow[-1]，无学生helper；不会以当前dense学生替代NB0。
