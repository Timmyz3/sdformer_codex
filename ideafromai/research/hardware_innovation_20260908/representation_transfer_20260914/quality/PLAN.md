# 空间R16新整数函数的完整网络评价

运行前计划，2026-09-14。因子为本轮一次冻结的Q1 signed8竖3×1、Q2 signed13横1×3，静态θ和逐rank第一尺度吸入第二因子；Z15、p32无中间RNE。保留原r0固定BN和真实FP32 identity的I24接口，使用对应新aQ40/bQ20。借鉴完整度与局部界由spatial_r16_integer记录，不继承浮点R16的AEE。

在已授权A800、原env312/PyTorch2.2.2、同matched-dense学生/原粗头/read_names协议运行。先diverse10并逐第一帧选中tile核对真实源、p/J/I24和本地NumPy gold；若优于同环境NB0子集，再跑官方valid825，不训练、不扫描rank/位宽。按FP32 EPE后FP64累加的现有evaluate_axis协议报告帧等权/像素全局。

两级卷积用FP64执行整数运算，每帧检查积分性和已证明的范围；GPU elapsed仅作运行收据，不是硬件速度。新I24必须被现有下一层reader实际读到。其余网络维持原TF32/固定BN配置，不宣称全网bittrue。原R8和NB0同环境已有825记录作明确分母，若首帧输入不一致则先定位协议，不把旧gold强塞当前运行。
