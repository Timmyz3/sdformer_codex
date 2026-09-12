# native投影W8：一个固定新函数的十帧

旧ordinary R24＋onepass父上，仅替换`sttmultires_unet.encoders.swin3d.patch_embed.proj.conv.weight`为已实测的逐输出行dyadic INT8展开函数。所有source/PED常量逐键保持，未混合新320步学生或PoT2。

实际diverse10 AEE **1.149276778**，去首帧九帧 **1.125297062**；相对同父分别−0.015529103/−0.015502202，均优于相应NB0。完整825尚未测；不将小集改善认定为普遍提高精度。

CPU compact权重数据解码/scale与GPU展开权重逐项一致。首帧两个4×4窗口的native raw、BN和最终PED输出共92,160值，加完整192000域的480统计量，对实际CPU结果全部0 bit差。见[实际核对](../../hardware/native_bn_join/native_w8/gpu_alignment.json)。没有采全域GPU raw，不把有限核对冒称全域张量证明；这是新的W8对照，未与旧原W捕获混比。

十帧推理与quality.json已成功保存后，首次NPZ端点保存因`file`键与NumPy参数冲突失败。改名`input_file`，用`--capture-only`只重放同一首帧并核对AEE/像素逐项相同，补齐端点文件；未重跑/挑选十帧结果，也未新增AEE臂。

完整后段少3.4507%服务是CPU同函数展开vs实际packed驻留，非新RTL/PPA；量化、驻留、目录复用是共同强底座，未申领标题X。
