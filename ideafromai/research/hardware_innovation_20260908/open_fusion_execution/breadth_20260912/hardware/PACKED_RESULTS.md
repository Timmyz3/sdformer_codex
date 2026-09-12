# W4/W8真实压缩权重执行

**完成固定四窗与一组压力，总25条完整局部链；压缩接口功能正确，当前布局没有服务净收益。** 同函数的expanded16与packed都对实际GPU部署字段逐项一致；不是拿原权重精度配新压缩拍数。全部updated I24、projection gate与PED检查共3,945,600值0差。

| 同负载 | 原R32／同函数expanded16 | 实际packed | 同函数整局部服务变化 |
|---|---:|---:|---:|
| ordinary corner ready W8 | 2,107,838 | 2,124,544 | +0.793% |
| ordinary corner ready W4 | 2,107,838 | 2,123,488 | +0.742% |
| ordinary interior ready W8 | 2,800,716 | 2,817,422 | +0.596% |
| ordinary interior ready W4 | 2,800,716 | 2,816,366 | +0.559% |
| ordinary interior stress W8 | 3,095,578 | 3,113,466 | +0.578% |
| ordinary interior stress W4 | 3,095,578 | 3,111,738 | +0.522% |
| lifting_raw corner ready W8 | 1,943,768 | 1,960,474 | +0.859% |
| lifting_raw corner ready W4 | 1,943,768 | 1,959,418 | +0.805% |
| lifting_raw interior ready W8 | 2,576,768 | 2,593,474 | +0.648% |
| lifting_raw interior ready W4 | 2,576,768 | 2,592,418 | +0.607% |

每窗W8实际少填2,976B、少读23,808B系数；W4少填4,512B、少读36,096B系数。SR64与SW64字节不变。代码/行尺度/32B头确实写入模拟coefficient memory，原expanded U16没有出现在packed存储体。U分别为3072B或1536B code＋64B尺度＋32B头，V还是6144B原q16，原bias保留。

代价出在哪里：ordinary/corner普通U为113,936槽，W8 packed131,200、W4 packed130,432。每窗3,072个H8解包，3,072次RF→既有staging搬移，1,280次16×24尺度乘积与640次移位合并，较普通多9,360个operand等待槽。减少的冷填W8为558槽、W4为846槽，抵不过这些真实步骤。完整局部服务分别增加16,706/15,650槽；固定压力例增加17,888/16,160槽。不同量化后的V输出同样逐值核对，未减少有义务的PED写出。

双方同96×8×48 RF、128KiB state/coef、SR64/SW64/CR256、8192B源ROM、32B/5slot DMA。代码解包到RF84，再付费读到共同64B staging的16B区域；source gather最多24B，预算不增加。MAC读取acc＋source两RF口，权重来自暂存，未开第三口。行尺度作用于宽点积，因此用signed24 hi/lo分解、两次共同16×24乘法和shift/add，之后才做原U RNE/sat；没有把宽乘法当一拍免费操作。V RNE及bias原位置不变。

每窗前级真实执行一次并复制其完整Machine状态/时间/待写回/仲裁，分别跑五个后级；原R32逐count复现上一阶段。没有旧表相加、没有新门替换旧SRAM输入。边界仍从真实I24到完整K864、BN2/rawI24、gate＋PED U32/V96；**native投影、全域BN与最终join不在此表**。

本轮的A是普通低位权重底座，不是X。只停止当前RF解包＋尺度回写布局的加速主张，保留已过十帧质量的低位参数。尚未试的接口是共用系数响应路径内解码／更低位乘法 datapath，但不能在本表中假设其无成本。这里无法证明ReverB/MiLo完整系统失败，也不能仅凭压缩比立贡献。

实现：[packed_weights.py](packed_weights.py)、[run_packed.py](run_packed.py)；同负载机器表：[packed_summary.json](packed_summary.json)。旧stage和生产目录保持只读，没有训练/EDA或新AEE。
