# Spatial R16：独立接口与强对照审阅

只读审阅 [RTL PLAN](../spatial_r16_rtl/PLAN.md) 与 [spatial_core.sv](../spatial_r16_rtl/spatial_core.sv) 的当前完整 factor 核；此时该分支尚无 TB/实测结果，本页不替代后续输出、BP及周期核验。整数接口已独立通过 [135个真实 tile 的 NumPy/Torch CPU 验证](torch_cpu_stats.json)。

**A 与当前明确建立的事实。** 借入 A 是普通空间低秩 `3×1→1×3`：从真实 W8-QDQ 出发，首级 per-rank scale 在第二级系数中静态对齐，signed8×binary → Z15，Z15×signed13 → P32，最后接固定 I24。它确实区别于之前平坦 R8 的连续表示及几次位图控制调整。实现给出了可执行的窄整数合同，未把 W32 AEE 当作整数链质量；完整网络 AEE 尚由根代理执行。当前已量化的 B 是 factor 接口的状态/复读义务：两个 R8 stripe 各1280B Z、纵向8位置而输出4位置、Q2三tap缓存、首条带部分和物化、第二条带真实读回。不能把这些必要工作叫作已验证的异常性能瓶颈，需等周期账确认主导项。

**资源核验。** 核内只有8条32bit carry链；ZADD 的 bit15 断carry形成双signed15独立加法，MAC使用同链完整32bit。只有8个19×13乘法表达式，Z高/低字段正确符号扩展。Z读在 ZREAD/ZSCAN/BASE_MAC，写在 ZCLEAR/ZADD，不同周期；MAC只读所选rank bank。psum读在第二stripe POSLOAD或DRAIN_READ，写在STORE，不同周期；stripe0完整覆盖480行后才允许最终退休，避免旧tile残留。Q1与Q2权重在互斥状态读，未见偷加并发权重端口。source每stripe按C重新读取16个原始门字，图像外位置实际遮零，所有三个ky复用该已装载窗口；此处没有 TB 赠送 Z。以上是源码结构判断，尚待仿真端口断言和全输出覆盖。

报告除1920B source、20B native窗口、4608B Q1、7488B Q2、1280B Z、312B Q2 cache、15360B psum之外，须记 `q1_live[576]+q2_live[576]` 的 **144B静态支持元数据**（由已收费配置生成）及 `position_live[80]×8` 的 **80B动态最终Z支持**、Q1/Z/acc/output holds 和少量控制状态。256bit物理权重接口上的Q1/Q2有效载荷分别64/104bit，不应把payload宽度当独立免费接口。`source_allow/weight_allow`目前表达存储服务停顿；真实外部请求延迟、在途队列能力及新增消费者端口须以实际接线和计数为准，不能从 ready 门控推得系统带宽。

**还缺的同资源反事实。** 当前核本身是强 factor A，还没有新的融合 X，也没有同函数 direct-expanded-W 硬件分母。旧平坦 R8 的不同函数/质量/halo不能证明本函数加速。若下一步声称融合减少物化，应让 native factor 与候选共享同一 ALU、乘法器、Z/psum/权重容量及端口权限，并实付 stripe复读、Zscan、Q2cache装载、psum读回和 I24 末输出；给旧臂相同可借资源。若用较大Z保存全R16来消除source重读，应单独标出2560B资源点，并给两臂同权限。普通full-frame行缓冲可以改变tile重复halo读，但也需真实行状态预算，不能当无成本强分母。

**新颖性判断。** 当前实现的研究新颖性主观为 **2/10**：正确实现连续分解、rank尺度吸收、静态位宽界、分段加法与固定点consumer属于必要且有用的工程底座。尚没有足以提升标题的“故障适配后新机制”。可继续挖的接口是：空间分解产生的连续latent有纵/横不同寿命和consumer可证明精度需求，能否在严格同资源条件下改变保存/退休表示，降低已实测的状态或服务税且维持同环境质量。仅删除数组、增加stripe调度、复用旧ALU或拿到几个百分点都不自动构成创新；小百分比也不足以否定这个连续表示家族。
