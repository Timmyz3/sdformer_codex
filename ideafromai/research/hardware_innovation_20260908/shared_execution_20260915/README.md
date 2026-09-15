本轮完成两项实测：把空间phase3候选接到真正共享执行资源的双context完整消费者链；把Claude T5证书迁到付费原始字/阈值供数接口。结论是空间当前端点在强R8分母下显著变慢，证书保留局部正收益，但两者均未形成TCAS-II强接收证据。没有改生产RTL、主稿或已有模型，没有进行EDA、GPU训练或新增valid825。

| 本轮对象 | 实际推进 | 结论 |
|---|---|---|
| 空间自由U3M/phase3 | 单实例共享ALU/乘法/W/Z/psum、双context、唯一FP32 identity→J20→wide64→I24、真实借链仲裁 | 两64集合及18seq36tile上比最佳R8慢74.08–93.62%，当前布局退出性能主线 |
| R8强对照 | native、宽链借用、count、bitmap四臂统一装载/原点协议；给予相同容量许可；重算跨序列gold | 96命令、2944次tile执行；各raw/J/wide/I24共11,304,960值通过；bitmap保持最强已测控制 |
| Claude T5付费迁移 | RTL生成指数/位平面、64bit唯一在途、实际Y/τ读取、可背压退休；删掉FX分母重复符号位 | 相对同电路BF full减少22.66–23.88%周期，背压下12.35–13.36%；外部字节不减 |
| Claude T5/T6独评 | 重编原核定向检查、原320万判决存档复核、反例和资源核查 | 保留合法域门证书；不接受部署BN已闭合、17%完整服务及T6全包络外推 |

完整可比较表在 [COMPARISON.md](COMPARISON.md)，原始逐命令收据在 [spatial_rr](spatial_rr/README.md) 和 [r8_reference](r8_reference/README.md)。空间small15与R8small12不同，未比较小集合计。三主集合source、FP32 identity和物理origin已同序逐字核对；输出由各自不同学生函数独立产生。二者使用共同容量/口/算术数量与消费者合同，进位切分和其他控制不同，不是等面积或等频率网表。

空间失败有具体计算原因。held64上自由U3M的Q2发射634,068次，R8 bitmap为198,720次，约3.19倍；disjoint与跨序列同样约3.1倍。空间分解把一部分原来在二值源侧完成的邻域组合放到了连续Q2侧，3M只相对自己的四乘结构省了一乘，未抵消相对R8的连续工作增长。借宽链held仅减575拍，无法消除这项差距。这个结论来自实际RTL和完整I24退休，不再用260→159节点数、MAC代理或单context自比代替。

已有网络质量另列：自由U3M valid825 AEE为1.258343102157，flatR8为1.327635022608，matched NB0为1.447936665574。它们来自上一阶段相同评估协议的[完整质量报告](../representation_transfer_20260914/quality/QUALITY_REPORT.md)，本轮没有重训或重跑825。自由U3M质量更好、执行更慢，不能把两种函数相除称作无损加速。native_tap此前质量和单context结果仍保留，尚未在本轮共同双context上测；本轮也没有穷尽低秩或空间分解家族。

Claude部分的完整报告在 [claude_review/README.md](claude_review/README.md)，实际迁移在 [cert_transport/README.md](cert_transport/README.md)、[RESULTS.md](cert_transport/RESULTS.md)。关键修正：

- 9.24%相对位深代理不进入我方服务表。四份T5的约17%原结果在其预生成输入和已知阈值的局部门核协议下成立。
- 原脚本从当前完整Y计算动态BN矩和τ，没有证明是训练期冻结常量。T6五点扰动仅证明样点敏感性，不能证明任何部署阈值、阈值供数或网络AEE。
- 负gamma转换与signed49→48保存存在可复现边界错误；四份已测trace均未触发，不能因此抹掉它们的正gamma结果。
- 新接口从原始signed24字实际生成元数据并付每组5个源传输词、10个阈值词。证书节省的是门核后半服务，所有模式仍读完整Y；不能当上游FC1、动态BN、连续消费者或整网收益。

独立评审暂将现有证书算法组合评2/10、系统接口评3/10，数字是技术判断而非接收概率；“有效位宽+精确终止”本身不足作标题。空间相对自己的普通因子化/Winograd结果也没有自动晋级为新颖性。保留可复用底座和负结果，按用户要求继续“先完整借入A，找到不适配的具体原因，再试一次接口改动”，后续工单见 [NEXT.md](NEXT.md)。

复核入口：`compare.py`合并冷服务表；`r8_reference/verify.py`与`cross_inputs.py`、`spatial_rr/verify.py`核独立值与服务账；`claude_review/check_cert_transport_billing.py`核64行供数恒等式。构建和原始数据路径见各目录。数据/生成fixture及构建目录不入Git；源码、参数、摘要、计费记录及审阅保留。没有使用哈希封存流程。
