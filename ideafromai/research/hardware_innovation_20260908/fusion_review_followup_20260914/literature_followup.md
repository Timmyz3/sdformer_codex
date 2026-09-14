本轮文献只围绕已经写出的硬件及其最近邻，不把检索数量当新颖性。下列“未复现”指作者完整实现；本轮自己的RTL与作者成果始终分开。

|先验|本轮读取深度|对当前融合的约束与用法|
|---|---|---|
|ULPPACK，MLSys2022|官方12页PDF，重点§3.2–3.3、Algorithm2|已有子字打包、无溢出条件及局部累加后提取。新试验不能以“窄域装进宽ALU”作贡献；我方测的是二值源选择的四空间位置加法与有符号高位义务，同原生Q1/Q2消费者闭合。没有移植作者SIMD库或复现其CPU速度。|
|Overflow-Aware Partial Sum Management，ISCAS2025|官方摘要及会议条目；本轮未得到全文|已经有窄本地psum将溢出时合入output buffer的机制。直接打掉“检测溢出后精确补偿”是新思想的说法；我方尚可检验的差别收窄到四P分段执行、共享ALU修复、一次规范化及强RR仲裁；这些差别是否足够新仍待证。|
|A2Q，作者预印本2308.13504|作者摘要|累加器位宽约束训练与溢出保证已有。真实Q1的静态正负界及三P10是应给对照的普通优化。本轮未训练A2Q，不把配置期求界写成新标题。|
|PQS，作者预印本2504.09064|作者HTML全文，重点方法段|N:M剪枝、量化和排序联合减少累加位宽已有。本轮不改变源顺序/模型，不能借它的有损质量或压缩率。未实现其完整训练/排序；若以后引入，必须和精确修复分开比较。|
|GKPD，AAAI2022|正式论文页面/摘要|Kronecker卷积压缩已有；本轮固定96×8算子不是新分解数学。实际8lane中间S布局与有界完整输出组回退只是执行候选。未复现作者完整训练。|
|Hybrid KPRNN，作者1906.02876v5|作者HTML全文，Algorithm1、§3.4/Algorithm3|已有两因子快速矩阵向量乘，也有未压缩行和Kronecker行混合。因此“Kronecker+少量精确行”不能当新增机制。本轮只能争取固定整数/输出组调度差异；实际十帧失败，不凭4倍参数故事恢复标题。|

来源：[ULPPACK官方PDF](https://proceedings.mlsys.org/paper_files/paper/2022/file/e09d45e14e9ece7142217550ddd3c4d0-Paper.pdf)，[ISCAS官方条目](https://epapers2.org/iscas2025/ESR/paper_details.php?paper_id=1103)，[A2Q作者页面](https://arxiv.org/abs/2308.13504)，[PQS作者全文](https://arxiv.org/html/2504.09064v1)，[GKPD正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/19958)，[Hybrid KPRNN全文](https://arxiv.org/html/1906.02876v5)。

对新颖性的结论是审阅判断：modular/强RR值得完成同资源实测，但仍没有理由宣称首创或强接收；Kronecker单项+两组escape既有强近邻又质量不合格，停止当前端点。把作者方法名称挂在本地变体上不等于“抄全”；现有共同底座完整性由明确的数据接口和硬件费用判定，不由引用多少篇判定。
