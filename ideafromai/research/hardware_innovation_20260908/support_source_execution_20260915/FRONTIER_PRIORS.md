# 沿本轮失配继续：生产粒度不是算法定律

本轮已经写成RTL并有负/薄结果：后端响应相等并不自动减少源最近码投影；按源任务选约束修正了这一点但只有约2.5%的源核增量；完整链增量更小。下一步应检查前轮自定的“一次channel必须产出T10所有门”是否仍是必要执行单位。

## 本网确切接口

`Student.source` 对每个输出t独立计算 `h[t,c]=sum_s A[t,s]*X[s,c]+bias[t]-center[t]`，再作门比较和最近码投影。A虽非因果满矩阵，**不同输出t并不互相依赖**；每个被请求的t仍必须保留完整十项输入，不能截断到过去s。

上游 `MS_Spiking_Mlp.forward` 的顺序是 sn1→drop1→fc1，推理eval关闭dropout。forced student部署hook只输出投影门×θ，原raw gate和soft没有应用消费者。更外层残差消费的是进入MLP前的x，并非这里raw gate/U。所以可以只算当前code决策路径实际查询的(t,c)，前提是完整等价投影仍完成，且不再把未算的U冒充有效值。

## 与其它领域相同、不同的部分

| A及可读出处 | 已解决的问题 | 本例还需实测的适配 |
|---|---|---|
| [LUT-DLA，HPCA2025](https://arxiv.org/abs/2501.10658) | 向量量化→查表累加，配套编码与硬件协同 | 此处被编码的位不是免费输入，而是另一个连续T10 PSN的输出；先生产全部门再编码是否过量 |
| [conifer FPU，作者开源](https://github.com/thesps/conifer) | 多个Tree Engine并行导航各自决策树 | 这里每次分支条件需要真实10项MAC，并且不同t的条件共享channel输入；不能把特征值已驻留的树硬件性能照搬 |
| [Traversal Caches，IJRC2010](https://doi.org/10.1155/2010/652620) | 共享指针结构的并行遍历、缓存与应用生成器分离 | 前沿遍历、缓存复用和应用kernel分离本身已有明确先验；本例额外付非因果T10源MAC和部分时间完成接口 |
| [SilvanForge，SOSP2024作者全文](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/sosp24-final168.pdf) | 调度/布局驱动树推理，不同批量和硬件的复用取舍 | 不把改变树顺序、节点压缩或按cache安排工作称新颖；检查减少的源算术是否被新的输入重读吞掉 |
| [GustavSNN，HPCA2026官方](https://2026.hpca-conf.org/details/hpca-2026-main-conference/83/GustavSNN-Unleashing-the-Power-of-Gustavson-s-Algorithm-on-SNN-Acceleration-with-Col) | 将SNN稀疏乘法按列并行和时间批处理组织 | 借它“改变执行单位并重做局部状态”的方法，不声称本例决策图就是Gustavson/完整GustavSNN |

以上是有界相关检索，不是穷尽先验；RF/树处理硬件可借但不是未存在的领域。一般按需特征、树并行、缓存优先调度均应归A。候选X仅是**共享源码与不同时间门需求分离后，有限holding下的部分生产/重用能否形成净服务**，需要在普通code与class两臂共同授权后才判断。

## 已完成的固定两步

第一版最多取四个不同前沿channel，用10个MAC各算所需的t；128B X holding、128B图cache、同8bank，一次bank一笔在途。相比当前32B源码holding增加96B，必须给普通基线同容量。

CPU机会已显示风险：code批次3359→3048，实际标量MAC335900→267920，但X取数6718→8282词。省计算引入23.28%额外X取数，不能把MAC降幅写成加速。这版已实际跑RTL；扩展后BP相对static几乎没有收益，见[结果](frontier_source/README.md)。

后续适配保持相同容量：四个X槽保留channel标签，优先消费驻留channel对应的合法前沿，缺额再补新channel；替换不得覆盖当前批引用的槽。源请求、重读、图请求、部分U有效位和最后码握手都计费。已扩展32训练帧并接完整FC1/PSN，[真实整链](frontier_joined/RESULTS.md)证实普通code仍解释主要收益，class额外只有约0.3%。因此驻留作为有效借入底座保留，不以换接口命名替代创新证明。

进一步发现PF仍只针对旧最小rank通道，已在[独立目录](frontier_source/active_prefetch/PLAN.md)改为当前active前沿并实测。它针对供数失配，且普通code同享；没有借此宣布原创或整网加速。
