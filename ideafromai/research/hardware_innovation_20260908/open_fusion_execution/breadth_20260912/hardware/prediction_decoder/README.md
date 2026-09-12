# Dg+c固定5+5解码器：实际新参数结果

**30例真实载荷执行全部通过，但当前查表放置比完整缓存条件加慢。** 已补普通常量传播：第二半全零行不取表，第一半pattern0=c通过实际CR读取/packed24解包后留在RF32/33。报告只用补齐后的结果，不用未优化表版来否决思想。

使用新train-only `representation/parameters/{ordinary,lifting_raw}.npz` 的full_g_q8_D/c，未使用旧验证拟合D；没有额外coefficient exponent。两个父臂均是current R24+onepass，输入为真实四窗4×4锚点、C96的projection-g。每轴穷举1024个T10门模式，完整Dg+c和两半表逐值相等，全部半表/总预测都在signed24内；因此这里只需整数加，没有隐去必要饱和。真实四窗加一组固定压力共460,800个预测值0差。

| 相同fullD函数 | 缓存条件加 | 固定5+5 | 服务变化 | 非默认实际查表行数 |
|---|---:|---:|---:|---:|
| ordinary corner ready | 13,204 | 16,382 | +24.07% | 208 |
| ordinary interior ready | 14,014 | 17,932 | +27.96% | 286 |
| lifting_raw corner ready | 12,794 | 15,600 | +21.93% | 170 |
| lifting_raw interior ready | 13,844 | 17,632 | +27.36% | 271 |
| ordinary interior 压力 | 15,045 | 19,142 | +27.23% | 286 |

条件加将完整c和非零D的352B载荷冷填，并通过真实CR/packed24解码缓存最多14个既有RF；每列付费选择g mask，只对实际有激活lane的非零列执行加法。center/affine/diagonal给同缓存与静态零项权限，三种c-only控制ready为7,694槽，diagonal为11,141–11,263槽；它们是不同预测函数，不能拿其快慢当fullD同函数优势。

表方案固定两个32×32B表，合计2,048B，包括每行10个signed24和2B padding，c折第一半。8个通道的半门码可能不同，按实际H8相等索引合并；每个剩余真实索引要CR256响应、两个解包RF和10个masked-add，不能写成每H8只有两次取表。虽然默认传播已去掉大量请求，ready仍有170–286次非默认逻辑查表；共同单字响应缓存将实际CR读降到123–245次（包括第一次c读取）。顺序条件加只冷读352B常量，稀疏g进一步减少它的实际加法。

每例gate输入已在SRAM，D相关臂实际读3,072B，c-only不读不使用的g；所有臂真实写46,080B预测。这是**共同的稠密预测出口接口**，不含其生成、量化编码、残差码的step乘法、PED，也没有将center默认值的未来消费者强迫物化作为全链分母。常量cold fill、所有SR/CR/RF解码/控制/写回均收费，预算仍96×8×48 RF、128KiB state/coef及同SR64/SW64/CR256。控制mask是付费RF读得到的8bit控制寄存器，条件加只读acc＋常量两个RF，未开第三口。

只停止“CR驻留表＋两RF行解包”这一放置的加速主张。随后唯一追加的跨行紧凑RF替代也已实跑：完整640个scalar占80个RF向量，总91/96活RF，不新增gather，但按真实不同索引串行且逐scalar收费后仍比条件加慢44.16%–49.82%。这停止第二个具体放置的性能主张，不杀5+5/查表家族，更不宣称LUT-DLA整体无效。详见[紧凑RF完整结果](COMPACT_RESULTS.md)。A是分表查表的完整Dg解码迁移；这里尚无独占X。

代码[execute.py](execute.py)，[ready原始费用](ready.json)，[固定压力](stress.json)，[摘要](summary.json)。CPU payload slot prototype，非RTL/PPA/AEE或整链证据；表值的新训练集拟合由根代理完成，本代理没有训练或调用GPU。
