# 从附加模块转向独立硬件子系统：本轮机会实测

2026-09-05。遵循最新方向：不以两周工程时间淘汰研究；保留 frozen
模型、精确计算、真实宏和公平PPA。**本轮没有改TeX、没有启动新EDA**，
未训练、未修改checkpoint。既有任务只收取已经完成的新思结果。

## 结论先行

压缩存储与模式复用仍是两条备选主线，不预先宣布任何一条立项成功。
融合累加器降为实现/消融。此次不是仅改优先级：新增三份可运行CPU
筛查器，测完32个权重张量、45,626,112个INT8 codeword，并重放
4320个冷G48块的FC权重需求。

最值得继续的问题是：**可随机寻址的小包怎样在减少存储的同时，不增加
128-bit宏读取和无用解码？** 不是“给现有TSBG串一个EBPC”。

## 1. 数据身份：不能把FC候选写成冻结无损模型

- 8个Conv/decoder：直接读取M2042已有INT8数组，未重新量化。
- 24个FC：按M2251已使用的逐输出、2的幂scale、RNE、[-127,127]
  规则从ep34生成完整矩阵；三份既有功耗tile逐code对上。
- 压缩/解压精确性针对既定INT8字节数组；**FC全模型AEE仍未准入**。
- 原始零比例在未补齐数组上统计；decoder的770/386/194源通道导致
  bank尾部补齐，不能计成真实weight sparsity。

| 权重族 | 原始字节 | 零比例 | 逐层加权边际熵 (bit/code) |
|---|---:|---:|---:|
| C1四Conv | 21,233,664 | 2.330% | 6.2344 |
| decoder四层 | 7,140,096 | 3.842% | 5.8450 |
| FC1候选 | 8,626,176 | 1.678% | 6.7771 |
| FC2候选 | 8,626,176 | 1.576% | 6.8162 |

边际熵是一个统计参考，不是任意有结构编码的普适下界。

## 2. 小块直接打包：容量与读取要分开

M2262比较有符号位宽打包、min+unsigned offset；逐块选择较短者，
保留8-bit原始路径。不是EBPC复现。每块4-byte可随机寻址目录；
16/32/64/128值，8个bank独立编码。

同时测source-major和**output-tile-major**，后者与96输出的当前执行粒度
更贴近。不能在全Cout source-major布置后只读tile0，拿人为地址空隙
作为新颖性。64/128粒度已随布局重新编码，不仅修改访存地址。

tile-major的简单编码最好紧凑容量：Conv约−7.87%，decoder约−10.77%；
FC1/FC2全族仍约+1.85～1.87%，应允许raw bypass。容量包含目录，但
不等于宏面积：bank末尾物理行、宏深度取整、解码器尚未计入物理实现。

M2263进一步给raw和packed相同的4/16个128-bit字/每bank缓存预算，
**目录和payload共用缓存**，避免免费完美目录cache。真实FC请求：

| block values | 强制压缩读取/raw | 按权重容量选择raw fallback后的读取/raw |
|---|---:|---:|
| 16 | 1.3231 | 1.0000 |
| 32 | 1.2535 | 1.0000 |
| 64 | 1.2277 | 1.0349 |
| 128 | 1.1805 | 1.0095 |

两种cache容量本批相同结果；标签/控制面积未计，因此是同payload容量、
不是同面积。只报底层读次数，不换算cycle/energy。该表只覆盖FC候选，
不能外推到未重放访存的Conv/decoder。

独立复核确认4320块、24FC层、267,405次source需求，包含73,965次首
G48之外地址；FC2 continuation没有错误回到group0。独立序列化目录与
payload解码182,784 scalar零失配。复核指出并已补测cache合并、
tile-major和raw bypass；没有因第一版读放大而杀整条压缩研究。

## 3. 更强编码对照：restart Huffman

M2264采用普通canonical Huffman，逐层256-byte码长表；每32/96/384
个值独立重新开始，每包4-byte offset/mode目录，逐包raw fallback。
这是参考软件codec，**不是新发明或已实现高速decoder**。

| 权重族 | 96-value包紧凑字节减少 | 384-value包紧凑字节减少 |
|---|---:|---:|
| C1四Conv | 16.92% | 20.39% |
| decoder四层 | 22.00% | 25.43% |
| FC1候选 | 10.37% | 13.84% |
| FC2候选 | 9.83% | 13.29% |

96值恰好对应单bank/单source/单96-output-tile需要的数据，没有额外解码
别的source。无包复用敏感性下，FC读次数（含目录）仍为raw的
1.1738/1.1787。384值整包解码策略会处理4倍数据，但这不是不可避免的
下界：补上前缀解到需求末尾即停、raw块直接读所需子范围后，FC1/FC2
解码量降到2.5126/2.5081倍，含目录读取降到2.4551/2.4627倍。
仍是无已解码包复用的策略敏感性，不是硬件周期。

96值包又补了raw/packed两边相同4/16个128-bit字/每bank的LRU缓存，
目录与payload竞争同一预算，沿4320个真实FC冷G48块重放：

| 每bank字缓存 | raw宏读 | packed宏读（含目录） | packed/raw |
|---|---:|---:|---:|
| 4 | 1,604,430 | 1,818,652 | 1.1335 |
| 16 | 1,604,430 | 1,730,101 | 1.0783 |

packed的数据读为1,551,247次，单看payload少3.315%，但目录读使总量
仍增加。不能把目录免费放寄存器后只报数据收益；另一方面，也不能拿最初
无缓存结果判定所有压缩存储无效。256-byte码长表只是序列化格式，未给
八bank并行解码表、标签或控制电路定价。这些字节和读次数均非PPA。

这提供了继续设计子系统的依据：**内容有压缩余量，但存储粒度与访问
粒度冲突真实存在。** 下一步优先研究分级索引、按消费者需求划分独立
解码包、包内restart/seek与有限缓冲协同；必须对普通缓存/同粒度
codec消融，不把改布局本身另命名成新贡献。

## 4. 模式复用备选不是从零开始

旧M70/M72/M76已经做过Phi式精确有符号残差。M76旧内部开发集上，
独立模拟比bit sparse为1.5031×（96B weight/144B PWP双宽接口），
共用96B口后为1.2136×；它不是当前ep34、不是同面积、不是RTL。
M351已有PWP DMA收费纠正；M361r4已有train-only宽分区catalog。

重新立项必须回答新问题：在同一权重tile寿命内，什么时候建表、保留、
重算比现有parent/zero更便宜？固定模式、在线模式、parent三者同账，
负残差、建表/换tile、索引、1RW争用、输出bank倍数全部计入。
固定模式不能用evaluation样本既校准又报泛化；在线表只用已到达数据。
不因它需要更久而淘汰，但不把旧PWP表缩小改名当新的数学机制。

## 5. 查明的近邻边界

- [EBPC, JETCAS 2019](https://arxiv.org/html/1908.11645v2)：特征图压缩，
  有[SystemVerilog](https://github.com/pulp-platform/stream-ebpc)，
  原设计约1个8-bit word/cycle，不能无成本替代本项目每bank16值的通道。
- [BPC, ISCA 2016](https://lph.ece.utexas.edu/merez/uploads/MattanErez/isca2016_bpc.pdf)
  已把低延迟、细粒度随机访存作为问题；“压缩与随机读取冲突”本身不是新发现。
- [2026 lossless LLM tile-ANS](https://arxiv.org/html/2606.15789v1)
  已讨论独立tile子流、offset目录、解码与GEMM布局耦合；这轮未确认其完整
  自有ASIC RTL，不能写成已可移植的宏级实现。
- [Unweight公开CUDA](https://github.com/cloudflareresearch/unweight-kernels)
  主要利用BF16指数冗余并融合重构矩阵乘；不能照搬其压缩率到INT8。
- [Phi](https://arxiv.org/html/2505.10909v1) §3.3明确区分无损与PAFT；
  §5.5.2明确额外PWP读取。有限表仍须正面对照它的存储与预取设计。

## 6. 既有硬件收尾没有停

group-demand / consumer-union cofill最终logic area分别
190,949.218549 / 191,635.036537µm²，增量0.3592%。3ns原约束下
setup分别+306.406/+5.909ps，hold +0.052/0.000ps（报告舍入）。
mapped→mapped Formality两轴各77,247点PASS；新pair功耗未完成，
不得借用M2018那组能量，也不是布线/CTS签核。

研究排序现在按信息增益与独立贡献潜力，不按9.20墙钟：压缩子系统优先
继续测，模式引擎保留对照，任务重组配合访问布局，融合加法只作配套。
**没有候选因此自动成为论文主线，更没有Strong Accept保证。**

复现入口：`system_simulator/scripts/m2262_sparse_weight_compression_screen.py`
（`--layout tile-major`）、`m2263_compressed_row_cache_probe.py`
（同布局参数）、`m2264_restart_huffman_screen.py`。用pytorch310_cpu环境。
结果分别在同编号`results/`目录。本次分析校验用于区分编码容量、逻辑读次数、
真实宏/PPA三种口径；不新增审批合同或反复哈希。
