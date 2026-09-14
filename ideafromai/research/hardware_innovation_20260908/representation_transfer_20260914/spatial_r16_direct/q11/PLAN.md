# Q11 同函数 OS → 原 FP32 identity → I24 完整分母

复用已测OS core，不修改微机制。系数唯一来源 `../../spatial_winograd_inputs/factors.npz`，直接expanded-W32与该Q11 ordinary/Winograd factor为同一函数；不继承Q13 AEE。重新检查原始spike任意prefix32界，按新aQ40/bQ20重算所有raw/J/wide/I24 gold。输入保留实际原始source与FP32identity，严禁向DUT喂连续中间值。

在本目录保存Q11系数/derived gold/结果，Q13全部结果不覆盖。OS物理1280B bitmap、1920B源、15360B p、8×32ALU、单256bitW读与既有实现相同；完整复用factor分支的i24_consumer与wide_phase_alu，consumer资源为原8×32×32乘法/8×64宽链、单context、原始identity和ordered480行端口。producer保留原raw背压，最后3840 I24与J/wide都实际比较；staticW10368+consumer24拍，source1536+origin1+start1逐tile收费。

先small15/readyBP/不同tile无reset两遍，再两64；根提供的18序列×2tile live集合就绪后同协议回放。若共用wrapper可直接闭合Q13，则同时补Q13同端点结果，不改变原raw账。无新量化/训练/GPU/EDA/生产修改/Git。
