# 双P窄latent打包：实现前合同

本点只改变完整R8第一因子的状态表示与更新粒度。原函数仍为 `z[p,t,r]=Σk Q1[r,k]g[p,t,k]`、`p[n,p,t]=Σr Q2[n,r]z[p,t,r]`，C=N=96、K=864、T=10、R=8、4×4原生源产生2×2输出。没有中间RNE，最终raw p保持signed32。Q1∈[-3,3]，故z∈[-2592,2592]，signed13足够；Q2为signed16。

B是同k、同t的相邻两个输出位置经常同时活动，逐位置更新浪费两个窄字段之间已经付出的32位ALU宽度。完整A是低秩、小整数第一因子、SIMD分段加法、原生源窗复用及完整R8 Q2缓存的OS。可能X仅是本网络窄latent允许共同源义务与双位置状态事务合并；普通carry切分本身不新。

固定mode14为scalar-packed强控制、mode15为dual-P-packed。两侧均有8bank×20row×26bit（4160bit）z，208bit向量读/写口与单bank26bit标量读口。两侧均有8条32bit加法链且ZADD时允许bit13断carry；第二级完整32bit累加。第一因子用同一个读→ALU→写状态机；14每次只更新一半，15按两侧source mask更新一半或两半。不能拿旧13bit端口的mode12当同面积分母。

两侧共同先为每Cin载入16个T10源字到160bit局部窗，再依tap gather四个P的源mask，支付1536字上限、边界判断及每tap gather拍数；Q1静态k_live在gather/权重请求之前共同跳过死k。第二级保持最终40×R支持及完整R8×N8 Q2驻留，跳过全零Q2向量，按位置OS并只写一次完整p。八个signed19×13乘法器与加法器、Q2块128B等权限相同；乘法和加法在同MAC周期，标量z读为组合reg/mux口，未做Fmax/物理SRAM映射。

输入/因子配置、源与权重许可背压、最终480个256bit输出和输出背压全部收费。TB只载原生源、固定Q1/Q2、k_live及整数gold，不计算动态z/p。回放旧14fixture，每侧双背压、两次无reset命令，共112runs。比较实际周期并独立复算字段更新数；不以操作数估算替代RTL结果。本点不训练、不新增布局/比例扫描、不承诺PPA。
