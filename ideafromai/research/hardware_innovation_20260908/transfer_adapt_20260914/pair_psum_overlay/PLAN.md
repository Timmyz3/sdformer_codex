# Count8 / psum阶段复用：实现前合同

2026-09-14。本目录独占；pair_sparse原源码/对照/记录只读，不复制原始归档或计算hash。无训练、EDA、生产修改、commit。沿用AT-LIF {0,theta}吸W后的二值输入、原Q1/Q2、R8/K864/N96/P4/T10与全3840 raw输出，不接I24/RR。

B是原R2字典额外6400B计数SRAM及160bit计数服务的低利用。候选mode20由静态class频次证明计数<=255，复用既有p_mem的Q1空闲期；同模块mode14保留原native双P13直接执行。旧mode19在原模块/旧metadata独立重跑参考，明确其不同存储/端口，不能叫已证同面积。当前分母仍是208bit z服务；520B/416bit z的三P/四P强执行未纳入这个分母，不扩本轮范围。

## 静态与物理合同

四个R2组，每组最多32个非零重复键，仅频次2..255可分配class1..32。超过255或未选中的非零键用direct63，零键0；无运行时截断、丢弃或计数饱和。static编译依Q1，与source/gold无关；完整代表32拍+class864拍+尺寸1拍，额外897配置拍与原字典一致。直接mode14不加载metadata。

p_mem仍为8bank×480row×32bit=15360B。Q1计数占每bank前160行，共5120B：row=class_slot*5+time_pair，bank=2*rank_group+time_in_pair，四个byte分别是P0/P1/P2/P3计数。每拍八条32bit ALU在bit8/16/24切carry，更新四P×双T。独立count_mem必须从候选模块消失。现有psum256bit口服务计数，替代旧独立160bit计数口；每bank至多一次单地址读或写，四组class不同地址需真实选择mux，不算免费端口或增加同时访问。

640bit count_live每命令启动清零；未触行返回0且不读p_mem，首C_ADD写入并置有效。count_hold变为8×32bit，披露比旧8×20bit多12B。完整计数退休结束之后才进入ZSCAN/Q2，Q2 STORE覆盖全部480行，然后DRAIN全输出。每命令重新开始有效位生命周期，不依赖清空p_mem或保留上个tile。

退休按一个time_pair的八个(P,T)位置支持执行；每次取一个空间P-pair，沿用原19×13乘法器将a+2048b乘signed3 q，a<=255且b<=127才双P退休，否则标量两位置。q<0且a!=0的高字段借位在已有13bit carry-in修正。四P通过两个P-pair依次退休，没有额外乘法器或z端口。

## 验证顺序与停线

重编原20fixture的metadata，保持所有原source/Q1/Q2/gold；另构造255/256类频次混合边界验证group与direct共存。独立从native source卷积坐标重建z/raw及每class计数、首次触及、单银行访问、退休数和状态服务公式。实际RTL逐输出比较，含零/密/负4/padding污染/source-weight-output背压/无reset重复，以及不同源与权重之间重配置。检查alias地址范围、每银行读写互斥、Q2后再无count访问与全部输出覆盖。

只有20+边界小测通过且真实八块正收益，才用pair_sparse已有tile128..191 source/gold跑64连续真实流。静态Q1/Q2/class一次配置、每tile1537拍source/origin、start拍全部记费；两轮无reset。旧八fixture与64捕获real_6身份差异分开报告，不拼倍率。机会统计68631→52595仅用于提出假设，不当RTL周期。

本轮新颖性保持3/10待证；计数位宽证明、阶段buffer复用和carry分段分别有先验，不能以aliasing首次提出命名。报告资源变化、计数/退休循环因果、功能与服务边界即可。
