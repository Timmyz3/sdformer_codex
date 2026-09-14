2026-09-14，controller实施前安排。只读spatial_r16_rtl和已冻结spatial_winograd_inputs；普通mode0与Winograd mode1共同使用q1 signed8／q2 signed11的同一函数。Q1保留两个R8 stripe、原theta-gate AAC、双P15与原始Z检查。mode1只改变连续1×3 Q2，不新增latent RNE。

每stripe先完成原Q1和40次原Zscan。随后20个(y,T)分别读取[d0,d1]和[d2,d3]两个256bit Z字；前字用原z_hold，后字新增32B holding。四拍依次用唯一八条32bit ALU算d0−d2、d1+d2、d2−d1、d1−d3，原acc临时保存前半，两个实际整字写把原15×2格式改为16×2。Z仍8bank×40×32=1280B，不加transform数组。D支持在实际ALU结果拍产生，覆盖原position_live；完成本stripe变换后才开始Q2。

Q2固定四tap变换权重来自冻结模型常量[2g0,g0+g1+g2,g0−g1+g2,2g2]，范围±2981。只保留一份8×768×13bit静态Q2表（9984B）：mode0装原g的576向量、mode1装变换系数768向量；不同时保留原g副本。Q1配置576向量，两臂原始配置总1152／1344拍。换布局须经同配置口重载Q2，不能用mode切换免费改变权重表示。额外静态Q2容量2496B、配置192拍明确计入；普通臂享同容量但未用尾部。

每output group的qcache从24→32个13bit×8lane向量，312→416B，额外104B共同预留。保留有效rank／向量支持过滤及W背压；单次W读仍一个104bit向量。输入D16符号扩到19、系数13，唯一8个19×13乘法器逐rank／component累加4个M。M0复用原acc，其余三组额外96B；无rank或component复制ALU／乘法。

每(y,T)pair完成M后，用同8ALU各两拍恢复M0+M1+M2与M1−M2−M3；每个最终和必须偶数，精确算术右移1，无RNE。原32bit前缀界和新M／重建界均检查。stripe0写两个p32，stripe1逐输出实际读原psum、再付一拍同ALU累加后写回。读回部分和借用此时闲置的原z_hold，四M在y1恢复完前保持。p_mem仍15360B，最终3840 raw按原og/P/T/N顺序退休。

原source1920B／native窗口20B、Z1280B、完整psum15360B、8ALU／8mult相同；新+32B变换尾字holding、+96B M及+104B cache、+2496B静态Q2单列。原holding在变换/Q2归约阶段按生命周期复用，读／加／写均有状态和计数。普通臂允许同union额外holding容量，不声称裁剪后绝对等面积。cache、变换和重建组合路径未做Fmax／能耗验证。

静态q2_live同表从576→768项，额外192bit=24B；qcache的block_live及remaining各从24→32bit，各额外1B，组合next_support也扩为32bit。Q1的576项支持保持原容量，不能共用旧Q2支持长度。

先同函数ordinary与Winograd raw small、无reset换源、short和两64（128–191、4000–4063）有／无BP；全raw与原Q1 Z逐值检查、D/M/偶数重建义务独立核对，再接实际FP32→J20→I24 wrapper。q11整网质量由root独立测，不继承q13 AEE。亏也完成一次真实RTL测量再评，不扫bit／rank／接口，不加第三算法。仅本目录写，Python3.12、Verilator4.028 --cc --exe＋make；无EDA、生产、训练、main.tex或Git提交。
