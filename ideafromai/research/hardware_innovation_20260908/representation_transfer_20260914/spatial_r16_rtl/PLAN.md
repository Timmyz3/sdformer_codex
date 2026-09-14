# Spatial R16 完整整数因子 RTL

先写合同，2026-09-14。A 是普通3×1→1×3空间分解，固定q1 signed8[16,96,3]、q2 signed13[96,16,3]；无中间RNE，source theta已经吸入系数。不是新颖X。B为两R8条带需要复读source、Q2三tap缓存及p32跨条带归约，所有费用实测。

输入为原96×4×4低10bit门字及原点；每条带Q1遍历96C×3ky，从已实际读取的16字窗口产生2×4×T10的位置掩码。跳过全零Q1向量及空源事件；固定相邻x双P15，唯一8×32 carry链在bit15断开。实际Z为8bank×40word×32=1280B，逻辑2×15bit；完整最终Z扫描生成80×8bit支持后执行Q2。两条带重读source明确收费。

Q2每output group预取R8×3tap×N8的13bit系数到312B缓存，利用最终秩/位置支持和整向量零过滤；唯一8个signed19×13乘法表达式，Z15符号扩到19，系数13bit。每输出累加器32bit。原p_mem为8×480×32=15360B，stripe0写部分和、stripe1实际读后累加，最后3840值按og×P4×T10×N8原顺序输出。raw消费者背压保持数据/地址。

静态Q1 4608B、Q2 7488B；256bit配置/权重服务，Q1实际64bit向量、Q2实际104bit向量，片内请求受weight_allow控制。源1920B与native窗口20B。Z单授权读地址、全向量256bit或单bank32bit；p_mem单向量读或写。所有源读取、Q1/Q2权重读取、Z/psum读写、配置、启动及缓存活动计数。共享producer ALU/乘法不按条带或位置复制。

只接受固定模型满足signed15静态界的Q1，不声称任意INT8矩阵安全；读取固定导出给出的逐rank Z界及按Q2加权的p全prefix界。合成源全零/全一/角落/图外污染及稀疏末位，保持实际合法factors，不把不合法极值矩阵截断。所有3840p32独立对导出整数因子/展开卷积gold；Q1 stripe边界检查Z。先small与BP/无reset换源，后held128–191与4000–4063两组64，两遍连续命令。

第一阶段仅完整factor RTL；尚无同函数直接展开W硬件分母，不编CPU周期、不混用旧R8函数性能。raw过后才可接原FP32→J20→I24消费者，算法网络质量由root报告。无训练、量化调整、EDA、生产修改或Git提交；仅写本目录。

实施闭环：raw 与完整消费者均各 572 命令通过，详见 README / SUMMARY / verification。source 原点在 fixture `origin.hex` 已是 output_origin−1，配置时直接使用。完整 wrapper 中 kind6 配置 24 个 a/b N8 行，FP identity 由独立 256bit 请求/有效接口实际输入；原消费者的 8×32×32 乘法、8×64 宽加链与 FP32→Q20/RNE/饱和逻辑真实保留并列入资源。模型配置后支持同模型跨源连续 start，不支持重叠命令/在线模型失效。独立同函数 expanded-W32 raw 对照已在 sibling `spatial_r16_direct` 完成，替代初始“无硬件分母”状态；不推断未运行的 direct I24 端点。
