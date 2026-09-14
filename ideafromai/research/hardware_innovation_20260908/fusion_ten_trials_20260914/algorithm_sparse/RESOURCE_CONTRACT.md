本次九个执行模式使用同一个综合顶层 `lossy_tile`，共同资源与控制权限相同。不是对既有 mode15 的等面积或等频率声明；新增编码器、原型表和 signed14 乘数口需要付硬件成本。本次仅报告周期，没有 EDA、Fmax、能耗或全层吞吐测量。

| 实体 | 物理组织/宽度 | 运行权限 |
|---|---|---|
| 原生源 | 1536×10bit 一读口；C1 的16×10bit本地窗 | 所有模式完整扫描C96×9tap；四P同T双半字更新 |
| Q1 | 8×864×3bit | 一次8lane共同K地址；整零K列跳过 |
| z | 8×20×26bit =520B | 单208bit向量读写；每字两P、同T，13bit间进位隔离 |
| Q2 | 8×96×16bit；8×8×16bit局部cache | 128bit权重字读；共同rank/v_live控制，og内驻留 |
| p | 8×480×32bit；8×32bit累加器 | 所有480字写回、读取、完整输出；时域复用只保留线性acc |
| 算术 | 8×32bit共用加减链，8×signed16×14乘法器 | Q1打包加、encoder差/绝对值/归约、Q2累计复用同8加法器；早先按通用±2592抽象界为AS2残差保守留14bit；固定Q1的实际残差≤667，13bit已经足够。本轮不改冻结RTL资源；不能拿旧19×13口视作免费同资源 |
| 原型codebook | 4×8×13bit =416bit寄存器 | 一组8lane组合读并接EDIFF；此小表有独立读口，全部模式都具备；不是借Q2外口多读 |
| Q2原型结果 | 8×48×32bit =12288bit | PROTO_READ单256bit字读，与Q1/Q2读互斥，服从weight_allow；code0已知整零合法跳过 |
| 编码暂存 | current/ref各8×13、difference8×14、work8×32 | 全部模式共同；参考向量每P的t0重新设置，命令不串状态 |
| 控制参数 | shift8×3、rank门两组8×32、τ×nz表9×32、time门32 | 配置一次；τ×nz查9项控制寄存器，未引入运行时乘法器 |
| 逐P/T元数据 | code40×2、rank40×3、residual40×14、refresh40、use_delta40 | 共840bit；signed13 Δ满足范围且变化rank更少时原位写z，否则完整重算，无额外全tile latent buffer |
| 选择逻辑 | nnz计数、signed13范围判断、8项最大值及rank优先选择、min-score比较 | 实际组合逻辑；在EREAD/ESCORE/EMAX/EWRITE等状态付周期。未测这些组合路径的时序 |
| 后继消费者 | 固定旧版i24_consumer原样；8×32bit乘法、8×64bit加法、8lane IEEE32→Q20转换 | 每个T/P/N值都实际消费其FP32 identity；所有模式每tile480转换、480乘法、960加法、480RNE，双输入ready/valid |

总新编码相关状态约15014bit（表、参数、metadata、暂存和encoder控制，不含通用计数器/FSM地址），其中12288bit为原型结果表。该近似清单不代表综合面积；完整实体以SV为准。consumer系数额外24×256bit常驻，原有后端buffer不重复算作本次新增。

冷配置每tile3450拍：source1536、origin1、Q1 864、Q2 96、k_live864、阈值13、codebook4、prototype48、consumer24。九模式配置同union；第二个连续命令配置0拍。本小fixture基准把每tile单独配置，因此主表另列计算总拍和“计算+一次配置”；不拿此冷启动数推整层。在完整wrapper复用静态参数可摊销，但本轮没有新全层仿真。

Q2 `BASE_MAC`仍采用异步z标量读取→cached Q2选择→乘法→8路加法的单状态；原型EDIFF是小codebook读→sub，ESCORE/EMAX为比较链。需要实现端时序验证才能把周期改称时间。输出背压时数据/地址稳定；TB每模式同pattern，两次命令不复位。
