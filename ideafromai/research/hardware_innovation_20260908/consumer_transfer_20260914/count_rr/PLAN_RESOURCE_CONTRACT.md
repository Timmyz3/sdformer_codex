2026-09-14，SV修改前的接口与共同资源合同。

本轮迁移已实测的正count8／psum复用mode20和固定T配对mode21，接到真实FP32 identity→J20→I24消费者及双context RR。保留原dualP13、证明有效三P10、四P低8／高5修复、consumer64借链及宽链组RR强控制。所有臂完整执行K864、P4、T10、R8、N96，raw原T顺序退休；不是把旧raw周期比例乘进consumer结果。

共同producer为唯一8条32bit carry chain和8个signed19×13乘法表达式；count更新请求这些链，退休乘法请求同8乘法器及链。退休需要逐lane的13bit乘数接口，普通Q2将原共同scalar广播，不新增乘法器；signed packed退休的乘积拆位及负数借位修正在同一条共享链完成。普通控制享有同一新增选择接口与状态预算。

共同z服务统一为416bit，每context仍8银行×10行×52bit=520B。候选两P更新只使用52bit字中的选定26bit字段，不能称半宽占用带来面积收益；每银行只有一个授权读地址。两context对source、共享权重、z、psum、producer ALU/乘法、borrow64请求原子grant，重叠资源由RR仲裁。计数读写也必须申请同一全局psum服务；C_ADD同时申请psum和producer ALU，不能获得免费计数ALU或端口。

source为每context1536×10bit=1920B；Q1为唯一2592B，Q2为唯一1536B，禁止按context复制权重。class为唯一4×864×6bit=2592B、代表为唯一8×32×3bit=96B、ngroups为6bit；这些静态表驻留top层，class和representative读取与Q1/Q2竞争同一个共享权重服务，不私设metadata读口。加载864+32+1拍实付，普通臂预留该容量但不强制加载不用内容。mode21共享唯一40bit固定排列，额外1拍配置；冷/暖及首次跨mode装载分开计数。

每context原8×480×32bit=15360B p_mem不增容，Q1阶段借前160行5120B存count8；八银行每拍至多一个读或写，允许原候选的逐bank class地址mux。count_live为每context4×160bit，另有class/count holding、group控制；共同union硬件显式保留这些状态，不能声称裁剪后等面积。两个context的输出所有权分别直到完整480beat被消费者接收，wrapper在该批全部I24退休后才重载或重启，禁止计数覆盖仍被消费者持有的psum。逆序只在完整Q2物化后的原psum单读地址上完成。

consumer沿用实际8×64主链和8个32×32乘法器、原FP32→J20及最终RNE/饱和边界。borrow控制真实竞争同一8×64主链，候选count本轮不默认额外借用。J20和I24与raw各逐值对既有fullgold，identity使用原FP32输入，不以预量化J20供数。

先小真实、全零/全一、padding poison、正负极值、255/256计数边界、逆映射与冷暖/跨mode，再两套64tile（128–191及4000–4063）有/无BP、同实例两遍。排列固定继承[8,2,6,3,9,4,1,5,7,0]，不再校准；后一集合与校准输入不重叠，但仍同帧。若实际完整consumer负，保留忠实臂并只对一个测得瓶颈适配一次。所有结果逐命令JSONL，资源grant/holding/所有权断言与算术独立模型核验。

原transfer_adapt与production只读；仅本目录实现及Verilator4.028 --cc --exe＋make。无EDA、训练、main.tex、hash或Git提交。普通RR、权重分组、正计数、布局/时间排序与功能单元共享本身不作新颖性主张。
