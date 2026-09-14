2026-09-14，SV实施前合同。

A为consumer_transfer/count_rr已有双context共同执行器，保留原生四P modular mode2、实际consumer64借链RR mode4和固定时间配对count mode21。B为bitmap_block_resident最终mode8的小位图／系数驻留布局，移入同一context和原完整FP32 identity→J20→I24消费者。先忠实移接，不沿用单context百分比。两套原64tile、换源、跨mode、冷暖与BP均逐raw/J20/I24验证。

共同producer只有top的一份8条32bit carry chain和8个signed19×13乘法器。B的八棵pop16也只在top一份，输入在ALU owner确定后选择；两context不能复制pop、数据加法器或乘法器。负权重plane使用同carry chain的补码减法。cache读、pop和ALU在同一授权周期完成，未测该组合路径Fmax。原consumer一份8×64主链及8×32×32乘法器，mode4真实共享宽链；B不额外借链。

全部臂共同416bit z服务。每context维持8bank×10row×52bit=520B；B的每活动P/T K16条带各占一次416bit读和完整416bit写，保存其余13bit字段，没有半字写口。W、z、psum、source、ALU仍原全局6bit请求的原子grant与RR，B无额外W/z许可；缓存有效读取占用本context原Q2 qblock单128bit读路径并绑定ALU授权。

source每context1920B、完整psum15360B、原qblock128B保持。B每context额外80B bitmap、5B bitmap-live、40bit pending、16bit holding、8×32bit条带累加holding和地址控制；qblock前三行48B仅Q1期间覆盖，Q2 VLOAD必须覆写全部八行后才消费。bitmap构造沿原40路bit写，是真实寄存器布局而非任意SRAM宏推断。

Q1 2592B、Q2 1536B、class2592B、rep96B、40bit排列在top各一份。B再有一份2592B plane系数和162bit plane-live；从原Q1加载拍实际构造，每拍额外24个plane bit写及live更新，存储和写扇出明确计入，静态服务仍原1848拍。运行时每非空K16的三拍预取槽付费，有效plane争用原W mux并受weight BP；空plane槽也占拍但不访问权重。class及排列沿原首次模式加载897+1拍，所有臂保留同表容量，不作裁剪等面积主张。

source在当前批全部I24完成后才回收、重载或启动下一批；count对psum的覆盖和bitmap对qblock的覆盖按原阶段所有权。原始raw全物化，count21内部排列在原输出单读地址逆映射，B输出原P/T/N顺序。无新中间舍入、近似或CPU工作量替代RTL周期。

先small真实／全零全一／−4／poison／FP舍入／跨mode与双context，再跨row和两套64（128–191、4000–4063）冷暖/BP。同模型静态驻留，换模型需reset／完整reload；两组仍同帧。若共享接口暴露负瓶颈，仅按实测做一次有界适配并保留B原臂，不扫参。普通稀疏枚举、位平面、驻留和RR不称新算法。

仅写representation_transfer_20260914/bitmap_rr。父树、生产与主稿只读。Python /opt/anaconda3/bin/python3.12，Verilator4.028 --cc --exe后make；无EDA／训练／main.tex／hash／Git提交。结果逐命令JSONL。
