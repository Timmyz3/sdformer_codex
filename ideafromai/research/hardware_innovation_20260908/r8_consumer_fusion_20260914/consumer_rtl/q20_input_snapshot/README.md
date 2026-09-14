# 已完成的 Q20 identity 入口收据

此处是加入 FP32→J20 硬件转换前的冻结代码与实跑结果。完整四模式每个单 go 连续19200tile，294912000个raw p与294912000个I24全通过；此数不含后来FP32入口主实验。identity输入是J20，不能用该周期代替最终FP32消费者结果。

原src/config/identity/结果接口均实际运行。完整模式6/7/9/11总周期分别1394460661/536906648/502422152/484105352；静态12504拍每job只一次，未分行重置。连续64包含背压与同实例warm，完整帧不含这两项。

冻结SV与TB可从父consumer_rtl工作目录以 `--Mdir obj_q20` 指向本目录的consumer_stream.sv/i24_consumer.sv/stream_tb.cpp、以及../fusion_rtl/r8_fusion.sv重新编译；TB输入为data/first_source_words.npy、identity_q20_full.npy、raw_p_full.npy、i24_new_full.npy以及父目录fixtures/real_0常量。使用Verilator4.028 `--cc --exe` 与make。归档的Python入口保留当时路径，不作为当前目录独立安装包；最终FP32入口的可复现实验从父README运行。
