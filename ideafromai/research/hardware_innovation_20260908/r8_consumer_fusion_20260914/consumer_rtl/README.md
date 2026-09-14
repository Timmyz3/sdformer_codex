# 完整 R8 消费者 RTL

[REPORT.md](REPORT.md)给出真实数值边界、强对照、实测和限制；[resource_contract.json](resource_contract.json)列共同状态、端口和费用。[PLAN.md](PLAN.md)保存本轮冻结函数和范围。最终输入为真实binary source与IEEE FP32 identity，输出signed24/f14 I24。

当前生产者直接只读依赖[../fusion_rtl/r8_fusion.sv](../fusion_rtl/r8_fusion.sv)，不修改作者文件。模式6是同函数expanded21、7是R8 V驻留、9是最终z支持强A、11是原生C1窗口。`integer_factor.sv`仅保留旧叶逐字副本以便溯源，最终编译实例为r8_fusion。消费者 `i24_consumer.sv` 及跨tile `consumer_stream.sv` 为本目录实现。

在本目录运行，Python需NumPy（已验证 `/opt/anaconda3/bin/python3.12`），Verilator4.028和make：

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 generate_wrapper.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 verify.py
/opt/anaconda3/bin/python3.12 make_stream_tb.py
/opt/anaconda3/bin/python3.12 run_stream.py
/opt/anaconda3/bin/python3.12 run_stream.py --full
/opt/anaconda3/bin/python3.12 verify_stream.py
/opt/anaconda3/bin/python3.12 summarize.py
```

`run.py`实际使用 `verilator --cc --exe --Mdir obj_fp` 后 `make`；16fixtures、4mode、2背压、2同实例命令。`run_stream.py`编译同一SV到obj_stream_fp；固定64tile ids128…191，9/11、背压和warm；`--full`每模式单go连续19200tile，无分行重置、无完整帧背压/重复。参数12504拍冷装一次，新tile全部source/origin/identity和FP转换都重新付费。

输入来自 `../data/consumer_first8.npz`、`consumer_coefficients.npz`、`consumer_integer_first8.npz`；连续数据为 `first_source_words.npy`、`identity_fp32_full.npy`、`identity_q20_full.npy`、`raw_p_full.npy`、`i24_new_full.npy`。stream TB按SV实际source/tile/row地址响应，只读mmap；检查rawp/J/I24，不能将gold当机制输入。

结果见[results.json](results.json)、[results_64.json](results_64.json)、[results_full.json](results_full.json)、[SUMMARY.json](SUMMARY.json)、[checks.json](checks.json)、[stream_checks.json](stream_checks.json)，计数表为real8_costs.csv和stream_*_costs.csv。详细stdout/stderr在logs/。旧J20入口完整结果独立归档[q20_input_snapshot](q20_input_snapshot/README.md)，不代替最终FP32入口性能。

本轮无EDA、无生产文件改动。质量记录由[data](../data/)独立产生：消费者新函数须使用deployed记录，不能借r8旧浮点消费者或别的剪枝臂。源、identity和数值范围已对齐同首帧捕获；全网其余算子仍不是全网定点RTL。

独立[Root消费者审阅](../REVIEW_CONSUMER_ROOT.md)已核最终FP转换、握手及整帧账，无阻断。下一cachedOS接入仅整理为[NEXT_CACHED_OS_INTERFACE.md](NEXT_CACHED_OS_INTERFACE.md)，本目录不新增重复RTL。

[data独立消费者审阅](../data/REVIEW_CONSUMER.md)已完成：5332项数值范围、状态账与连续几何复核，无阻断；未重跑RTL。
