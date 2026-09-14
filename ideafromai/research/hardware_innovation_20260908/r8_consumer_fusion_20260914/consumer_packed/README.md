# packed14/15 接真实 FP32→I24 完成边界

[REPORT](REPORT.md)、[资源合同](resource_contract.json)、[函数核对](function_contract.json)和[SUMMARY](SUMMARY.json)给出最终范围。核心只读依赖 [packed_r8.sv](../packed_rtl/packed_r8.sv)，本目录复用已验证消费者并实现连续wrapper/TB。旧consumer_rtl完全不改。

在本目录执行（NumPy、Verilator4.028 `--cc --exe` + make，无EDA）：

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 generate_wrapper.py
/opt/anaconda3/bin/python3.12 make_stream_tb.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 verify.py
/opt/anaconda3/bin/python3.12 run_stream.py
/opt/anaconda3/bin/python3.12 run_stream.py --full
/opt/anaconda3/bin/python3.12 verify_stream.py
/opt/anaconda3/bin/python3.12 summarize.py
```

小块数据读取本轮`../data/consumer_first8.npz`和`consumer_coefficients.npz`，连续输入读取`first_source_words.npy / identity_fp32_full.npy / identity_q20_full.npy / raw_p_full.npy / i24_new_full.npy`。身份输入为实际IEEE FP32，不以预量化J代输入；J文件只作诊断oracle。raw p由SV完整K生成，TB只响应原生地址。

配置4Q1、5Q2、6k_live、7a/b，静态1848拍，只在cold一次装入；每tile源/原点/identity/转换重新实付。两个模式共享208bit z口及19×13乘法，不以旧9/11作同面积比较。

结果：[128个小命令](results.json)、[连续64](results_64.json)、[两臂整帧](results_full.json)、[小块检查](checks.json)、[连续检查](stream_checks.json)、[全帧CSV](stream_full_costs.csv)。每模式完整帧是一条go19200tile，未拆行独立实例；完整帧不含背压和warm，二者另在64tile实测。日志在logs/；`adapt.py`只是最初适配记录，不是重复运行步骤。

共同后端已有[root](../REVIEW_CONSUMER_ROOT.md)和[data](../data/REVIEW_CONSUMER.md)独立审阅，但这些旧审阅不自动给新packed核或本次新配置映射背书；新核审阅由本阶段独立作者另交。

本次新证据的独立审阅已完成：[packed核心](../data/REVIEW_PACKED.md)、[消费者新映射与整帧](../data/REVIEW_CONSUMER_PACKED.md)，未发现阻断；独立整图源交集与真实周期差一致。
